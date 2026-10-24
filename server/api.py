"""HTTP surface: static web UI, JSON API, SSE progress and result downloads.

Built on `http.server` only. The project rule is no new dependencies, and the
whole surface is a handful of endpoints -- a framework would add an install step
to a checkout that currently runs on `cv2` + `torch` alone.

Three upload body shapes are accepted for the same endpoint, because three kinds
of client exist: the browser (raw body, parameters in the query), a shell script
(`multipart/form-data`, the way `curl -F` wants it), and a Python caller
(base64 JSON, which survives being embedded in another request body).
"""

from __future__ import annotations

import base64
import hmac
import json
import mimetypes
import os
import re
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit

from . import config, repaint
from .config import BadParams, Config, JOB_ID_RE, VERSION, coerce_params, parse_query_params
from .engine import Engine
from .jobs import JobManager, JobNotFound, QueueFull
from .store import JobStore, StoreError

WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")

# kind -> (artifact key, fallback content type, disposition)
RESULT_KINDS = {
    "out": ("out", "image/png", "attachment"),
    "mask": ("mask", "image/png", "inline"),
    "source": ("source", "application/octet-stream", "attachment"),
    "preview_out": ("preview_out", "image/jpeg", "inline"),
    "preview_mask": ("preview_mask", "image/png", "inline"),
    "preview_source": ("preview_source", "image/jpeg", "inline"),
}

# artifact key -> file name; meta.json stores keys only, so a job recovered from
# disk needs the mapping back. Keep in sync with JobManager._encode().
_ARTIFACT_FILES = {
    "out": "out.png",
    "mask": "mask.png",
    "preview_out": "out.preview.jpg",
    "preview_mask": "mask.preview.png",
    "preview_source": "source.preview.jpg",
}

_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".webmanifest": "application/manifest+json",
}

_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^;]+))', re.I)
_DISPO_NAME_RE = re.compile(r'name="([^"]*)"', re.I)
_DISPO_FILE_RE = re.compile(r'filename="([^"]*)"', re.I)


class HttpError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}


def _parse_cookies(header: str) -> dict:
    out = {}
    for piece in (header or "").split(";"):
        key, sep, value = piece.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _safe_download_name(source_name: str, suffix: str, ext: str) -> str:
    stem = os.path.splitext(os.path.basename(source_name or "image"))[0]
    stem = re.sub(r'[^\w\-. ()\u4e00-\u9fff]', "_", stem).strip(" .") or "image"
    return "%s%s%s" % (stem[:80], suffix, ext)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SkinRetouch/" + VERSION
    sys_version = ""
    # A stalled client must not pin a worker thread for ever.
    timeout = 60

    # -- plumbing --------------------------------------------------------

    @property
    def cfg(self) -> Config:
        return self.server.cfg  # type: ignore[attr-defined]

    @property
    def manager(self) -> JobManager:
        return self.server.manager  # type: ignore[attr-defined]

    @property
    def store(self) -> JobStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # noqa: A003 - base class name
        if self.cfg.verbose:
            self.server.log("%s %s" % (self.address_string(), fmt % args))  # type: ignore[attr-defined]

    def handle_one_request(self):
        # A client that walks away mid-response (a probed health check, a closed
        # tab, a killed curl) is routine, not an error. Without this the socket
        # error escapes into socketserver's handle_error and prints a traceback
        # for every such disconnect, which buries the real failures in the log.
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, TimeoutError):
            self.close_connection = True

    def _send(self, status, body=b"", ctype="application/octet-stream", headers=None,
              close=False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, payload, headers=None, close=False):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        extra = {"Cache-Control": "no-store"}
        extra.update(headers or {})
        self._send(status, body, "application/json; charset=utf-8", extra, close=close)

    def _json_error(self, status, code, message, headers=None):
        # A rejected POST may still have its body on the wire. Answering without
        # reading it makes the OS reset the connection while the client is still
        # writing, so a plain 401 arrives as a network error instead. Swallow the
        # body first, then reply.
        self._drain_body()
        self._json(status, {"error": {"code": code, "message": message}}, headers=headers)

    # -- authentication --------------------------------------------------

    def _presented_tokens(self, query) -> list[str]:
        found = []
        auth = self.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            found.append(auth[7:].strip())
        header = self.headers.get("X-Auth-Token")
        if header:
            found.append(header.strip())
        if query:
            for value in query.get("token", []):
                found.append(value)
        cookie = _parse_cookies(self.headers.get("Cookie", "")).get("skin_token")
        if cookie:
            found.append(cookie)
        return [t for t in found if t]

    def _authorized(self, query) -> bool:
        expected = self.cfg.token
        if not expected:
            return True
        for candidate in self._presented_tokens(query):
            # compare_digest on equal-length byte strings; encode first so a
            # non-ASCII token cannot raise inside the comparison.
            if hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8")):
                return True
        return False

    def _unauthorized(self, message="a token is required"):
        return self._json_error(401, "unauthorized", message,
                                headers={"WWW-Authenticate": 'Bearer realm="skin-retouch"'})

    def _check_origin(self):
        """Reject cross-site form posts; SameSite=Lax is not enough on its own."""
        origin = self.headers.get("Origin")
        if not origin:
            return
        host = (self.headers.get("Host") or "").strip()
        if self.cfg.allow_origin:
            allowed = [o.strip() for o in self.cfg.allow_origin.split(",") if o.strip()]
            if origin in allowed:
                return
        if origin in ("http://" + host, "https://" + host):
            return
        raise HttpError(403, "bad_origin", "cross-origin request refused")

    # -- request body ----------------------------------------------------

    def _read_body(self) -> bytes:
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            raise HttpError(411, "length_required",
                            "send Content-Length; chunked bodies are not supported")
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise HttpError(411, "length_required", "Content-Length is required")
        try:
            length = int(raw)
        except ValueError:
            raise HttpError(400, "bad_length", "Content-Length is not a number") from None
        if length < 0:
            raise HttpError(400, "bad_length", "negative Content-Length")
        if length > self.cfg.max_upload_bytes:
            raise HttpError(413, "too_large",
                            "body is %d bytes; the limit is %d" % (length, self.cfg.max_upload_bytes),
                            headers={"Connection": "close"})
        data = self.rfile.read(length)
        self._body_read = True
        if len(data) != length:
            raise HttpError(400, "truncated", "request body ended early")
        return data

    def _drain_body(self):
        """Consume an unread request body so the reply is not lost to a reset."""
        if getattr(self, "_body_read", False):
            return
        self._body_read = True
        if self.command not in ("POST", "PUT", "PATCH", "DELETE"):
            return
        if (self.headers.get("Transfer-Encoding") or "").lower():
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        if length > self.cfg.max_upload_bytes:
            # Already refused as 413; do not spend bandwidth swallowing it.
            self.close_connection = True
            return
        budget = max(length, 0)
        while budget > 0:
            try:
                chunk = self.rfile.read(min(65536, budget))
            except (OSError, ValueError):
                return
            if not chunk:
                return
            budget -= len(chunk)

    def _multipart_fields(self) -> dict:
        ctype = self.headers.get("Content-Type") or ""
        match = _BOUNDARY_RE.search(ctype)
        if not match:
            raise HttpError(400, "bad_multipart", "multipart body without a boundary")
        boundary = (match.group(1) or match.group(2)).strip().encode("utf-8")
        body = self._read_body()
        fields = {}
        for chunk in body.split(b"--" + boundary)[1:]:
            if chunk[:2] == b"--":
                break
            if chunk.startswith(b"\r\n"):
                chunk = chunk[2:]
            head, sep, rest = chunk.partition(b"\r\n\r\n")
            if not sep:
                continue
            if rest.endswith(b"\r\n"):
                rest = rest[:-2]
            headers = {}
            for line in head.split(b"\r\n"):
                key, colon, value = line.partition(b":")
                if colon:
                    headers[key.strip().lower().decode("latin-1")] = value.strip().decode("latin-1")
            disposition = headers.get("content-disposition", "")
            name_match = _DISPO_NAME_RE.search(disposition)
            if not name_match:
                continue
            file_match = _DISPO_FILE_RE.search(disposition)
            fields[name_match.group(1)] = {
                "filename": file_match.group(1) if file_match else None,
                "ctype": headers.get("content-type", ""),
                "data": rest,
            }
        return fields

    def _extract_upload(self, query) -> tuple[bytes, str, dict]:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        params = parse_query_params(query)

        if ctype in ("application/json", "text/json"):
            raw = self._read_body()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HttpError(400, "bad_json", "body is not valid JSON: %s" % (exc,)) from None
            if not isinstance(payload, dict):
                raise HttpError(400, "bad_json", "JSON body must be an object")
            blob = payload.get("image_base64") or payload.get("image")
            if not blob:
                raise HttpError(400, "no_image",
                                "JSON body needs 'image_base64' (raw base64 or a data: URL)")
            if isinstance(blob, str) and blob.startswith("data:"):
                blob = blob.split(",", 1)[-1]
            try:
                data = base64.b64decode(blob, validate=False)
            except Exception as exc:  # noqa: BLE001 - b64decode raises binascii.Error
                raise HttpError(400, "bad_base64", "image_base64 is not valid base64: %s"
                                % (exc,)) from None
            params.update(payload.get("params") or {})
            name = payload.get("name") or "upload"
            return data, str(name), params

        if ctype == "multipart/form-data":
            fields = self._multipart_fields()
            chosen = None
            for key in ("image", "file", "photo"):
                if key in fields and fields[key]["filename"] is not None:
                    chosen = fields[key]
                    break
            if chosen is None:
                for field in fields.values():
                    if field["filename"] is not None:
                        chosen = field
                        break
            if chosen is None:
                raise HttpError(400, "no_image",
                                "multipart body has no file field (expected 'image')")
            if "params" in fields and fields["params"]["filename"] is None:
                try:
                    extra = json.loads(fields["params"]["data"].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise HttpError(400, "bad_params_json",
                                    "the 'params' field is not valid JSON: %s" % (exc,)) from None
                if isinstance(extra, dict):
                    params.update(extra)
            return chosen["data"], chosen["filename"] or "upload", params

        if ctype.startswith("image/") or ctype in ("application/octet-stream", ""):
            data = self._read_body()
            name = (query.get("name") or ["upload"])[0]
            return data, str(name), params

        raise HttpError(415, "unsupported_type",
                        "send the image as image/*, multipart/form-data or application/json")

    # -- routing ---------------------------------------------------------

    def do_OPTIONS(self):  # noqa: N802 - base class contract
        if not self.cfg.allow_origin:
            return self._json_error(405, "method_not_allowed", "CORS is not enabled")
        self._send(204, b"", "text/plain", {
            "Access-Control-Allow-Origin": self.cfg.allow_origin,
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Auth-Token",
            "Access-Control-Max-Age": "600",
        })

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self):  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str):
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        try:
            if method == "POST":
                self._check_origin()
            if path in ("/", "/index.html"):
                return self._serve_static("index.html")
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            if path == "/favicon.ico":
                return self._serve_static("favicon.svg")
            if not path.startswith("/api/"):
                raise HttpError(404, "not_found", "no such endpoint: %s" % (path,))
            return self._handle_api(method, path, query)
        except HttpError as exc:
            return self._json_error(exc.status, exc.code, exc.message, headers=exc.headers)
        except BadParams as exc:
            return self._json_error(400, "bad_params", str(exc))
        except JobNotFound:
            return self._json_error(404, "job_not_found", "unknown job id")
        except StoreError as exc:
            return self._json_error(400, "bad_request", str(exc))
        except QueueFull as exc:
            return self._json_error(429, "queue_full", str(exc))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The peer left mid-response. Windows reports that as
            # ConnectionAbortedError (10053), not ConnectionResetError, so it
            # has to be named here or it gets logged as an internal error.
            self.close_connection = True
            return
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the server
            self.server.log("unhandled error on %s %s: %s\n%s"  # type: ignore[attr-defined]
                            % (method, path, exc, traceback.format_exc()))
            return self._json_error(500, "internal", "%s: %s" % (type(exc).__name__, exc))

    def _handle_api(self, method: str, path: str, query):
        tail = [segment for segment in path.split("/") if segment][1:]  # drop 'api'

        if tail == ["health"] and method == "GET":
            return self._health(query)
        if tail == ["schema"] and method == "GET":
            return self._json(200, self.cfg.serve_schema())
        if tail == ["login"] and method == "POST":
            return self._login()
        if tail == ["repaint", "engine"]:
            if method == "GET":
                return self._json(200, self.server.engine.status())  # type: ignore[attr-defined]
            if method == "POST":
                return self._engine_command()

        if not tail or tail[0] != "jobs":
            raise HttpError(404, "not_found", "no such endpoint: %s" % (path,))

        if len(tail) == 1:
            if not self._authorized(query):
                return self._unauthorized()
            if method == "POST":
                return self._create_job(query)
            if method == "GET":
                return self._list_jobs(query)
            raise HttpError(405, "method_not_allowed", "use GET or POST here")

        job_id = tail[1]
        if not JOB_ID_RE.match(job_id):
            raise HttpError(400, "bad_job_id", "job ids are 16 lowercase hex characters")
        if not self._authorized(query):
            return self._unauthorized()

        if len(tail) == 2:
            if method == "GET":
                return self._job_snapshot(job_id)
            if method == "DELETE":
                return self._cancel(job_id)
            raise HttpError(405, "method_not_allowed", "use GET or DELETE here")

        action = tail[2]
        if action == "events" and method == "GET":
            return self._events(job_id, query)
        if action == "result" and method == "GET":
            return self._result(job_id, query)
        if action == "meta" and method == "GET":
            return self._meta(job_id)
        raise HttpError(404, "not_found", "no such job action: %s" % (action,))

    # -- endpoints -------------------------------------------------------

    def _health(self, query):
        authorized = self._authorized(query)
        payload = {
            "ok": True,
            "version": VERSION,
            "auth_required": bool(self.cfg.token),
            "authorized": authorized,
        }
        if authorized:
            payload.update({
                "caps": self.server.caps(),  # type: ignore[attr-defined]
                "queue": self.manager.stats(),
                "repaint": self.server.repaint_probe(),  # type: ignore[attr-defined]
                "uptime_s": round(time.time() - self.server.started_at, 1),  # type: ignore[attr-defined]
                "preview_max_px": self.cfg.preview_max_px,
                "limits": {
                    "upload_bytes": self.cfg.max_upload_bytes,
                    "megapixels": self.cfg.max_pixels // 1_000_000,
                    "queue": self.cfg.queue_limit,
                    "job_ttl_s": self.cfg.job_ttl_s,
                },
            })
        self._json(200, payload)

    def _engine_command(self):
        """Start or stop the local ComfyUI behind the repaint pass.

        `start` returns immediately with state `starting`; the page polls
        `GET /api/repaint/engine` because loading SDXL takes tens of seconds and
        holding the request open would just look like a hang.
        """
        raw = self._read_body()
        payload = {}
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise BadParams('body must be JSON like {"action": "start"}')
            if not isinstance(payload, dict):
                raise BadParams('body must be a JSON object like {"action": "start"}')
        action = str(payload.get("action") or "").strip().lower()
        if action == "start":
            return self._json(200, self.server.engine.start(wait=False))  # type: ignore[attr-defined]
        if action == "stop":
            return self._json(200, self.server.engine.stop())  # type: ignore[attr-defined]
        raise BadParams("action must be 'start' or 'stop'")

    def _login(self):
        if not self.cfg.token:
            return self._json(200, {"ok": True, "auth_required": False})
        client = self.address_string()
        now = time.time()
        attempts, blocked_until = self.server.login_attempts.get(client, (0, 0.0))  # type: ignore[attr-defined]
        if now < blocked_until:
            raise HttpError(429, "throttled",
                            "too many failed attempts; retry in %d s" % (blocked_until - now))
        raw = self._read_body()
        try:
            payload = json.loads(raw.decode("utf-8"))
            token = str(payload.get("token") or "")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            token = ""
        if not token or not hmac.compare_digest(token.encode("utf-8"), self.cfg.token.encode("utf-8")):
            attempts += 1
            blocked = now + 60.0 if attempts >= 5 else 0.0
            self.server.login_attempts[client] = (0 if attempts >= 5 else attempts, blocked)  # type: ignore[attr-defined]
            raise HttpError(401, "bad_token", "that token does not match")
        self.server.login_attempts.pop(client, None)  # type: ignore[attr-defined]
        cookie = "skin_token=%s; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800" % (quote(token),)
        self._json(200, {"ok": True}, headers={"Set-Cookie": cookie})

    def _create_job(self, query):
        # parse_query_params drops anything it does not recognise, so a
        # misspelled knob (strengh=0.7) would otherwise run with the default and
        # hide its own bug -- the exact failure coerce_params refuses to produce
        # for body-supplied values. Reject it here too.
        unknown = sorted(set(query) - set(config.PARAM_KEYS) - {"name", "token"})
        if unknown:
            raise HttpError(400, "unknown_parameter",
                            "unknown parameter(s): %s" % ", ".join(unknown))
        data, name, params_raw = self._extract_upload(query)
        if not data:
            raise HttpError(400, "no_image", "the image payload is empty")
        params = coerce_params(params_raw)
        job = self.manager.submit(params, data, os.path.basename(name)[:120] or "upload")
        self._json(202, self.manager.snapshot(job),
                   headers={"Location": "/api/jobs/" + job.id})

    def _list_jobs(self, query):
        try:
            limit = int((query.get("limit") or ["12"])[0])
        except ValueError:
            limit = 12
        limit = max(1, min(60, limit))
        jobs = [self.manager.snapshot(job) for job in self.manager.list(limit)]
        self._json(200, {"jobs": jobs, "queue": self.manager.stats()})

    def _stored_meta(self, job_id):
        """A finished job survives a restart: meta.json holds the same snapshot."""
        try:
            raw = self.store.read(job_id, "meta.json")
        except (OSError, StoreError):
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _job_snapshot(self, job_id):
        try:
            job = self.manager.get(job_id)
        except JobNotFound:
            meta = self._stored_meta(job_id)
            if meta is None:
                raise
            return self._json(200, meta)
        self._json(200, self.manager.snapshot(job))

    def _meta(self, job_id):
        try:
            job = self.manager.get(job_id)
        except JobNotFound:
            try:
                raw = self.store.read(job_id, "meta.json")
            except (OSError, StoreError):
                raise JobNotFound(job_id) from None
            return self._send(200, raw, "application/json; charset=utf-8",
                              {"Cache-Control": "no-store"})
        try:
            raw = self.store.read(job.id, "meta.json")
            return self._send(200, raw, "application/json; charset=utf-8",
                              {"Cache-Control": "no-store"})
        except OSError:
            return self._json(200, self.manager.snapshot(job))

    def _cancel(self, job_id):
        canceled, note = self.manager.cancel(job_id)
        self._json(200, {"canceled": canceled, "note": note,
                         "job": self.manager.snapshot(self.manager.get(job_id))})

    def _events(self, job_id, query):
        job = self.manager.get(job_id)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        deadline = time.time() + 900.0
        last_push = 0.0
        try:
            while True:
                snapshot = self.manager.snapshot(job)
                now = time.time()
                finished = snapshot["status"] in ("done", "error", "canceled")
                if finished or now - last_push >= 1.0:
                    frame = "data: %s\n\n" % json.dumps(snapshot, ensure_ascii=False)
                    self.wfile.write(frame.encode("utf-8"))
                    self.wfile.flush()
                    last_push = now
                if finished:
                    self.wfile.write(b"event: done\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                if now > deadline:
                    self.wfile.write(b"event: timeout\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _result(self, job_id, query):
        kind = (query.get("kind") or ["out"])[0]
        if kind not in RESULT_KINDS:
            raise HttpError(400, "bad_kind",
                            "kind must be one of: " + ", ".join(sorted(RESULT_KINDS)))
        artifact_key, fallback_type, disposition = RESULT_KINDS[kind]
        try:
            job = self.manager.get(job_id)
            artifacts, source_name = job.artifacts, job.source_name
        except JobNotFound:
            meta = self._stored_meta(job_id)
            if meta is None:
                raise
            artifacts = {key: _ARTIFACT_FILES[key] for key in (meta.get("artifacts") or [])
                         if key in _ARTIFACT_FILES}
            if meta.get("source_artifact"):
                artifacts["source"] = meta["source_artifact"]
            source_name = (meta.get("source") or {}).get("name") or "image"
        name = artifacts.get(artifact_key)
        if not name:
            raise HttpError(409, "not_ready", "no %s artifact for this job yet" % (kind,))
        try:
            data = self.store.read(job_id, name)
        except OSError:
            raise HttpError(410, "expired",
                            "this job's files have been cleaned up; run it again") from None
        ctype = _STATIC_TYPES.get(os.path.splitext(name)[1].lower()) or fallback_type
        if kind == "source":
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        suffix = {"out": "_skin_retouch", "mask": "_mask", "source": "_source"}.get(kind, "_preview")
        filename = _safe_download_name(source_name, suffix, os.path.splitext(name)[1])
        headers = {"Cache-Control": "private, max-age=600"}
        if disposition == "attachment" or (query.get("download") or [""])[0] in ("1", "true"):
            headers["Content-Disposition"] = "attachment; filename*=UTF-8''%s" % (quote(filename),)
        self._send(200, data, ctype, headers)

    def _serve_static(self, relative: str):
        name = relative.strip("/")
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HttpError(404, "not_found", "no such file")
        path = os.path.abspath(os.path.join(WEBUI_DIR, name))
        if os.path.commonpath([path, WEBUI_DIR]) != WEBUI_DIR or not os.path.isfile(path):
            raise HttpError(404, "not_found", "no such file")
        with open(path, "rb") as fh:
            data = fh.read()
        ctype = _STATIC_TYPES.get(os.path.splitext(name)[1].lower()) or "application/octet-stream"
        # no-store on the shell: editing app.js during development must not need
        # a hard refresh, and there is no build step to bust the cache for us.
        self._send(200, data, ctype, {"Cache-Control": "no-store"})


class SkinHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, cfg: Config, manager: JobManager, store: JobStore, log,
                 engine: Engine | None = None):
        self.cfg = cfg
        self.manager = manager
        self.store = store
        self.log = log
        self.engine = engine or Engine(url=cfg.repaint_url or repaint.DEFAULT_URL,
                                       data_dir=cfg.resolved_data_dir())
        self.started_at = time.time()
        self.login_attempts = {}
        self._caps = None
        self._repaint_probe = {"at": 0.0, "status": {}}
        super().__init__(address, Handler)

    def repaint_probe(self) -> dict:
        """Reachability of the optional repaint engine, cached for a few seconds.

        Deliberately not part of `caps()`: that cache never expires, while
        ComfyUI is something the user starts and stops during a session. The
        timeout is short because this rides on the health endpoint the page
        polls.
        """
        now = time.time()
        if now - self._repaint_probe["at"] > 3.0:
            self._repaint_probe = {"at": now, "status": self.engine.status()}
        return dict(self._repaint_probe["status"])

    def caps(self) -> dict:
        """Environment facts the UI shows in the footer; probed once."""
        if self._caps is not None:
            return self._caps
        caps = {"device_default": "auto", "cuda": False, "torch": "", "opencv": "", "models": {}}
        try:
            import cv2
            import torch

            caps["opencv"] = cv2.__version__
            caps["torch"] = torch.__version__
            caps["cuda"] = bool(torch.cuda.is_available())
            if caps["cuda"]:
                caps["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - diagnostics must never break the endpoint
            pass
        try:
            import skin_core

            # Reading the module cache is the only way to tell whether the
            # 50 MB detector is resident without triggering a load; the public
            # loader has no "is loaded" query.
            caps["models"]["face"] = getattr(skin_core, "_FACE_MODEL", None) is not None
        except Exception:  # noqa: BLE001
            pass
        try:
            import skin_body

            caps["models"]["person"] = getattr(skin_body, "_PERSON_MODEL", None) is not None
        except Exception:  # noqa: BLE001
            caps["models"]["person"] = False
        self._caps = caps
        return caps


def create_server(cfg: Config, log=None):
    log = log or (lambda message: print(message, flush=True))
    store = JobStore(cfg.resolved_data_dir())
    manager = JobManager(cfg, store, log)
    engine = Engine(
        url=cfg.repaint_url or repaint.DEFAULT_URL,
        comfyui_dir=cfg.comfyui_dir,
        python=cfg.comfyui_python,
        autostart=cfg.repaint_autostart,
        data_dir=cfg.resolved_data_dir(),
    )
    return SkinHTTPServer((cfg.host, cfg.port), cfg, manager, store, log, engine)
