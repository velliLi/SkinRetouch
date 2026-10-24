"""End-to-end check of the HTTP console and its remote API.

Runs a real server on a random port against a real detector, submits one asset
through the same raw-body upload path the browser uses, and reads the results
back. Everything here goes through HTTP on purpose: the point is to prove the
wire contract, not the pipeline (blemish_test.py already pins that).

    python tests/server_test.py
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from server.api import create_server  # noqa: E402
from server.config import Config, PARAM_KEYS  # noqa: E402

ASSET = os.path.join(HERE, "assets", "t01_headshoulders_135_00001_.png")
TOKEN = "srv-test-token-2f8a"
PREVIEW_PX = 640
failures = []
checks = 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    if ok:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        failures.append(name)


def call(base, path, method="GET", data=None, headers=None, token=TOKEN):
    req = urllib.request.Request(base + path, data=data, method=method)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def payload(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def decode(raw):
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)


def main():
    if not os.path.exists(ASSET):
        print("missing test asset: %s" % ASSET)
        return 2

    workdir = tempfile.mkdtemp(prefix="skin-server-")
    cfg = Config(
        host="127.0.0.1",
        port=0,
        token=TOKEN,
        max_upload_bytes=64 * 1024 * 1024,
        job_ttl_s=3600,
        preview_max_px=PREVIEW_PX,
        queue_limit=4,
        data_dir=workdir,
    )
    server = create_server(cfg, log=lambda message: None)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                              daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % server.server_address[1]
    source = cv2.imread(ASSET, cv2.IMREAD_COLOR)
    with open(ASSET, "rb") as handle:
        blob = handle.read()
    print("server %s  asset %dx%d" % (base, source.shape[1], source.shape[0]))

    try:
        # -- auth surface ---------------------------------------------------
        status, _, raw = call(base, "/api/health", token="")
        body = payload(raw)
        check("health rejects anonymity", status == 200 and body.get("authorized") is False,
              body)
        check("health reports auth_required", body.get("auth_required") is True, body)
        check("health hides caps before auth", "caps" not in body, sorted(body))

        status, _, raw = call(base, "/api/jobs?strength=0.75", method="POST", data=blob,
                              headers={"Content-Type": "image/png"}, token="")
        check("upload without token is 401", status == 401, status)

        status, headers, _ = call(base, "/api/login", method="POST",
                                  data=json.dumps({"token": "wrong"}).encode(),
                                  headers={"Content-Type": "application/json"}, token="")
        check("wrong token is 401", status == 401, status)

        status, headers, _ = call(base, "/api/login", method="POST",
                                  data=json.dumps({"token": TOKEN}).encode(),
                                  headers={"Content-Type": "application/json"}, token="")
        cookie = headers.get("Set-Cookie", "")
        check("login sets an HttpOnly cookie",
              status == 200 and "skin_token=" in cookie and "HttpOnly" in cookie, cookie)

        status, _, raw = call(base, "/api/health")
        body = payload(raw)
        caps = body.get("caps") or {}
        check("health exposes caps when authorized",
              status == 200 and body.get("authorized") is True and "opencv" in caps, caps.keys())
        check("health reports queue stats", "queue" in body and "pending" in body["queue"],
              body.get("queue"))
        check("health reports the repaint engine",
              isinstance((body.get("repaint") or {}).get("available"), bool),
              body.get("repaint"))

        # -- schema and static assets --------------------------------------
        status, _, raw = call(base, "/api/schema")
        schema = payload(raw)
        checks_keys = {spec["key"] for spec in schema.get("params", [])}
        check("schema lists every parameter",
              status == 200 and checks_keys == set(PARAM_KEYS),
              sorted(checks_keys ^ set(PARAM_KEYS)))
        check("schema carries defaults and presets",
              "strength" in (schema.get("defaults") or {}) and len(schema.get("presets") or []) == 3,
              schema.get("presets"))
        check("repaint is off by default",
              (schema.get("defaults") or {}).get("repaint") is False,
              (schema.get("defaults") or {}).get("repaint"))

        import run_server

        args = run_server.parse_args(["--repaint-url", "http://127.0.0.1:9999/"])
        check("--repaint-url is accepted",
              args.repaint_url.rstrip("/") == "http://127.0.0.1:9999", args.repaint_url)

        status, headers, raw = call(base, "/", token="")
        check("index page is served", status == 200 and b"Skin" in raw,
              headers.get("Content-Type"))
        status, _, raw = call(base, "/static/app.css", token="")
        check("css is served", status == 200 and b"--amber" in raw, status)
        status, _, raw = call(base, "/static/app.js", token="")
        check("js is served", status == 200 and b"/api/jobs" in raw, status)
        status, _, raw = call(base, "/static/../server/api.py", token="")
        check("static traversal is refused", status in (400, 404), status)

        # -- upload --------------------------------------------------------
        status, headers, raw = call(base, "/api/jobs?strength=0.7&include_body=0&name="
                                    + urllib.request.quote("测试 图.png"),
                                    method="POST", data=blob,
                                    headers={"Content-Type": "image/png"})
        job = payload(raw)
        job_id = job.get("id", "")
        check("upload accepted", status == 202 and len(job_id) == 16, (status, job.get("error")))
        check("upload returns Location", headers.get("Location") == "/api/jobs/" + job_id,
              headers.get("Location"))
        check("chinese filename survives", (job.get("source") or {}).get("name") == "测试 图.png",
              (job.get("source") or {}).get("name"))
        check("params are echoed", (job.get("params") or {}).get("strength") == 0.7
              and (job.get("params") or {}).get("include_body") is False, job.get("params"))

        status, _, raw = call(base, "/api/jobs?strength=75", method="POST", data=blob,
                              headers={"Content-Type": "image/png"})
        check("out-of-range parameter is 400", status == 400, status)
        status, _, raw = call(base, "/api/jobs?nonsense=1", method="POST", data=blob,
                              headers={"Content-Type": "image/png"})
        check("unknown parameter is 400", status == 400, status)
        status, _, raw = call(base, "/api/jobs", method="POST", data=b"",
                              headers={"Content-Type": "image/png"})
        check("empty body is 400", status == 400, status)
        status, _, raw = call(base, "/api/jobs", method="POST", data=blob,
                              headers={"Content-Type": "text/plain"})
        check("wrong content type is 415", status == 415, status)

        # -- job ids -------------------------------------------------------
        status, _, raw = call(base, "/api/jobs/not-a-job-id")
        check("malformed job id is 400", status == 400, status)
        status, _, raw = call(base, "/api/jobs/0123456789abcdef")
        check("unknown job id is 404", status == 404, status)
        status, _, raw = call(base, "/api/jobs/%s/result?kind=../../../etc/passwd" % job_id)
        check("bad result kind is 400", status == 400, status)

        # -- wait for the worker ------------------------------------------
        deadline = time.time() + 180
        snapshot = job
        while time.time() < deadline:
            status, _, raw = call(base, "/api/jobs/" + job_id)
            snapshot = payload(raw)
            if snapshot.get("status") in ("done", "error", "canceled"):
                break
            time.sleep(0.3)
        check("job reached done", snapshot.get("status") == "done",
              (snapshot.get("status"), snapshot.get("error")))
        if snapshot.get("status") != "done":
            return 1

        image = snapshot.get("image") or {}
        check("image dimensions are reported",
              image.get("width") == source.shape[1] and image.get("height") == source.shape[0],
              image)
        timing = snapshot.get("timing") or {}
        check("timing covers all three phases",
              all(isinstance(timing.get(key), (int, float))
                  for key in ("decode_ms", "pipeline_ms", "encode_ms")), timing)
        check("sha256 is recorded", len(snapshot.get("sha256") or "") == 64,
              snapshot.get("sha256"))
        artifacts = set(snapshot.get("artifacts") or [])
        check("preview artifacts exist",
              {"out", "mask", "preview_out", "preview_mask", "preview_source"}
              <= artifacts, sorted(artifacts))
        check("source artifact is recorded",
              str(snapshot.get("source_artifact") or "").startswith("source"),
              snapshot.get("source_artifact"))
        check("info mentions the face", isinstance(snapshot.get("info"), list)
              and len(snapshot["info"]) >= 1, snapshot.get("info"))
        check("a plain job skips the repaint pass",
              snapshot.get("repaint") == {} and "repaint_ms" not in timing,
              snapshot.get("repaint"))

        # -- the repaint pass is optional and must never fail a job ---------
        status, _, raw = call(base, "/api/jobs?repaint=1&include_body=0", method="POST",
                              data=blob, headers={"Content-Type": "image/png"})
        repaint_job = payload(raw)
        check("repaint job accepted",
              status == 202 and len(repaint_job.get("id", "")) == 16,
              (status, repaint_job.get("error")))
        repaint_snap = repaint_job
        if repaint_job.get("id"):
            deadline = time.time() + 300
            while time.time() < deadline:
                status, _, raw = call(base, "/api/jobs/" + repaint_job["id"])
                repaint_snap = payload(raw)
                if repaint_snap.get("status") in ("done", "error", "canceled"):
                    break
                time.sleep(0.5)
        check("repaint job still finishes", repaint_snap.get("status") == "done",
              (repaint_snap.get("status"), repaint_snap.get("error")))
        engine = repaint_snap.get("repaint") or {}
        if engine.get("available") is False:
            # ComfyUI is not running here: the job keeps the pipeline result and
            # says why, which is the whole point of the degrade path.
            check("missing engine degrades to a warning",
                  bool(repaint_snap.get("warnings")) and bool(engine.get("reason")),
                  (repaint_snap.get("warnings"), engine))
        else:
            check("repaint pass reports its work",
                  isinstance(engine.get("tiles_done"), int)
                  and isinstance((repaint_snap.get("timing") or {}).get("repaint_ms"),
                                 (int, float)), engine)

        # -- results -------------------------------------------------------
        status, headers, raw = call(base, "/api/jobs/%s/result?kind=out" % job_id)
        full = decode(raw)
        check("full result decodes", status == 200 and full is not None
              and full.shape[:2] == source.shape[:2],
              None if full is None else full.shape)
        check("full result is png", headers.get("Content-Type") == "image/png",
              headers.get("Content-Type"))

        status, headers, raw = call(base, "/api/jobs/%s/result?kind=preview_out" % job_id)
        preview = decode(raw)
        check("preview decodes", status == 200 and preview is not None,
              None if preview is None else preview.shape)
        if preview is not None:
            check("preview respects the size cap",
                  max(preview.shape[:2]) == PREVIEW_PX, preview.shape)
            check("preview keeps the aspect ratio",
                  abs(preview.shape[1] / preview.shape[0]
                      - source.shape[1] / source.shape[0]) < 0.02,
                  preview.shape)

        status, _, raw = call(base, "/api/jobs/%s/result?kind=preview_source" % job_id)
        check("preview of the source exists", status == 200 and decode(raw) is not None, status)
        status, _, raw = call(base, "/api/jobs/%s/result?kind=preview_mask" % job_id)
        mask = decode(raw)
        check("mask preview exists", status == 200 and mask is not None, status)

        status, headers, _ = call(base, "/api/jobs/%s/result?kind=out&download=1" % job_id)
        disposition = headers.get("Content-Disposition", "")
        check("download asks for a filename", "attachment" in disposition, disposition)

        status, _, raw = call(base, "/api/jobs/%s/meta" % job_id)
        meta = payload(raw)
        check("meta.json is served", status == 200 and meta.get("id") == job_id, status)

        # -- listing and cancel -------------------------------------------
        status, _, raw = call(base, "/api/jobs?limit=5")
        listing = payload(raw)
        check("job list includes the job",
              status == 200 and any(item["id"] == job_id for item in listing.get("jobs", [])),
              status)
        status, _, raw = call(base, "/api/jobs/" + job_id, method="DELETE")
        canceled = payload(raw)
        check("finished job refuses to be canceled",
              status == 200 and canceled.get("canceled") is False, canceled.get("note"))

        status, headers, raw = call(base, "/api/jobs/%s/events" % job_id)
        check("sse endpoint answers with a stream",
              status == 200 and b"data:" in raw, headers.get("Content-Type"))

        # -- survival across a restart -------------------------------------
        # A fresh manager knows nothing in memory; a job id must still resolve
        # through meta.json on disk, otherwise a reload after a restart 404s.
        second = create_server(cfg, log=lambda message: None)
        second_thread = threading.Thread(target=second.serve_forever,
                                         kwargs={"poll_interval": 0.2}, daemon=True)
        second_thread.start()
        base2 = "http://127.0.0.1:%d" % second.server_address[1]
        try:
            status, _, raw = call(base2, "/api/jobs/" + job_id)
            revived = payload(raw)
            check("finished job survives a restart",
                  status == 200 and revived.get("id") == job_id, status)
            status, _, image = call(base2, "/api/jobs/%s/result?kind=preview_out" % job_id)
            check("result survives a restart", status == 200 and len(image) > 512, status)
            status, _, raw = call(base2, "/api/jobs/" + "0" * 16)
            check("unknown job still 404s after restart", status == 404, status)
        finally:
            second.shutdown()
            second.manager.shutdown()
            second.server_close()
    finally:
        server.shutdown()
        server.manager.shutdown()
        server.server_close()
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print("SERVER FAIL  %d of %d checks failed: %s"
              % (len(failures), checks, ", ".join(failures)))
        return 1
    print("SERVER OK  %d checks" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
