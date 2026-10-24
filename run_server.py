#!/usr/bin/env python
"""Start the Skin Retouch HTTP console.

    python run_server.py                     # http://127.0.0.1:8791, no token
    python run_server.py --lan --open        # LAN access, token auto-generated
    python run_server.py --port 9000 --token hunter2

The GUI and the remote API are the same process: the page at `/` drives
`/api/jobs`, and any script can drive the identical endpoints with curl.

A non-loopback bind without a token would hand the GPU to everyone on the
network, so that combination is refused instead of quietly starting an open
service. Pass --token, or let the launcher mint one and print it.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import VERSION  # noqa: E402
from server.api import create_server  # noqa: E402
from server.config import DEFAULT_PREVIEW_MAX_PX, Config, new_token  # noqa: E402


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults, except for the options whose default is 'unset'."""

    def _get_help_string(self, action):
        text = super()._get_help_string(action)
        if action.default in (None, "") and "%(default)" in text:
            # argparse has not interpolated yet; drop the empty default marker
            # and keep whatever the option spelled out for itself.
            return action.help or ""
        return text


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_server.py",
        description="Graphical console and remote API for the Skin Retouch node.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    # 7861 sits inside a Windows reserved range on this machine (7779-7878 is
    # held by Hyper-V/WSL, so binding it fails with WinError 10013 before the
    # app ever starts). 8791 is outside every excluded block here.
    parser.add_argument("--port", type=int, default=8791, help="bind port, 0 picks a free one")
    parser.add_argument("--lan", action="store_true",
                        help="shortcut for --host 0.0.0.0 with a generated token")
    parser.add_argument("--token", default="", help="require this token for every API call")
    parser.add_argument("--no-auth", action="store_true",
                        help="disable the token check (loopback binds only)")
    parser.add_argument("--data-dir", default="", help="where job files live (default: server/data)")
    parser.add_argument("--preview", type=int, default=DEFAULT_PREVIEW_MAX_PX,
                        help="longest edge of the preview images the browser loads")
    parser.add_argument("--ttl", type=int, default=3600, help="seconds a job survives on disk")
    parser.add_argument("--queue", type=int, default=8, help="max jobs waiting behind the worker")
    parser.add_argument("--max-upload-mb", type=int, default=128, help="upload size ceiling")
    parser.add_argument("--repaint-url", default="",
                        help="ComfyUI base URL for the optional repaint pass "
                             "(default: $SKIN_REPAINT_URL or http://127.0.0.1:8188)")
    parser.add_argument("--repaint-autostart", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="keep the repaint engine warm: reuse a ComfyUI that "
                             "already answers on --repaint-url, otherwise start a "
                             "private one (--no-repaint-autostart to opt out)")
    parser.add_argument("--comfyui-dir", default="",
                        help="directory containing ComfyUI's main.py, needed only "
                             "when the console has to start the engine itself")
    parser.add_argument("--comfyui-python", default="",
                        help="interpreter to run ComfyUI with (default: the "
                             "embedded one next to --comfyui-dir)")
    parser.add_argument("--allow-origin", default="",
                        help="echo this Origin in CORS headers (off by default)")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="open the console in the default browser")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    parser.add_argument("--version", action="version", version="skin-retouch server " + VERSION)
    return parser.parse_args(argv)


def resolve_token(args) -> tuple[str, str]:
    """Return (token, note). The note explains where the token came from."""
    if args.no_auth:
        return "", "auth disabled"
    if args.token:
        return args.token, "token from --token"
    host = "0.0.0.0" if args.lan else args.host
    if host in ("127.0.0.1", "localhost", "::1"):
        return "", "loopback only, no token needed"
    return new_token(), "generated token"


def lan_address() -> str:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1, never routed
            return probe.getsockname()[0]
        finally:
            probe.close()
    except OSError:
        return ""


def engine_note(status: dict, autostart: bool) -> str:
    """One banner line answering "can the repaint pass run right now?"."""
    if status.get("available"):
        if status.get("managed"):
            return "standby (private instance, closed when this console exits)"
        return "standby (reusing the instance already running)"
    if not autostart:
        return "idle (--no-repaint-autostart)"
    detail = status.get("detail") or ""
    if detail:
        return "idle: %s" % (detail,)
    return "idle: nothing answering on %s" % (status.get("url", ""),)


def main(argv=None) -> int:
    args = parse_args(argv)
    host = "0.0.0.0" if args.lan else args.host
    token, note = resolve_token(args)

    if args.no_auth and host not in ("127.0.0.1", "localhost", "::1"):
        print("[refused] --no-auth on a non-loopback bind would expose the GPU "
              "to the whole network.", file=sys.stderr)
        print("          Drop --no-auth to get an auto-generated token, or bind 127.0.0.1.",
              file=sys.stderr)
        return 2

    cfg = Config(
        host=host,
        port=args.port,
        token=token,
        allow_origin=args.allow_origin,
        max_upload_bytes=max(1, args.max_upload_mb) * 1024 * 1024,
        job_ttl_s=max(60, args.ttl),
        preview_max_px=max(256, args.preview),
        queue_limit=max(1, args.queue),
        data_dir=args.data_dir,
        repaint_url=args.repaint_url.rstrip("/"),
        repaint_autostart=bool(args.repaint_autostart),
        comfyui_dir=args.comfyui_dir,
        comfyui_python=args.comfyui_python,
        verbose=args.verbose,
    )

    log = (lambda message: print("[server] " + message, flush=True)) if args.verbose else None
    try:
        server = create_server(cfg, log=log)
    except OSError as exc:
        # 10013 means Windows reserved the range (Hyper-V / WSL / Docker do this
        # silently), which reads like a permission bug unless we spell it out.
        reason = "port %d is taken or reserved by the system" % cfg.port
        if getattr(exc, "winerror", None) == 10013:
            reason += " (WinError 10013: pick another --port, e.g. 8791)"
        print("[refused] %s: %s" % (reason, exc), file=sys.stderr)
        return 3
    bound_host, bound_port = server.server_address[:2]

    display_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::", "") else bound_host
    url = "http://%s:%d/" % (display_host, bound_port)
    query = "?token=" + token if token else ""

    def say(line=""):
        # flushed line by line so the banner survives redirection to a log file
        print(line, flush=True)

    say()
    say("  Skin Retouch  v%s" % VERSION)
    say("  console   %s%s" % (url, query))
    if bound_host in ("0.0.0.0", "::"):
        wan = lan_address()
        if wan:
            say("  network   http://%s:%d/%s" % (wan, bound_port, query))
    say("  api       POST %sapi/jobs?strength=0.75  (body: image bytes)" % url)
    say("  schema    GET  %sapi/schema" % url)
    if token:
        say("  token     %s   (%s)" % (token, note))
        say("            the console at ?token=... stores it in a cookie and strips the URL")
    else:
        say("  auth      %s" % note)
    say("  data      %s" % cfg.resolved_data_dir())
    say("  repaint   %s" % server.engine.url)
    status = server.engine.status()
    if cfg.repaint_autostart and not status["available"] and status["spawnable"]:
        say("            warming up: starting a private ComfyUI "
            "(first run loads a 6.9 GB checkpoint)")
        status = server.engine.start()
    say("            %s" % engine_note(status, cfg.repaint_autostart))
    say("  stop      Ctrl+C")
    say()

    if args.open_browser and bound_host in ("127.0.0.1", "localhost", "0.0.0.0", "::"):
        try:
            webbrowser.open(url + query)
        except Exception:  # noqa: BLE001 - a missing browser must not stop the server
            pass

    try:
        server.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        print("\n[server] stopping", flush=True)
    finally:
        server.engine.shutdown()
        server.manager.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
