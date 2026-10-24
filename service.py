#!/usr/bin/env python
"""One-touch switch for the Skin Retouch console service.

    python service.py                # toggle: start when stopped, stop when running
    python service.py start          # start in the background, wait until it answers
    python service.py stop           # stop and wait for the port to free up
    python service.py restart
    python service.py status
    python service.py open           # open the console, token included
    python service.py logs           # tail the service log

`run_server.py` stays the only thing that speaks HTTP; this module only
supervises that child process. The pid, port and token live in
`server/data/service.json` (and the banner in `server/data/service.log`), so a
second double-click can stop the exact process this script started instead of
guessing from `netstat` and killing whatever owns the port.

Only the standard library is used: importing the server package would drag in
OpenCV and torch, which costs seconds before the switch even prints a line.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "run_server.py")
DEFAULT_DATA = os.path.join(HERE, "server", "data")
SERVER_HEADER_PREFIX = "SkinRetouch/"
LOOPBACK = ("127.0.0.1", "localhost", "::1", "")
START_TIMEOUT = 40.0
STOP_TIMEOUT = 15.0

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


# --------------------------------------------------------------------------- io


def state_path(data_dir: str) -> str:
    return os.path.join(data_dir, "service.json")


def log_path(data_dir: str) -> str:
    return os.path.join(data_dir, "service.log")


def read_state(data_dir: str) -> dict:
    try:
        with open(state_path(data_dir), "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def write_state(data_dir: str, state: dict) -> None:
    os.makedirs(data_dir, exist_ok=True)
    target = state_path(data_dir)
    temporary = target + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, target)


def clear_state(data_dir: str) -> None:
    try:
        os.remove(state_path(data_dir))
    except OSError:
        pass


def tail(path: str, lines: int = 30) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return ""
    rows = content.splitlines()[-lines:]
    return "\n".join(row.rstrip() for row in rows)


# ------------------------------------------------------------------- processes


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    synchronize = 0x00100000
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return False
    try:
        # WAIT_TIMEOUT (0x102) means the process object is still unsignalled,
        # i.e. still running. 0 means it finished but the handle stayed valid.
        return kernel32.WaitForSingleObject(handle, 0) == 0x102
    finally:
        kernel32.CloseHandle(handle)


def spawn_server(argv: list[str], log) -> subprocess.Popen:
    kwargs = dict(
        cwd=HERE,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        # A detached child has no console, so Python falls back to the locale
        # code page (cp936 here) and the banner lands in the log as GBK. Pin it
        # to UTF-8 so the log stays readable by anything, including us.
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    if os.name == "nt":
        # Detached or the console window would take the server down with it when
        # the double-clicked launcher closes.
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(argv, **kwargs)


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


# ------------------------------------------------------------------- http probe


def probe(base_url: str, token: str = "", timeout: float = 3.0):
    """Return the parsed /api/health body, or None when nothing answers."""
    request = urllib.request.Request(base_url + "api/health")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not (response.headers.get("Server") or "").startswith(SERVER_HEADER_PREFIX):
                return None
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    if not port:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.settimeout(0.4)
        return probe_socket.connect_ex((host, port)) == 0


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
            probe_socket.connect(("192.0.2.1", 9))  # TEST-NET-1, never routed
            return probe_socket.getsockname()[0]
    except OSError:
        return ""


def console_url(host: str, port: int) -> str:
    display = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return "http://%s:%d/" % (display, port)


def with_token(url: str, token: str) -> str:
    return url + ("?token=" + token if token else "")


# ------------------------------------------------------------------ operations


def resolve_data_dir(value: str) -> str:
    return os.path.abspath(value) if value else DEFAULT_DATA


def current(data_dir: str):
    """Return (state, health) for a live service started by this script."""
    state = read_state(data_dir)
    pid = int(state.get("pid") or 0)
    if pid and pid_alive(pid):
        health = probe(console_url(state.get("host", "127.0.0.1"), state.get("port", 0)),
                       state.get("token", ""))
        return state, health
    return {}, None


def describe(health: dict) -> str:
    if not health:
        return "no answer from /api/health"
    parts = ["v" + str(health.get("version", "?"))]
    caps = health.get("caps") or {}
    if caps:
        device = "cuda" if caps.get("cuda") else "cpu"
        gpu = caps.get("gpu")
        parts.append("[%s]" % device)
        if gpu:
            parts.append(gpu)
        queue = health.get("queue") or {}
        parts.append("queue %s/%s" % (queue.get("running", 0), queue.get("pending", 0)))
    elif health.get("auth_required"):
        parts.append("[token required]")
    return "  ".join(parts)


def print_state(state: dict, health) -> None:
    url = console_url(state.get("host", "127.0.0.1"), state.get("port", 0))
    token = state.get("token", "")
    uptime = time.time() - float(state.get("started_at") or time.time())
    print("  console   %s" % with_token(url, token))
    network = state.get("network") or ""
    if network:
        print("  network   %s" % with_token(network, token))
    print("  health    %s" % describe(health))
    print("  pid       %d   up %s" % (int(state.get("pid") or 0), duration(uptime)))
    print("  data      %s" % state.get("data_dir", ""))
    print("  log       %s" % state.get("log", ""))


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "%d s" % seconds
    if seconds < 3600:
        return "%d min %02d s" % (seconds // 60, seconds % 60)
    return "%d h %02d min" % (seconds // 3600, (seconds % 3600) // 60)


def start(args) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    host = "0.0.0.0" if args.lan else args.host
    port = args.port
    token = args.token

    state, health = current(data_dir)
    if state:
        print("  status    already running, nothing to do")
        print_state(state, health)
        if args.open_browser:
            open_console(data_dir)
        return 0

    # A leftover state file can point at a pid the OS already recycled.
    if read_state(data_dir):
        clear_state(data_dir)

    if args.no_auth and host not in LOOPBACK:
        print("  refused   --no-auth on %s would expose the GPU to the whole network" % host)
        print("            drop --no-auth to get a generated token, or bind 127.0.0.1")
        return 2
    if host not in LOOPBACK and not token and not args.no_auth:
        token = secrets.token_urlsafe(18)
    if args.no_auth:
        token = ""

    if port and port_in_use(port):
        print("  refused   port %d is already taken; pass --port to pick another" % port)
        return 3
    if not port:
        print("  refused   --port 0 cannot be tracked; give the switch a fixed port")
        return 3

    argv = [sys.executable, LAUNCHER, "--host", host, "--port", str(port)]
    if token:
        argv += ["--token", token]
    if args.no_auth:
        argv += ["--no-auth"]
    if args.verbose:
        argv += ["--verbose"]
    if args.data_dir:
        argv += ["--data-dir", data_dir]

    os.makedirs(data_dir, exist_ok=True)
    logfile = log_path(data_dir)
    # One rotation is enough: the log only holds banners and request tracing,
    # and an unbounded append would grow forever on a machine left running.
    try:
        if os.path.getsize(logfile) > 2 * 1024 * 1024:
            os.replace(logfile, logfile + ".1")
    except OSError:
        pass
    with open(logfile, "ab", buffering=0) as log:
        log.write(("\n==== start %s ====\n" % time.strftime("%Y-%m-%d %H:%M:%S")).encode())
        process = spawn_server(argv, log)

    print("  status    starting (pid %d)" % process.pid)
    deadline = time.time() + START_TIMEOUT
    url = console_url(host, port)
    health = None
    while time.time() < deadline:
        if process.poll() is not None:
            print("  failed    the launcher exited with code %d" % process.returncode)
            print(indent(tail(logfile, 14)))
            return 4
        health = probe(url, token, timeout=1.5)
        if health is not None:
            break
        time.sleep(0.25)

    if health is None:
        print("  failed    no answer on %s after %.0f s; the pid is left running" % (url, START_TIMEOUT))
        print(indent(tail(logfile, 14)))
        return 5

    # run_server may have bound port 0 to a free one; trust the banner.
    network = ""
    if host in ("0.0.0.0", "::"):
        address = lan_address()
        if address:
            network = "http://%s:%d/" % (address, port)
    state = {
        "pid": process.pid,
        "host": host,
        "port": port,
        "token": token,
        "network": network,
        "data_dir": data_dir,
        "log": logfile,
        "started_at": time.time(),
        "argv": argv[1:],
    }
    write_state(data_dir, state)
    print("  status    running")
    print_state(state, health)
    if args.open_browser:
        open_console(data_dir)
    return 0


def stop(args) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    state, health = current(data_dir)

    if not state:
        stale = read_state(data_dir)
        clear_state(data_dir)
        if stale:
            print("  status    not running (cleared a stale pid file)")
        else:
            print("  status    not running")
        leftover_port = int(stale.get("port") or args.port or 0)
        if leftover_port and port_in_use(leftover_port):
            print("  note      port %d is still taken by another process" % leftover_port)
            print("            it was not started by this switch, so its pid is unknown:")
            print("            stop it from its own console, or find it with netstat -ano")
        return 0

    pid = int(state["pid"])
    if not args.yes:
        print("  warning   stopping the service aborts any job still running")
        print("            Ctrl+C to cancel, stopping in ", end="", flush=True)
        try:
            for remaining in (3, 2, 1):
                print("%d " % remaining, end="", flush=True)
                time.sleep(1)
            print()
        except KeyboardInterrupt:
            print("\n  status    cancelled, still running")
            return 0

    print("  status    stopping (pid %d)" % pid)
    kill_tree(pid)
    deadline = time.time() + STOP_TIMEOUT
    while time.time() < deadline:
        if not pid_alive(pid):
            break
        time.sleep(0.2)

    if pid_alive(pid):
        print("  failed    pid %d survived the kill; stop it manually" % pid)
        return 6

    port = int(state.get("port") or 0)
    while time.time() < deadline and port and port_in_use(port):
        time.sleep(0.2)

    clear_state(data_dir)
    print("  status    stopped")
    if port:
        state_word = "free" if not port_in_use(port) else "still held"
        print("  port      %d %s" % (port, state_word))
    return 0


def restart(args) -> int:
    args.yes = True  # the user already said restart; asking twice is noise
    code = stop(args)
    if code != 0:
        return code
    time.sleep(0.4)
    return start(args)


def status(args) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    state, health = current(data_dir)
    if state:
        print("  status    running")
        print_state(state, health)
        return 0
    stale = read_state(data_dir)
    print("  status    stopped")
    if stale:
        print("  note      leftover pid file from pid %s, already gone" % stale.get("pid"))
    port = int(stale.get("port") or args.port or 0)
    if port and port_in_use(port):
        print("  note      port %d is held by another process" % port)
    print("  start     python service.py start")
    return 1


def open_console(data_dir: str) -> int:
    state, _health = current(data_dir)
    if not state:
        print("  status    not running, nothing to open")
        return 1
    url = with_token(console_url(state.get("host", "127.0.0.1"), state.get("port", 0)),
                     state.get("token", ""))
    print("  open      %s" % url)
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - a missing browser must not fail the command
        pass
    return 0


def logs(args) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    path = log_path(data_dir)
    body = tail(path, args.lines)
    if not body:
        print("  empty     no log yet at %s" % path)
        return 1
    print(indent(body))
    return 0


def toggle(args) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    state, _health = current(data_dir)
    if state:
        return stop(args)
    return start(args)


def indent(text: str, prefix: str = "            ") -> str:
    return "\n".join(prefix + row for row in text.splitlines())


# -------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="service.py",
        description="One-touch switch for the Skin Retouch console service.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default="toggle",
                        choices=["toggle", "start", "stop", "restart", "status", "open", "logs"],
                        help="toggle when omitted")
    parser.add_argument("--port", type=int, default=8791, help="port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--lan", action="store_true",
                        help="bind 0.0.0.0 and mint a token for the network")
    parser.add_argument("--token", default="", help="use this token instead of generating one")
    parser.add_argument("--no-auth", action="store_true",
                        help="disable the token check (loopback binds only)")
    parser.add_argument("--data-dir", default="", help="where state, log and jobs live")
    parser.add_argument("--no-open", action="store_false", dest="open_browser",
                        help="do not open the browser after starting")
    parser.add_argument("--yes", "-y", action="store_true", help="skip the stop confirmation")
    parser.add_argument("--verbose", action="store_true", help="pass --verbose to the server")
    parser.add_argument("--lines", type=int, default=30, help="log lines for the logs command")
    parser.set_defaults(open_browser=True)
    return parser


def main(argv=None) -> int:
    # Logs can carry bytes the console code page cannot represent (a GBK tail
    # from an older run, a stray replacement char). Reporting a status must
    # never die on that, so degrade instead of raising.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    if args.command == "open":
        return open_console(resolve_data_dir(args.data_dir))
    handlers = {
        "toggle": toggle,
        "start": start,
        "stop": stop,
        "restart": restart,
        "status": status,
        "logs": logs,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
