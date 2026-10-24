#!/usr/bin/env python
"""End-to-end check for the one-touch switch (service.py).

    python tests/service_test.py

Starts a real service on a scratch data directory, drives it through
start/status/toggle/stop with the same code path the double-clicked .bat uses,
and asserts the things a switch has to get right: it comes up, it is
idempotent, stopping frees the port, the pid file never lies, and a job
submitted before a restart is still readable after one.

No image is processed here on purpose: tests/server_test.py already covers the
pipeline, and loading torch twice would double the runtime of both.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE = os.path.join(ROOT, "service.py")
sys.path.insert(0, ROOT)

import service  # noqa: E402  the module under test

CHECKS = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError("%s%s" % (label, (": " + detail) if detail else ""))
    print("  ok   %s" % label)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(*args: str) -> subprocess.CompletedProcess:
    # --no-repaint-autostart on every call: the switch would otherwise spawn a
    # real ComfyUI and load a 6.9 GB checkpoint just to prove it can start a
    # service, which is minutes of test time for a feature covered by
    # server_test.py and tests/probe_engine.py.
    return subprocess.run(
        [sys.executable, SERVICE, *[str(item) for item in args], "--no-repaint-autostart"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        check=False,
    )


def http_json(url: str, token: str = ""):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    port = free_port()
    workdir = tempfile.mkdtemp(prefix="skin-service-")
    data_dir = os.path.join(workdir, "data")
    state_file = os.path.join(data_dir, "service.json")
    print("switch test on port %d, data %s" % (port, data_dir))

    try:
        stopped = run("status", "--port", str(port), "--data-dir", data_dir)
        check("status on a fresh data dir reports stopped", stopped.returncode == 1,
              stopped.stdout + stopped.stderr)
        check("status on a fresh data dir says stopped", "stopped" in stopped.stdout,
              stopped.stdout)

        started = run("start", "--port", str(port), "--data-dir", data_dir, "--no-open")
        check("start exits 0", started.returncode == 0, started.stdout + started.stderr)
        check("start reports running", "running" in started.stdout, started.stdout)
        check("start writes the pid file", os.path.exists(state_file))

        with open(state_file, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        pid = int(state["pid"])
        check("recorded pid is alive", service.pid_alive(pid))
        check("recorded port matches", int(state["port"]) == port, str(state))

        health = http_json("http://127.0.0.1:%d/api/health" % port)
        check("health answers after start", health.get("ok") is True, str(health))
        check("health reports the version", str(health.get("version", "")).startswith("1."),
              str(health.get("version")))

        twice = run("start", "--port", str(port), "--data-dir", data_dir, "--no-open")
        check("starting twice is a no-op", twice.returncode == 0, twice.stdout)
        check("no-op start says already running", "already running" in twice.stdout,
              twice.stdout)
        with open(state_file, "r", encoding="utf-8") as handle:
            again = json.load(handle)
        check("no-op start keeps the same pid", int(again["pid"]) == pid,
              "%s vs %s" % (again["pid"], pid))

        busy = run("start", "--port", port, "--data-dir", os.path.join(workdir, "other"),
                   "--no-open")
        check("a second data dir refuses the busy port", busy.returncode == 3,
              busy.stdout + busy.stderr)

        running = run("status", "--port", str(port), "--data-dir", data_dir)
        check("status exits 0 while running", running.returncode == 0, running.stdout)
        check("status shows the console url", "console" in running.stdout, running.stdout)
        check("status shows the health line", "health" in running.stdout, running.stdout)

        tailed = run("logs", "--data-dir", data_dir, "--lines", "40")
        check("logs exits 0", tailed.returncode == 0, tailed.stdout + tailed.stderr)
        check("logs contains the server banner", "Skin Retouch" in tailed.stdout,
              tailed.stdout[-400:])

        stopped_again = run("stop", "-y", "--port", str(port), "--data-dir", data_dir)
        check("stop exits 0", stopped_again.returncode == 0,
              stopped_again.stdout + stopped_again.stderr)
        check("stop reports stopped", "stopped" in stopped_again.stdout, stopped_again.stdout)
        check("stop removes the pid file", not os.path.exists(state_file))
        check("stop frees the port", not service.port_in_use(port))
        check("stop killed the process", not service.pid_alive(pid))

        repeat = run("stop", "-y", "--port", str(port), "--data-dir", data_dir)
        check("stopping a stopped service exits 0", repeat.returncode == 0,
              repeat.stdout + repeat.stderr)
        check("stopping a stopped service says not running", "not running" in repeat.stdout,
              repeat.stdout)

        # toggle is what the .bat calls: stopped -> start, running -> stop.
        first = run("toggle", "--port", str(port), "--data-dir", data_dir, "--no-open")
        check("toggle starts when stopped", "running" in first.stdout, first.stdout)
        second = run("toggle", "-y", "--port", str(port), "--data-dir", data_dir)
        check("toggle stops when running", "stopped" in second.stdout, second.stdout)
        check("toggle left the port free", not service.port_in_use(port))

        # A restart must not orphan work: the id is a durable address.
        run("start", "--port", str(port), "--data-dir", data_dir, "--no-open")
        jobs_dir = os.path.join(data_dir, "jobs")
        os.makedirs(jobs_dir, exist_ok=True)
        fake_id = "0123456789abcdef"
        fake_dir = os.path.join(jobs_dir, fake_id)
        os.makedirs(fake_dir, exist_ok=True)
        with open(os.path.join(fake_dir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"id": fake_id, "status": "done", "image": {"width": 2, "height": 2},
                       "source": {"name": "before.png"}, "artifacts": ["out"],
                       "source_artifact": ""}, handle)
        with open(os.path.join(fake_dir, "out.png"), "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

        fetched = http_json("http://127.0.0.1:%d/api/jobs/%s" % (port, fake_id))
        check("a job written to disk is readable", fetched.get("id") == fake_id, str(fetched))

        run("restart", "--port", str(port), "--data-dir", data_dir, "--no-open")
        survived = http_json("http://127.0.0.1:%d/api/jobs/%s" % (port, fake_id))
        check("the job survives a switch restart", survived.get("id") == fake_id, str(survived))
        try:
            http_json("http://127.0.0.1:%d/api/jobs/%s" % (port, "0" * 16))
            check("unknown job still 404s", False, "expected 404")
        except urllib.error.HTTPError as exc:
            check("unknown job still 404s", exc.code == 404, str(exc.code))

        final = run("stop", "-y", "--port", str(port), "--data-dir", data_dir)
        check("final stop exits 0", final.returncode == 0, final.stdout + final.stderr)
        check("final stop frees the port", not service.port_in_use(port))

        # A pid file can outlive its process (crash, reboot, pid reuse). The
        # switch must read that as "stopped" instead of refusing to start.
        ghost = 999999
        while service.pid_alive(ghost):
            ghost += 1
        with open(state_file, "w", encoding="utf-8") as handle:
            json.dump({"pid": ghost, "port": port, "host": "127.0.0.1", "token": ""}, handle)
        stale = run("status", "--port", str(port), "--data-dir", data_dir)
        check("a dead pid is reported as stopped", stale.returncode == 1, stale.stdout)
        check("a dead pid is named in the note", str(ghost) in stale.stdout, stale.stdout)
        revived = run("start", "--port", str(port), "--data-dir", data_dir, "--no-open")
        check("a stale pid file does not block a start", revived.returncode == 0,
              revived.stdout + revived.stderr)
        swept = run("stop", "-y", "--port", str(port), "--data-dir", data_dir)
        check("the revived service stops again", swept.returncode == 0, swept.stdout)
    finally:
        subprocess.run([sys.executable, SERVICE, "stop", "-y", "--port", str(port),
                        "--data-dir", data_dir], capture_output=True, cwd=ROOT, check=False)
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    print("SERVICE OK  %d checks" % CHECKS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
