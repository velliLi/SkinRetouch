"""Keep a local ComfyUI warm so the repaint pass is always ready.

The repaint pass talks to ComfyUI over HTTP. Two situations have to work
without the user thinking about them:

* they already run ComfyUI (the normal case, the console is usually started
  from the same box) -- use that instance and never spawn a second one, because
  two SDXL checkpoints on a 6 GB card is an out-of-memory waiting to happen;
* nothing is running -- optionally spawn a private instance and hold it until
  the console exits.

The spawned instance runs with ``--disable-all-custom-nodes`` on purpose. The
graph only uses core nodes, and a full node pack can block startup for minutes
(ComfyUI-Manager reaches for the network on start, which never returns on an
offline machine). Override the argument list if a workflow ever needs more.

Nothing here is allowed to raise into the request path: every failure becomes a
status dict the UI can show and the job can skip past.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from . import repaint

DEFAULT_PORT = 8188
READY_TIMEOUT_S = 180.0
STOP_TIMEOUT_S = 15.0

# Windows shows a console window for every child process unless asked not to.
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _guess_dir(explicit: str = "") -> str:
    """Where ComfyUI lives, for the case where we have to start it ourselves."""
    for candidate in (
        explicit,
        os.environ.get("SKIN_COMFYUI_DIR", ""),
        r"E:\comfyui\ComfyUI",
        r"C:\comfyui\ComfyUI",
        r"D:\comfyui\ComfyUI",
        os.path.join(os.path.expanduser("~"), "ComfyUI"),
    ):
        if candidate and os.path.isfile(os.path.join(candidate, "main.py")):
            return candidate
    return ""


def _guess_python(comfyui_dir: str, explicit: str = "") -> str:
    """The interpreter ComfyUI should run under, ideally its embedded one."""
    candidates = [explicit, os.environ.get("SKIN_COMFYUI_PYTHON", "")]
    if comfyui_dir:
        parent = os.path.dirname(comfyui_dir)
        candidates.append(os.path.join(parent, "python_embeded", "python.exe"))
        candidates.append(os.path.join(comfyui_dir, "venv", "Scripts", "python.exe"))
    candidates.append(sys.executable)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


def _port_of(url: str) -> int:
    try:
        return int(urlsplit(url).port or DEFAULT_PORT)
    except (TypeError, ValueError):
        return DEFAULT_PORT


class Engine:
    """Owns at most one spawned ComfyUI process plus its status bookkeeping."""

    def __init__(self, url: str = "", comfyui_dir: str = "", python: str = "",
                 autostart: bool = False, data_dir: str = ""):
        self.url = (url or repaint.DEFAULT_URL).rstrip("/")
        self.comfyui_dir = _guess_dir(comfyui_dir)
        self.python = _guess_python(self.comfyui_dir, python)
        self.autostart = bool(autostart)
        self.data_dir = data_dir or ""
        self._proc = None
        self._state = "idle"      # idle | starting | ready | failed
        self._detail = ""
        self._lock = threading.Lock()

    # -- queries ----------------------------------------------------------
    def _spawnable(self) -> tuple[bool, str]:
        if not self.comfyui_dir:
            return False, "找不到 ComfyUI 目录（用 --comfyui-dir 指定含 main.py 的目录）"
        if not self.python:
            return False, "找不到可用的 Python 解释器（用 --comfyui-python 指定）"
        return True, ""

    @property
    def managed(self) -> bool:
        """True only when we own the process, so we never kill someone else's."""
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        """What the page shows. Reachability is probed, ownership is tracked."""
        with self._lock:
            state = self._state
            detail = self._detail
            proc_alive = self.managed
        if proc_alive and state == "starting":
            if repaint.probe(self.url, timeout=0.4):
                with self._lock:
                    self._state, self._detail = "ready", ""
                state = "ready"
            elif self._proc.poll() is not None:
                state = "failed"
                with self._lock:
                    self._state = "failed"
                    self._detail = self._log_tail()
                    detail = self._detail
        available = state == "ready" or (
            state != "starting" and repaint.probe(self.url, timeout=0.4))
        if available and state != "starting":
            state = "ready"
        spawnable, why = self._spawnable()
        return {
            "url": self.url,
            "available": bool(available),
            "managed": proc_alive,
            "state": state,
            "autostart": self.autostart,
            "spawnable": spawnable,
            "detail": detail or why,
        }

    def wait_ready(self, timeout: float = READY_TIMEOUT_S) -> dict:
        """Block until the engine answers, or give up. Never raises."""
        deadline = time.time() + max(1.0, timeout)
        while time.time() < deadline:
            if repaint.probe(self.url, timeout=0.5):
                with self._lock:
                    self._state, self._detail = "ready", ""
                return self.status()
            if self._proc is not None and self._proc.poll() is not None:
                with self._lock:
                    self._state, self._detail = "failed", self._log_tail()
                return self.status()
            time.sleep(0.5)
        return self.status()

    # -- lifecycle --------------------------------------------------------
    def start(self, timeout: float = READY_TIMEOUT_S, wait: bool = True) -> dict:
        """Spawn a private instance, unless something already answers.

        `wait=False` returns as soon as the process is up so an HTTP call is not
        held for the whole model load; the page then polls `status()` and sees
        `starting` until the instance answers.
        """
        if repaint.probe(self.url, timeout=0.5):
            with self._lock:
                self._state, self._detail = "ready", ""
            return self.status()
        if self.managed:
            return self.wait_ready(timeout) if wait else self.status()
        spawnable, why = self._spawnable()
        if not spawnable:
            with self._lock:
                self._state, self._detail = "failed", why
            return self.status()
        with self._lock:
            self._state, self._detail = "starting", ""
        try:
            self._spawn()
        except OSError as exc:
            with self._lock:
                self._state, self._detail = "failed", "启动失败：%s" % (exc,)
            return self.status()
        return self.wait_ready(timeout) if wait else self.status()

    def _spawn(self):
        port = _port_of(self.url)
        outdir = os.path.join(self.data_dir or ".", "comfyui")
        os.makedirs(outdir, exist_ok=True)
        log_path = os.path.join(self.data_dir or ".", "comfyui.log")
        self._log = open(log_path, "wb")
        cmd = [
            self.python, "main.py",
            "--port", str(port),
            "--listen", "127.0.0.1",
            "--disable-auto-launch",
            # see the module docstring: a full node pack can hang startup
            "--disable-all-custom-nodes",
            "--output-directory", outdir,
        ]
        self._proc = subprocess.Popen(
            cmd, cwd=self.comfyui_dir, stdin=subprocess.DEVNULL,
            stdout=self._log, stderr=subprocess.STDOUT,
            creationflags=_NO_WINDOW,
        )

    def _log_tail(self, lines: int = 6) -> str:
        path = os.path.join(self.data_dir or ".", "comfyui.log")
        try:
            with open(path, "rb") as handle:
                data = handle.read()[-4000:]
        except OSError:
            return ""
        text = data.decode("utf-8", "replace").strip().splitlines()
        return " / ".join(line.strip() for line in text[-lines:] if line.strip())

    def stop(self) -> dict:
        """Only ever stops the process we started."""
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            with self._lock:
                self._proc = None
                self._state = "idle"
            return self.status()
        try:
            proc.terminate()
            proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
        with self._lock:
            self._proc = None
            self._state = "idle"
            self._detail = ""
        return self.status()

    def shutdown(self):
        """Console exit: a child we spawned must not outlive us."""
        if self.managed:
            self.stop()
        log = getattr(self, "_log", None)
        if log is not None:
            try:
                log.close()
            except OSError:
                pass
            self._log = None
