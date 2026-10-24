"""Exercise Engine start/wait/stop against the real ComfyUI install.

The server's autostart path is the one thing unit tests cannot cover: it shells
out to ComfyUI, waits for a 6.9 GB checkpoint to load and then has to be shut
down again. This probe does exactly that, in a throwaway data dir, and prints
every status transition. Run it with the embedded interpreter:

    E:\\comfyui\\python_embeded\\python.exe tests\\probe_engine.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import repaint  # noqa: E402
from server.engine import Engine  # noqa: E402

URL = os.environ.get("SKIN_REPAINT_URL", "http://127.0.0.1:8188")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "engine_data")


def show(tag, status):
    print("%-9s %s" % (tag, status))


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    engine = Engine(url=URL, autostart=True, data_dir=DATA)
    show("before", engine.status())

    spawnable, why = engine._spawnable()
    print("spawnable %s  %s" % (spawnable, why))
    if not spawnable:
        return 1

    t0 = time.time()
    show("start", engine.start(wait=False))
    if not engine.managed:
        print("error: no process was spawned")
        return 1
    print("pid       %d" % engine._proc.pid)

    status = engine.wait_ready(timeout=float(os.environ.get("ENGINE_TIMEOUT", "240")))
    show("waited", status)
    print("seconds   %.1f" % (time.time() - t0))
    print("probe     %s" % repaint.probe(URL, timeout=2.0))

    ok = bool(status.get("available"))
    show("stop", engine.stop())
    time.sleep(1.0)
    after = engine.status()
    show("after", after)
    print("log tail:")
    for line in (engine._log_tail(lines=8) or "").splitlines():
        print("  " + line)
    engine.shutdown()
    print("RESULT    %s" % ("READY" if ok else "NOT READY"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
