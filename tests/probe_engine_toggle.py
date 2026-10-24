"""Stop the repaint engine, start it again, and watch it come back."""

from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8799"


def call(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as res:
        return json.load(res)


def show(tag):
    status = call("/api/repaint/engine")
    print("%-10s available=%-5s managed=%-5s state=%s"
          % (tag, status["available"], status["managed"], status["state"]))
    return status


show("now")
print("--- stop")
print("stop  ->", call("/api/repaint/engine", {"action": "stop"})["state"])
time.sleep(2)
show("after-stop")

print("--- start")
print("start ->", call("/api/repaint/engine", {"action": "start"})["state"])
t0 = time.time()
for _ in range(80):
    time.sleep(1.5)
    status = call("/api/repaint/engine")
    if status["available"] or status["state"] == "failed":
        print("back  -> state=%s in %.1fs" % (status["state"], time.time() - t0))
        break
else:
    print("timed out waiting for the engine")
print("final  ", call("/api/repaint/engine"))
