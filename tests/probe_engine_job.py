"""Submit one repaint job over HTTP and print what the engine actually did.

This is the end-to-end check for the autostarted engine: the server behind
`http://127.0.0.1:8799` must have a live ComfyUI on 8188 for it to pass.
"""

from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8799"
ASSET = r"E:\SkinRetouch\tests\assets\t01_headshoulders_135_00001_.png"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as res:
        return json.load(res)


with open(ASSET, "rb") as fh:
    image = fh.read()

req = urllib.request.Request(
    BASE + "/api/jobs?repaint=1&include_body=0&strength=0.75&blemish=0.7",
    data=image, method="POST",
    headers={"Content-Type": "application/octet-stream"})
with urllib.request.urlopen(req, timeout=60) as res:
    job = json.load(res)
print("job       %s  status=%s" % (job["id"], job["status"]))

t0 = time.time()
snap = job
while True:
    time.sleep(2)
    snap = get("/api/jobs/" + job["id"])
    if snap["status"] in ("done", "error", "canceled"):
        break
    if time.time() - t0 > 300:
        raise SystemExit("timed out")
print("finished  %s in %.1fs" % (snap["status"], time.time() - t0))
print("timing    %s" % json.dumps(snap.get("timing") or {}))
print("repaint   %s" % json.dumps(snap.get("repaint") or {}, ensure_ascii=False))
print("warnings  %s" % json.dumps(snap.get("warnings") or [], ensure_ascii=False))
if snap.get("error"):
    print("error     %s" % snap["error"])
