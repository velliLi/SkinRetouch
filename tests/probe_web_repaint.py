"""Visual proof that the web console's 局部重绘 switch does something.

Runs the real HTTP path twice on the same injected frame -- once with
`repaint=0`, once with `repaint=1` -- then lays the two results next to the
clean reference and the blemished input, cropped around the spots at 4x.

Everything goes through `/api/jobs` exactly as the browser does it, so this
also exercises the wiring `server_test.py` cannot show: the numbers in
`REPAINT_EXPLORE.md` prove the engine works, this proves the *feature* works.

    python tests/probe_web_repaint.py                       # default migan engine
    python tests/probe_web_repaint.py --repaint-engine lama
    python tests/probe_web_repaint.py --repaint-engine sdxl --url http://127.0.0.1:8188
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from probe_repaint_local import build_mask, hf_energy, inject  # noqa: E402
from server import repaint  # noqa: E402
from server.api import create_server  # noqa: E402
from server.config import Config  # noqa: E402

import skin_core  # noqa: E402

OUT = os.path.join(HERE, "output")
ASSETS = os.path.join(HERE, "assets")


def label(img, text):
    bar = np.full((30, img.shape[1], 3), 18, np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def call(base, path, method="GET", data=None, headers=None):
    req = urllib.request.Request(base + path, data=data, method=method,
                                headers=headers or {})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.status, dict(resp.headers), resp.read()


def submit(base, blob, query):
    status, _, body = call(base, "/api/jobs" + query, "POST", blob,
                           {"Content-Type": "image/png"})
    job = json.loads(body.decode("utf-8"))
    if status != 202:
        raise SystemExit("submit failed: %s %s" % (status, job))
    deadline = time.time() + 300
    while time.time() < deadline:
        _, _, raw = call(base, "/api/jobs/" + job["id"])
        snap = json.loads(raw.decode("utf-8"))
        if snap["status"] in ("done", "error", "canceled"):
            return snap
        time.sleep(0.4)
    raise SystemExit("job %s never finished" % job["id"])


def fetch(base, job_id, kind="out"):
    _, _, raw = call(base, "/api/jobs/%s/result?kind=%s" % (job_id, kind))
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="t01_headshoulders_135_00001_.png")
    parser.add_argument("--url", default="", help="ComfyUI base URL (sdxl only)")
    parser.add_argument("--repaint-engine", default=repaint.DEFAULT_ENGINE,
                        choices=list(repaint.ENGINES))
    parser.add_argument("--crop", type=int, default=260, help="crop side in pixels")
    parser.add_argument("--zoom", type=int, default=4)
    args = parser.parse_args(argv)

    engine = args.repaint_engine
    if engine == "sdxl":
        if not repaint.probe((args.url or repaint.DEFAULT_URL).rstrip("/"), timeout=1.0):
            raise SystemExit("no ComfyUI for the sdxl engine; start it first")
    elif not repaint.net_ready(engine):
        raise SystemExit("%s weights missing; run tests/fetch_repaint_weights.py "
                         "--model %s" % (engine, engine))
    url = (args.url or repaint.DEFAULT_URL).rstrip("/")

    clean = cv2.imread(os.path.join(ASSETS, args.asset), cv2.IMREAD_COLOR)
    if clean is None:
        raise SystemExit("cannot read %s" % args.asset)
    os.makedirs(OUT, exist_ok=True)

    # Reuse the CPU pipeline's own skin mask so the injection lands where the
    # benchmark put it, then run the detector path once more for the real job.
    _, skin, info = skin_core.skin_retouch(clean, include_body=False, device="cpu")
    bad, discs = inject(clean, skin)
    spots = np.argwhere(discs > 0)
    cy, cx = spots.mean(axis=0).astype(int) if len(spots) else (clean.shape[0] // 2,
                                                               clean.shape[1] // 2)

    ok, buf = cv2.imencode(".png", bad)
    blob = buf.tobytes()

    data_dir = tempfile.mkdtemp(prefix="skin-web-")
    cfg = Config(host="127.0.0.1", port=0, token="", data_dir=data_dir,
                 preview_max_px=2400, repaint_url=url)
    server = create_server(cfg)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                     daemon=True).start()
    base = "http://127.0.0.1:%d" % server.server_address[1]
    print("console %s  engine %s" % (base, engine))

    try:
        query = "?include_body=0&strength=0.75&blemish=0.7&repaint_engine=" + engine
        plain = submit(base, blob, query)
        plus = submit(base, blob, query + "&repaint=1")
        out_plain = fetch(base, plain["id"])
        out_plus = fetch(base, plus["id"])
    finally:
        server.manager.shutdown()
        server.server_close()
        shutil.rmtree(data_dir, ignore_errors=True)

    sel = discs > 127
    # The pass only touches what its own detector finds, and on this synthetic
    # frame that is a minority of the injected disc area (the spots are blown
    # soft by the injection blur). Measure both: the whole discs, and the part
    # the mask actually covered. The second number is what the engine is
    # responsible for; the first one is diluted by the 70% nobody asked about.
    mask, mask_stats = repaint.blemish_mask(bad, skin, int(info[0]["face_px"]),
                                            blemish=0.7)
    hit = sel & (mask > 127)

    def resid(img, where):
        return round(float(np.abs(img[where].astype(np.float32) - clean[where]).mean()), 2)

    report = {
        "asset": args.asset,
        "engine": engine,
        "mask": mask_stats,
        "coverage_of_injected_discs": round(float(hit.sum()) / max(1, int(sel.sum())), 3),
        "blemished": {"resid": resid(bad, sel), "hf": round(hf_energy(bad, sel), 2),
                      "resid_hit": resid(bad, hit)},
        "retouch": {"ms": plain["timing"].get("pipeline_ms"),
                    "resid": resid(out_plain, sel), "hf": round(hf_energy(out_plain, sel), 2),
                    "resid_hit": resid(out_plain, hit)},
        "retouch_repaint": {"ms": plus["timing"].get("pipeline_ms"),
                            "repaint_ms": plus["timing"].get("repaint_ms"),
                            "resid": resid(out_plus, sel),
                            "hf": round(hf_energy(out_plus, sel), 2),
                            "resid_hit": resid(out_plus, hit),
                            "stats": plus.get("repaint")},
        "clean_hf": round(hf_energy(clean, sel), 2),
    }

    half = args.crop // 2
    y0 = max(0, min(cy - half, clean.shape[0] - args.crop))
    x0 = max(0, min(cx - half, clean.shape[1] - args.crop))
    box = (slice(y0, y0 + args.crop), slice(x0, x0 + args.crop))

    def zoom(img):
        crop = img[box]
        return cv2.resize(crop, (args.crop * args.zoom, args.crop * args.zoom),
                          interpolation=cv2.INTER_NEAREST)

    panels = [
        label(zoom(clean), "reference"),
        label(zoom(bad), "blemished"),
        label(zoom(out_plain), "retouch"),
        label(zoom(out_plus), "retouch+repaint"),
    ]
    sheet = np.hstack(panels)
    sheet_name = "web_repaint_compare_%s.png" % engine
    cv2.imwrite(os.path.join(OUT, sheet_name), sheet)

    path = os.path.join(OUT, "web_repaint_report_%s.json" % engine)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report["retouch_repaint"], ensure_ascii=False, indent=2)[:900])
    print("resid  blemished %-6s retouch %-6s retouch+repaint %-6s"
          % (report["blemished"]["resid"], report["retouch"]["resid"],
             report["retouch_repaint"]["resid"]))
    print("hf     clean %-6s blemished %-6s retouch %-6s retouch+repaint %s"
          % (report["clean_hf"], report["blemished"]["hf"], report["retouch"]["hf"],
             report["retouch_repaint"]["hf"]))
    print("wrote %s" % os.path.join(OUT, sheet_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
