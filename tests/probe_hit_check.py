"""Is the weak web result the detector or the engine?

`probe_web_repaint.py` uses the automatic blemish mask, which only covers about
a quarter of the injected spot area on the synthetic frame. This script feeds
the ground-truth disc mask through the same `apply()` path, so the two numbers
separate detector quality from repaint quality.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from probe_repaint_local import inject  # noqa: E402
from server import repaint  # noqa: E402
import skin_core  # noqa: E402


def hf_energy(img, sel):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    return float(lap[sel].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=repaint.DEFAULT_URL)
    ap.add_argument("--asset", default="t01_headshoulders_135_00001_.png")
    args = ap.parse_args()

    asset = os.path.join(HERE, "assets", args.asset)
    clean = cv2.imread(asset, cv2.IMREAD_COLOR)
    _, skin, info = skin_core.skin_retouch(clean, include_body=False, device="cpu")
    bad, discs = inject(clean, skin)

    out, _, _ = skin_core.skin_retouch(bad, include_body=False, device="cpu")
    truth = repaint.prep_mask(discs, repaint.GROW_PX, repaint.BLUR_PX)
    params = {"blemish": 0.7, "repaint_denoise": 0.60, "repaint_steps": 8}

    auto_img, auto_stats = repaint.apply(bad, out, skin, info, params,
                                         base_url=args.url)
    true_img, true_stats = repaint.apply(bad, out, skin, info, params,
                                         base_url=args.url, mask=truth)

    sel = discs > 127

    def resid(img):
        return round(float(np.abs(img[sel].astype(np.float32) - clean[sel]).mean()), 2)

    print("face_px", int(info[0]["face_px"]), "disc px", int(sel.sum()))
    print("auto mask  frac %.4f tiles %s ms %s"
          % (auto_stats["mask_frac"], auto_stats.get("tiles_done"), auto_stats.get("ms")))
    print("truth mask frac %.4f tiles %s ms %s"
          % (true_stats["mask_frac"], true_stats.get("tiles_done"), true_stats.get("ms")))
    for name, img in (("blemished", bad), ("retouch", out),
                      ("retouch+auto", auto_img), ("retouch+truth", true_img)):
        print("%-16s resid %5.2f  hf %5.2f" % (name, resid(img), hf_energy(img, sel)))

    cv2.imwrite(os.path.join(HERE, "output", "hit_check_truth.png"), true_img)


if __name__ == "__main__":
    main()
