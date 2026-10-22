"""CPU and GPU paths must stay in the same neighbourhood.

They are not bit-identical by design: the GPU does morphology with a rectangular
kernel (torch has no ellipse structuring element) and computes one global tau
where the CPU recomputes it per ROI/strip. That shifts mask edges by a pixel or
two, which shows up as a small p99 on the output and a handful of hot pixels at
the boundary. This script pins the size of that gap so it cannot silently grow.

Run: python tests/compare_paths.py
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import skin_core

ASSETS = os.path.join(HERE, "assets")

# Measured on DSC_7300 after the chroma-radius fix: rel 1.44%, p99 13.0,
# max 73 (a few hundred boundary pixels). Budgets below leave ~2x headroom.
MAX_COV_REL = 0.030
MAX_P99 = 15.0
MAX_MEAN = 2.0
WEIGHT = dict(strength=0.75, texture_keep=0.35, blemish=0.7,
              tone_even=0.5, feather=0.5, conf=0.25, include_body=True)


def run(bgr, device):
    out, mask, info = skin_core.skin_retouch(bgr, device=device, **WEIGHT)
    return out, mask, [i for i in info if i.get("kind") == "body"][0]


def main():
    name = "DSC_7300.JPG"
    bgr = cv2.imdecode(np.fromfile(os.path.join(ASSETS, name), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print("MISS", name)
        return 1

    out_c, mask_c, bi_c = run(bgr, "cpu")
    out_g, mask_g, bi_g = run(bgr, "auto")

    if bi_g["backend"] != "cuda":
        print("SKIP: CUDA backend not available (backend=%s)" % bi_g["backend"])
        return 0

    cov_c = float((mask_c > 127).mean())
    cov_g = float((mask_g > 127).mean())
    rel = abs(cov_g - cov_c) / max(cov_c, 1e-9)

    d = np.abs(out_g.astype(np.int16) - out_c.astype(np.int16))
    p99 = float(np.percentile(d, 99))
    mean = float(d.mean())

    print("%s cov cpu=%.4f gpu=%.4f rel=%.2f%%" % (name, cov_c, cov_g, rel * 100))
    print("%s out max=%d mean=%.3f p99=%.1f" % (name, d.max(), mean, p99))
    print("%s ms cpu=%s gpu=%s" % (name, bi_c["ms"], bi_g["ms"]))

    fails = []
    if rel > MAX_COV_REL:
        fails.append("coverage rel %.2f%% > %.2f%%" % (rel * 100, MAX_COV_REL * 100))
    if p99 > MAX_P99:
        fails.append("output p99 %.1f > %.1f" % (p99, MAX_P99))
    if mean > MAX_MEAN:
        fails.append("output mean diff %.2f > %.2f" % (mean, MAX_MEAN))

    for line in fails:
        print("FAIL", line)
    if fails:
        return 1
    print("PATHS OK (rel %.2f%%, p99 %.1f)" % (rel * 100, p99))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
