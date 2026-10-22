"""1:1 before/after crops for the regions that matter."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

REGIONS = {
    "DSC_7300.JPG": {
        "face": (1500, 950, 2400, 2250),
        "waist_false": (850, 4200, 1650, 5000),
        "arm_right": (3050, 3300, 3642, 4400),
        "hand": (2700, 4000, 3300, 4700),
    },
    "DSC_4081.JPG": {
        "face": (2400, 850, 3650, 2500),
        "arm": (300, 1300, 1400, 2400),
        "chest": (3200, 2400, 4300, 3500),
    },
}

for name, regions in REGIONS.items():
    stem = os.path.splitext(name)[0]
    src = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    out = cv2.imdecode(np.fromfile(os.path.join(OUT, f"final_out_{stem}.png"), np.uint8), cv2.IMREAD_COLOR)
    if out is None:
        print("missing", stem)
        continue
    h, w = src.shape[:2]
    for key, (x1, y1, x2, y2) in regions.items():
        x2, y2 = min(x2, w), min(y2, h)
        a = src[y1:y2, x1:x2]
        b = out[y1:y2, x1:x2]
        gap = np.full((a.shape[0], 8, 3), 20, np.uint8)
        pair = np.hstack([a, gap, b])
        scale = min(1.0, 800.0 / max(pair.shape[:2]))
        if scale < 1.0:
            pair = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        path = os.path.join(OUT, f"cmp_{stem}_{key}.png")
        cv2.imwrite(path, pair)
        print(path, pair.shape)
