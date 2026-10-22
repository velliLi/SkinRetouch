"""Dump 1:1 crops so the exact pixel content of "suit" and "skin" areas is verifiable."""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

CROPS = {
    "DSC_7300.JPG": {
        "torso": (0.30, 0.55, 0.70, 0.85),
        "face": (0.36, 0.20, 0.56, 0.36),
        "left_arm": (0.02, 0.60, 0.16, 0.80),
        "hand": (0.70, 0.74, 0.84, 0.86),
    },
    "DSC_4081.JPG": {
        "torso": (0.45, 0.72, 0.85, 0.95),
        "face": (0.44, 0.28, 0.62, 0.44),
        "arm": (0.06, 0.38, 0.24, 0.56),
    },
}

for name, crops in CROPS.items():
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    for key, (x1, y1, x2, y2) in crops.items():
        crop = bgr[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        ch, cw = crop.shape[:2]
        scale = min(1.0, 900.0 / max(ch, cw))
        if scale < 1.0:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(OUT, f"crop_{os.path.splitext(name)[0]}_{key}.png"), crop)
        print(name, key, "px", (cw, ch))
