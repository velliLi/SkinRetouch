"""Why does the lower-left torso still pass the chroma gate? Sample it directly."""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

# (x, y) centres of 60 px windows, in original pixels.
POINTS = {
    "DSC_7300.JPG": {
        "ref_face_cheek": (1950, 1250),
        "suspect_waist_a": (1250, 4600),
        "suspect_waist_b": (1500, 5000),
        "suspect_waist_c": (1000, 4200),
        "suspect_waist_d": (1700, 5300),
        "cloth_white_chest": (2200, 3600),
        "cloth_white_hip": (2400, 5000),
        "arm_left": (200, 3800),
        "hand": (2900, 4300),
    },
    "DSC_4081.JPG": {
        "ref_face_cheek": (2700, 1300),
        "suspect_armpit": (2050, 2600),
        "suspect_chest": (3300, 2600),
        "arm_upper": (700, 1750),
        "cloth_black": (3000, 3400),
    },
}

R = 60
out = {}
for name, pts in POINTS.items():
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    row = {}
    for key, (x, y) in pts.items():
        win = ycc[y - R:y + R, x - R:x + R].reshape(-1, 3).astype(np.float32)
        row[key] = {
            "Y": round(float(np.median(win[:, 0])), 0),
            "Cr": round(float(np.median(win[:, 1])), 0),
            "Cb": round(float(np.median(win[:, 2])), 0),
            "Cr-Cb": round(float(np.median(win[:, 1]) - np.median(win[:, 2])), 0),
            "Cr_iqr": [round(float(np.percentile(win[:, 1], 25)), 0), round(float(np.percentile(win[:, 1], 75)), 0)],
            "Cb_iqr": [round(float(np.percentile(win[:, 2], 25)), 0), round(float(np.percentile(win[:, 2], 75)), 0)],
        }
    out[name] = row

print(json.dumps(out, indent=2, ensure_ascii=False))
