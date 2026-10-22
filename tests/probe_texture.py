"""Is local texture energy enough to separate skin from the pale suit?

Skin: fine random pore noise. Suit: woven vertical ribbing plus seams.
Measures local std on known skin and known cloth patches.
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

# Hand-picked 1:1 windows (x1,y1,x2,y2) in pixels, verified against dump_crops.py.
PATCHES = {
    "DSC_7300.JPG": {
        "skin_face": (1700, 1150, 2000, 1400),
        "skin_neck": (1700, 2050, 2000, 2250),
        "skin_arm_L": (60, 3400, 300, 3900),
        "skin_hand": (2800, 4150, 3100, 4450),
        "cloth_chest": (1300, 3200, 1700, 3600),
        "cloth_belly": (1300, 4000, 1700, 4400),
        "cloth_side": (2200, 3300, 2500, 3700),
    },
    "DSC_4081.JPG": {
        "skin_face": (2450, 1150, 2800, 1450),
        "skin_arm": (500, 1600, 900, 1900),
        "skin_chest": (3700, 3100, 4000, 3400),
        "cloth_black": (2900, 3300, 3200, 3600),
        "hair_red": (1900, 1900, 2300, 2300),
    },
}


def local_stats(gray, k):
    m = cv2.blur(gray, (k, k))
    m2 = cv2.blur(gray * gray, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


report = {}
for name, patches in PATCHES.items():
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row = {}
    for key, (x1, y1, x2, y2) in patches.items():
        patch = bgr[y1:y2, x1:x2]
        if patch.size == 0:
            row[key] = "out of range"
            continue
        g = gray[y1:y2, x1:x2]
        ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(np.float32)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        entry = {
            "Y": round(float(np.median(ycc[:, 0])), 0),
            "Cr": round(float(np.median(ycc[:, 1])), 0),
            "Cb": round(float(np.median(ycc[:, 2])), 0),
            "S": round(float(np.median(hsv[:, 1]) / 255), 2),
        }
        for k in (5, 9, 15):
            entry[f"std{k}"] = round(float(np.median(local_stats(g, k))), 2)
        # High-frequency energy from Laplacian at multiple scales.
        entry["lap3"] = round(float(np.median(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)))), 2)
        row[key] = entry
    report[name] = row

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "texture_stats.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
