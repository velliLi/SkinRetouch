"""Directional texture: woven cloth has one dominant gradient axis, skin does not.

aniso = (Ex - Ey) / (Ex + Ey) on a 9 px window. Ribbed fabric trends positive,
skin hovers near zero with low energy.
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

PATCHES = {
    "DSC_7300.JPG": {
        "skin_face": (1700, 1150, 2000, 1400),
        "skin_neck": (1700, 2050, 2000, 2250),
        "skin_arm_L": (60, 3400, 300, 3900),
        "skin_hand": (2800, 4150, 3100, 4450),
        "cloth_chest": (1300, 3200, 1700, 3600),
        "cloth_belly": (1300, 4000, 1700, 4400),
        "cloth_side": (2200, 3300, 2500, 3700),
        "cloth_waist_hi": (900, 4450, 1300, 4900),
    },
    "DSC_4081.JPG": {
        "skin_face": (2450, 1150, 2800, 1450),
        "skin_arm": (500, 1600, 900, 1900),
        "skin_chest": (3700, 3100, 4000, 3400),
        "cloth_black": (2900, 3300, 3200, 3600),
        "hair_red": (1900, 1900, 2300, 2300),
    },
}


def aniso_energy(gray, k=9):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ex = cv2.blur(gx * gx, (k, k))
    ey = cv2.blur(gy * gy, (k, k))
    aniso = (ex - ey) / (ex + ey + 1e-3)
    return aniso, ex + ey


report = {}
for name, patches in PATCHES.items():
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    aniso, energy = aniso_energy(gray)
    row = {}
    for key, (x1, y1, x2, y2) in patches.items():
        a = aniso[y1:y2, x1:x2]
        e = energy[y1:y2, x1:x2]
        row[key] = {
            "aniso_med": round(float(np.median(a)), 3),
            "aniso_p90": round(float(np.percentile(a, 90)), 3),
            "energy_med": round(float(np.median(e)), 1),
            "energy_p90": round(float(np.percentile(e, 90)), 1),
        }
    report[name] = row

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "aniso_stats.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
