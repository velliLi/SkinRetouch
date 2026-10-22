"""Measure YCrCb/HSV statistics of hand-picked regions so thresholds are evidence-based.

Regions are given in fractions of the frame, so they survive any resize.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

# (name, cx, cy, half_w, half_h) in frame fractions
REGIONS = {
    "DSC_7300.JPG": [
        ("face_cheek", 0.470, 0.295, 0.035, 0.030),
        ("face_forehead", 0.470, 0.215, 0.045, 0.018),
        ("neck", 0.470, 0.375, 0.030, 0.020),
        ("arm_left_out", 0.120, 0.700, 0.020, 0.040),
        ("hand_right", 0.770, 0.800, 0.020, 0.025),
        ("red_suit_chest", 0.500, 0.620, 0.060, 0.050),
        ("red_suit_belly", 0.470, 0.780, 0.060, 0.050),
        ("white_shoulder", 0.300, 0.520, 0.030, 0.030),
        ("hair_dark", 0.300, 0.300, 0.030, 0.040),
        ("hair_lit", 0.680, 0.400, 0.025, 0.040),
        ("background_wall", 0.900, 0.200, 0.040, 0.040),
    ],
    "DSC_4081.JPG": [
        ("face_cheek", 0.520, 0.360, 0.030, 0.030),
        ("face_forehead", 0.510, 0.265, 0.040, 0.018),
        ("arm_upper", 0.150, 0.480, 0.030, 0.030),
        ("shoulder_skin", 0.780, 0.780, 0.040, 0.040),
        ("chest_skin", 0.700, 0.880, 0.040, 0.030),
        ("black_choker", 0.640, 0.620, 0.020, 0.012),
        ("black_bodice", 0.560, 0.900, 0.040, 0.040),
        ("hair_red", 0.430, 0.450, 0.030, 0.040),
        ("background_dark", 0.200, 0.150, 0.040, 0.040),
        ("background_skin", 0.030, 0.120, 0.015, 0.020),
    ],
}

report = {}
for name, regions in REGIONS.items():
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rows = {}
    for rname, cx, cy, hw, hh in regions:
        x1, x2 = int((cx - hw) * w), int((cx + hw) * w)
        y1, y2 = int((cy - hh) * h), int((cy + hh) * h)
        patch_y = ycc[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
        patch_h = hsv[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
        rows[rname] = {
            "YCrCb_med": [round(float(v), 0) for v in np.median(patch_y, axis=0)],
            "Cb_range": [round(float(np.percentile(patch_y[:, 2], 5)), 0), round(float(np.percentile(patch_y[:, 2], 95)), 0)],
            "Cr_range": [round(float(np.percentile(patch_y[:, 1], 5)), 0), round(float(np.percentile(patch_y[:, 1], 95)), 0)],
            "Y_range": [round(float(np.percentile(patch_y[:, 0], 5)), 0), round(float(np.percentile(patch_y[:, 0], 95)), 0)],
            "H_med_deg": round(float(np.median(patch_h[:, 0]) * 2), 0),
            "S_med": round(float(np.median(patch_h[:, 1]) / 255.0), 2),
            "V_med": round(float(np.median(patch_h[:, 2]) / 255.0), 2),
        }
    report[name] = rows

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "region_stats.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
