"""Steady-state timing plus a presentation comparison at default settings."""

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import skin_core

OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(HERE, "assets", "t01_headshoulders_135_00001_.png")

bgr = cv2.imdecode(np.fromfile(SRC, np.uint8), cv2.IMREAD_COLOR)

# Warm the detector so timing excludes first-load cost.
skin_core.skin_retouch(bgr, max_faces=1)

for scale in (0.5, 1.0, 1.5):
    img = cv2.resize(bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        skin_core.skin_retouch(img, strength=0.75, texture_keep=0.35, blemish=0.7,
                               tone_even=0.5, feather=0.5)
        times.append((time.perf_counter() - t0) * 1000)
    print("size %dx%d  median %.0f ms" % (img.shape[1], img.shape[0], sorted(times)[1]))

out, mask, info = skin_core.skin_retouch(
    bgr, strength=0.75, texture_keep=0.35, blemish=0.7, tone_even=0.5, feather=0.5,
)
print("default run info", info)

cv2.imwrite(os.path.join(OUT, "deliver_pair.png"), np.hstack([bgr, out]))

ys, xs = np.where(mask > 180)
cx, cy = int(np.median(xs)), int(np.median(ys))
side = min(260, bgr.shape[0], bgr.shape[1])
x1 = int(np.clip(cx - side // 2, 0, bgr.shape[1] - side))
y1 = int(np.clip(cy - side // 2, 0, bgr.shape[0] - side))
crops = [
    bgr[y1:y1 + side, x1:x1 + side],
    out[y1:y1 + side, x1:x1 + side],
    cv2.applyColorMap(mask[y1:y1 + side, x1:x1 + side], cv2.COLORMAP_INFERNO),
]
crops = [cv2.resize(c, (side * 2, side * 2), interpolation=cv2.INTER_NEAREST) for c in crops]
for c, t in zip(crops, ("original", "retouched", "skin mask")):
    cv2.rectangle(c, (0, 0), (c.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(c, t, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(os.path.join(OUT, "deliver_zoom.png"), np.hstack(crops))
print("DONE")
