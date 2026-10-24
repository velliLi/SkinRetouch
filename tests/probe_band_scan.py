"""Which band actually separates spots from ordinary skin?

The mask built on `fine - mid` (8..42 px) is dominated by large blotches, so it
lands on a couple of red patches and misses the small spots entirely. A spot of
r=3..8 px, blurred by the injection, has an effective width of 10..16 px, so the
right band is narrower. Scan a few mid radii and report hit/spill for each.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from probe_repaint_local import inject  # noqa: E402
import skin_core  # noqa: E402

ASSET = os.path.join(HERE, "assets", "t01_headshoulders_135_00001_.png")
clean = cv2.imread(ASSET, cv2.IMREAD_COLOR)
_, skin, info = skin_core.skin_retouch(clean, include_body=False, device="cpu")
bad, discs = inject(clean, skin)
face_px = int(info[0]["face_px"])

active = skin > 180
in_disc = (discs > 127) & active
in_skin_only = active & ~in_disc

red = bad[:, :, 2].astype(np.float32)
r_fine = int(np.clip(round(face_px * 0.010), 2, 64))
fine = skin_core._guided(red, red, r_fine, (0.010 * 255.0) ** 2)
print("face_px %d  r_fine %d" % (face_px, r_fine))


def report(name, band):
    response = np.where(active, np.clip(band, 0.0, None), 0.0)
    for pct in (90.0, 95.0, 97.0):
        thr = float(np.percentile(response[active], pct))
        hit = 100 * (response[in_disc] > thr).mean()
        spill = 100 * (response[in_skin_only] > thr).mean()
        print("  %-18s p%-5.1f hit %5.1f%%  spill %5.2f%%" % (name, pct, hit, spill))


for ratio in (0.012, 0.018, 0.025, 0.035, 0.055):
    r_mid = int(np.clip(round(face_px * ratio), 3, 160))
    band = fine - skin_core._guided(fine, fine, r_mid, (0.030 * 255.0) ** 2)
    report("fine-mid r=%d" % r_mid, band)

raw_band = red - skin_core._guided(red, red, int(np.clip(round(face_px * 0.018), 3, 160)),
                                   (0.030 * 255.0) ** 2)
report("raw-mid r18", raw_band)

# --- top-hat: is the pixel redder than its own neigbourhood? -----------------
base_band = fine - skin_core._guided(fine, fine, 19, (0.030 * 255.0) ** 2)
for k in (11, 17, 25, 35):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(base_band.astype(np.float32), cv2.MORPH_TOPHAT, kernel)
    report("tophat k=%d" % k, tophat)
