"""Diagnostic: how well does the automatic blemish mask hit the injected spots?"""

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

ASSET = os.path.join(HERE, "assets", "t01_headshoulders_135_00001_.png")
clean = cv2.imread(ASSET, cv2.IMREAD_COLOR)
_, skin, info = skin_core.skin_retouch(clean, include_body=False, device="cpu")
bad, discs = inject(clean, skin)
face_px = int(info[0]["face_px"])

mid = repaint._redness_midband(bad, face_px, 0.7)
active = skin > 180
in_disc = (discs > 127) & active
in_skin_only = active & ~in_disc

print("face_px", face_px)
print("skin px", int(active.sum()), "disc px", int(in_disc.sum()))
for name, sel in (("discs", in_disc), ("skin-not-disc", in_skin_only)):
    vals = mid[sel]
    print("%-14s n=%-7d mean %6.2f  p50 %6.2f  p90 %6.2f  p97 %6.2f  max %6.2f"
          % (name, len(vals), vals.mean(), np.percentile(vals, 50),
             np.percentile(vals, 90), np.percentile(vals, 97), vals.max()))

for pct in (90.0, 95.0, 97.0, 98.5):
    thr = float(np.percentile(np.clip(mid[active], 0, None), pct))
    hit = (mid[in_disc] > thr).mean()
    spill = ((mid[in_skin_only] > thr)).mean()
    print("pct %5.1f -> thr %5.2f   disc hit %5.1f%%   skin spill %5.2f%%"
          % (pct, thr, 100 * hit, 100 * spill))

mask, stats = repaint.blemish_mask(bad, skin, face_px)
print("blemish_mask", stats, "px", int((mask > 127).sum()))
cover = ((mask > 127) & in_disc).sum() / max(1, in_disc.sum())
print("coverage of injected discs: %.1f%%" % (100 * cover))

# --- adaptive (MAD) threshold instead of a fixed percentile ------------------
sig = skin_core._mad_sigma(np.clip(mid[active], 0, None))
print("MAD sigma of positive midband: %.3f" % sig)
for k in (0.5, 1.0, 1.5, 2.0, 2.5):
    thr = k * sig
    hit = (mid[in_disc] > thr).mean()
    spill = (mid[in_skin_only] > thr).mean()
    print("MAD k %.1f -> thr %5.2f   disc hit %5.1f%%   skin spill %5.2f%%"
          % (k, thr, 100 * hit, 100 * spill))

out, _, _ = skin_core.skin_retouch(bad, include_body=False, device="cpu")
diff = np.abs(bad.astype(np.float32) - out.astype(np.float32)).mean(axis=2)
for name, sel in (("discs", in_disc), ("skin-not-disc", in_skin_only)):
    vals = diff[sel]
    print("DIFF %-14s mean %6.2f  p50 %6.2f  p90 %6.2f  p97 %6.2f  max %6.2f"
          % (name, vals.mean(), np.percentile(vals, 50), np.percentile(vals, 90),
             np.percentile(vals, 97), vals.max()))
for pct in (90.0, 95.0, 97.0):
    thr = float(np.percentile(diff[active], pct))
    print("DIFF pct %5.1f -> thr %5.2f   disc hit %5.1f%%   skin spill %5.2f%%"
          % (pct, thr, 100 * (diff[in_disc] > thr).mean(),
             100 * (diff[in_skin_only] > thr).mean()))
