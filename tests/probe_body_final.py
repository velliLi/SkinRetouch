"""End-to-end body retouch on the two reference shots."""
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

import skin_core

DEVICE = os.environ.get("SKIN_DEVICE", "cpu")

for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    t0 = time.perf_counter()
    out, mask, info = skin_core.skin_retouch(
        bgr, strength=0.7, texture_keep=0.45, blemish=0.6, tone_even=0.5,
        feather=0.5, conf=0.25, include_body=True, device=DEVICE,
    )
    ms = (time.perf_counter() - t0) * 1000

    stem = os.path.splitext(name)[0]
    cv2.imwrite(os.path.join(OUT, f"final_out_{stem}.png"), out)
    cv2.imwrite(os.path.join(OUT, f"final_mask_{stem}.png"), mask)
    scale = 1400.0 / max(h, w)
    small_o = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_r = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gap = np.full((small_o.shape[0], 16, 3), 24, np.uint8)
    cv2.imwrite(os.path.join(OUT, f"final_side_{stem}.png"), np.hstack([small_o, gap, small_r]))
    ms_s = cv2.resize(mask, (small_o.shape[1], small_o.shape[0]), interpolation=cv2.INTER_AREA)
    tint = np.zeros_like(small_r)
    tint[..., 1] = 255
    tint[..., 2] = 40
    cv2.imwrite(
        os.path.join(OUT, f"final_overlay_{stem}.png"),
        np.where(ms_s[..., None] > 60, (small_r * 0.35 + tint * 0.65).astype(np.uint8), small_r),
    )

    print(json.dumps({
        "image": name,
        "device": DEVICE,
        "ms_total": round(ms, 1),
        "mask_cov": round(float((mask > 127).mean()), 4),
        "info": info,
    }, indent=2, ensure_ascii=False, default=str))
