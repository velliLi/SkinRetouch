"""Second probe: correct mask coordinates, plus a CPU baseline of the current pipeline.

ultralytics returns masks in letterbox space; the first probe resized them to the
full frame without undoing the letterbox, which produced garbage overlays.
"""
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

from ultralytics import YOLO

PERSON = r"E:\comfyui\ComfyUI\models\ultralytics\segm\person_yolov8m-seg.pt"
person = YOLO(PERSON)


def masks_to_frame(data, orig_shape):
    """Undo ultralytics letterbox padding: network-space masks -> frame size."""
    n, mh, mw = data.shape
    h, w = orig_shape
    gain = min(mh / h, mw / w)
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    pad_w, pad_h = (mw - new_w) // 2, (mh - new_h) // 2
    out = np.empty((n, h, w), np.float32)
    for i in range(n):
        m = data[i, pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        out[i] = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return out

report = {}

for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    t0 = time.perf_counter()
    r = person.predict(bgr, imgsz=640, conf=0.25, verbose=False, device=0, retina_masks=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    entry = {"predict_ms": round(ms, 1), "n": int(len(r.boxes))}
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None
    entry["raw_mask_shape"] = None if masks is None else [int(v) for v in masks.shape]
    entry["xy_counts"] = None if r.masks is None else [int(len(p)) for p in r.masks.xy]

    # Undo the letterbox: scale_masks maps network-space masks back to the frame.
    full = None
    if masks is not None:
        full = masks_to_frame(masks, bgr.shape[:2])
        entry["full_mask_shape"] = [int(v) for v in full.shape]
        areas = full.reshape(len(full), -1).mean(axis=1)
        entry["area_frac"] = [round(float(a), 4) for a in areas]
        best = int(np.argmax(areas))
        entry["best_area_frac"] = round(float(areas[best]), 4)

    # Previews at 1400 px long edge: filled person mask + boxes.
    scale = 1400.0 / max(h, w)
    sw, sh = int(w * scale), int(h * scale)
    prev = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    if full is not None:
        mm = cv2.resize(full[best], (sw, sh), interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(prev)
        tint[..., 2] = 255
        prev = np.where(mm[..., None] > 0.5, (prev * 0.5 + tint * 0.5).astype(np.uint8), prev)
        cv2.imwrite(os.path.join(OUT, f"mask_{os.path.splitext(name)[0]}.png"), (mm * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUT, f"probe2_{os.path.splitext(name)[0]}.png"), prev)

    # Current pipeline cost on the full 20 MP frame (faces only).
    import skin_core
    model = skin_core.load_face_detector()
    t0 = time.perf_counter()
    out, m, info = skin_core.skin_retouch(bgr, strength=0.75, texture_keep=0.35,
                                          blemish=0.7, tone_even=0.5, feather=0.5)
    entry["face_pipeline_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    entry["face_info"] = [{k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in i.items()} for i in info]
    entry["mask_cov"] = round(float((m > 127).mean()), 5) if m is not None else None
    report[f"{name} {w}x{h}"] = entry

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "probe2.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
