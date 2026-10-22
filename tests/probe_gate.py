"""Chroma-ellipse gate: the only colour cue that separates this pale suit from skin.

Measured medians (tests/output/texture_stats.json):
  skin   dCr ~ 0..4,   dCb ~ -3..0   (relative to the face reference)
  suit   dCr ~ -8..-18, dCb ~ +6..+18
So a tight ellipse in (dCr, dCb) separates them where a plain RGB/YCrCb distance
does not.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

from ultralytics import YOLO
import skin_core

PERSON = r"E:\comfyui\ComfyUI\models\ultralytics\segm\person_yolov8m-seg.pt"
person = YOLO(PERSON)
face_model = skin_core.load_face_detector()


def masks_to_frame(data, orig_shape):
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


def gate_mask(bgr, person_mask, ref, cr_tol, cb_tol, y_lo=0.30, sat_hi=0.45):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    dcr = ycc[..., 1] - ref[1]
    dcb = ycc[..., 2] - ref[2]
    q = np.sqrt((dcr / cr_tol) ** 2 + (dcb / cb_tol) ** 2)
    soft = np.clip((1.15 - q) / 0.40, 0.0, 1.0)

    Y = ycc[..., 0]
    soft *= np.clip((Y - y_lo * ref[0]) / (0.12 * ref[0]), 0.0, 1.0)

    sat = hsv[..., 1] / 255.0
    soft *= np.clip((sat_hi + 0.12 - sat) / 0.12, 0.0, 1.0)

    soft *= person_mask
    m8 = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    return m8


VARIANTS = {"G1": (9, 10), "G2": (11, 12), "G3": (13, 14)}

report = {}
for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    r = person.predict(bgr, imgsz=640, conf=0.25, verbose=False, device=0)[0]
    masks = masks_to_frame(r.masks.data.cpu().numpy(), bgr.shape[:2])
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    pm = masks[int(np.argmax(areas))]

    faces = skin_core.detect_faces(bgr, face_model, conf=0.25, max_faces=4)
    refs = [skin_core._reference_skin(bgr, b) for b in faces]
    ref = np.median(np.stack(refs), axis=0) if refs else np.array([140.0, 146.0, 121.0], np.float32)
    entry = {"ref": [round(float(v), 1) for v in ref]}

    for vname, (cr, cb) in VARIANTS.items():
        m8 = gate_mask(bgr, pm, ref, cr, cb)
        entry[vname] = {
            "frac_in_person": round(float((m8 > 127)[pm > 0.5].mean()), 4),
            "frac_frame": round(float((m8 > 127).mean()), 4),
        }
        scale = 1300.0 / max(h, w)
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m8, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        tint = np.zeros_like(small)
        tint[..., 1] = 255
        tint[..., 2] = 40
        prev = np.where(ms[..., None] > 60, (small * 0.35 + tint * 0.65).astype(np.uint8), small)
        cv2.imwrite(os.path.join(OUT, f"gate_{vname}_{os.path.splitext(name)[0]}.png"), prev)
    report[name] = entry

print(json.dumps(report, indent=2, ensure_ascii=False))
