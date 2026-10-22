"""Iterate on the body-skin mask: person mask x YCrCb band x saturation suppression.

Saves both a full preview and 1:1 crops so threshold errors are visible.
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


def body_skin_mask(bgr, person_mask, ref, lo=16.0, hi=46.0,
                   cr_lo=124.0, cr_hi=174.0, cb_lo=104.0, cb_hi=138.0,
                   y_rel=0.16, sat_cut=0.55):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    d = ycc - ref[None, None, :]
    dist = np.sqrt((d[..., 0] * 0.35) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    soft = np.clip(1.0 - (dist - lo) / (hi - lo), 0.0, 1.0)

    Cr = ycc[..., 1]
    Cb = ycc[..., 2]
    soft *= np.clip((cr_hi - Cr) / 10.0, 0.0, 1.0) * np.clip((Cr - cr_lo) / 8.0, 0.0, 1.0)
    soft *= np.clip((cb_hi - Cb) / 10.0, 0.0, 1.0) * np.clip((Cb - cb_lo) / 8.0, 0.0, 1.0)

    Y = ycc[..., 0]
    ref_y = max(float(ref[0]), 1e-3)
    soft *= np.clip((Y - y_rel * ref_y) / (0.10 * ref_y), 0.0, 1.0)

    sat = hsv[..., 1] / 255.0
    soft *= np.where(sat > sat_cut, np.clip((0.80 - sat) / 0.25, 0.05, 1.0), 1.0)

    soft *= person_mask
    m8 = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    return m8


VARIANTS = {
    "A_base": {},
    "B_tight": {"cr_lo": 130.0, "cr_hi": 170.0, "cb_lo": 108.0, "cb_hi": 134.0, "hi": 40.0},
    "C_loose": {"hi": 56.0, "cr_lo": 118.0, "cr_hi": 186.0, "cb_lo": 98.0, "cb_hi": 142.0},
}

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
    ref = np.median(np.stack(refs), axis=0) if refs else np.array([140.0, 148.0, 120.0], np.float32)

    entry = {"ref": [round(float(v), 1) for v in ref], "faces": len(faces), "person_frac": round(float((pm > 0.5).mean()), 4)}
    for vname, kw in VARIANTS.items():
        m8 = body_skin_mask(bgr, pm, ref, **kw)
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
        cv2.imwrite(os.path.join(OUT, f"body_{vname}_{os.path.splitext(name)[0]}.png"), prev)

    report[name] = entry

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "probe_body.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
