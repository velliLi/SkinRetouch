"""Chroma ellipse + directional-texture suppression, evaluated at half resolution.

Measured (tests/output/aniso_stats.json):
  skin   aniso -0.08..0.05, energy 27..284
  cloth  aniso  0.31..0.41, energy 441..2048
So (aniso, energy) cleanly marks woven fabric while leaving skin untouched.
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


def body_skin(bgr, person_mask, ref, cr_tol=9.0, cb_tol=10.0,
              y_lo=0.30, sat_hi=0.45, dir_on=True):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = ycc[..., 0]

    dcr = ycc[..., 1] - ref[1]
    dcb = ycc[..., 2] - ref[2]
    q = np.sqrt((dcr / cr_tol) ** 2 + (dcb / cb_tol) ** 2)
    soft = np.clip((1.15 - q) / 0.40, 0.0, 1.0)

    soft *= np.clip((gray - y_lo * ref[0]) / (0.12 * ref[0]), 0.0, 1.0)
    sat = hsv[..., 1] / 255.0
    soft *= np.clip((sat_hi + 0.12 - sat) / 0.12, 0.0, 1.0)

    suppress = np.zeros_like(soft)
    if dir_on:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        k = 5
        ex = cv2.blur(gx * gx, (k, k))
        ey = cv2.blur(gy * gy, (k, k))
        aniso = (ex - ey) / (ex + ey + 1e-3)
        energy = ex + ey
        c_dir = np.clip((aniso - 0.14) / 0.18, 0.0, 1.0)
        c_en = np.clip((energy - 150.0) / 200.0, 0.0, 1.0)
        suppress = c_dir * c_en
        soft *= 1.0 - 0.95 * suppress

    soft *= person_mask
    m8 = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k7)
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k15)
    return m8, suppress


SCALE = 0.5
report = {}
for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    t0 = time.perf_counter()
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    r = person.predict(bgr, imgsz=640, conf=0.25, verbose=False, device=0)[0]
    masks = masks_to_frame(r.masks.data.cpu().numpy(), bgr.shape[:2])
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    pm_full = masks[int(np.argmax(areas))]
    t1 = time.perf_counter()

    faces = skin_core.detect_faces(bgr, face_model, conf=0.25, max_faces=4)
    refs = [skin_core._reference_skin(bgr, b) for b in faces]
    ref = np.median(np.stack(refs), axis=0) if refs else np.array([140.0, 146.0, 121.0], np.float32)

    small = cv2.resize(bgr, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_AREA)
    pm_s = cv2.resize(pm_full, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    pm_s = cv2.erode(pm_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    t2 = time.perf_counter()

    m_on, sup = body_skin(small, pm_s, ref, dir_on=True)
    m_off, _ = body_skin(small, pm_s, ref, dir_on=False)
    t3 = time.perf_counter()

    entry = {
        "ms_detect": round((t1 - t0) * 1000, 1),
        "ms_mask": round((t3 - t2) * 1000, 1),
        "ref": [round(float(v), 1) for v in ref],
        "dir_frac_person": round(float((m_on > 127)[pm_s > 0.5].mean()), 4),
        "nodir_frac_person": round(float((m_off > 127)[pm_s > 0.5].mean()), 4),
        "suppressed_px": int((sup > 0.5).sum()),
    }
    for tag, m in (("dir", m_on), ("nodir", m_off)):
        prev = np.where(m[..., None] > 60, (small * 0.35 + np.array([40, 255, 0], np.uint8) * 0.65).astype(np.uint8), small)
        cv2.imwrite(os.path.join(OUT, f"g2_{tag}_{os.path.splitext(name)[0]}.png"), prev)
    report[name] = entry

print(json.dumps(report, indent=2, ensure_ascii=False))
