"""Same gate at full resolution.

The 0.5x trial lost the ribbing: INTER_AREA averaged the 2-3 px weave away, so the
directional term went blind on the torso. Texture must be measured at native scale.
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


def body_skin_full(bgr, person_mask, ref, cr_tol=9.0, cb_tol=10.0,
                   y_lo=0.30, sat_hi=0.45, k_win=9, open_k=11, close_k=31):
    t = {}
    t0 = time.perf_counter()
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[..., 0].astype(np.float32)
    dcr = ycc[..., 1].astype(np.float32) - ref[1]
    dcb = ycc[..., 2].astype(np.float32) - ref[2]
    q = np.sqrt((dcr / cr_tol) ** 2 + (dcb / cb_tol) ** 2)
    soft = np.clip((1.15 - q) / 0.40, 0.0, 1.0)
    soft *= np.clip((Y - y_lo * ref[0]) / (0.12 * ref[0]), 0.0, 1.0)
    t["chroma"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    soft *= np.clip((sat_hi + 0.12 - sat) / 0.12, 0.0, 1.0)
    keep_strict = soft.copy()
    t["sat"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
    ex = cv2.blur(gx * gx, (k_win, k_win))
    ey = cv2.blur(gy * gy, (k_win, k_win))
    aniso = (ex - ey) / (ex + ey + 1e-3)
    energy = ex + ey
    c_dir = np.clip((aniso - 0.14) / 0.18, 0.0, 1.0)
    c_en = np.clip((energy - 150.0) / 200.0, 0.0, 1.0)
    suppress = c_dir * c_en
    soft *= 1.0 - 0.95 * suppress
    t["texture"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    soft *= person_mask
    m8 = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    t["morph"] = round((time.perf_counter() - t0) * 1000, 1)
    return m8, suppress, keep_strict, t


report = {}
for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    r = person.predict(bgr, imgsz=640, conf=0.25, verbose=False, device=0)[0]
    masks = masks_to_frame(r.masks.data.cpu().numpy(), bgr.shape[:2])
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    pm = masks[int(np.argmax(areas))]
    pm = cv2.erode(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    faces = skin_core.detect_faces(bgr, face_model, conf=0.25, max_faces=4)
    refs = [skin_core._reference_skin(bgr, b) for b in faces]
    ref = np.median(np.stack(refs), axis=0) if refs else np.array([140.0, 146.0, 121.0], np.float32)

    m8, sup, strict, timing = body_skin_full(bgr, pm, ref)

    entry = {
        "timing": timing,
        "ref": [round(float(v), 1) for v in ref],
        "frac_person": round(float((m8 > 127)[pm > 0.5].mean()), 4),
        "frac_frame": round(float((m8 > 127).mean()), 4),
    }
    report[name] = entry

    scale = 1300.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m8, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    prev = np.where(ms[..., None] > 60, (small * 0.35 + np.array([40, 255, 0], np.uint8) * 0.65).astype(np.uint8), small)
    cv2.imwrite(os.path.join(OUT, f"g3_{os.path.splitext(name)[0]}.png"), prev)
    sups = cv2.resize(np.clip(sup * 255, 0, 255).astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, f"g3_sup_{os.path.splitext(name)[0]}.png"), sups)

print(json.dumps(report, indent=2, ensure_ascii=False))
