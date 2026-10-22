"""Body-skin candidacy probe: inside the person mask, how well does YCrCb separate skin?

Writes tinted previews plus connected-component statistics so thresholds can be
tuned against numbers instead of guesses.
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


def dominant_ref(ycc, sel, iters=2, lo=17.0, hi=52.0):
    ref = np.median(ycc[sel], axis=0)
    hist = []
    for _ in range(iters):
        d = (ycc - ref[None, None, :]) * np.array([0.35, 1.0, 1.0], np.float32)
        dist = np.linalg.norm(d, axis=2)
        w = (dist < hi) & sel
        if w.sum() < 2000:
            break
        ref = np.median(ycc[w], axis=0)
        hist.append({"ref": [round(float(v), 1) for v in ref], "n": int(w.sum())})
    return ref, hist


report = {}
for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    r = person.predict(bgr, imgsz=640, conf=0.25, verbose=False, device=0)[0]
    masks = masks_to_frame(r.masks.data.cpu().numpy(), bgr.shape[:2])
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    best = int(np.argmax(areas))
    pm = masks[best] > 0.5

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    faces = skin_core.detect_faces(bgr, face_model, conf=0.25, max_faces=4)
    entry = {"person_frac": round(float(pm.mean()), 4), "faces": [[int(v) for v in b[:4]] for b in faces]}

    face_ref = None
    if faces:
        face_ref = skin_core._reference_skin(bgr, faces[0])
    entry["face_ref"] = None if face_ref is None else [round(float(v), 1) for v in face_ref]

    ref, hist = dominant_ref(ycc, pm)
    entry["body_ref"] = [round(float(v), 1) for v in ref]
    entry["ref_iters"] = hist

    # Skin score relative to the body reference.
    for lo, hi in ((17.0, 52.0),):
        d = (ycc - ref[None, None, :]) * np.array([0.35, 1.0, 1.0], np.float32)
        dist = np.linalg.norm(d, axis=2)
        soft = np.clip(1.0 - (dist - lo) / (hi - lo), 0, 1)
        Y = ycc[..., 0]
        ref_y = max(float(ref[0]), 1e-3)
        soft *= np.clip((Y - 0.20 * ref_y) / (0.12 * ref_y), 0, 1)
        sat = hsv[..., 1] / 255.0
        hue = hsv[..., 0] * 2.0
        soft *= np.where(((hue < 30.0) | (hue > 330.0)) & (sat > 0.55), 0.1, 1.0)
        soft *= pm
        m8 = (soft * 255).astype(np.uint8)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k3)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k3)
        binm = (m8 > 127).astype(np.uint8)
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(binm, 8)
        comps = []
        for i in range(1, nlab):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < 2000:
                continue
            comps.append({
                "area": a,
                "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
            })
        comps.sort(key=lambda c: -c["area"])
        entry[f"skin_frac_in_person_{lo}_{hi}"] = round(float((m8 > 127)[pm].mean()), 4)
        entry[f"skin_frac_frame_{lo}_{hi}"] = round(float((m8 > 127).mean()), 4)
        entry[f"components_{lo}_{hi}"] = comps[:12]
        entry[f"ncomp_{lo}_{hi}"] = len(comps)

        scale = 1300.0 / max(h, w)
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m8, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        tint = np.zeros_like(small)
        tint[..., 1] = 255
        prev = np.where(ms[..., None] > 90, (small * 0.45 + tint * 0.55).astype(np.uint8), small)
        cv2.imwrite(os.path.join(OUT, f"skin_{os.path.splitext(name)[0]}.png"), prev)
        cv2.imwrite(os.path.join(OUT, f"skinmask_{os.path.splitext(name)[0]}.png"), m8)

    report[name] = entry

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "probe_skin.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
