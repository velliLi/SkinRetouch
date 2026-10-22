"""Cluster the person mask in YCrCb to see whether skin and clothing separate at all.

If they do not separate, colour alone cannot solve the torso and a geometric
prior is required; the numbers here decide that.
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


rng = np.random.default_rng(7)
report = {}
for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    r = person.predict(bgr, imgsz=640, conf=0.25, verbose=False, device=0)[0]
    masks = masks_to_frame(r.masks.data.cpu().numpy(), bgr.shape[:2])
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    pm = masks[int(np.argmax(areas))] > 0.5

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ys, xs = np.where(pm)
    idx = rng.choice(len(ys), size=min(300000, len(ys)), replace=False)
    samp = np.stack([ycc[ys[idx], xs[idx], 0] * 0.35, ycc[ys[idx], xs[idx], 1], ycc[ys[idx], xs[idx], 2]], axis=1)

    k = 6
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(samp.astype(np.float32), k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=k))
    entry = {"k": k, "clusters": []}
    for ci in order:
        cnt = int((labels.ravel() == ci).sum())
        entry["clusters"].append({
            "frac": round(cnt / len(labels), 3),
            "Y*0.35": round(float(centers[ci, 0]), 1),
            "Cr": round(float(centers[ci, 1]), 1),
            "Cb": round(float(centers[ci, 2]), 1),
        })

    faces = skin_core.detect_faces(bgr, face_model, conf=0.25, max_faces=4)
    if faces:
        fr = skin_core._reference_skin(bgr, faces[0])
        entry["face_ref"] = [round(float(v), 1) for v in fr]
        d = np.stack([ycc[..., 0] * 0.35 - fr[0] * 0.35, ycc[..., 1] - fr[1], ycc[..., 2] - fr[2]], axis=-1)
        dist = np.linalg.norm(d, axis=2)
        entry["dist_in_person"] = {p: round(float(np.percentile(dist[pm], p)), 1) for p in (5, 25, 50, 75, 90, 95, 99)}

        # Which cluster does each labelled pixel fall in? Visualise the two biggest.
        flat = np.stack([ycc[..., 0][ys[idx], xs[idx]] * 0.35, ycc[..., 1][ys[idx], xs[idx]], ycc[..., 2][ys[idx], xs[idx]]], axis=1)
        d2 = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
        lab = np.argmin(d2, axis=1)
        vis = (bgr * 0.35).astype(np.uint8)
        palette = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)]
        for j, ci in enumerate(order):
            sel = ys[idx][lab == ci], xs[idx][lab == ci]
            vis[sel] = palette[j % len(palette)]
        scale = 1000.0 / max(h, w)
        small = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, f"cluster_{os.path.splitext(name)[0]}.png"), small)

    report[name] = entry

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "probe_cluster.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
