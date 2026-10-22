"""Probe the two 20 MP samples: detector behaviour, GPU speed, segmentation quality.

Writes downscaled previews with boxes and a person mask overlay into tests/output.
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
os.makedirs(OUT, exist_ok=True)

import skin_core

ASSETS = os.path.join(HERE, "assets")
PERSON = r"E:\comfyui\ComfyUI\models\ultralytics\segm\person_yolov8m-seg.pt"

report = {}
face_model = skin_core.load_face_detector()

from ultralytics import YOLO
person = YOLO(PERSON)

import torch
cuda = torch.cuda.is_available()


def timed(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(sorted(ts)[len(ts) // 2], 1)


for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    path = os.path.join(ASSETS, name)
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    key = f"{name} {w}x{h}"
    entry = {}

    for dev in ("cpu", "cuda0"):
        device = "cpu" if dev == "cpu" else 0
        for imgsz in (640, 1280):
            def run_faces():
                return face_model.predict(bgr, imgsz=imgsz, conf=0.25, verbose=False, device=device)
            try:
                r = run_faces()
                boxes = r[0].boxes
                entry[f"face_{dev}_{imgsz}"] = {
                    "ms": timed(run_faces),
                    "n": int(0 if boxes is None else len(boxes)),
                    "conf": [round(float(c), 3) for c in (boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else [])],
                    "wh": [[round(float(a), 0) for a in b] for b in (boxes.xywh.cpu().numpy() if boxes is not None and len(boxes) else [])],
                }
            except Exception as exc:  # noqa: BLE001
                entry[f"face_{dev}_{imgsz}"] = f"error: {exc}"

            def run_person():
                return person.predict(bgr, imgsz=imgsz, conf=0.25, verbose=False, device=device, retina_masks=False)
            try:
                r = run_person()
                b = r[0].boxes
                entry[f"person_{dev}_{imgsz}"] = {
                    "ms": timed(run_person),
                    "n": int(0 if b is None else len(b)),
                    "conf": [round(float(c), 3) for c in (b.conf.cpu().numpy() if b is not None and len(b) else [])],
                    "wh": [[round(float(a), 0) for a in x] for x in (b.xywh.cpu().numpy() if b is not None and len(b) else [])],
                }
                if b is not None and len(b):
                    m = r[0].masks
                    entry[f"person_{dev}_{imgsz}"]["masks"] = None if m is None else [int(v) for v in np.array(m.data.shape)]
            except Exception as exc:  # noqa: BLE001
                entry[f"person_{dev}_{imgsz}"] = f"error: {exc}"

    # Preview at 1600 px long edge with person mask from the best run.
    scale = 1600.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    r = person.predict(bgr, imgsz=1280, conf=0.25, verbose=False, device=0 if cuda else "cpu")
    prev = small.copy()
    if r[0].masks is not None and len(r[0].masks):
        masks = r[0].masks.data.cpu().numpy()
        areas = masks.reshape(len(masks), -1).sum(axis=1)
        best = int(np.argmax(areas))
        mm = cv2.resize(masks[best], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        entry["person_area_frac"] = round(float(masks[best].mean()), 4)
        tint = np.zeros_like(prev)
        tint[..., 2] = 255
        prev = np.where(mm[..., None] > 0.5, (prev * 0.45 + tint * 0.55).astype(np.uint8), prev)
    for i, box in enumerate(skin_core.detect_faces(bgr, face_model, conf=0.25, max_faces=8)):
        x1, y1, x2, y2 = [int(v * scale) for v in box[:4]]
        cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(prev, f"f{i} {box[4]:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT, f"probe_{os.path.splitext(name)[0]}.png"), prev)

    report[key] = entry

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "probe_big.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
