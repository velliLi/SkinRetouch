"""Does SAM (ViT-B, already on disk) give usable part masks at 20 MP?

Prompts: face box, a point on the back of the hand, and a point on the suit.
If the suit separates cleanly from the hand, subtraction beats colour rules.
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

from segment_anything import sam_model_registry, SamPredictor
import skin_core

CKPT = r"E:\comfyui\ComfyUI\models\sams\sam_vit_b_01ec64.pth"
t0 = time.perf_counter()
sam = sam_model_registry["vit_b"](checkpoint=CKPT).to("cuda")
print("load_ms", round((time.perf_counter() - t0) * 1000))

pred = SamPredictor(sam)
report = {}

PROMPTS = {
    "DSC_7300.JPG": {
        "face_box": None,  # filled from the face detector
        "hand_pt": (0.775, 0.795),
        "suit_pt": (0.50, 0.63),
        "arm_pt": (0.12, 0.70),
        "neck_pt": (0.47, 0.375),
    },
    "DSC_4081.JPG": {
        "face_box": None,
        "arm_pt": (0.15, 0.48),
        "suit_pt": (0.56, 0.90),
        "chest_pt": (0.72, 0.86),
    },
}

face_model = skin_core.load_face_detector()

for name, pset in PROMPTS.items():
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    pred.set_image(rgb)
    set_ms = (time.perf_counter() - t0) * 1000
    entry = {"set_image_ms": round(set_ms)}

    def run(pts, box=None, label=""):
        kwargs = {"multimask_output": True}
        if pts is not None:
            kwargs["point_coords"] = np.array(pts, np.float32)
            kwargs["point_labels"] = np.ones(len(pts), np.int32)
        if box is not None:
            kwargs["box"] = np.array(box, np.float32)[None]
        t = time.perf_counter()
        masks, scores, _ = pred.predict(**kwargs)
        ms = (time.perf_counter() - t) * 1000
        return masks, scores, round(ms, 1)

    faces = skin_core.detect_faces(bgr, face_model, conf=0.25, max_faces=1)
    if faces:
        box = faces[0][:4]
        masks, scores, ms = run(None, box, "face")
        entry["face_box"] = {
            "ms": ms, "scores": [round(float(s), 3) for s in scores],
            "frac": [round(float(m.mean()), 4) for m in masks],
        }
        m = masks[int(np.argmax(scores))]
        cv2.imwrite(os.path.join(OUT, f"sam_face_{os.path.splitext(name)[0]}.png"),
                    (m.astype(np.uint8) * 255))

    for key in ("hand_pt", "arm_pt", "neck_pt", "chest_pt", "suit_pt"):
        if key not in pset:
            continue
        fx, fy = pset[key]
        pt = [(fx * w, fy * h)]
        masks, scores, ms = run(pt, None, key)
        entry[key] = {
            "ms": ms, "scores": [round(float(s), 3) for s in scores],
            "frac": [round(float(m.mean()), 4) for m in masks],
        }
        m = masks[int(np.argmax(scores))]
        small = (m.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(OUT, f"sam_{key}_{os.path.splitext(name)[0]}.png"),
                    cv2.resize(small, (w // 4, h // 4), interpolation=cv2.INTER_AREA))

    report[name] = entry

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(OUT, "probe_sam.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
