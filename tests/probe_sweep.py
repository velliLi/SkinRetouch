"""Sweep the chroma window and look at what each setting actually selects."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

import skin_body
import skin_core

GRID = [(9.0, 7.0), (12.0, 9.0), (14.0, 10.0), (16.0, 11.0)]
CFG = {
    "DSC_7300.JPG": dict(ref=(145.0, 145.0, 122.0), box=(1361, 851, 2336, 2154)),
    "DSC_4081.JPG": dict(ref=(183.0, 147.0, 124.0), box=(2400, 1000, 3600, 2600)),
}

seg = skin_body.load_person_segmenter()

for name, cfg in CFG.items():
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    pmask, _, _ = skin_body.detect_primary_person(bgr, seg, device=0)
    pbin = (pmask > 0.5)
    ref = np.array(cfg["ref"], np.float32)
    stem = os.path.splitext(name)[0]
    sc = 1000.0 / max(h, w)
    base = cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    for cr, cb in GRID:
        m = skin_body.body_skin_mask(bgr, pmask, ref, cr_tol=cr, cb_tol=cb)
        frac_p = float((m > 127).sum()) / max(1.0, float(pbin.sum()))
        frac_f = float((m > 127).mean())
        mk = cv2.resize(m, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
        tint = np.zeros_like(base)
        tint[..., 2] = 255
        vis = np.where(mk[..., None] > 127, (base * 0.35 + tint * 0.65).astype(np.uint8), base)
        path = os.path.join(OUT, f"sw_{stem}_cr{int(cr)}_cb{int(cb)}.png")
        cv2.imwrite(path, vis)
        print(f"{stem} cr={cr} cb={cb} person={frac_p:.4f} frame={frac_f:.4f} -> {os.path.basename(path)}")
