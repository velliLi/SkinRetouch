"""Blemish-level comparison sheet: reference / blemished / CPU retouch / repaint / combo.

Reads the inpaint artefacts written by probe_repaint_local.py so the decision is
made on pixels at blemish scale, not on tile-wide averages.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import skin_core  # noqa: E402
import probe_repaint_local as P  # noqa: E402

OUT = os.path.join(HERE, "output")
SIDE = 56
SCALE = 5


def disc_centres(rel, want=6):
    """Largest isolated discs, at least one crop apart."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats((rel > 0).astype(np.uint8))
    picked = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 20:
            picked.append((area, int(cent[i][0]), int(cent[i][1])))
    picked.sort(reverse=True)
    keep = []
    for _area, cx, cy in picked:
        if all((cx - kx) ** 2 + (cy - ky) ** 2 > (SIDE * 1.2) ** 2 for kx, ky in keep):
            keep.append((cx, cy))
        if len(keep) >= want:
            break
    return keep


def crop(img, cx, cy, side=SIDE, scale=SCALE):
    h, w = img.shape[:2]
    half = side // 2
    x0 = min(max(cx - half, 0), w - side)
    y0 = min(max(cy - half, 0), h - side)
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (side * scale, side * scale),
                      interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grow", type=int, default=8)
    ap.add_argument("--blur", type=int, default=4)
    ap.add_argument("--denoise", default="0.60")
    args = ap.parse_args()

    tag = "repaint_g%d_b%d_d%s" % (args.grow, args.blur, args.denoise)
    panels = [("reference", None), ("blemished", None), ("skin_retouch", None)]
    repaint = cv2.imread(os.path.join(OUT, tag + "_full.png"))
    combo = cv2.imread(os.path.join(OUT, tag + "_pre_full.png"))
    assert repaint is not None and combo is not None, "run probe_repaint_local.py first"

    path = os.path.join(P.ASSETS, "t01_headshoulders_135_00001_.png")
    clean = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    model = skin_core.load_face_detector()
    skin = P.build_mask(clean, model)
    bad, discs = P.inject(clean, skin)
    fixed, _m, _i = skin_core.skin_retouch(bad, strength=0.75, texture_keep=0.35,
                                          blemish=0.7, tone_even=0.5, feather=0.5)

    rows = []
    for cx, cy in disc_centres(discs):
        row = [P.label(crop(clean, cx, cy), "reference"),
               P.label(crop(bad, cx, cy), "blemished"),
               P.label(crop(fixed, cx, cy), "skin_retouch"),
               P.label(crop(repaint, cx, cy), "repaint"),
               P.label(crop(combo, cx, cy), "retouch+repaint")]
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    dst = os.path.join(OUT, "repaint_zoom.png")
    cv2.imwrite(dst, sheet)
    print("zoom sheet ->", dst, sheet.shape)

    # wide view of the blemish cloud, same five panels
    ys, xs = np.where(discs > 0)
    cx, cy = int(np.median(xs)), int(np.median(ys))
    wide = [P.label(crop(clean, cx, cy, 400, 1), "reference"),
            P.label(crop(bad, cx, cy, 400, 1), "blemished"),
            P.label(crop(fixed, cx, cy, 400, 1), "skin_retouch"),
            P.label(crop(repaint, cx, cy, 400, 1), "repaint"),
            P.label(crop(combo, cx, cy, 400, 1), "retouch+repaint")]
    dst2 = os.path.join(OUT, "repaint_compare.png")
    cv2.imwrite(dst2, np.hstack(wide))
    print("wide sheet ->", dst2)


if __name__ == "__main__":
    main()
