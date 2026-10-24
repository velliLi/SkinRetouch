"""Wide-mask comparison: the case that actually produces visible blotches.

A whole-face skin mask is wider than one 896 tile, so the ComfyUI path has to
sample several tiles independently and stitch them. Every tile draws its own
noise and its own shading, which is what shows up as patchwork on real skin.
MI-GAN and LaMa run the frame in one pass, so there is nothing to stitch.

    python tests/probe_wide_mask.py
"""

import os
import sys
import time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import skin_core  # noqa: E402
from server import repaint as rp  # noqa: E402
from probe_small_inpaint import graft_hf, hf_energy, load_lama, load_migan, inpaint_lama, inpaint_migan  # noqa: E402

ASSET = os.path.join(ROOT, "tests", "assets", "t01_headshoulders_135_00001_.png")
OUT = os.path.join(ROOT, "tests", "output")


def seams(img, mask, tile, band=2):
    """Mean vertical gradient on each tile boundary vs the rows around it."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = np.abs(np.diff(g, axis=0)).mean(axis=1)
    out = []
    for y in range(tile, h, tile):
        if y - band < 0 or y + band >= len(rows):
            continue
        above = rows[y - band - 2:y - band].mean() if y - band - 2 >= 0 else rows[y - band]
        here = rows[y - band:y + band].mean()
        below = rows[y + band:y + band + 2].mean() if y + band + 2 <= len(rows) else rows[y + band]
        out.append((y, float(above), float(here), float(below)))
    return out


def block_means(img, mask, tile):
    """Average colour of the masked pixels inside each tile: patchwork detector."""
    h, w = img.shape[:2]
    vals = []
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            sel = mask[y0:y0 + tile, x0:x0 + tile] > 127
            if sel.sum() < 500:
                continue
            vals.append(img[y0:y0 + tile, x0:x0 + tile][sel].astype(np.float32).mean(axis=0))
    if len(vals) < 2:
        return 0.0, []
    stacked = np.stack(vals)
    spread = float(np.abs(stacked - stacked.mean(axis=0)).mean())
    return spread, [tuple(round(float(v), 2) for v in row) for row in stacked]


def main():
    os.makedirs(OUT, exist_ok=True)
    clean = cv2.imread(ASSET)
    _, skin, info = skin_core.skin_retouch(clean, strength=0.0)
    face_px = int(info[0]["face_px"])
    fixed, _, info2 = skin_core.skin_retouch(clean, strength=0.75, texture_keep=0.35,
                                             blemish=0.7, tone_even=0.5, feather=0.5)
    wide = rp.prep_mask((skin > 180).astype(np.uint8) * 255, grow=2, blur=3.0)
    core = wide > 127
    ring = wide == 0
    print("asset %dx%d  face_px %d  wide mask %.1f%%"
          % (clean.shape[1], clean.shape[0], face_px, 100.0 * core.mean()))

    rows = [("retouch", fixed, None)]
    if os.path.exists(rp.net_path("migan")):
        net = load_migan("cpu")
        img, ms = inpaint_migan(fixed, wide, net)
        rows.append(("migan", graft_hf(img, fixed, wide, strength=1.15), ms))
    if os.path.exists(rp.net_path("lama")):
        net = load_lama("cpu")
        img, ms = inpaint_lama(fixed, wide, net)
        rows.append(("lama", graft_hf(img, fixed, wide, strength=1.15), ms))
    if rp.probe():
        t0 = time.perf_counter()
        img, st = rp.apply(fixed, fixed, skin, info2,
                           {"blemish": 0.7, "repaint_engine": "sdxl",
                            "repaint_denoise": 0.6, "repaint_steps": 8}, mask=wide)
        rows.append(("sdxl", img, (time.perf_counter() - t0) * 1000.0))
        print("sdxl tiles %s skipped %s" % (st.get("tiles_done"), st.get("tiles_skipped")))

    print("\n%-8s %7s %7s %8s %10s %11s %8s"
          % ("state", "hf", "tex", "spread", "seam_here", "seam_around", "ms"))
    keep = []
    for name, img, ms in rows:
        spread, means = block_means(img, wide, rp.TILE)
        se = seams(img, wide, rp.TILE)
        here = se[0][2] if se else float("nan")
        around = ((se[0][1] + se[0][3]) / 2.0) if se else float("nan")
        hf = hf_energy(img, core)
        ring_hf = hf_energy(img, ring)
        print("%-8s %7.3f %7.3f %8.3f %10.3f %11.3f %8s"
              % (name, hf, hf / max(ring_hf, 1e-6), spread, here, around,
                 "-" if ms is None else "%.0f" % ms))
        if means:
            print("         block means %s" % (means,))
        keep.append((name, img))

    strip = []
    for name, img in keep:
        c = cv2.resize(img, (448, round(448 * img.shape[0] / img.shape[1])),
                       interpolation=cv2.INTER_AREA)
        cv2.putText(c, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(c, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        strip.append(c)
    path = os.path.join(OUT, "wide_mask_compare.png")
    cv2.imwrite(path, np.concatenate(strip, axis=1))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
