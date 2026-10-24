"""Same comparison as probe_small_inpaint, but on a real photograph.

The injected fixture cannot reproduce what YG reported: an injected pimple sits
on a smooth patch of skin, so every engine looks fine there. A real frame has
pores, fine hair and lens grain inside the detected mask, and that is where a
sampling model starts inventing texture. So this probe takes a real portrait,
lets the production detector pick the mask, and measures the repaired patch
against the skin immediately around it.

    python tests/probe_real_small_inpaint.py --asset DSC_4081.JPG
    python tests/probe_real_small_inpaint.py --asset DSC_7300.JPG --skip-sdxl
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import skin_core  # noqa: E402
from server import repaint as rp  # noqa: E402
from probe_small_inpaint import (  # noqa: E402
    LAMA_PATH,
    MIGAN_PATH,
    graft_hf,
    inpaint_lama,
    inpaint_migan,
    load_lama,
    load_migan,
    ring_of,
    speckle,
    hf_energy,
)

OUT = os.path.join(HERE, "output")
ASSETS = os.path.join(HERE, "assets")


def face_tile(photo, side=896, expand=1.55):
    model = skin_core.load_face_detector()
    faces = skin_core.detect_faces(photo, model, max_faces=1)
    if not faces:
        raise SystemExit("no face detected")
    x1, y1, x2, y2, score = faces[0]
    fw, fh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = int(round(max(fw, fh) * expand / 2))
    tx1, ty1 = max(cx - half, 0), max(cy - half, 0)
    tx2, ty2 = min(cx + half, photo.shape[1]), min(cy + half, photo.shape[0])
    tile = photo[ty1:ty2, tx1:tx2]
    scale = side / float(max(tile.shape[:2]))
    tile = cv2.resize(tile, (side, side), interpolation=cv2.INTER_AREA)
    face_px = int(round(max(fw, fh) * scale))
    skin_box = (max(int((x1 - tx1) * scale), 0), max(int((y1 - ty1) * scale), 0),
                min(int((x2 - tx1) * scale), side - 1),
                min(int((y2 - ty1) * scale), side - 1))
    return tile, face_px, scale, face_px and (fw, fh, score)


def stats(name, img, fixed, core, ring, ms):
    sel = core
    diff = np.abs(img.astype(np.float32) - fixed.astype(np.float32))
    core_hf = hf_energy(img, core)
    ring_hf = hf_energy(img, ring)
    # Sampling engines round-trip the whole tile through the VAE, so every pixel
    # outside the mask moves a little. Erase-style engines leave them bit-exact.
    outside = ~cv2.dilate(core.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    out_diff = diff[outside]
    return {
        "name": name,
        "hf": round(core_hf, 3),
        "tex_ratio": round(core_hf / max(ring_hf, 1e-6), 3),
        "speckle_core": round(speckle(img, core, ring)[0], 3),
        "speckle_ratio": round(speckle(img, core, ring)[2], 3),
        "changed_in": round(float(diff[sel].mean()), 3),
        "changed_p99": round(float(np.percentile(diff[sel], 99)), 2),
        "outside_mean": round(float(out_diff.mean()), 4),
        "outside_p99": round(float(np.percentile(out_diff, 99)), 2),
        "outside_max": round(float(out_diff.max()), 1),
        "ms": round(ms, 1) if ms is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="DSC_4081.JPG")
    ap.add_argument("--side", type=int, default=896)
    ap.add_argument("--expand", type=float, default=1.55)
    ap.add_argument("--denoise", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--skip-sdxl", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(ASSETS, args.asset)
    photo = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if photo is None:
        raise SystemExit("missing asset %s" % path)
    tile, face_px, scale, face = face_tile(photo, args.side, args.expand)
    print("photo %dx%d  face %s  tile %dx%d  scale %.3f  face_px %d"
          % (photo.shape[1], photo.shape[0], face, tile.shape[1], tile.shape[0],
             scale, face_px))

    t0 = time.perf_counter()
    fixed, skin, info = skin_core.skin_retouch(tile, strength=0.75, texture_keep=0.35,
                                              blemish=0.7, tone_even=0.5, feather=0.5)
    skin_ms = (time.perf_counter() - t0) * 1000.0
    print("skin_retouch %.0f ms  skin cov %.4f" % (skin_ms, float((skin > 180).mean())))

    mask, minfo = rp.blemish_mask(fixed, skin, face_px, blemish=0.7)
    core = mask > 127
    ring = ring_of(mask, skin)
    print("mask px %d frac %.4f thr %s raw %s ring %d"
          % (int(core.sum()), float(core.mean()), minfo.get("thr"),
             minfo.get("raw_px"), int(ring.sum())))
    if not core.any():
        raise SystemExit("detector found nothing on this frame")

    rows = [("retouch", fixed, skin_ms)]

    if os.path.exists(MIGAN_PATH):
        model = load_migan(args.device)
        img, ms = inpaint_migan(fixed, mask, model, device=args.device)
        rows.append(("migan", img, ms))
        rows.append(("migan+hf1.15", graft_hf(img, fixed, mask, strength=1.15), ms))
    else:
        print("migan weights missing")

    if os.path.exists(LAMA_PATH):
        model = load_lama(args.device)
        img, ms = inpaint_lama(fixed, mask, model, device=args.device)
        rows.append(("lama", img, ms))
        rows.append(("lama+hf1.15", graft_hf(img, fixed, mask, strength=1.15), ms))
    else:
        print("lama weights missing")

    if not args.skip_sdxl:
        if rp.probe():
            t0 = time.perf_counter()
            out = rp.run_tile(fixed, mask, steps=args.steps, denoise=args.denoise)
            ms = (time.perf_counter() - t0) * 1000.0
            rows.append(("sdxl", out, ms))
        else:
            print("comfyui not answering, sdxl column skipped")

    print("\n%-14s %7s %8s %8s %9s %9s %10s %9s %8s"
          % ("state", "hf", "tex", "spk_rat", "chg_in", "chg_p99", "out_mean",
             "out_p99", "ms"))
    results = []
    for name, img, ms in rows:
        r = stats(name, img, fixed, core, ring, ms)
        r["_img"] = img
        results.append(r)
        print("%-14s %7.3f %8.3f %8.3f %9.3f %9.2f %10.4f %9.2f %8s"
              % (r["name"], r["hf"], r["tex_ratio"], r["speckle_ratio"],
                 r["changed_in"], r["changed_p99"], r["outside_mean"],
                 r["outside_p99"],
                 "-" if r["ms"] is None else "%.0f" % r["ms"]))

    stem = os.path.splitext(args.asset)[0]
    ys, xs = np.where(core)
    cy, cx = int(ys.mean()), int(xs.mean())
    side = 200
    y0 = int(min(max(cy - side // 2, 0), tile.shape[0] - side))
    x0 = int(min(max(cx - side // 2, 0), tile.shape[1] - side))
    strip = []
    for r in results:
        img = r.pop("_img")
        c = img[y0:y0 + side, x0:x0 + side]
        c = cv2.resize(c, (side * 3, side * 3), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, r["name"], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(c, r["name"], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        strip.append(c)
    sheet = os.path.join(OUT, "real_small_%s.png" % stem)
    cv2.imwrite(sheet, np.concatenate(strip, axis=1))
    print("\nwrote %s" % sheet)

    rep = os.path.join(OUT, "real_small_%s.json" % stem)
    with open(rep, "w", encoding="utf-8") as fh:
        json.dump({"asset": args.asset, "face_px": face_px,
                   "mask_frac": float(core.mean()), "rows": results},
                  fh, ensure_ascii=False, indent=2)
    print("wrote %s" % rep)


if __name__ == "__main__":
    main()
