"""Real-photo probe: does a locally-repainted blemish patch beat the CPU pipeline?

Unlike probe_repaint_local.py this does not inject synthetic blemishes.  The mask
comes from the *redness midband* that skin_core.retouch_channel already isolates,
so the experiment measures the whole chain the node would ship: CPU detection ->
SDXL inpaint patch -> composite.  Both inputs are real photographs.
"""
import argparse
import json
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
RED = 2  # BGR


def redness_midband(tile_bgr, face_px, blemish=0.7):
    """Same scale space retouch_channel uses; positive values = redder than local base."""
    r = tile_bgr[:, :, RED].astype(np.float32)
    r_fine = int(np.clip(round(face_px * 0.010), 2, 64))
    r_mid = int(np.clip(round(face_px * (0.030 + 0.035 * blemish)), 4, 160))
    fine = skin_core._guided(r, r, r_fine, (0.010 * 255.0) ** 2)
    mid = skin_core._guided(fine, fine, r_mid, (0.030 * 255.0) ** 2)
    return fine - mid, r_fine, r_mid


def blemish_mask(tile_bgr, skin, face_px, pct=97.0, max_frac=0.05, grow=8, blur=4.0):
    midband, r_fine, r_mid = redness_midband(tile_bgr, face_px)
    active = skin > 180
    resp = np.clip(midband, 0, None)
    if not np.any(active):
        return np.zeros(tile_bgr.shape[:2], np.uint8), {"r_fine": r_fine, "r_mid": r_mid}
    lo = float(np.percentile(resp[active], pct))
    thr = lo
    binm = ((resp > thr) & active).astype(np.uint8) * 255
    # keep the patch budget honest: raise the threshold until the mask fits
    while (binm > 0).mean() > max_frac and thr < float(resp[active].max()):
        thr += 1.0
        binm = ((resp > thr) & active).astype(np.uint8) * 255
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = P.prep_mask(binm, grow, blur)
    return mask, {"r_fine": r_fine, "r_mid": r_mid, "thr": round(thr, 3),
                  "pct": pct, "raw_px": int((binm > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="DSC_4081.JPG")
    ap.add_argument("--side", type=int, default=896)
    ap.add_argument("--expand", type=float, default=1.55,
                    help="tile side as a multiple of the face box width")
    ap.add_argument("--denoise", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--pct", type=float, default=97.0)
    ap.add_argument("--max-frac", type=float, default=0.05)
    args = ap.parse_args()

    path = os.path.join(P.ASSETS, args.asset)
    photo = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    assert photo is not None, path
    model = skin_core.load_face_detector()
    faces = skin_core.detect_faces(photo, model, max_faces=1)
    assert faces, "no face detected"
    x1, y1, x2, y2, score = faces[0]
    fw, fh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = int(round(max(fw, fh) * args.expand / 2))
    tx1, ty1 = max(cx - half, 0), max(cy - half, 0)
    tx2, ty2 = min(cx + half, photo.shape[1]), min(cy + half, photo.shape[0])
    tile = photo[ty1:ty2, tx1:tx2]
    scale = args.side / float(max(tile.shape[:2]))
    tile = cv2.resize(tile, (args.side, args.side), interpolation=cv2.INTER_AREA)
    print("photo %dx%d face %dx%d score %.2f -> tile %dx%d (scale %.3f)"
          % (photo.shape[1], photo.shape[0], fw, fh, score,
             tile.shape[1], tile.shape[0], scale))

    face_px = int(round(max(fw, fh) * scale))
    skin_box = (max(int((x1 - tx1) * scale), 0), max(int((y1 - ty1) * scale), 0),
                min(int((x2 - tx1) * scale), args.side - 1),
                min(int((y2 - ty1) * scale), args.side - 1))
    skin = skin_core.face_skin_mask(tile, skin_box)
    print("skin cov %.4f face_px %d" % (float((skin > 180).mean()), face_px))

    mask, info = blemish_mask(tile, skin, face_px, pct=args.pct,
                              max_frac=args.max_frac)
    core = mask > 127
    print("mask px %d frac %.4f thr %.3f (r_fine %d r_mid %d)"
          % (int(core.sum()), float(core.mean()), info["thr"], info["r_fine"], info["r_mid"]))

    out_tile, sec = P.run_repaint(tile, mask, args.steps, args.denoise)
    alpha = cv2.GaussianBlur(mask, (0, 0), 2.0).astype(np.float32) / 255.0
    blended = tile.astype(np.float32) * (1 - alpha[..., None]) + \
        out_tile.astype(np.float32) * alpha[..., None]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    outside = ~cv2.dilate(mask, np.ones((9, 9), np.uint8)).astype(bool)
    drift = np.abs(out_tile.astype(np.float32) - tile.astype(np.float32))[outside]
    inside = np.abs(out_tile.astype(np.float32) - tile.astype(np.float32))[core]
    report = {
        "asset": args.asset,
        "tile_scale": round(scale, 4),
        "face_px": face_px,
        "mask_frac": float(core.mean()),
        "denoise": args.denoise,
        "steps": args.steps,
        "sec": round(sec, 2),
        "drift_outside_mean": float(drift.mean()),
        "drift_outside_p999": float(np.percentile(drift, 99.9)),
        "inside_mean": float(inside.mean()),
        "inside_p99": float(np.percentile(inside, 99)),
        "mask_info": info,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "mask_info"},
                     ensure_ascii=False))

    stem = os.path.splitext(args.asset)[0]
    for tag, im in (("tile", tile), ("mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
                    ("repaint", out_tile), ("blended", blended)):
        cv2.imwrite(os.path.join(OUT, "real_%s_%s.png" % (stem, tag)), im)

    # strongest blemishes, in the same order for every panel
    resp = np.clip(redness_midband(tile, face_px)[0], 0, None)
    picks = []
    work = resp.copy()
    for _ in range(3):
        idx = int(np.argmax(np.where(core, work, 0)))
        py, px = divmod(idx, work.shape[1])
        if work[py, px] <= 0:
            break
        picks.append((px, py))
        cv2.circle(work, (px, py), 40, 0, -1)
    panels = []
    for px, py in picks:
        row = []
        for img, txt in ((tile, "original"), (blended, "repainted")):
            half = 32
            x0 = min(max(px - half, 0), args.side - 64)
            y0 = min(max(py - half, 0), args.side - 64)
            crop = cv2.resize(img[y0:y0 + 64, x0:x0 + 64], (320, 320),
                              interpolation=cv2.INTER_NEAREST)
            row.append(P.label(crop, txt))
        panels.append(np.hstack(row))
    if panels:
        cv2.imwrite(os.path.join(OUT, "real_%s_zoom.png" % stem), np.vstack(panels))

    with open(os.path.join(OUT, "real_%s_report.json" % stem), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("artifacts ->", OUT)


if __name__ == "__main__":
    main()
