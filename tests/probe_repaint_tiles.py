"""Multi-tile stitch probe: does repainting a full frame in tiles leave a seam?

A 20 MP photo needs several 896 px tiles, and every tile pays its own VAE round
trip.  This measures the resulting tone step and edge gradient across the seam so
the node can decide whether tiles are safe to concatenate blindly.
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


def split_rows(height, tile):
    """Tile origins along one axis so the last tile is flush with the border."""
    if height <= tile:
        return [0]
    starts = list(range(0, height - tile + 1, tile))
    if starts[-1] + tile < height:
        starts.append(height - tile)
    return starts


def seam_report(img, seam_y, band=3):
    """Mean |vertical gradient| on the seam row and on its neighbours."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = {}
    for name, y in (("above", seam_y - band - 1), ("seam", seam_y - 1),
                    ("below", seam_y + band - 1)):
        y = int(np.clip(y, 1, g.shape[0] - 2))
        rows[name] = float(np.abs(g[y + 1] - g[y]).mean())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="t01_headshoulders_135_00001_.png")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--denoise", type=float, default=0.60)
    ap.add_argument("--grow", type=int, default=8)
    ap.add_argument("--blur", type=float, default=4.0)
    ap.add_argument("--mask-mode", choices=("discs", "skin"), default="discs",
                    help="discs = injected blemishes; skin = every skin pixel "
                         "(forces tile overlap on the seam, the worst case)")
    args = ap.parse_args()

    path = os.path.join(P.ASSETS, args.asset)
    clean = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    model = skin_core.load_face_detector()
    skin = P.build_mask(clean, model)
    bad, discs = P.inject(clean, skin)
    mask_all = P.prep_mask((skin > 180).astype(np.uint8) * 255 if args.mask_mode == "skin"
                           else discs, args.grow, args.blur)
    h, w = clean.shape[:2]
    tile = args.tile

    out = bad.copy().astype(np.float32)
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 1), np.float32)
    tiles, times = [], []
    for y0 in split_rows(h, tile):
        for x0 in split_rows(w, tile):
            t_bad = bad[y0:y0 + tile, x0:x0 + tile]
            t_mask = mask_all[y0:y0 + tile, x0:x0 + tile]
            if int((t_mask > 127).sum()) == 0:
                continue
            t_out, sec = P.run_repaint(t_bad, t_mask, args.steps, args.denoise)
            times.append(sec)
            if t_out.shape[:2] != (tile, tile):
                t_out = cv2.resize(t_out, (tile, tile), interpolation=cv2.INTER_AREA)
            a = cv2.GaussianBlur(t_mask, (0, 0), 2.0).astype(np.float32)[..., None] / 255.0
            acc[y0:y0 + tile, x0:x0 + tile] += t_out * a
            wsum[y0:y0 + tile, x0:x0 + tile] += a
            tiles.append({"y0": y0, "x0": x0, "sec": round(sec, 2),
                          "mask_px": int((t_mask > 127).sum())})
    # overlap-aware composite: tiles may cover the same pixel twice
    mean = acc / np.maximum(wsum, 1e-6)
    cover = np.clip(wsum, 0.0, 1.0)
    out = out * (1.0 - cover) + mean * cover
    stitched = np.clip(out, 0, 255).astype(np.uint8)

    seam_y = next((t["y0"] for t in tiles if t["y0"] > 0), None)
    report = {
        "asset": args.asset, "frame": [w, h], "tile": tile, "mask_mode": args.mask_mode,
        "tiles": tiles, "total_sec": round(sum(times), 2),
        "covered_frac": float((cover > 0.01).mean()),
        "resid_disc_bad": float(np.abs(bad[discs > 0].astype(np.float32)
                                       - clean[discs > 0].astype(np.float32)).mean()),
        "resid_disc_stitched": float(np.abs(stitched[discs > 0].astype(np.float32)
                                            - clean[discs > 0].astype(np.float32)).mean()),
        "ssim_stitched": P.ssim_gray(stitched, clean),
    }
    if seam_y is not None:
        report["seam_stitched"] = seam_report(stitched, seam_y)
        report["seam_bad"] = seam_report(bad, seam_y)
        report["seam_clean"] = seam_report(clean, seam_y)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    cv2.imwrite(os.path.join(OUT, "tiles_stitched.png"), stitched)
    if seam_y is not None:
        band = 120
        y1, y2 = max(seam_y - band, 0), min(seam_y + band, h)
        strip = [P.label(bad[y1:y2], "blemished"),
                 P.label(stitched[y1:y2], "stitched repaint"),
                 P.label(clean[y1:y2], "reference")]
        cv2.imwrite(os.path.join(OUT, "tiles_seam.png"), np.hstack(strip))
    with open(os.path.join(OUT, "tiles_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("artifacts ->", OUT)


if __name__ == "__main__":
    main()
