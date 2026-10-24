"""Erase-style small inpaint nets (MI-GAN / LaMa) against the SDXL inpaint pass.

Why this exists: the SDXL 9-channel inpaint node leaves speckle on skin. That is
a sampling artefact - a diffusion net asked for "fine pores" at denoise 0.6 will
invent them, and the VAE round trip adds its own grain. Erase models regress a
plausible patch instead of sampling one, so the whole class is worth measuring
against the same injected blemishes.

The metric that matters is a speckle ratio, because "it removed the spot but
left three new ones" is invisible to a mean-difference number:

    speckle = median |Laplacian| inside the mask
              / median |Laplacian| on the skin ring just outside it

Around 1.0 the repaired patch carries the same texture as its surroundings;
well above 1.0 it drew something new.

    python tests/probe_small_inpaint.py
    python tests/probe_small_inpaint.py --skip-sdxl     # no ComfyUI needed
    python tests/probe_small_inpaint.py --asset t07_headshoulders_180_00001_.png
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import skin_core
from server.repaint import prep_mask

OUT = os.path.join(HERE, "output")
ASSETS = os.path.join(HERE, "assets")
WEIGHTS = os.path.join(ROOT, "models", "repaint")

MIGAN_PATH = os.path.join(WEIGHTS, "migan_traced.pt")
LAMA_PATH = os.path.join(WEIGHTS, "big-lama.pt")

GROW, BLUR = 8, 4.0


# ------------------------------------------------------------------ fixtures
def inject(bgr, mask, seed=20240521):
    """Same recipe as blemish_test.inject, plus the disc mask it never returned."""
    rng = np.random.default_rng(seed)
    out = bgr.astype(np.float32).copy()
    h, w = bgr.shape[:2]
    ys, xs = np.where(mask > 200)
    if len(xs) == 0:
        return bgr.copy(), np.zeros((h, w), np.uint8)

    discs = np.zeros((h, w), np.uint8)
    for _ in range(26):
        i = rng.integers(0, len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        r = int(rng.integers(3, 8))
        patch = np.zeros((h, w), np.float32)
        cv2.circle(patch, (cx, cy), r, 1.0, -1)
        cv2.circle(discs, (cx, cy), r + 2, 255, -1)
        patch = cv2.GaussianBlur(patch, (0, 0), max(1.0, r * 0.45))[..., None]
        tint = np.array([90.0, 110.0, 190.0], np.float32)  # BGR reddish
        out = out + (tint - out) * patch * 0.55

    low = np.zeros((h, w), np.float32)
    for _ in range(10):
        i = rng.integers(0, len(xs))
        cv2.circle(low, (int(xs[i]), int(ys[i])), int(rng.integers(18, 46)), 1.0, -1)
    low = cv2.GaussianBlur(low, (0, 0), 14.0) * (mask.astype(np.float32) / 255.0)
    tint = np.array([110.0, 130.0, 205.0], np.float32)
    out = out + (tint - out) * low[..., None] * 0.30

    out = out + rng.normal(0.0, 3.0, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), discs


# ------------------------------------------------------------------- metrics
def hf_energy(bgr, sel):
    g = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    return float(np.abs(g[sel]).mean())


def ssim_gray(a, b):
    return float(ssim(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), data_range=255))


def ring_of(mask_bin, skin, inner=17, outer=51):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner, inner))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer, outer))
    grown = cv2.dilate(mask_bin, k2) > 0
    hole = cv2.dilate(mask_bin, k1) > 0
    return grown & ~hole & (skin > 180)


def speckle(bgr, core, ring):
    """Median Laplacian inside the repaired patch vs the untouched skin ring."""
    g = np.abs(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    a = float(np.median(g[core])) if core.any() else 0.0
    b = float(np.median(g[ring])) if ring.any() else 0.0
    return a, b, a / max(b, 1e-6)


# --------------------------------------------------------------- small nets
def load_migan(device="cpu"):
    return torch.jit.load(MIGAN_PATH, map_location=device).eval()


def load_lama(device="cpu"):
    return torch.jit.load(LAMA_PATH, map_location=device).eval()


def _square_window(shape, cx, cy, side):
    h, w = shape[:2]
    side = int(min(max(side, 64), max(h, w)))
    x0 = int(min(max(cx - side // 2, 0), max(w - side, 0)))
    y0 = int(min(max(cy - side // 2, 0), max(h - side, 0)))
    return x0, y0, min(x0 + side, w), min(y0 + side, h)


def _pad_square(img, value=0):
    h, w = img.shape[:2]
    s = max(h, w)
    pt = (s - h) // 2
    pb = s - h - pt
    pl = (s - w) // 2
    pr = s - w - pl
    if img.ndim == 3:
        return cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REFLECT_101), pt, pl
    return cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REFLECT_101), pt, pl


def _tiles_for(shape, core, side, min_px=32):
    h, w = shape[:2]
    out = []
    for y0 in range(0, h, side):
        for x0 in range(0, w, side):
            if core[y0:y0 + side, x0:x0 + side].sum() >= min_px:
                out.append((y0, x0))
    return out


def inpaint_migan(bgr, mask, model, side=512, device="cpu", min_px=32):
    """Tile the mask with native-512 windows so the net never sees resampled detail.

    Squeezing a whole face crop into one 512 window was the first attempt; it
    threw away the pores the patch was supposed to keep. The traced graph is
    fixed at 512, so the only way to stay 1:1 is to feed it 512-pixel windows.
    """
    core = mask > 127
    if not core.any():
        return bgr.copy(), 0.0
    h, w = bgr.shape[:2]
    res = bgr.copy()
    total = 0.0
    for y0, x0 in _tiles_for(bgr.shape, core, side, min_px):
        y1, x1 = min(y0 + side, h), min(x0 + side, w)
        ph, pw = y1 - y0, x1 - x0
        crop = bgr[y0:y1, x0:x1]
        cmask = mask[y0:y1, x0:x1]
        if ph < side or pw < side:
            crop = cv2.copyMakeBorder(crop, 0, side - ph, 0, side - pw,
                                      cv2.BORDER_REFLECT_101)
            cmask = cv2.copyMakeBorder(cmask, 0, side - ph, 0, side - pw,
                                       cv2.BORDER_REFLECT_101)

        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0 * 2 - 1
        m = (cmask > 120).astype(np.float32)
        erased = rgb * (1 - m[..., None])
        inp = np.concatenate([(0.5 - m)[..., None], erased], axis=2)
        t = torch.from_numpy(inp.transpose(2, 0, 1)[None].copy()).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(t)
        total += (time.perf_counter() - t0) * 1000.0
        arr = ((out[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
               .round().clip(0, 255).astype(np.uint8))
        got = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)[:ph, :pw]

        sel = cmask[:ph, :pw] > 127
        region = res[y0:y1, x0:x1]
        region[sel] = got[sel]
        res[y0:y1, x0:x1] = region
    return res, total


def inpaint_lama(bgr, mask, model, pad=32, device="cpu"):
    """LaMa takes any size; pad to a multiple of `pad` and crop the result back."""
    h, w = bgr.shape[:2]
    ph = (pad - h % pad) % pad
    pw = (pad - w % pad) % pad
    img = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    msk = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    m = (msk > 127).astype(np.float32)
    ti = torch.from_numpy(rgb.transpose(2, 0, 1)[None].copy()).to(device)
    tm = torch.from_numpy(m[None, None].copy()).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ti, tm)
    ms = (time.perf_counter() - t0) * 1000.0
    arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    got = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)[:h, :w]

    res = bgr.copy()
    sel = mask > 127
    res[sel] = got[sel]
    return res, ms


# ------------------------------------------------------------------- report
def graft_hf(base, texture_src, mask, sigma=2.5, strength=1.0):
    """Put the original high-frequency back on top of a smooth repair.

    Erase nets deliver a clean low-frequency patch but no pores, so their tex
    ratio lands near 0.45 while real skin sits at 0.9. The texture that belongs
    there is still present in the source frame - only its low frequencies were
    blemished - so transplant the high-pass band instead of synthesising one.
    """
    a = (mask.astype(np.float32) / 255.0)[..., None] * float(strength)
    low = cv2.GaussianBlur(base, (0, 0), sigma).astype(np.float32)
    src = texture_src.astype(np.float32)
    hf = src - cv2.GaussianBlur(src, (0, 0), sigma).astype(np.float32)
    mixed = low + hf
    out = base.astype(np.float32) * (1.0 - a) + mixed * a
    return np.clip(out, 0, 255).astype(np.uint8)


def evaluate(name, img, clean, discs, core, ring, ms):
    sel = discs > 127
    core_hf = hf_energy(img, core)
    ring_hf = hf_energy(img, ring)
    return {
        "name": name,
        "hf": round(core_hf, 3),
        "resid_disc": round(float(np.abs(img[sel].astype(np.float32)
                                        - clean[sel].astype(np.float32)).mean()), 3),
        "ssim": round(ssim_gray(img, clean), 4),
        "speckle_core": round(speckle(img, core, ring)[0], 3),
        "speckle_ratio": round(speckle(img, core, ring)[2], 3),
        # detail carried by the repaired patch relative to the skin around it;
        # 1.0 keeps the texture it replaced, far below 1.0 means it painted a
        # smooth patch, far above means it invented speckle.
        "tex_ratio": round(core_hf / max(ring_hf, 1e-6), 3),
        "ms": round(ms, 1) if ms is not None else None,
    }


def contact_sheet(rows, out_path, crop=260, zoom=4, cols=4, centre=None):
    """Zoomed crops centred on the blemishes: reference / blemished / retouch / +net."""
    pick = rows[:cols]
    tiles = []
    for r in pick:
        img = r.pop("_img")
        tiles.append(img)
    h, w = tiles[0].shape[:2]
    side = min(crop, h, w)
    cy, cx = centre if centre else (h // 2, w // 2)
    y0 = int(min(max(cy - side // 2, 0), h - side))
    x0 = int(min(max(cx - side // 2, 0), w - side))
    strip = []
    for img, r in zip(tiles, pick):
        c = img[y0:y0 + side, x0:x0 + side]
        c = cv2.resize(c, (side * zoom, side * zoom), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, r["name"], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(c, r["name"], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        strip.append(c)
    cv2.imwrite(out_path, np.concatenate(strip, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="t01_headshoulders_135_00001_.png")
    ap.add_argument("--skip-sdxl", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    stem = os.path.splitext(args.asset)[0]
    path = os.path.join(ASSETS, args.asset)
    clean = cv2.imread(path)
    if clean is None:
        raise SystemExit("missing asset %s" % path)
    h, w = clean.shape[:2]
    print("asset %s  %dx%d" % (args.asset, w, h))

    t0 = time.time()
    _, skin, info = skin_core.skin_retouch(clean, strength=0.0)
    print("skin mask %.2fs  covers %.4f  faces %d" % (time.time() - t0,
          float((skin > 180).mean()), len(info)))
    bad, discs = inject(clean, skin)
    mask = prep_mask(discs, grow=GROW, blur=BLUR)

    core = mask > 127
    ring = ring_of(discs, skin)
    print("mask %.3f%%  discs %d px  ring %d px" % (100.0 * core.mean(),
          int((discs > 127).sum()), int(ring.sum())))

    t0 = time.time()
    fixed, _, _ = skin_core.skin_retouch(bad, strength=0.75, texture_keep=0.35,
                                         blemish=0.7, tone_even=0.5, feather=0.5)
    pipeline_ms = (time.time() - t0) * 1000.0
    print("skin_retouch %.0f ms" % pipeline_ms)

    rows = [("reference", clean, None),
            ("blemished", bad, None),
            ("retouch", fixed, pipeline_ms)]

    if os.path.exists(MIGAN_PATH):
        model = load_migan(args.device)
        img, ms = inpaint_migan(fixed, mask, model, device=args.device)
        rows.append(("retouch+migan", img, ms))
        for k in (1.0, 1.15, 1.3):
            rows.append(("migan+hf%.1f" % k, graft_hf(img, fixed, mask, strength=k), ms))
    else:
        print("migan weights missing at %s" % MIGAN_PATH)

    if os.path.exists(LAMA_PATH):
        model = load_lama(args.device)
        img, ms = inpaint_lama(fixed, mask, model, device=args.device)
        rows.append(("retouch+lama", img, ms))
        rows.append(("lama+hf1.15", graft_hf(img, fixed, mask, strength=1.15), ms))
    else:
        print("lama weights missing at %s" % LAMA_PATH)

    if not args.skip_sdxl:
        try:
            from server import repaint as rp
            if rp.probe():
                t0 = time.perf_counter()
                out = rp.run_tile(fixed, mask, steps=8, denoise=0.6)
                sdxl_ms = (time.perf_counter() - t0) * 1000.0
                res = fixed.copy()
                res[core] = out[core]
                rows.append(("retouch+sdxl", res, sdxl_ms))
            else:
                print("comfyui not answering, sdxl column skipped")
        except Exception as exc:  # noqa: BLE001 - a probe must not die on the optional column
            print("sdxl column failed: %s: %s" % (type(exc).__name__, exc))

    print("\n%-16s %7s %8s %7s %8s %8s %8s %8s" %
          ("state", "hf", "resid", "ssim", "spk_core", "spk_rat", "tex", "ms"))
    results = []
    for name, img, ms in rows:
        r = evaluate(name, img, clean, discs, core, ring, ms)
        r["_img"] = img
        results.append(r)
        print("%-16s %7.3f %8.3f %7.4f %8.3f %8.3f %8.3f %8s" %
              (r["name"], r["hf"], r["resid_disc"], r["ssim"],
               r["speckle_core"], r["speckle_ratio"], r["tex_ratio"],
               "-" if r["ms"] is None else "%.0f" % r["ms"]))

    dy, dx = np.where(discs > 127)
    centre = (int(dy.mean()), int(dx.mean())) if len(dy) else (h // 2, w // 2)
    sheet = os.path.join(OUT, "small_inpaint_%s.png" % stem.replace("_00001_", ""))
    contact_sheet(results, sheet, centre=centre, crop=200, zoom=3, cols=6)
    print("\nwrote %s" % sheet)

    rep = os.path.join(OUT, "small_inpaint_%s.json" % stem.replace("_00001_", ""))
    with open(rep, "w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in r.items() if k != "_img"} for r in results],
                  fh, ensure_ascii=False, indent=2)
    print("wrote %s" % rep)


if __name__ == "__main__":
    main()
