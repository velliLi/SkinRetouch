"""Local small-model repaint probe.

Question: can an on-disk inpaint model retouch skin *inside a blemish-only
mask* without damaging pixels it was not asked to touch?

Protocol matches blemish_test.py so numbers stay comparable: inject 26 pimples
+ 10 low-frequency blotches + grain inside the skin mask, then repair and score
against the clean reference.

  bad       injected frame (lower bound)
  pipeline  skin_core.skin_retouch, current shipping path
  repaint   local SDXL 9-channel inpaint, discs only

The repaint runs through ComfyUI's own core nodes (InpaintModelConditioning ->
KSampler -> VAEDecode) served on 127.0.0.1, because that is the engine this
machine already has; no Hub access, no config guessing.

Reported per run:
  hf          mean |Laplacian| inside the skin mask (pore-level texture energy)
  resid_disc  mean |out - clean| inside the discs (did the blemish die?)
  ssim        whole-frame SSIM against clean
  vae_drift   mean/999th |model - input| inside the tile *outside* the mask,
              i.e. how much pixels we never asked about got rewritten
  seam        mean |grad| on the disc boundary ring vs the same ring in clean
  sec, peak_vram

Usage:
  python tests/probe_repaint_local.py --tile 1024 --denoise 0.30 0.45 0.60
"""

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import skin_core

OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")
ASSETS = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8189")
CKPT = "utopianPony_v1Inpainting.safetensors"

PROMPT = ("ultra detailed natural skin, fine pores, soft studio light, "
          "photorealistic portrait detail")
NEG = ("blemish, pimple, acne, red spots, blotchy, plastic, waxy, blurry, "
       "painting, cartoon, text, watermark, deformed")


# ---------------------------------------------------------------- frame prep
def build_mask(bgr, model):
    faces = skin_core.detect_faces(bgr, model, conf=0.35, max_faces=4)
    h, w = bgr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for box in faces:
        m = np.maximum(m, skin_core.face_skin_mask(bgr, box))
    return m


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


def grad_mean(bgr, sel):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
                  cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    return float(mag[sel].mean()) if sel.any() else 0.0


def label(img, text):
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def crop_square(img, cx, cy, side):
    h, w = img.shape[:2]
    side = min(side, h, w)
    x1 = int(np.clip(cx - side // 2, 0, w - side))
    y1 = int(np.clip(cy - side // 2, 0, h - side))
    return img[y1:y1 + side, x1:x1 + side].copy(), x1, y1, side


# -------------------------------------------------------------- comfy client
def _post_json(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def prep_mask(mask, grow=0, blur=0.0):
    """Grow/soften a binarised blemish mask and renormalise it to full weight.

    ComfyUI resamples the mask bilinearly down to latent resolution (1/8), so a
    4 px blemish disc keeps only a small fraction of its peak weight there and
    KSamplerX0Inpaint blends most of the original latent back in. Growing and
    renormalising restores weight at latent scale.
    """
    m = mask
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        m = cv2.dilate(m, k)
    m = m.astype(np.float32)
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    peak = float(m.max())
    if peak > 0:
        m = m * (255.0 / peak)
    return np.clip(m, 0, 255).astype(np.uint8)


def upload_png(bgr_or_mask, name):
    """POST /upload/image with a raw multipart body (no requests dependency)."""
    ok, buf = cv2.imencode(".png", bgr_or_mask)
    assert ok
    boundary = "----probe" + uuid.uuid4().hex
    body = b""
    body += ("--%s\r\nContent-Disposition: form-data; name=\"image\"; "
             "filename=\"%s\"\r\nContent-Type: image/png\r\n\r\n"
             % (boundary, name)).encode()
    body += buf.tobytes()
    body += ("\r\n--%s\r\nContent-Disposition: form-data; name=\"overwrite\""
             "\r\n\r\ntrue\r\n--%s--\r\n" % (boundary, boundary)).encode()
    req = urllib.request.Request(
        COMFY + "/upload/image", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=120) as r:
        info = json.loads(r.read())
    return info["name"]


def run_repaint(tile_bgr, tile_mask, steps, denoise, seed=1234, timeout=600):
    img_name = upload_png(tile_bgr, "probe_tile.png")
    mask_name = upload_png(tile_mask, "probe_mask.png")
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "5": {"class_type": "LoadImageMask",
              "inputs": {"image": mask_name, "channel": "red"}},
        "6": {"class_type": "InpaintModelConditioning",
              "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                         "vae": ["1", 2], "pixels": ["4", 0],
                         "mask": ["5", 0], "noise_mask": True}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0],
                         "negative": ["6", 1], "latent_image": ["6", 2],
                         "seed": seed, "steps": steps, "cfg": 5.0,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": denoise}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "probe_repaint"}},
    }
    t0 = time.time()
    res = _post_json(COMFY + "/prompt", {"prompt": wf, "client_id": "skinprobe"})
    pid = res["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.5)
        hist = _get_json("%s/history/%s" % (COMFY, pid))
        entry = hist.get(pid)
        if not entry:
            continue
        st = entry.get("status", {})
        if st.get("status_str") == "error" or st.get("completed") is False and \
                any(m[0] == "execution_error" for m in st.get("messages", [])):
            raise RuntimeError("comfy error: %s" % json.dumps(st)[:800])
        images = entry.get("outputs", {}).get("9", {}).get("images")
        if not images:
            continue
        img = images[0]
        url = ("%s/view?filename=%s&subfolder=%s&type=%s"
               % (COMFY, urllib.parse.quote(img["filename"]),
                  urllib.parse.quote(img.get("subfolder", "")),
                  img.get("type", "output")))
        with urllib.request.urlopen(url, timeout=120) as r:
            buf = np.frombuffer(r.read(), np.uint8)
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return out, time.time() - t0
    raise TimeoutError("comfy prompt %s timed out after %ss" % (pid, timeout))


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="t01_headshoulders_135_00001_.png")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--denoise", type=float, nargs="+", default=[0.30, 0.45, 0.60])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--no-pipeline", action="store_true")
    ap.add_argument("--mask-grow", type=int, default=0,
                    help="dilate the blemish mask by N px before inpainting")
    ap.add_argument("--mask-blur", type=float, default=0.0,
                    help="gaussian sigma applied to the grown mask")
    ap.add_argument("--pre-pipeline", action="store_true",
                    help="feed the CPU retouch output into the inpaint step")
    args = ap.parse_args()

    path = os.path.join(ASSETS, args.asset)
    clean = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    assert clean is not None, path
    model = skin_core.load_face_detector()
    skin = build_mask(clean, model)
    assert skin.max() > 0
    bad, discs = inject(clean, skin)
    sel = skin > 180
    print("frame %dx%d skin_cov=%.4f discs_px=%d"
          % (clean.shape[1], clean.shape[0], sel.mean(), int((discs > 0).sum())))

    got = {"frame": os.path.basename(path), "ckpt": CKPT,
           "clean_hf": hf_energy(clean, sel), "bad_hf": hf_energy(bad, sel)}
    print("hf clean=%.2f bad=%.2f" % (got["clean_hf"], got["bad_hf"]))

    t0 = time.time()
    if args.no_pipeline:
        fixed, m2 = bad.copy(), skin
    else:
        fixed, m2, _info = skin_core.skin_retouch(
            bad, strength=0.75, texture_keep=0.35, blemish=0.7,
            tone_even=0.5, feather=0.5)
    got["pipeline"] = {
        "hf": hf_energy(fixed, m2 > 180),
        "resid_disc": float(np.abs(fixed[discs > 0].astype(np.float32)
                                   - clean[discs > 0].astype(np.float32)).mean()),
        "ssim": ssim_gray(fixed, clean),
        "sec": round(time.time() - t0, 3),
    }
    got["bad"] = {
        "hf": got["bad_hf"],
        "resid_disc": float(np.abs(bad[discs > 0].astype(np.float32)
                                   - clean[discs > 0].astype(np.float32)).mean()),
        "ssim": ssim_gray(bad, clean),
    }
    print("pipeline hf=%.2f resid=%.2f ssim=%.4f %.2fs"
          % (got["pipeline"]["hf"], got["pipeline"]["resid_disc"],
             got["pipeline"]["ssim"], got["pipeline"]["sec"]))
    print("bad      hf=%.2f resid=%.2f ssim=%.4f"
          % (got["bad"]["hf"], got["bad"]["resid_disc"], got["bad"]["ssim"]))

    # tile centred on the disc cloud so every blemish participates
    source = fixed if args.pre_pipeline else bad
    ys, xs = np.where(discs > 0)
    cx, cy = int(np.median(xs)), int(np.median(ys))
    roi_bad, x1, y1, side = crop_square(source, cx, cy, args.tile)
    roi_clean, _, _, _ = crop_square(clean, cx, cy, args.tile)
    roi_mask = (discs[y1:y1 + side, x1:x1 + side] > 0).astype(np.uint8) * 255
    mask_used = prep_mask(roi_mask, args.mask_grow, args.mask_blur)
    print("tile %dx%d at (%d,%d) mask_px=%d grown_px=%d frac=%.4f"
          % (side, side, x1, y1, int((roi_mask > 0).sum()),
             int((mask_used > 127).sum()), float((mask_used > 127).mean())))

    core = mask_used > 127
    k = 9
    ring = (cv2.dilate(mask_used, np.ones((k, k), np.uint8)) > 0) & \
           (cv2.dilate(mask_used, np.ones((3, 3), np.uint8)) == 0)
    outside = ~cv2.dilate(mask_used, np.ones((k, k), np.uint8)).astype(bool)

    runs = []
    for dn in args.denoise:
        out_bgr, sec = run_repaint(roi_bad, mask_used, args.steps, dn)
        if out_bgr.shape[:2] != (side, side):
            out_bgr = cv2.resize(out_bgr, (side, side), interpolation=cv2.INTER_AREA)
        alpha = cv2.GaussianBlur(mask_used, (0, 0), 2.0).astype(np.float32) / 255.0
        blended = roi_bad.astype(np.float32) * (1 - alpha[..., None]) + \
            out_bgr.astype(np.float32) * alpha[..., None]
        repainted = source.copy()
        repainted[y1:y1 + side, x1:x1 + side] = np.clip(blended, 0, 255).astype(np.uint8)

        drift = np.abs(out_bgr.astype(np.float32) - roi_bad.astype(np.float32))[outside]
        run = {
            "denoise": dn,
            "sec": round(sec, 2),
            "mask_frac": float(core.mean()),
            "hf": hf_energy(repainted, sel),
            "hf_tile": hf_energy(repainted[y1:y1 + side, x1:x1 + side],
                                 (skin > 180)[y1:y1 + side, x1:x1 + side]),
            "resid_disc": float(np.abs(repainted[discs > 0].astype(np.float32)
                                       - clean[discs > 0].astype(np.float32)).mean()),
            "ssim": ssim_gray(repainted, clean),
            "vae_drift_mean": float(drift.mean()) if drift.size else 0.0,
            "vae_drift_p999": float(np.percentile(drift, 99.9)) if drift.size else 0.0,
            "vae_drift_max": float(drift.max()) if drift.size else 0.0,
            "seam_repainted": grad_mean(repainted[y1:y1 + side, x1:x1 + side], ring),
            "seam_clean": grad_mean(roi_clean, ring),
            "core_abs_change": float(np.abs(
                out_bgr[core].astype(np.float32) - roi_bad[core].astype(np.float32)).mean()),
        }
        runs.append(run)
        print("denoise=%.2f %.1fs hf=%.2f resid=%.2f ssim=%.4f "
              "drift=%.2f/%.1f/%.1f seam=%.1f(clean %.1f)"
              % (dn, sec, run["hf"], run["resid_disc"], run["ssim"],
                 run["vae_drift_mean"], run["vae_drift_p999"],
                 run["vae_drift_max"], run["seam_repainted"], run["seam_clean"]))
        for tag, im in (("roi", out_bgr), ("full", repainted)):
            cv2.imwrite(os.path.join(OUT, "repaint_g%d_b%.0f_d%.2f%s_%s.png"
                                     % (args.mask_grow, args.mask_blur, dn,
                                        "_pre" if args.pre_pipeline else "", tag)), im)

    got["repaint"] = runs
    got["tile"] = {"side": side, "x1": x1, "y1": y1, "steps": args.steps,
                   "pre_pipeline": bool(args.pre_pipeline),
                   "mask_px": int((roi_mask > 0).sum()),
                   "grown_px": int((mask_used > 127).sum()),
                   "grow": args.mask_grow, "blur": args.mask_blur}
    got["inject_ok"] = abs(got["clean_hf"] - 5.9) < 0.15

    best = runs[0]
    tiles = [
        label(crop_square(clean, cx, cy, 360)[0], "1 reference"),
        label(crop_square(bad, cx, cy, 360)[0], "2 blemished"),
        label(crop_square(fixed, cx, cy, 360)[0], "3 skin_retouch"),
        label(crop_square(repainted, cx, cy, 360)[0],
              "4 %srepaint d=%.2f" % ("retouch+" if args.pre_pipeline else "",
                                      best["denoise"])),
    ]
    cv2.imwrite(os.path.join(OUT, "repaint_compare.png"), np.hstack(tiles))

    with open(os.path.join(OUT, "repaint_report.json"), "w", encoding="utf-8") as f:
        json.dump(got, f, indent=2, ensure_ascii=False)
    print("report ->", os.path.join(OUT, "repaint_report.json"))
    print("inject_matches_blemish_test:", got["inject_ok"])


if __name__ == "__main__":
    main()
