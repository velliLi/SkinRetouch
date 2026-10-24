"""Optional repaint pass: clear blemishes with a local inpaint model.

Why this exists
---------------
`skin_retouch` removes blemishes by smoothing them away -- the pixel is
reconstructed from its neighbourhood, so a large spot survives as a soft
discoloured patch. An inpaint model instead *redraws* the masked area, which
recovers a flat skin patch instead of a blurred one. Measured on the injected
blemish benchmark that halves the residual error inside the blemish mask
(7.95 -> 4.48) at the cost of ~10 s per tile, and it leaves every pixel outside
the mask untouched.

Why it is opt-in
----------------
It needs a live ComfyUI process holding a 6.9 GB SDXL inpaint checkpoint (about
2 GB of VRAM) and it is 10x slower than the CPU pipeline. A job with
`repaint=False` must behave exactly as before, so nothing here runs unless asked.

Where the numbers come from
---------------------------
`REPAINT_EXPLORE.md` has the full measurement table. The two constants that are
not free parameters are load-bearing:

* the mask is grown by `GROW_PX` before upload, because ComfyUI resamples it
  bilinearly down to latent resolution (1/8) and a 4 px blemish disc keeps only
  a sub-pixel of weight there -- it then blends the original latent back in and
  the blemish survives untouched;
* only tiles that actually contain mask pixels are sent. Repainting whole skin
  regions drifts the tone and leaves visible seams at tile borders (SSIM 0.954
  -> 0.899 in the same benchmark); sparse masks do not.

Everything is standard library, cv2 and numpy: the project runs on those alone.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import cv2
import numpy as np

DEFAULT_URL = os.environ.get("SKIN_REPAINT_URL", "http://127.0.0.1:8189").rstrip("/")

# Tile side in *source* pixels. 896 is what the benchmark measured; it is close
# to SDXL's native resolution so the redrawn skin texture matches its surround.
TILE = 896
GROW_PX = 8
BLUR_PX = 4.0
MIN_TILE_MASK_PX = 48        # below this a tile is not worth a 10 s request
MAX_TILES = 16               # budget guard; a full-frame mask would need ~35

CKPT = "utopianPony_v1Inpainting.safetensors"
PROMPT = ("ultra detailed natural skin, fine pores, soft studio light, "
          "photorealistic portrait detail")
NEG = ("blemish, pimple, acne, red spots, blotchy, plastic, waxy, blurry, "
       "painting, cartoon, text, watermark, deformed")

RED = 2  # BGR


class RepaintError(RuntimeError):
    """The repaint engine could not be reached or refused the request."""


# ------------------------------------------------------------------ comfy client

def probe(base_url: str = "", timeout: float = 2.0) -> bool:
    """True when a ComfyUI instance answers on `base_url`."""
    url = (base_url or DEFAULT_URL).rstrip("/") + "/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _upload(image, base_url: str) -> str:
    """POST an image to /upload/image with a hand-built multipart body."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RepaintError("PNG encoding failed for upload")
    # A unique name per request: ComfyUI's input dir is shared with the user's
    # own graphs, and `overwrite` would otherwise let a concurrent run clobber
    # this tile mid-flight.
    filename = "skin_%s.png" % uuid.uuid4().hex[:12]
    boundary = "----skinretouch" + uuid.uuid4().hex
    body = b""
    body += ('--%s\r\nContent-Disposition: form-data; name="image"; '
             'filename="%s"\r\nContent-Type: image/png\r\n\r\n'
             % (boundary, filename)).encode("utf-8")
    body += buf.tobytes()
    body += ('\r\n--%s\r\nContent-Disposition: form-data; name="overwrite"'
             '\r\n\r\ntrue\r\n--%s--\r\n' % (boundary, boundary)).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/upload/image", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["name"]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        raise RepaintError("upload to ComfyUI failed: %s" % (exc,)) from None


def _workflow(img_name: str, mask_name: str, steps: int, denoise: float, seed: int) -> dict:
    """9-channel inpaint graph, the same one the benchmark measured."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "5": {"class_type": "LoadImageMask",
              "inputs": {"image": mask_name, "channel": "red"}},
        "6": {"class_type": "InpaintModelConditioning",
              "inputs": {"positive": ["2", 0], "negative": ["3", 0], "vae": ["1", 2],
                         "pixels": ["4", 0], "mask": ["5", 0], "noise_mask": True}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1],
                         "latent_image": ["6", 2], "seed": int(seed), "steps": int(steps),
                         "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal",
                         "denoise": float(denoise)}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "skin_repaint"}},
    }


def run_tile(tile_bgr, tile_mask, steps: int, denoise: float,
             base_url: str = "", seed: int = 1234, timeout: float = 600.0):
    """Repaint one tile; returns the decoded BGR image."""
    base = (base_url or DEFAULT_URL).rstrip("/")
    img_name = _upload(tile_bgr, base)
    mask_name = _upload(tile_mask, base)
    try:
        submitted = _post_json(base + "/prompt",
                               {"prompt": _workflow(img_name, mask_name, steps, denoise, seed),
                                "client_id": "skinretouch"})
        prompt_id = submitted["prompt_id"]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        raise RepaintError("ComfyUI rejected the workflow: %s" % (exc,)) from None

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.4)
        try:
            entry = _get_json("%s/history/%s" % (base, prompt_id)).get(prompt_id)
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if not entry:
            continue
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            raise RepaintError("ComfyUI execution error: %s"
                               % json.dumps(status)[:400])
        images = (entry.get("outputs") or {}).get("9", {}).get("images")
        if not images:
            continue
        image = images[0]
        view = "%s/view?filename=%s&subfolder=%s&type=%s" % (
            base, urllib.parse.quote(image["filename"]),
            urllib.parse.quote(image.get("subfolder", "")),
            image.get("type", "output"))
        with urllib.request.urlopen(view, timeout=120) as resp:
            raw = np.frombuffer(resp.read(), np.uint8)
        out = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if out is None:
            raise RepaintError("ComfyUI returned an undecodable image")
        return out
    raise RepaintError("ComfyUI prompt %s timed out after %.0f s"
                       % (prompt_id, timeout))


# ------------------------------------------------------------------------ masks

def prep_mask(mask, grow: int = 0, blur: float = 0.0):
    """Grow and renormalise a blemish mask so it keeps weight at latent scale.

    ComfyUI resamples the mask bilinearly to 1/8 resolution. A 3-8 px blemish
    disc drops to a sub-pixel there, KSamplerX0Inpaint then blends the original
    latent back and the blemish comes out unchanged. Dilating and rescaling the
    peak to 255 restores enough weight for the sampler to actually redraw.
    """
    out = mask
    if grow > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        out = cv2.dilate(out, kernel)
    out = out.astype(np.float32)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    peak = float(out.max())
    if peak > 0:
        out = out * (255.0 / peak)
    return np.clip(out, 0, 255).astype(np.uint8)


def _redness_midband(bgr, face_px: int, blemish: float):
    """Red channel `fine - mid`: positive where skin is redder than its surround.

    Same guided-filter construction as `skin_core.retouch_channel`, but with a
    much narrower mid scale. `retouch_channel` tunes `mid` to the blotch scale
    (about 40 px at a 780 px face) because it wants to flatten whole patches,
    and that scale throws away spots: a 3..8 px spot, blurred by its halo, is
    10..16 px wide and barely survives a 40 px band-pass. Measured hit rate on
    the injected frame goes from 20.3% to 29.6% at the same spill budget when
    the band is narrowed to about 20 px (`tests/probe_band_scan.py`).
    """
    import skin_core

    red = bgr[:, :, RED].astype(np.float32)
    fine_r = int(np.clip(round(face_px * 0.010), 2, 64))
    mid_r = int(np.clip(round(face_px * (0.014 + 0.020 * float(blemish))), 4, 64))
    fine = skin_core._guided(red, red, fine_r, (0.010 * 255.0) ** 2)
    mid = skin_core._guided(fine, fine, mid_r, (0.030 * 255.0) ** 2)
    return fine - mid


def blemish_mask(bgr, skin, face_px: int, blemish: float = 0.7,
                 pct=None, max_frac=None,
                 grow: int = GROW_PX, blur: float = BLUR_PX):
    """Binarise the strongest redness response inside the skin mask.

    `pct` is a percentile of the positive response rather than an absolute
    threshold, so exposure and skin tone do not need to be tuned per photo.
    `max_frac` caps the patch budget: a red-heavy frame would otherwise mask
    most of the face and drift its tone.

    Both defaults follow `blemish`, so the user-facing slider widens the patch
    instead of only changing the detection scale. At the shipped default (0.7)
    the mask covers roughly 30% of a synthetic spot's area; going much below
    p95 starts repainting ordinary cheek flush as well, which is where the
    tone drift measured in REPAINT_EXPLORE.md comes from
    (`tests/probe_mask_diag.py` prints the hit/spill curve).
    """
    if pct is None:
        pct = 98.0 - 4.0 * float(blemish)
    if max_frac is None:
        max_frac = 0.05 + 0.06 * float(np.clip(blemish, 0.0, 1.0))
    height, width = bgr.shape[:2]
    active = skin > 180
    if not np.any(active):
        return np.zeros((height, width), np.uint8), {"raw_px": 0, "thr": 0.0}
    midband = _redness_midband(bgr, face_px, blemish)
    response = np.where(active, np.clip(midband, 0.0, None), 0.0)
    threshold = float(np.percentile(response[active], pct))
    binary = (response > threshold).astype(np.uint8) * 255
    ceiling = float(response[active].max())
    while (binary > 0).mean() > max_frac and threshold < ceiling:
        threshold += 1.0
        binary = (response > threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    raw_px = int((binary > 0).sum())
    mask = prep_mask(binary, grow, blur)
    return mask, {"raw_px": raw_px, "thr": round(threshold, 2)}


# ------------------------------------------------------------------------ driver

def _starts(length: int, tile: int) -> list[int]:
    """Tile origins along one axis; the last tile stays flush with the border."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, tile))
    if origins[-1] + tile < length:
        origins.append(length - tile)
    return origins


def _plan(h: int, w: int, mask, tile: int):
    """Pick the tiles worth a request, densest first."""
    candidates = []
    for y0 in _starts(h, tile):
        for x0 in _starts(w, tile):
            local = mask[y0:y0 + tile, x0:x0 + tile]
            count = int((local > 127).sum())
            if count >= MIN_TILE_MASK_PX:
                candidates.append((count, y0, x0))
    candidates.sort(reverse=True)
    skipped = max(0, len(candidates) - MAX_TILES)
    return [(y0, x0, count) for count, y0, x0 in candidates[:MAX_TILES]], skipped


def apply(bgr, out, skin, info, params: dict, base_url: str = "",
          timeout_per_tile: float = 300.0, mask=None) -> tuple:
    """Run the repaint pass over `out` and return (image, stats).

    `bgr` is the untouched source: the blemish mask is detected there, because
    the CPU pipeline has already softened the spots and would under-report them.
    `out` is the pipeline result and is what gets repainted, so the model starts
    from clean skin instead of from the blemished pixels.
    `info` is `skin_retouch`'s per-face report, used only for the face scale.
    `mask` overrides the detector: tests feed a known ground truth through it,
    and any future smarter detector plugs into the same place.
    """
    started = time.perf_counter()
    base = (base_url or DEFAULT_URL).rstrip("/")
    height, width = out.shape[:2]

    face_px = 0
    for item in info or []:
        face_px = max(face_px, int(round(float(
            item.get("face_px") or item.get("scale_px") or 0))))
    if face_px <= 0:
        face_px = max(8, min(height, width) // 8)

    if mask is None:
        mask, mask_stats = blemish_mask(
            bgr, skin, face_px, blemish=float(params.get("blemish", 0.7)),
            grow=GROW_PX, blur=BLUR_PX)
    else:
        mask = np.ascontiguousarray(np.asarray(mask, np.uint8))
        mask_stats = {"raw_px": int((mask > 127).sum()), "thr": None}
    plan, skipped = _plan(height, width, mask, TILE)
    stats = {
        "available": True,
        "tiles": len(plan),
        "tiles_skipped": skipped,
        "mask_px": int((mask > 127).sum()),
        "mask_frac": round(float((mask > 127).mean()), 5),
        "face_px": face_px,
        "seed": int(params.get("repaint_seed", 0) or 1234),
        "denoise": float(params.get("repaint_denoise", 0.60)),
        "steps": int(params.get("repaint_steps", 8)),
        **mask_stats,
    }
    if not plan:
        stats["ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        stats["note"] = "no blemish above threshold"
        return out, stats

    result = out.copy()
    per_tile = []
    for y0, x0, count in plan:
        tile_bgr = out[y0:y0 + TILE, x0:x0 + TILE]
        tile_mask = mask[y0:y0 + TILE, x0:x0 + TILE]
        clock = time.perf_counter()
        try:
            repainted = run_tile(tile_bgr, tile_mask,
                                 steps=stats["steps"], denoise=stats["denoise"],
                                 base_url=base, seed=stats["seed"],
                                 timeout=timeout_per_tile)
        except RepaintError:
            stats["tiles_failed"] = stats.get("tiles_failed", 0) + 1
            if not per_tile:
                raise
            continue
        if repainted.shape[:2] != tile_bgr.shape[:2]:
            repainted = cv2.resize(repainted, (tile_bgr.shape[1], tile_bgr.shape[0]),
                                   interpolation=cv2.INTER_AREA)
        alpha = tile_mask.astype(np.float32)[..., None] / 255.0
        region = result[y0:y0 + TILE, x0:x0 + TILE].astype(np.float32)
        result[y0:y0 + TILE, x0:x0 + TILE] = np.clip(
            region * (1.0 - alpha) + repainted.astype(np.float32) * alpha, 0, 255
        ).astype(np.uint8)
        per_tile.append({"y": y0, "x": x0, "mask_px": count,
                         "ms": round((time.perf_counter() - clock) * 1000.0, 1)})

    stats["tiles_done"] = len(per_tile)
    stats["per_tile"] = per_tile
    stats["ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return result, stats
