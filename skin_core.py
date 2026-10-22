"""Lightweight portrait skin retouching.

Only one model is involved: a YOLOv8 face detector that is already present in
ComfyUI/models/ultralytics/bbox/face_yolov8m.pt. Its sole job is to bound the
search area, so background skin-toned objects are never touched. Everything
else is deterministic OpenCV math, which keeps the result reproducible and
lets this run on CPU in well under a second for a 1 MP portrait.

Per face:
  1. YOLOv8 face box.
  2. Face-ellipse prior trimmed at the hairline.
  3. Adaptive YCrCb skin probability referenced to both cheeks, nose bridge and
     chin, with dark pixels (eyes, brows, nostril shadow) and saturated pixels
     (lips) suppressed.
  4. Cascade scale-space smoothing with a guided filter:
       fine = G(I)     removes sensor noise below pore scale
       mid  = G(fine)  removes blemishes at 3-15 px -> clean base
       midband = fine - mid   (blemish scale, dropped)
       highband = I - fine    (pore scale, kept)
  5. Amplitude-adaptive detail restore: weak texture is suppressed, strong
     gradients (lashes, lip line, nostril) survive.
  6. Local chroma flattening for redness and blotches, mean-preserving.
"""

from __future__ import annotations

import os
import warnings

import cv2
import numpy as np

_FACE_MODEL = None
_FACE_MODEL_PATH = None


def _optional_module(name):
    """Import a sibling module whether or not this file is loaded as a package.

    ComfyUI imports the folder as a package, so `import skin_body` finds
    nothing there; the tests import this file directly, where the relative form
    fails instead. Both entry points have to work.
    """
    import importlib

    if __package__:
        return importlib.import_module("." + name, __package__)
    return importlib.import_module(name)


def _default_face_model():
    # Prefer the copy inside this project so it runs standalone; fall back to
    # ComfyUI's model library when installed as a custom node.
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_yolov8m.pt")
    if os.path.exists(local):
        return local
    import folder_paths

    return os.path.join(folder_paths.models_dir, "ultralytics", "bbox", "face_yolov8m.pt")


def load_face_detector(model_path=None):
    """Load and cache the face detector; loading costs ~1.5 s, inference ~0.1 s."""
    global _FACE_MODEL, _FACE_MODEL_PATH
    path = model_path or _default_face_model()
    if _FACE_MODEL is not None and _FACE_MODEL_PATH == path:
        return _FACE_MODEL
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Face detector not found at %s. Put face_yolov8m.pt in models/ or pass model_path."
            % (path,)
        )
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yolo"))
    _warm_cuda()
    from ultralytics import YOLO

    _FACE_MODEL = YOLO(path)
    _FACE_MODEL_PATH = path
    return _FACE_MODEL


def _warm_cuda():
    """Force the CUDA device count to be computed before ultralytics runs.

    torch caches the count the first time it is asked. ultralytics masks
    CUDA_VISIBLE_DEVICES with '-1' for a CPU request, so if that cached count is
    gathered while the mask is in place it stays 0 for the rest of the process
    and torch.cuda.is_available() reports False forever -- which silently
    demotes the GPU filter to CPU. Asking once here, before the import, pins the
    real count.
    """
    try:
        import torch

        torch.cuda.is_available()
    except Exception:
        pass


def _keep_cuda_visible():
    """Snapshot CUDA_VISIBLE_DEVICES so a detector call cannot leak its own value.

    ultralytics' select_device sets CUDA_VISIBLE_DEVICES to '-1' whenever it is
    asked for the CPU, and leaves it that way. A torch import that happens after
    that reads the mask and reports no CUDA device at all -- which silently
    demoted every later GPU stage to CPU while still printing device="cuda".
    Returns a restore callable.
    """
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")

    def restore():
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
        return saved

    return restore


def detect_faces(bgr, model, conf=0.35, max_faces=4):
    """Return [(x1, y1, x2, y2, score)] sorted by score, in pixel coords."""
    h, w = bgr.shape[:2]
    restore = _keep_cuda_visible()
    try:
        res = model.predict(bgr, imgsz=640, conf=conf, verbose=False, device="cpu")
    finally:
        restore()
    boxes = []
    if res and res[0].boxes is not None and len(res[0].boxes) > 0:
        xyxy = res[0].boxes.xyxy.cpu().numpy()
        cf = res[0].boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, cf):
            x1 = int(np.clip(x1, 0, w - 2))
            y1 = int(np.clip(y1, 0, h - 2))
            x2 = int(np.clip(x2, x1 + 2, w - 1))
            y2 = int(np.clip(y2, y1 + 2, h - 1))
            boxes.append((x1, y1, x2, y2, float(c)))
    boxes.sort(key=lambda t: t[4], reverse=True)
    return boxes[: int(max_faces)]


def _face_ellipse_mask(shape, box):
    h, w = shape[:2]
    x1, y1, x2, y2 = box[:4]
    fw = float(x2 - x1)
    fh = float(y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy + fh * 0.03))),
        (max(2, int(round(fw * 0.505))), max(2, int(round(fh * 0.60)))),
        0, 0, 360, 255, -1,
    )
    # A YOLO face box includes forehead hair; cut the cap above the brow line.
    hair_y = int(round(cy - fh * 0.38))
    if hair_y > 0:
        mask[:hair_y, :] = 0
    return mask


def _reference_skin(bgr, box):
    """Median YCrCb over both cheeks plus the nose bridge.

    A single centre patch lands on nostril shadow or eye for small faces, which
    is how a full-body shot ends up with an almost empty mask.
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box[:4]
    fw, fh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    samples = []
    for ox, oy in ((-0.20, 0.14), (0.20, 0.14), (0.0, 0.00), (0.0, 0.22)):
        pcx = cx + fw * ox
        pcy = cy + fh * oy
        hw = max(2, int(fw * 0.11))
        hh = max(2, int(fh * 0.075))
        sx1 = int(np.clip(pcx - hw, 0, w - 1))
        sx2 = int(np.clip(pcx + hw, 0, w - 1))
        sy1 = int(np.clip(pcy - hh, 0, h - 1))
        sy2 = int(np.clip(pcy + hh, 0, h - 1))
        if sx2 - sx1 >= 2 and sy2 - sy1 >= 2:
            patch = cv2.cvtColor(bgr[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2YCrCb)
            samples.append(patch.reshape(-1, 3).astype(np.float32))
    if not samples:
        return np.array([140.0, 150.0, 110.0], np.float32)
    return np.median(np.concatenate(samples, axis=0), axis=0)


def face_skin_mask(bgr, box, lo=17.0, hi=52.0):
    """Per-face soft skin mask (uint8 0-255), before feathering."""
    prior = _face_ellipse_mask(bgr.shape, box)
    if prior.max() == 0:
        return prior

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ref = _reference_skin(bgr, box)
    # Skin identity lives in chroma. Weighting Y fully would reject half of a
    # backlit face, where Y swings widely across cheeks and jaw.
    diff = (ycc - ref[None, None, :]) * np.array([0.35, 1.0, 1.0], np.float32)
    dist = np.linalg.norm(diff, axis=2)
    soft = np.clip(1.0 - (dist - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    Y = ycc[..., 0]
    ref_y = max(float(ref[0]), 1e-3)
    # Eyes, brows, lashes and nostril shadow sit far below skin luminance.
    dark = np.clip((Y - 0.26 * ref_y) / (0.12 * ref_y), 0.0, 1.0)
    soft = soft * dark

    # Lips are skin-hued but strongly saturated; dropping them keeps lip texture.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    hue = hsv[..., 0] * 2.0
    lipish = ((hue < 22.0) | (hue > 330.0)) & (sat > 0.52)
    soft[lipish] *= 0.18

    soft = soft * (prior.astype(np.float32) / 255.0)
    out = np.clip(soft * 255.0, 0, 255).astype(np.uint8)

    # Remove specks, then fill pinholes left by thresholding.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k3)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k3)
    return out


def _guided(guide, src, radius, eps):
    return cv2.ximgproc.guidedFilter(guide, src, int(max(1, radius)), float(eps))


def _mad_sigma(x):
    med = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - med))) + 1e-6


def retouch_channel(ch, weight, strength, texture_keep, blemish, face_px):
    """Cascade scale-space smoothing on one float32 channel in 0-255 scale."""
    r_fine = int(np.clip(round(face_px * 0.010), 2, 64))
    r_mid = int(np.clip(round(face_px * (0.030 + 0.035 * blemish)), 4, 160))

    fine = _guided(ch, ch, r_fine, (0.010 * 255.0) ** 2)
    mid = _guided(fine, fine, r_mid, (0.030 * 255.0) ** 2)

    # mid   = clean base (blemishes >= r_mid are gone)
    # midband = 8..30 px, i.e. exactly the blemish/blotch scale -> drop it
    # highband = under pore scale -> the texture worth keeping
    midband = fine - mid
    highband = ch - fine

    keep = float(np.clip(texture_keep, 0.0, 1.0))
    active = weight > 0.01
    tau = max(2.0, 2.5 * _mad_sigma(highband[active])) if np.any(active) else 6.0
    # exp(-(hf/tau)^2) is ~1 on pore-level noise and ~0 on lashes, lip line, nostril.
    w_edge = np.exp(-(highband / tau) ** 2)
    kill = float(np.clip(strength * (1.0 - keep), 0.0, 1.0))

    return mid + midband * (0.12 * keep) + highband * (1.0 - kill * w_edge)


def retouch_face(bgr_f, weight, face_px, strength=0.7, texture_keep=0.45,
                 blemish=0.6, tone_even=0.5, mean_ref=None):
    """Retouch one face. bgr_f is float32 0-255; weight is HxW 0-1.

    mean_ref pins the target mean luminance. Body retouching runs in horizontal
    strips, and a per-strip mean would leave visible banding, so the caller
    passes one value measured over the whole skin region.
    """
    lab = cv2.cvtColor(bgr_f / 255.0, cv2.COLOR_BGR2LAB)
    L = lab[..., 0] * (255.0 / 100.0)
    a = lab[..., 1]
    b = lab[..., 2]

    L2 = retouch_channel(L, weight, strength, texture_keep, blemish, face_px)

    if tone_even > 0.0:
        r_ch = int(np.clip(round(face_px * (0.08 + 0.10 * tone_even)), 8, 200))
        # Blotches are a chroma defect and their amplitude is small, so the
        # tolerance curve is lifted: sqrt keeps low settings useful.
        t = float(np.sqrt(np.clip(tone_even, 0.0, 1.0))) * strength * weight
        a = a + (cv2.blur(a, (r_ch, r_ch)) - a) * t
        b = b + (cv2.blur(b, (r_ch, r_ch)) - b) * t

    # Keep the original mean luminance inside the mask: no global lift.
    m = weight > 0.02
    if np.any(m):
        target = float(L[m].mean()) if mean_ref is None else float(mean_ref)
        L2 = L2 + (target - float(L2[m].mean()))

    lab2 = np.stack([L2 * (100.0 / 255.0), a, b], axis=-1).astype(np.float32)
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR) * 255.0
    out = np.clip(out, 0.0, 255.0)
    # Clamp here: faces are processed in series, and an out-of-range buffer
    # would be fed back into cvtColor for the next face.
    return np.clip(bgr_f + (out - bgr_f) * weight[..., None], 0.0, 255.0)


def feather_mask(mask, px):
    if px <= 0:
        return mask.astype(np.float32) / 255.0
    k = int(px) * 2 + 1
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), max(1.0, px * 0.5))


def _mean_luma(bgr, weight):
    """Mean L over the mask, measured on a reduced copy so it can be global."""
    h, w = weight.shape[:2]
    scale = max(1, int(round(max(h, w) / 512.0)))
    if scale > 1:
        small = cv2.resize(bgr, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
        wm = cv2.resize(weight, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        small, wm = bgr, weight
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    sel = wm > 0.02
    if not np.any(sel):
        return None
    return float((lab[..., 0] * (255.0 / 100.0))[sel].mean())


def _retouch_body_strips(result, weight, scale_px, strength, texture_keep,
                         blemish, tone_even, strip=640, mean_ref=None):
    """Smooth a body mask in horizontal strips.

    A 20 MP frame in float32 with a dozen intermediates overflows any sane
    budget, and the face path already avoids filtering whole frames for speed.
    Strips carry `pad` rows of context so the guided filters see a complete
    neighbourhood at the seam.
    """
    h, w = weight.shape[:2]
    r_ch = int(np.clip(round(scale_px * (0.08 + 0.10 * tone_even)), 8, 200))
    r_mid = int(np.clip(round(scale_px * (0.030 + 0.035 * blemish)), 4, 160))
    pad = int(max(192, max(r_mid, r_ch) + 16))
    touched = 0
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        ry0 = max(0, y0 - pad)
        ry1 = min(h, y1 + pad)
        wsub = weight[ry0:ry1]
        if float(wsub.max()) <= 0.004:
            continue
        c0, c1 = y0 - ry0, y1 - ry0
        wcore = wsub[c0:c1]
        sel = wcore > 0.0
        if not np.any(sel):
            continue
        sub = result[ry0:ry1].astype(np.float32)
        fixed = retouch_face(
            sub, wsub, scale_px, strength=strength, texture_keep=texture_keep,
            blemish=blemish, tone_even=tone_even, mean_ref=mean_ref,
        )
        src = np.clip(fixed[c0:c1], 0, 255).astype(np.uint8)
        dst = result[y0:y1]
        dst[sel] = src[sel]
        touched += 1
    return touched


def skin_retouch(bgr, strength=0.7, texture_keep=0.45, blemish=0.6,
                 tone_even=0.5, feather=0.5, conf=0.35, max_faces=4,
                 model_path=None, include_body=False, person_model=None,
                 body_strength=None, device=None):
    """Run the full pipeline on a uint8 BGR image.

    feather is relative to face size; a fixed pixel radius erodes the mask of a
    small face in a full-body shot down to nothing.

    Each face is processed inside its own region of interest. Filtering the
    whole frame instead costs 21 s on a 20 MP portrait versus a fraction of a
    second for the face crop, and scales with frame size rather than face size.

    include_body adds whole-person skin (arms, shoulders, chest, legs): the
    largest person is segmented, its skin found by chroma ellipse plus weave
    direction, then smoothed in strips. Face masks are unioned on top and the
    faces are still retouched last, at face scale.

    Returns (bgr_out uint8, mask uint8, info list).
    """
    h, w = bgr.shape[:2]
    if device is None:
        device = "cpu"
    yolo_device = "cpu" if str(device) == "cpu" else 0
    import time

    clock = time.perf_counter()
    model = load_face_detector(model_path)
    faces = detect_faces(bgr, model, conf=conf, max_faces=max_faces)
    t_faces = (time.perf_counter() - clock) * 1000.0

    result = bgr.copy()
    full_mask = np.zeros((h, w), np.float32)
    info = []

    if include_body:
        try:
            skin_body = _optional_module("skin_body")
        except Exception as exc:
            # HANDOFF promises that a missing skin_body.py degrades to face-only.
            warnings.warn(
                "skin_body unavailable, falling back to face-only path: %s" % (exc,),
                RuntimeWarning, stacklevel=2,
            )
            include_body = False

    if include_body:
        clock = time.perf_counter()
        seg = skin_body.load_person_segmenter(person_model)
        pmask, pbox, pscore = skin_body.detect_primary_person(bgr, seg, device=yolo_device)
        t_person = (time.perf_counter() - clock) * 1000.0
        max_face_px = max((float(max(b[2] - b[0], b[3] - b[1])) for b in faces), default=0.0)
        refs = [_reference_skin(bgr, b) for b in faces]
        if refs:
            ref = np.median(np.stack(refs), axis=0)
        else:
            ref = np.array([145.0, 145.0, 122.0], np.float32)
        clock = time.perf_counter()
        bmask = None
        if str(device) != "cpu":
            try:
                skin_gpu = _optional_module("skin_gpu")

                if skin_gpu.pick_device(device) == "cuda":
                    bmask = skin_gpu.body_skin_mask_gpu(
                        bgr, pmask, ref, smooth_px=float(np.clip(max_face_px * 0.004, 1.5, 8.0)),
                        device=device,
                    )
                    bmask = skin_body._fill_small_holes(
                        bmask, int(float((pmask > 0.5).sum()) * 0.02), hole_below=160
                    )
            except Exception as exc:
                # Silent fallback used to hide a dead GPU: 3.6 s quietly became 10 s.
                warnings.warn(
                    "skin_gpu body mask failed, falling back to CPU: %s" % (exc,),
                    RuntimeWarning, stacklevel=2,
                )
                bmask = None
        if bmask is None:
            bmask = skin_body.body_skin_mask(bgr, pmask, ref)
        t_mask = (time.perf_counter() - clock) * 1000.0
        coverage = float((bmask > 127).sum()) / float(h * w)
        backend = "cpu"
        strips = 0
        t_body = 0.0
        if bmask.max() > 0:
            scale_px = skin_body.body_scale_px(bgr, max_face_px)
            fp_b = float(np.clip(scale_px * (0.012 + 0.030 * feather), 3.0, 90.0))
            clock = time.perf_counter()
            soft_b = feather_mask(bmask, fp_b)
            # Hand the face back to the face pass. Without this the face is
            # retouched twice, once at body scale and again at face scale, and
            # the double pass reads as plastic.
            if faces:
                fcut = np.zeros((h, w), np.uint8)
                for fb in faces:
                    np.maximum(fcut, _face_ellipse_mask((h, w), fb), out=fcut)
                soft_b = soft_b * (1.0 - fcut.astype(np.float32) / 255.0)
            t_feather = (time.perf_counter() - clock) * 1000.0
            bs = float(strength if body_strength is None else body_strength)
            fixed = None
            clock = time.perf_counter()
            if str(device) != "cpu":
                try:
                    skin_gpu = _optional_module("skin_gpu")

                    if skin_gpu.pick_device(device) == "cuda":
                        fixed, _ = skin_gpu.retouch_image(
                            bgr, soft_b, scale_px, strength=bs,
                            texture_keep=texture_keep, blemish=blemish,
                            tone_even=tone_even, device=device,
                        )
                        backend = "cuda"
                except Exception as exc:
                    warnings.warn(
                        "skin_gpu retouch failed, falling back to CPU strips: %s" % (exc,),
                        RuntimeWarning, stacklevel=2,
                    )
                    fixed = None
            if fixed is None:
                mref = _mean_luma(result, soft_b)
                strips = _retouch_body_strips(
                    result, soft_b, scale_px, bs, texture_keep, blemish, tone_even,
                    mean_ref=mref,
                )
            else:
                np.copyto(result, fixed, where=soft_b[..., None] > 0.0)
            t_body = (time.perf_counter() - clock) * 1000.0
            full_mask = np.maximum(full_mask, soft_b)
        info.append({
            "kind": "body",
            "box": pbox,
            "score": round(float(pscore), 4),
            "scale_px": round(skin_body.body_scale_px(bgr, max_face_px), 1),
            "ref": [round(float(v), 1) for v in ref],
            "coverage": round(coverage, 5),
            "backend": backend,
            "strips": strips,
            "ms": {
                "person": round(t_person, 1),
                "mask": round(t_mask, 1),
                "body": round(t_body, 1),
                "faces": round(t_faces, 1),
            },
        })

    for box in faces:
        x1, y1, x2, y2, _ = box
        face_px = float(max(x2 - x1, y2 - y1))
        pad = int(round(face_px * 0.30))
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)
        crop = result[ry1:ry2, rx1:rx2]
        local = (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1, 1.0)
        if crop.size == 0:
            continue

        m = face_skin_mask(crop, local)
        if m.max() == 0:
            info.append({"box": box[:4], "coverage": 0.0, "face_px": face_px})
            continue
        fp = float(np.clip(face_px * (0.004 + 0.020 * feather), 1.0, 24.0))
        soft = feather_mask(m, fp)
        fixed = retouch_face(
            crop.astype(np.float32), soft, face_px, strength=strength,
            texture_keep=texture_keep, blemish=blemish, tone_even=tone_even,
        )
        # retouch_face already blended by mask, so only its coverage is pasted.
        sel = soft > 0.0
        # A slice of an ndarray is a view, so writing through it already lands
        # in result; copying it back would be a no-op over the whole ROI.
        sub = result[ry1:ry2, rx1:rx2]
        sub[sel] = np.clip(fixed, 0, 255).astype(np.uint8)[sel]
        full_mask[ry1:ry2, rx1:rx2] = np.maximum(full_mask[ry1:ry2, rx1:rx2], soft)
        info.append({
            "box": box[:4],
            "face_px": round(face_px, 1),
            "feather_px": round(fp, 1),
            "roi": (rx2 - rx1, ry2 - ry1),
            "coverage": round(float((m > 127).mean()), 5),
        })

    return result, np.clip(full_mask * 255.0, 0, 255).astype(np.uint8), info
