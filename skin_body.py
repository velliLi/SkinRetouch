"""Whole-body skin detection for large portraits.

The face pipeline only needs a box, because the ellipse prior does the heavy
lifting. A body has no such prior, so this module answers one question: which
pixels inside the primary person belong to skin?

Colour alone is not enough. On the reference shots the subject wears a pale
rose bodysuit whose chroma sits a few units from her own skin; a plain YCrCb
distance marks the whole torso. Two extra cues settle it:

  chroma ellipse  skin sits at |dCr| <= 5, |dCb| <= 4 around the face
                  reference, the fabric drifts to dCb +6..+16.
  weave direction ribbed knit has one dominant gradient axis
                  (aniso = (Ex-Ey)/(Ex+Ey) ~ 0.31..0.41), skin is isotropic
                  (-0.08..0.05). Measured at native scale: downscaling averages
                  the 2-3 px weave away and the cue dies.

Whatever survives both cues is closed morphologically, which refills the
specular highlights the chroma test drops (nose bridge, chin, forearm).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

_PERSON_MODEL = None
_PERSON_MODEL_PATH = None


def _default_person_model():
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "person_yolov8m-seg.pt")
    if os.path.exists(local):
        return local
    try:
        import folder_paths

        cand = os.path.join(folder_paths.models_dir, "ultralytics", "segm", "person_yolov8m-seg.pt")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "person_yolov8m-seg.pt"
    )


def load_person_segmenter(model_path=None):
    global _PERSON_MODEL, _PERSON_MODEL_PATH
    path = model_path or _default_person_model()
    if _PERSON_MODEL is not None and _PERSON_MODEL_PATH == path:
        return _PERSON_MODEL
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yolo"))
    # Import torch before ultralytics: on this box ultralytics' device probe
    # reports "CUDA unavailable" when torch has not been initialised yet, and it
    # masks CUDA_VISIBLE_DEVICES for a CPU request, which would pin the cached
    # device count at 0. Asking once here sorts both out.
    try:
        import torch  # noqa: F401

        torch.cuda.is_available()
    except Exception:
        pass
    from ultralytics import YOLO

    if not os.path.exists(path):
        raise FileNotFoundError(
            "person segmentation weights not found: %s\n"
            "Place person_yolov8m-seg.pt in %s or pass person_model."
            % (path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
        )
    _PERSON_MODEL = YOLO(path)
    _PERSON_MODEL_PATH = path
    return _PERSON_MODEL


def resolve_device(device):
    """Return a device ultralytics accepts, downgrading when CUDA is unusable."""
    if device is None or str(device) == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return device
    except Exception:
        pass
    return "cpu"


def masks_to_frame(data, orig_shape):
    """Undo the letterbox: ultralytics masks live in network space, not frame space."""
    n, mh, mw = data.shape
    h, w = int(orig_shape[0]), int(orig_shape[1])
    gain = min(mh / float(h), mw / float(w))
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    pad_w, pad_h = (mw - new_w) // 2, (mh - new_h) // 2
    out = np.empty((n, h, w), np.float32)
    for i in range(n):
        m = data[i, pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        out[i] = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def detect_primary_person(bgr, model, conf=0.25, imgsz=640, device=None):
    """Mask of the largest person, plus a box and a score.

    imgsz=640 beats 1280 here: at 1280 these full-length shots fragment into
    several boxes and the top score drops from 0.95 to 0.42.
    """
    h, w = bgr.shape[:2]
    if device is None:
        device = "cpu"
    saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        res = model.predict(bgr, imgsz=imgsz, conf=conf, verbose=False,
                            device=resolve_device(device))
    finally:
        # ultralytics rewrites CUDA_VISIBLE_DEVICES for its own device choice and
        # never restores it; leaving it masked breaks a later torch import.
        if saved_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
    if not res or res[0].masks is None or len(res[0].masks) == 0:
        return np.zeros((h, w), np.float32), None, 0.0
    data = res[0].masks.data.cpu().numpy()
    scores = res[0].boxes.conf.cpu().numpy() if res[0].boxes is not None else np.zeros(len(data), np.float32)
    masks = masks_to_frame(data, (h, w))
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    idx = int(np.argmax(areas))
    mask = np.clip(masks[idx], 0.0, 1.0)
    ys, xs = np.nonzero(mask > 0.5)
    if ys.size == 0:
        return mask, None, float(scores[idx])
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return mask, box, float(scores[idx])


def _fill_small_holes(mask_u8, max_hole_px, hole_below=1):
    """Fill enclosed gaps that are too small to be real background.

    hole_below is the cut-off for what counts as a gap. Anything under it
    (including the feather skirt, whose values straddle the threshold) is
    treated as a hole, which is what rescues the shadowed patch on a forearm
    that sits just outside the chroma window.
    """
    if max_hole_px <= 0:
        return mask_u8
    holes = (mask_u8 < int(hole_below)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    if num <= 1:
        return mask_u8
    h, w = mask_u8.shape[:2]
    keep = np.zeros(num, bool)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area > max_hole_px:
            continue
        # A hole touching the frame edge is background leaking in, not a highlight.
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        keep[i] = True
    ids = np.nonzero(keep)[0]
    if not ids.size:
        return mask_u8
    out = mask_u8.copy()
    out[np.isin(labels, ids)] = 255
    return out


def texture_suppression(Y, k=9, aniso_lo=0.14, aniso_span=0.18, energy_lo=150.0, energy_span=200.0):
    """0 = looks like skin, 1 = looks like woven fabric."""
    gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
    ex = cv2.blur(gx * gx, (k, k))
    ey = cv2.blur(gy * gy, (k, k))
    aniso = (ex - ey) / (ex + ey + 1e-3)
    energy = ex + ey
    c_dir = np.clip((aniso - aniso_lo) / aniso_span, 0.0, 1.0)
    c_en = np.clip((energy - energy_lo) / energy_span, 0.0, 1.0)
    return c_dir * c_en


def chroma_gate(ycc, ref, cr_tol=9.0, cb_tol=7.0, y_lo=0.30, soft_hi=1.15,
                soft_span=0.40, bgr=None, hot_lo=150.0, hot_span=70.0,
                cr_hot_gain=2.2, cb_hot_gain=0.6):
    """Skin colour window, widened where channels approach clipping.

    A lit forearm runs to Y 200+ on these frames. As one channel saturates the
    chroma is pulled toward neutral, so a fixed +-9 window that fits the cheek
    rejects the highlight on the same arm.

    Cr and Cb are treated differently on purpose. Blown-out skin keeps a Cb
    offset of about +5 from the cheek reference, while the pale suit sits at
    +12 or more. Relaxing Cb as far as Cr would drag the garment back in, so Cb
    gets a much gentler gain.
    """
    Y = ycc[..., 0]
    dcr = ycc[..., 1] - float(ref[1])
    dcb = ycc[..., 2] - float(ref[2])
    cr_gain = 1.0
    cb_gain = 1.0
    if bgr is not None and (cr_hot_gain > 0.0 or cb_hot_gain > 0.0):
        mn = bgr.min(axis=2).astype(np.float32)
        hot = np.clip((mn - hot_lo) / max(hot_span, 1e-6), 0.0, 1.0)
        cr_gain = 1.0 + cr_hot_gain * hot
        cb_gain = 1.0 + cb_hot_gain * hot
    q = np.sqrt((dcr / (cr_tol * cr_gain)) ** 2 + (dcb / (cb_tol * cb_gain)) ** 2)
    soft = np.clip((soft_hi - q) / soft_span, 0.0, 1.0)
    soft *= np.clip((Y - y_lo * float(ref[0])) / (0.12 * float(ref[0])), 0.0, 1.0)
    return soft


def body_skin_mask(bgr, person_mask, ref, cr_tol=9.0, cb_tol=7.0, y_lo=0.30,
                   sat_hi=0.45, use_texture=True, open_px=11, close_px=31,
                   edge_shrink=9, hole_frac=0.02):
    """Soft body-skin mask (uint8 0-255) confined to the person."""
    if person_mask is None or float(person_mask.max()) <= 0.0:
        return np.zeros(bgr.shape[:2], np.uint8)

    pm = (person_mask > 0.75).astype(np.uint8) * 255
    if edge_shrink > 1:
        pm = cv2.erode(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(edge_shrink), int(edge_shrink))))

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    soft = chroma_gate(ycc, ref, cr_tol=cr_tol, cb_tol=cb_tol, y_lo=y_lo, bgr=bgr)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    soft *= np.clip((sat_hi + 0.12 - sat) / 0.12, 0.0, 1.0)

    if use_texture:
        sup = texture_suppression(ycc[..., 0])
        soft *= 1.0 - 0.95 * sup

    soft *= pm.astype(np.float32) / 255.0
    out = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_px), int(open_px))))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_px), int(close_px))))
    out = cv2.bitwise_and(out, pm)

    person_area = float((pm > 127).sum())
    out = _fill_small_holes(out, int(person_area * hole_frac), hole_below=160)
    return out


def body_scale_px(bgr, face_px=None):
    """Feature scale for body skin, in pixels.

    Pore and blemish scale on a body is the same as on a face, so the largest
    detected face is the right ruler; the fallback keeps a full-length shot with
    no detectable face from being smoothed at a 3 px radius.
    """
    short = min(bgr.shape[:2])
    floor = 0.20 * short
    if face_px and face_px > 0:
        return float(max(face_px, floor))
    return float(floor)
