"""Torch backend for the retouch math.

cv2 in this environment is a CPU-only build (no cv2.cuda), so the guided filter
is the one thing that actually costs seconds on a 20 MP frame. Reimplementing
the same operators on tensor cores changes that:

  box mean      cumulative sums, O(1) per pixel regardless of radius
  guided filter He's fast variant, identical formula to cv2.ximgproc
  LAB           sRGB -> XYZ -> Lab, matching cv2's float32 path

Values are close to cv2 but not bit-identical, so the CPU path stays the default
for the face-only pipeline that the regression tests pin.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pick_device(prefer=None):
    if prefer is None or prefer == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if prefer == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return prefer


def to_tensor(bgr_u8, device):
    t = torch.from_numpy(np.ascontiguousarray(bgr_u8)).to(device=device, dtype=torch.float32)
    return t.permute(2, 0, 1).unsqueeze(0) / 255.0


def to_numpy_u8(t):
    t = t.clamp(0.0, 1.0)[0].permute(1, 2, 0).mul(255.0).round()
    return t.to("cpu", torch.uint8).numpy()


def _box_mean(x, r):
    """Mean over a (2r+1)^2 window, edge-replicated.

    Separated on purpose. A single 2-D cumulative sum accumulates to
    255 * W * H, which on a 20 MP frame is 5e9: float32 has seven digits, so the
    window difference loses several units and the guided filter stamps faint
    contour lines across smooth skin. Dividing after each 1-D pass keeps every
    intermediate at 255 or 255 * H at most, where the error is a hundredth of a
    grey level.
    """
    if r <= 0:
        return x
    h, w = x.shape[-2:]
    k = 2 * r + 1
    xp = F.pad(x, (r + 1, r, 0, 0), mode="replicate")
    s = xp.cumsum(-1)
    hsum = (s[..., k:] - s[..., :w]) / float(k)
    hp = F.pad(hsum, (0, 0, r + 1, r), mode="replicate")
    sv = hp.cumsum(-2)
    return (sv[..., k:, :] - sv[..., :h, :]) / float(k)


def guided_filter(guide, src, r, eps):
    """Self-guided when guide is src. Both are (1,1,H,W) or (1,C,H,W)."""
    mI = _box_mean(guide, r)
    mp = _box_mean(src, r)
    corrI = _box_mean(guide * guide, r)
    corrIp = _box_mean(guide * src, r)
    varI = corrI - mI * mI
    covIp = corrIp - mI * mp
    a = covIp / (varI + eps)
    b = mp - a * mI
    return _box_mean(a, r) * guide + _box_mean(b, r)


def box_blur(x, r):
    return _box_mean(x, r)


_XYZ_FROM_RGB = None
_RGB_FROM_XYZ = None


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = c.clamp(min=0.0)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def bgr_to_lab(t):
    """t is (1,3,H,W) BGR in 0..1; returns L 0..100, a/b in -128..127."""
    global _XYZ_FROM_RGB
    dev = t.device
    rgb = t[:, [2, 1, 0]]
    lin = _srgb_to_linear(rgb)
    if _XYZ_FROM_RGB is None or _XYZ_FROM_RGB.device != dev:
        _XYZ_FROM_RGB = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=torch.float32, device=dev)
    m = _XYZ_FROM_RGB
    xyz = torch.einsum("ij,njhw->nihw", m, lin)
    white = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    v = xyz / white
    delta = 6.0 / 29.0
    f = torch.where(v > delta ** 3, v.clamp(min=1e-12) ** (1.0 / 3.0), v / (3 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], dim=1)


def lab_to_bgr(lab):
    global _RGB_FROM_XYZ
    dev = lab.device
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0
    def inv(f):
        return torch.where(f > delta, f ** 3, 3 * delta ** 2 * (f - 4.0 / 29.0))
    xyz = torch.stack([inv(fx), inv(fy), inv(fz)], dim=1)
    white = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    xyz = xyz * white
    if _RGB_FROM_XYZ is None or _RGB_FROM_XYZ.device != dev:
        _RGB_FROM_XYZ = torch.tensor([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ], dtype=torch.float32, device=dev)
    lin = torch.einsum("ij,njhw->nihw", _RGB_FROM_XYZ, xyz)
    rgb = _linear_to_srgb(lin).clamp(0.0, 1.0)
    return rgb[:, [2, 1, 0]]


def mad_sigma(x, mask):
    sel = mask
    if int(sel.sum()) == 0:
        return 6.0
    v = x[sel]
    med = v.median()
    return float((1.4826 * (v - med).abs().median()).item()) + 1e-6


def gaussian_blur(x, sigma):
    """Separable Gaussian; sigma in pixels. Used for mask feathering."""
    if sigma <= 0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    k = (k / k.sum()).view(1, 1, 1, -1)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, k)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(x, k.view(1, 1, -1, 1))


def retouch_channel(ch, weight, strength, texture_keep, blemish, scale_px, tau):
    r_fine = int(np.clip(round(scale_px * 0.010), 2, 64))
    r_mid = int(np.clip(round(scale_px * (0.030 + 0.035 * blemish)), 4, 160))
    fine = guided_filter(ch, ch, r_fine, (0.010 * 255.0) ** 2)
    mid = guided_filter(fine, fine, r_mid, (0.030 * 255.0) ** 2)
    midband = fine - mid
    highband = ch - fine
    keep = float(np.clip(texture_keep, 0.0, 1.0))
    w_edge = torch.exp(-(highband / tau) ** 2)
    kill = float(np.clip(strength * (1.0 - keep), 0.0, 1.0))
    return mid + midband * (0.12 * keep) + highband * (1.0 - kill * w_edge)


def _dilate(t, k):
    k = int(k) | 1  # max_pool2d with an even kernel grows the tensor by one
    p = k // 2
    return F.max_pool2d(t, k, 1, p)


def _erode(t, k):
    k = int(k) | 1
    p = k // 2
    return -F.max_pool2d(-t, k, 1, p)


_SOBEL_X = None
_SOBEL_Y = None


def _sobel_energy(Y, k=9):
    global _SOBEL_X, _SOBEL_Y
    dev = Y.device
    if _SOBEL_X is None or _SOBEL_X.device != dev:
        _SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                                dtype=torch.float32, device=dev).view(1, 1, 3, 3)
        _SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                                dtype=torch.float32, device=dev).view(1, 1, 3, 3)
    yp = F.pad(Y, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(yp, _SOBEL_X)
    gy = F.conv2d(yp, _SOBEL_Y)
    r = max(1, k // 2)
    ex = _box_mean(gx * gx, r)
    ey = _box_mean(gy * gy, r)
    return ex, ey


def body_skin_mask_gpu(bgr_u8, person_mask, ref, cr_tol=9.0, cb_tol=7.0,
                       y_lo=0.30, sat_hi=0.45, use_texture=True,
                       open_px=11, close_px=31, edge_shrink=9, smooth_px=0.0,
                       device="cuda"):
    """GPU port of skin_body.body_skin_mask, without the hole fill.

    The caller runs the connected-component hole fill on the CPU result, which
    is cheaper than an iterative reconstruction on the card.
    """
    dev = pick_device(device)
    h, w = bgr_u8.shape[:2]
    t = torch.from_numpy(np.ascontiguousarray(bgr_u8)).to(dev, torch.float32)
    t = t.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
    B, G, R = t[:, 0:1], t[:, 1:2], t[:, 2:3]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128.0
    Cb = (B - Y) * 0.564 + 128.0

    dcr = Cr - float(ref[1])
    dcb = Cb - float(ref[2])
    # Cr relaxes hard where channels approach clipping, Cb only gently: blown
    # skin keeps a small Cb offset while the pale suit sits 12+ away.
    mn = torch.minimum(torch.minimum(B, G), R)
    hot = ((mn - 150.0) / 70.0).clamp(0.0, 1.0)
    q = torch.sqrt((dcr / (cr_tol * (1.0 + 2.2 * hot))) ** 2
                   + (dcb / (cb_tol * (1.0 + 0.6 * hot))) ** 2)
    soft = ((1.15 - q) / 0.40).clamp(0.0, 1.0)
    soft = soft * ((Y - y_lo * float(ref[0])) / (0.12 * float(ref[0]))).clamp(0.0, 1.0)

    mx = torch.maximum(torch.maximum(B, G), R)
    mn = torch.minimum(torch.minimum(B, G), R)
    sat = (mx - mn) / mx.clamp(min=1.0)
    soft = soft * ((sat_hi + 0.12 - sat) / 0.12).clamp(0.0, 1.0)

    if use_texture:
        ex, ey = _sobel_energy(Y, k=9)
        aniso = (ex - ey) / (ex + ey + 1e-3)
        energy = ex + ey
        c_dir = ((aniso - 0.14) / 0.18).clamp(0.0, 1.0)
        c_en = ((energy - 150.0) / 200.0).clamp(0.0, 1.0)
        soft = soft * (1.0 - 0.95 * c_dir * c_en)

    pm = torch.from_numpy(np.ascontiguousarray(person_mask)).to(dev, torch.float32).view(1, 1, h, w)
    pm_bin = (pm > 0.75).float()
    if edge_shrink > 1:
        pm_bin = _erode(pm_bin, int(edge_shrink))
    soft = soft * pm_bin

    out = _erode(soft, int(open_px))
    out = _dilate(out, int(open_px))
    out = _dilate(out, int(close_px))
    out = _erode(out, int(close_px))
    out = out * pm_bin
    if smooth_px > 0.5:
        out = gaussian_blur(out, float(smooth_px))
    res = (out.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)[0, 0]
    return res.to("cpu").numpy()


def retouch_image(bgr_u8, weight_np, scale_px, strength=0.7, texture_keep=0.45,
                  blemish=0.6, tone_even=0.5, device="cuda"):
    """Whole-frame equivalent of skin_core.retouch_face, on the GPU.

    Returns (bgr_out uint8, elapsed_ms). The frame goes up once and comes back
    once; nothing is tiled, because 6 GB holds the whole 20 MP working set.
    """
    dev = pick_device(device)
    import time

    if dev != "cuda":
        raise RuntimeError("retouch_image needs a CUDA device; use the CPU path instead")
    t_start = time.perf_counter()

    t = to_tensor(bgr_u8, dev)
    w = torch.from_numpy(np.ascontiguousarray(weight_np)).to(dev, torch.float32).view(1, 1, *weight_np.shape)

    lab = bgr_to_lab(t)
    L = lab[:, 0:1] * 2.55
    a = lab[:, 1:2]
    b = lab[:, 2:3]

    active = w > 0.01
    r_fine = int(np.clip(round(scale_px * 0.010), 2, 64))
    fine_for_tau = _box_mean(L, r_fine)
    high_for_tau = L - fine_for_tau
    tau = mad_sigma(high_for_tau, active)
    del fine_for_tau, high_for_tau

    L2 = retouch_channel(L, w, strength, texture_keep, blemish, scale_px, tau)
    del L

    if tone_even > 0.0:
        r_ch = int(np.clip(round(scale_px * (0.08 + 0.10 * tone_even)), 8, 200))
        amt = float(np.sqrt(np.clip(tone_even, 0.0, 1.0))) * strength
        # cv2.blur(a, (r_ch, r_ch)) takes a kernel *width*; box_blur takes a
        # *radius*. Passing r_ch straight through doubled the chroma scale.
        br = max(1, r_ch // 2)
        a = a + (box_blur(a, br) - a) * (amt * w)
        b = b + (box_blur(b, br) - b) * (amt * w)

    if int(active.sum()) > 0:
        L2 = L2 + (lab[:, 0:1][active].mean() * 2.55 - L2[active].mean())

    out = lab_to_bgr(torch.cat([L2 * (1.0 / 2.55), a, b], dim=1))
    blended = (t + (out - t) * w).clamp(0.0, 1.0)
    result = to_numpy_u8(blended)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - t_start) * 1000.0
