"""GPU backend parity and speed check against the cv2 path."""
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

import skin_core
import skin_gpu

dev = skin_gpu.pick_device("cuda")
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", "DSC_7300.JPG"), np.uint8), cv2.IMREAD_COLOR)
h, w = bgr.shape[:2]

# 1. guided filter parity on a mid-size patch.
patch = bgr[1200:2200, 1200:2200]
g = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
r, eps = 13, (0.010 * 255.0) ** 2
t0 = time.perf_counter()
ref_cpu = cv2.ximgproc.guidedFilter(g, g, r, eps)
cpu_ms = (time.perf_counter() - t0) * 1000
tg = torch.from_numpy(g).to(dev).view(1, 1, *g.shape)
t0 = time.perf_counter()
ref_gpu = skin_gpu.guided_filter(tg, tg, r, eps).cpu().numpy()[0, 0]
gpu_ms = (time.perf_counter() - t0) * 1000
print("guided parity  max|d|=%.3f  mean|d|=%.3f  cpu=%.0fms gpu=%.0fms"
      % (np.abs(ref_cpu - ref_gpu).max(), np.abs(ref_cpu - ref_gpu).mean(), cpu_ms, gpu_ms))

# 2. Full-frame guided filter, the actual bottleneck on a 20 MP frame.
L_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
t0 = time.perf_counter()
_ = cv2.ximgproc.guidedFilter(L_full, L_full, 66, (0.030 * 255.0) ** 2)
print("full cv2 guided r=66  %.0f ms" % ((time.perf_counter() - t0) * 1000))
tl = torch.from_numpy(L_full).to(dev).view(1, 1, h, w)
if dev == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
_ = skin_gpu.guided_filter(tl, tl, 66, (0.030 * 255.0) ** 2)
if dev == "cuda":
    torch.cuda.synchronize()
print("full torch guided r=66 %.0f ms" % ((time.perf_counter() - t0) * 1000))

# 3. LAB round-trip accuracy against cv2.
t = skin_gpu.to_tensor(bgr[1200:2200, 1200:2200], dev)
lab_cv = cv2.cvtColor(bgr[1200:2200, 1200:2200].astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
lab_gpu = skin_gpu.bgr_to_lab(t)[0].permute(1, 2, 0).cpu().numpy()
print("lab  L max|d|=%.3f  a max|d|=%.3f  b max|d|=%.3f"
      % (np.abs(lab_cv[..., 0] - lab_gpu[..., 0]).max(),
         np.abs(lab_cv[..., 1] - lab_gpu[..., 1]).max(),
         np.abs(lab_cv[..., 2] - lab_gpu[..., 2]).max()))

# 4. End-to-end retouch_face timing on a 4 MP strip.
crop = bgr[1500:3500, 800:2800].copy()
crop_f = crop.astype(np.float32)
weight = np.zeros(crop.shape[:2], np.float32)
cv2.ellipse(weight, (crop.shape[1] // 2, crop.shape[0] // 2), (700, 800), 0, 0, 360, 1.0, -1)

t0 = time.perf_counter()
_ = skin_core.retouch_face(crop_f, weight, 1303.0, 0.7, 0.45, 0.6, 0.5)
print("cpu retouch_face 4MP %.0f ms" % ((time.perf_counter() - t0) * 1000))
