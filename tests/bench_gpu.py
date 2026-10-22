"""Micro-benchmark: guided filter and box blur at 20 MP, CPU (OpenCV) vs GPU (torch).

The body region is far larger than a face ROI, so the old "GPU is not worth it"
conclusion has to be re-measured rather than inherited.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Benchmark the shipped implementation, not a copy of it. An earlier version
# reimplemented the box filter with a 2-D double cumsum, which is exactly the
# form that lost float32 precision on 20 MP and was replaced.
import skin_gpu

box_mean = skin_gpu._box_mean
guided_torch = skin_gpu.guided_filter


def bench(fn, n=3, warm=1):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize() if DEV.type == "cuda" else None
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(sorted(ts)[len(ts) // 2], 1)


shape = (3712, 5568)
img = np.random.default_rng(0).integers(0, 255, (shape[0], shape[1], 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
report = {"frame": f"{shape[1]}x{shape[0]}", "device": str(DEV)}

g = torch.from_numpy(gray).to(DEV)[None, None]
src = g.clone()

for r in (8, 20, 40):
    report[f"cv2_guided_r{r}_ms"] = bench(lambda r=r: cv2.ximgproc.guidedFilter(gray, gray, r, (0.03 * 255) ** 2))
    report[f"torch_guided_r{r}_ms"] = bench(lambda r=r: guided_torch(g, src, r, (0.03 * 255) ** 2))

report["cv2_blur_r100_ms"] = bench(lambda: cv2.blur(gray, (201, 201)))
report["torch_boxmean_r100_ms"] = bench(lambda: box_mean(g, 100))
report["torch_h2d_ms"] = bench(lambda: torch.from_numpy(gray).to(DEV))
report["torch_d2h_ms"] = bench(lambda: g[0, 0].cpu().numpy())

# Whole 20 MP color conversion cost, both lanes.
report["cv2_bgr2lab_ms"] = bench(lambda: cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
report["cv2_bgr2ycrcb_ms"] = bench(lambda: cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
img_gpu = torch.from_numpy(img).permute(2, 0, 1)[None].to(DEV)
report["torch_h2d_rgb_ms"] = bench(lambda: torch.from_numpy(img).to(DEV))

# Peak memory for a 20 MP float32 guided filter.
if DEV.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    guided_torch(g, src, 20, (0.03 * 255) ** 2)
    torch.cuda.synchronize()
    report["peak_gb_guided"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)

print(json.dumps(report, indent=2))
with open(os.path.join(OUT, "bench_gpu.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
