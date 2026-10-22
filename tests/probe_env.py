"""One-shot environment probe: what this machine can actually do.

Prints JSON so results can be quoted verbatim in the handoff notes.
"""
import json
import os
import sys

out = {"python": sys.version.split()[0], "executable": sys.executable[:60]}

try:
    import cv2
    out["cv2"] = cv2.__version__
    out["cv2_build_info_cuda"] = "CUDA" in cv2.getBuildInformation()
    try:
        out["cv2_cuda_devices"] = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception as exc:  # noqa: BLE001
        out["cv2_cuda_devices"] = f"error: {exc}"
    out["cv2_has_ximgproc"] = hasattr(cv2, "ximgproc")
except Exception as exc:  # noqa: BLE001
    out["cv2"] = f"error: {exc}"

try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda_built"] = torch.version.cuda
    out["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        out["gpu_mem_free_gb"] = round(free / 1024 ** 3, 2)
        out["gpu_mem_total_gb"] = round(total / 1024 ** 3, 2)
except Exception as exc:  # noqa: BLE001
    out["torch"] = f"error: {exc}"

try:
    import ultralytics
    out["ultralytics"] = ultralytics.__version__
except Exception as exc:  # noqa: BLE001
    out["ultralytics"] = f"error: {exc}"

try:
    import onnxruntime
    out["onnxruntime"] = onnxruntime.__version__
    out["onnxruntime_providers"] = onnxruntime.get_available_providers()
except Exception as exc:  # noqa: BLE001
    out["onnxruntime"] = f"error: {exc}"

# Model zoo: what segmentation / parsing weights are already on disk.
roots = [
    r"E:\comfyui\ComfyUI\models",
    r"E:\SkinRetouch\models",
]
found = []
for root in roots:
    if not os.path.isdir(root):
        continue
    for dirpath, _dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 3:
            continue
        for fn in filenames:
            low = fn.lower()
            if low.endswith((".pt", ".onnx", ".pth", ".safetensors")):
                p = os.path.join(dirpath, fn)
                try:
                    mb = round(os.path.getsize(p) / 1024 / 1024, 1)
                except OSError:
                    mb = -1
                found.append({"path": p, "mb": mb})
out["weights"] = found

# Sample image geometry.
imgs = []
for name in ("DSC_4081.JPG", "DSC_7300.JPG"):
    p = os.path.join(r"E:\SkinRetouch\tests\assets", name)
    if os.path.exists(p):
        import numpy as np
        data = np.fromfile(p, np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        imgs.append({"name": name, "shape": list(im.shape), "mb": round(os.path.getsize(p) / 1024 / 1024, 1)})
out["images"] = imgs

print(json.dumps(out, indent=2, ensure_ascii=False))
