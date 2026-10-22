"""Load the custom node the way ComfyUI does and run it end to end."""

import importlib.util
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "ComfyUI_SkinRetouch", os.path.join(ROOT, "__init__.py"),
    submodule_search_locations=[ROOT],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["ComfyUI_SkinRetouch"] = mod
spec.loader.exec_module(mod)

print("mappings", list(mod.NODE_CLASS_MAPPINGS))
node_cls = mod.NODE_CLASS_MAPPINGS["SkinRetouch"]
node = node_cls()
print("display", mod.NODE_DISPLAY_NAME_MAPPINGS["SkinRetouch"])
print("inputs", sorted(node_cls.INPUT_TYPES()["required"]))
print("returns", node_cls.RETURN_TYPES, node_cls.RETURN_NAMES)

bgr = cv2.imdecode(
    np.fromfile(os.path.join(HERE, "assets", "t07_laughing_candid_135_00001_.png"), np.uint8),
    cv2.IMREAD_COLOR,
)
img_in = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)[None]
gray = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)[None, ..., None] / 255.0)

for label, inp in (("rgb", img_in), ("gray", gray)):
    out, mask = node.run(inp, 0.75, 0.35, 0.7, 0.5, 0.5, 4, 0.35)
    assert out.dtype == torch.float32, out.dtype
    assert out.shape[0] == inp.shape[0]
    assert mask.shape[:3] == inp.shape[:3], (mask.shape, inp.shape)
    print(label, "out", tuple(out.shape), "mask", tuple(mask.shape),
          "range", round(float(out.min()), 3), round(float(out.max()), 3),
          "maskmax", round(float(mask.max()), 3))

print("NODE OK")
