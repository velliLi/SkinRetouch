"""1:1 crops of the regions the gate keeps misclassifying."""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")

CROPS = {
    "DSC_7300.JPG": {
        "waist": (760, 3800, 1800, 5000),
        "thigh": (1600, 4600, 2700, 5568),
        "shoulder": (600, 2600, 1600, 3600),
    },
    "DSC_4081.JPG": {
        "chest": (3200, 2400, 4300, 3500),
    },
}

for name, crops in CROPS.items():
    bgr = cv2.imdecode(np.fromfile(os.path.join(HERE, "assets", name), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    for key, (x1, y1, x2, y2) in crops.items():
        x2 = min(x2, w)
        y2 = min(y2, h)
        crop = bgr[y1:y2, x1:x2]
        scale = min(1.0, 900.0 / max(crop.shape[:2]))
        if scale < 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        path = os.path.join(OUT, f"crop2_{os.path.splitext(name)[0]}_{key}.png")
        cv2.imwrite(path, crop)
        print(path, crop.shape)
