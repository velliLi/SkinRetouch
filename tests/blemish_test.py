"""Inject known blemishes, then measure how well the pipeline removes them.

This is the decisive test: portraits produced by a generator are already
retouched, so a clean pair proves nothing. We add controlled blemishes inside
the skin mask, then compare against a plain Gaussian-blur baseline.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import skin_core

OUT = os.environ.get("SKIN_OUT") or os.path.join(HERE, "output")
ASSETS = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20240521)

# Locked CPU-path baselines. skin_retouch defaults to device=None -> "cpu", so
# these numbers pin the face path exactly; the GPU body path is deliberately not
# covered here (see compare_paths.py).
BASE = {
    "t01": {"clean": 5.9, "blur": 1.3, "ours": 7.5, "coverage": 0.320},
    "t07": {"clean": 6.1, "blur": 2.0, "ours": 9.4, "coverage": 0.264},
    "t05": {"clean": 7.4, "blur": 2.0, "ours": 8.3, "coverage": 0.011},
}
TOL_HF = 0.11
TOL_COV = 0.0025
results = {}


def build_mask(bgr, model):
    faces = skin_core.detect_faces(bgr, model, conf=0.35, max_faces=4)
    h, w = bgr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for box in faces:
        m = np.maximum(m, skin_core.face_skin_mask(bgr, box))
    return m


def inject(bgr, mask, n_pimples=26, n_blotch=10, noise_sigma=3.0):
    out = bgr.astype(np.float32).copy()
    h, w = bgr.shape[:2]
    ys, xs = np.where(mask > 200)
    if len(xs) == 0:
        return bgr.copy()

    # Pimples: small red-dark discs, the classic high-frequency skin defect.
    for _ in range(n_pimples):
        i = rng.integers(0, len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        r = int(rng.integers(3, 8))
        patch = np.zeros((h, w), np.float32)
        cv2.circle(patch, (cx, cy), r, 1.0, -1)
        patch = cv2.GaussianBlur(patch, (0, 0), max(1.0, r * 0.45))[..., None]
        tint = np.array([90.0, 110.0, 190.0], np.float32)  # BGR reddish
        out = out + (tint - out) * patch * 0.55

    # Blotches: low-frequency redness, the scale tone_even must flatten.
    if n_blotch:
        low = np.zeros((h, w), np.float32)
        for _ in range(n_blotch):
            i = rng.integers(0, len(xs))
            cv2.circle(low, (int(xs[i]), int(ys[i])), int(rng.integers(18, 46)), 1.0, -1)
        low = cv2.GaussianBlur(low, (0, 0), 14.0) * (mask.astype(np.float32) / 255.0)
        tint = np.array([110.0, 130.0, 205.0], np.float32)
        out = out + (tint - out) * low[..., None] * 0.30

    # Sensor grain, so pore-level texture also has to be judged.
    out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def blur_baseline(bgr, mask, k=15):
    soft = cv2.GaussianBlur(bgr, (k, k), 0)
    w = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(bgr * (1 - w) + soft * w, 0, 255).astype(np.uint8)


def crop_square(img, cx, cy, side):
    h, w = img.shape[:2]
    side = min(side, h, w)
    x1 = int(np.clip(cx - side // 2, 0, w - side))
    y1 = int(np.clip(cy - side // 2, 0, h - side))
    return img[y1:y1 + side, x1:x1 + side].copy()


def label(img, text):
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


CASES = [
    ("t01_headshoulders_135_00001_.png", "t01"),
    ("t07_laughing_candid_135_00001_.png", "t07"),
    ("05_wall_lean_jacket_00001_.png", "t05"),
]

for filename, name in CASES:
    path = os.path.join(ASSETS, filename)
    if not os.path.exists(path):
        print("MISS", path)
        continue
    clean = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    model = skin_core.load_face_detector()
    mask = build_mask(clean, model)
    if mask.max() == 0:
        print("NO FACE", name)
        continue

    bad = inject(clean, mask)
    fixed, m2, info = skin_core.skin_retouch(
        bad, strength=0.75, texture_keep=0.35, blemish=0.7,
        tone_even=0.5, feather=0.5,
    )
    print(name, "info", info)
    base = blur_baseline(bad, m2)

    ys, xs = np.where(m2 > 180)
    cx, cy = int(np.median(xs)), int(np.median(ys))
    tiles = [
        label(crop_square(bad, cx, cy, 360), "1 blemished"),
        label(crop_square(base, cx, cy, 360), "2 gauss blur"),
        label(crop_square(fixed, cx, cy, 360), "3 skin retouch"),
        label(crop_square(clean, cx, cy, 360), "4 reference"),
    ]
    cv2.imwrite(os.path.join(OUT, f"{name}_compare.png"), np.hstack(tiles))

    g_clean = cv2.Laplacian(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    g_base = cv2.Laplacian(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    g_fix = cv2.Laplacian(cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    sel = m2 > 180
    got = {
        "clean": float(np.abs(g_clean[sel]).mean()),
        "blur": float(np.abs(g_base[sel]).mean()),
        "ours": float(np.abs(g_fix[sel]).mean()),
        "coverage": float(sel.mean()),
    }
    results[name] = got
    print(
        name,
        "hf-energy clean=%.1f blur=%.1f ours=%.1f" % (got["clean"], got["blur"], got["ours"]),
        "coverage=%.3f" % (got["coverage"]),
    )

print("DONE")

fails = []
for name, got in results.items():
    exp = BASE.get(name)
    if exp is None:
        continue
    for key in ("clean", "blur", "ours"):
        if abs(got[key] - exp[key]) > TOL_HF:
            fails.append("%s %s: %.2f != %.2f" % (name, key, got[key], exp[key]))
    if abs(got["coverage"] - exp["coverage"]) > TOL_COV:
        fails.append("%s coverage: %.4f != %.4f" % (name, got["coverage"], exp["coverage"]))

missing = [k for k in BASE if k not in results]
if fails or missing:
    for line in fails:
        print("FAIL", line)
    for k in missing:
        print("FAIL", k, "did not run")
    raise SystemExit(1)
print("BASELINE OK (%d cases)" % len(BASE))
