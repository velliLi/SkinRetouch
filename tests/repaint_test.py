"""Checks for the optional repaint pass in `server/repaint.py`.

Two halves, on purpose:

* the pure geometry and mask helpers always run, no network and no GPU;
* the live half needs a ComfyUI instance holding the SDXL inpaint checkpoint.
  When none answers the live checks are reported as SKIP rather than failures,
  because the test suite has to stay runnable on a machine without the engine.

The live half asserts the two promises the feature makes to a user:

1. every pixel outside the mask comes back byte-identical -- repaint is a
   surgical pass, not a global re-render;
2. pixels under the mask actually change, otherwise the switch is a lie.

    python tests/repaint_test.py
    python tests/repaint_test.py --url http://127.0.0.1:8189
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from server import repaint  # noqa: E402

ASSET = os.path.join(HERE, "assets", "t01_headshoulders_135_00001_.png")

failures = []
checks = 0
skipped = 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    if ok:
        print("  ok   %s" % name)
    else:
        failures.append(name)
        print("  FAIL %s   %s" % (name, detail))


def skip(name, why):
    global skipped
    skipped += 1
    print("  skip %s   (%s)" % (name, why))


# --------------------------------------------------------------------- offline

def test_mask_helpers():
    print("\n[geometry and mask helpers]")

    check("_starts keeps the last tile flush", repaint._starts(1000, 896) == [0, 104],
          repaint._starts(1000, 896))
    check("_starts returns one tile when the image is smaller",
          repaint._starts(500, 896) == [0] and repaint._starts(896, 896) == [0],
          repaint._starts(500, 896))

    disc = np.zeros((256, 256), np.uint8)
    cv2.circle(disc, (128, 128), 4, 255, -1)
    raw_px = int((disc > 0).sum())

    grown = repaint.prep_mask(disc, grow=8, blur=4.0)
    check("prep_mask grows the mask", int((grown > 0).sum()) > raw_px,
          (raw_px, int((grown > 0).sum())))
    check("prep_mask normalises the peak to 255", int(grown.max()) == 255, int(grown.max()))

    # ComfyUI resamples to 1/8 with bilinear interpolation; a 4 px disc keeps
    # almost no weight there. This is the reason GROW_PX exists, so pin it.
    def latent_peak(mask):
        small = cv2.resize(mask.astype(np.float32) / 255.0,
                           (mask.shape[1] // 8, mask.shape[0] // 8),
                           interpolation=cv2.INTER_LINEAR)
        return float(small.max())

    check("an ungrown disc loses its weight at latent scale", latent_peak(disc) < 0.35,
          round(latent_peak(disc), 3))
    check("a grown disc survives latent downsampling", latent_peak(grown) > 0.85,
          round(latent_peak(grown), 3))

    plan, skipped_tiles = repaint._plan(256, 256, grown, 896)
    check("_plan keeps the single tile that holds the mask",
          len(plan) == 1 and skipped_tiles == 0, (plan, skipped_tiles))
    empty, empty_skipped = repaint._plan(256, 256, np.zeros((256, 256), np.uint8), 896)
    check("_plan drops tiles with no mask pixels", not empty and empty_skipped == 0,
          empty)

    check("probe is False for a dead endpoint",
          repaint.probe("http://127.0.0.1:1", timeout=0.5) is False)


def make_case():
    """Load the asset and paint three deterministic blemishes on the cheeks."""
    bgr = cv2.imread(ASSET, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit("cannot read %s" % ASSET)
    import skin_core

    out, skin, info = skin_core.skin_retouch(
        bgr, strength=0.75, texture_keep=0.35, blemish=0.7, tone_even=0.5,
        feather=0.5, conf=0.35, max_faces=1, include_body=False, device="cpu")
    height, width = bgr.shape[:2]
    blemished = bgr.copy()
    face_px = int((info[0] or {}).get("face_px") or min(height, width) // 6)
    cx = width // 2
    cy = height // 3
    for dx, dy in ((-face_px // 5, face_px // 6), (face_px // 5, face_px // 6),
                   (0, -face_px // 5)):
        cv2.circle(blemished, (cx + dx, cy + dy), max(4, face_px // 24), (90, 110, 190), -1)
    # The mask is detected on `blemished` and the redraw runs on `out`, exactly
    # as jobs.py does it: spots are still visible on the source, the model gets
    # clean skin to start from.
    return blemished, out, skin, info


def test_live(url, engine="sdxl"):
    print("\n[live repaint: %s]" % engine)
    blemished, out, skin, info = make_case()
    mask, mask_stats = repaint.blemish_mask(blemished, skin, int(info[0]["face_px"]))
    covered = int((mask > 127).sum())
    check("blemish_mask finds the painted spots", covered > 0,
          (covered, mask_stats))

    params = {"blemish": 0.7, "repaint_engine": engine}
    if engine == "sdxl":
        params.update({"repaint_denoise": 0.6, "repaint_steps": 8})
    else:
        params["repaint_hf"] = 1.15

    try:
        result, stats = repaint.apply(blemished, out, skin, info, params,
                                      base_url=url, timeout_per_tile=300.0)
    except repaint.RepaintError as exc:
        check("apply runs against a live engine", False, exc)
        return

    check("apply honours the requested engine", stats.get("engine") == engine, stats)
    check("apply reports tiles", stats.get("tiles_done", 0) >= 1, stats)
    check("apply reports timing", isinstance(stats.get("ms"), float), stats.get("ms"))

    solid = mask > 0
    outside = ~solid
    changed_outside = int((result[outside] != out[outside]).sum())
    check("pixels outside the mask are untouched", changed_outside == 0, changed_outside)

    inside = mask > 127
    changed_inside = int((result[inside] != out[inside]).any(axis=-1).sum())
    check("pixels under the mask are redrawn", changed_inside > 0,
          (changed_inside, int(inside.sum())))

    # Skin texture should stay in the same ballpark: a redraw that halves the
    # detail or doubles the contrast would be worse than the blemish.
    def hf(img, sel):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        return float(lap[sel].mean()) if sel.any() else 0.0

    before, after = hf(out, inside), hf(result, inside)
    check("redrawn skin keeps its detail level",
          after > 0 and 0.4 < before / max(after, 1e-6) < 2.5, (before, after))

    diff = cv2.absdiff(out, result)
    print("     mask %.2f%%  changed inside %d px  mean|diff| %.2f  hf %.2f -> %.2f"
          % (100.0 * covered / mask.size, changed_inside, diff[inside].mean()
             if inside.any() else 0.0, before, after))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="", help="ComfyUI base URL; probed before use")
    args = parser.parse_args(argv)

    test_mask_helpers()

    url = (args.url or repaint.DEFAULT_URL).rstrip("/")
    # The two TorchScript nets need only their weight file, so they are checked
    # whenever the weights are present. SDXL still needs a ComfyUI instance.
    for engine in ("migan", "lama"):
        if repaint.net_ready(engine):
            test_live("", engine)
        else:
            skip("live repaint (%s)" % engine,
                 "weights missing at %s" % repaint.net_path(engine))
    if repaint.probe(url, timeout=1.0):
        test_live(url, "sdxl")
    else:
        skip("live repaint (sdxl)", "no ComfyUI on %s" % url)

    print()
    if failures:
        print("REPAINT FAIL  %d of %d checks failed: %s"
              % (len(failures), checks, ", ".join(failures)))
        return 1
    print("REPAINT OK  %d checks%s" % (checks, "  (%d skipped)" % skipped if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
