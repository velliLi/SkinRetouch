"""Fetch the small erase-style inpaint models (LaMa / MI-GAN) from GitHub releases.

Both checkpoints are TorchScript, so they load with `torch.jit.load` and need
nothing beyond torch. The bare GitHub release host sits around 7-9 KB/s here;
the community proxies listed below are 10-15x faster, so we pick a mirror by
measured throughput and then pull the file in parallel byte ranges.

    python tests/fetch_repaint_weights.py --model migan
    python tests/fetch_repaint_weights.py --model all --workers 12
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST_DIR = os.path.join(ROOT, "models", "repaint")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# md5/size come from the iopaint sources bundled with ComfyUI-LayerStyle
# (custom_nodes/comfyui_layerstyle/py/iopaint/model/{lama,mi_gan}.py).
MODELS = {
    "migan": {
        "url": "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt",
        "md5": "76eb3b1a71c400ee3290524f7a11b89c",
        "size": 27048675,
        "note": "MI-GAN, 512px fixed, ~26 MB, mobile-class",
    },
    "lama": {
        "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "md5": "e3aa4aaa15225a33ec84f9f4bc47e500",
        "size": 205669692,
        "note": "big-lama, any size padded to 8, ~196 MB",
    },
}

MIRRORS = [
    "https://edgeone.gh-proxy.com/",
    "https://gh-proxy.com/",
    "https://ghproxy.net/",
    "",  # bare github, last resort
]


def _open(url, start=None, end=None, timeout=30):
    headers = dict(UA)
    if start is not None:
        headers["Range"] = "bytes=%d-%s" % (start, "" if end is None else end)
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def probe(url, chunk=262144, timeout=20):
    """Return (bytes_per_second, total_size) for a mirror, or (0, 0) if dead."""
    t0 = time.time()
    try:
        r = _open(url, 0, chunk - 1, timeout=timeout)
        data = r.read()
        dt = max(time.time() - t0, 1e-6)
        span = (r.headers.get("content-range") or "").split("/")[-1]
        total = int(span) if span.isdigit() else int(r.headers.get("content-length") or 0)
        return len(data) / dt, total
    except Exception as exc:
        print("    %-34s dead (%s)" % (url.split("/")[2] if "://" in url else url, type(exc).__name__))
        return 0.0, 0


def build_mirrors(remote):
    return [prefix + remote for prefix in MIRRORS]


def fetch_ranges(url, dest, size, workers=8, chunk=2 << 20):
    """Pull `size` bytes into `dest` from a queue of small range requests.

    The proxies stall on wide ranges - one 12.8 MB range sat at zero bytes for
    80 s and killed the whole run - while ~2 MB slices stream steadily. Keep the
    slices small and let the workers drain a shared queue, so one dead slice
    retries instead of poisoning a 16-way split of the entire file.

    Finished slices are journaled to `<dest>.progress`, so a mid-file timeout
    (the 196 MB model died at 60 % after six minutes) resumes instead of
    restarting from byte zero.
    """
    if size <= 0:
        raise RuntimeError("unknown size for %s" % url)
    spans = [(i, min(i + chunk, size) - 1) for i in range(0, size, chunk)]
    lock = threading.Lock()
    done = [0]
    errors = []
    cursor = [0]

    journal = dest + ".progress"
    finished = set()
    if os.path.exists(dest) and os.path.getsize(dest) == size and os.path.exists(journal):
        try:
            with open(journal, "r", encoding="utf-8") as fh:
                finished = set(json.load(fh).get("done", []))
        except (OSError, ValueError):
            finished = set()
        for i in finished:
            if 0 <= i < len(spans):
                done[0] += spans[i][1] - spans[i][0] + 1

    if not finished:
        with open(dest, "wb") as fh:
            fh.truncate(size)
    elif os.path.getsize(dest) != size:
        finished = set()
        with open(dest, "wb") as fh:
            fh.truncate(size)
    cursor[0] = 0
    pending = [i for i in range(len(spans)) if i not in finished]
    print("    %d/%d slices pending%s" % (len(pending), len(spans),
                                          " (resuming)" if finished else ""))

    def save_journal():
        tmp = journal + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"done": sorted(finished)}, fh)
        os.replace(tmp, journal)

    def worker():
        while True:
            with lock:
                if not pending or errors:
                    return
                idx = pending.pop(0)
                start, end = spans[idx]
            try:
                with open(dest, "r+b") as fh:
                    pos = start
                    while pos <= end:
                        got = _open(url, pos, end, timeout=120).read()
                        if not got:
                            break
                        with lock:
                            fh.seek(pos)
                            fh.write(got)
                        pos += len(got)
                        with lock:
                            done[0] += len(got)
                with lock:
                    finished.add(idx)
                    save_journal()
            except Exception as exc:  # noqa: BLE001 - report and let the caller retry
                with lock:
                    errors.append("%s: %s" % (type(exc).__name__, exc))
                return

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    t0 = time.time()
    for th in threads:
        th.start()
    while any(th.is_alive() for th in threads):
        time.sleep(1.0)
        pct = 100.0 * done[0] / size
        rate = done[0] / 1024.0 / max(time.time() - t0, 1e-6)
        sys.stdout.write("\r    %5.1f%%  %6.1f KB/s  %.1fs" % (pct, rate, time.time() - t0))
        sys.stdout.flush()
    for th in threads:
        th.join()
    sys.stdout.write("\r" + " " * 46 + "\r")
    if errors:
        raise RuntimeError(errors[0])


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, workers=12, force=False):
    spec = MODELS[name]
    dest = os.path.join(DEST_DIR, "%s.pt" % ("big-lama" if name == "lama" else "migan_traced"))
    print("\n[%s] %s" % (name, spec["note"]))
    print("  -> %s" % dest)
    if os.path.exists(dest) and not force:
        if md5_of(dest) == spec["md5"]:
            print("  already present and verified, skipping")
            return dest
        print("  existing file fails md5, refetching")

    os.makedirs(DEST_DIR, exist_ok=True)
    scored = []
    for url in build_mirrors(spec["url"]):
        rate, total = probe(url)
        if rate > 0:
            print("    %-34s %7.1f KB/s  total=%s" % (url.split("/")[2], rate / 1024, total or "?"))
            scored.append((rate, url, total))
    if not scored:
        raise RuntimeError("no mirror answered for %s" % name)
    scored.sort(reverse=True)
    rate, url, total = scored[0]
    size = spec["size"] or total
    print("  using %s (%.1f KB/s, %.1f MB)" % (url.split("/")[2], rate / 1024, size / 1048576))

    tmp = dest + ".part"
    if force and os.path.exists(tmp + ".progress"):
        os.remove(tmp + ".progress")
    fetch_ranges(url, tmp, size, workers=workers)
    got = md5_of(tmp)
    if got != spec["md5"]:
        os.remove(tmp)
        raise RuntimeError("md5 mismatch: got %s expected %s" % (got, spec["md5"]))
    os.replace(tmp, dest)
    print("  ok  md5=%s  %.1f MB" % (got, os.path.getsize(dest) / 1048576))
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="migan", choices=sorted(MODELS) + ["all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    names = sorted(MODELS) if args.model == "all" else [args.model]
    for name in names:
        try:
            fetch(name, workers=args.workers, force=args.force)
        except Exception as exc:  # noqa: BLE001 - a failed model must not stop the others
            print("  FAILED %s: %s" % (name, exc))
            if len(names) == 1:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
