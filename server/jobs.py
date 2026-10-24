"""Job queue, worker thread and result encoding.

One worker, always. The GPU is a 6 GB laptop part that usually has ComfyUI
resident, and a 20 MP body pass needs the whole frame at once; two concurrent
jobs would trade a 3.6 s queue wait for an out-of-memory crash. Everything the
HTTP layer needs is a status snapshot, so the CLI, a script and the browser all
observe the same state machine.

`skin_core.skin_retouch` takes no progress callback and is deliberately left
untouched (its pixel output is pinned by tests/blemish_test.py). Phases are
therefore coarse and honest: decode, pipeline, repaint, encode. The
fine-grained timings the pipeline does report, per face and per body stage, are
collected from its `info` list and shipped once the job finishes.

The repaint phase is optional and best-effort. It needs a live ComfyUI process
with an inpaint checkpoint resident; when that is missing the job still
succeeds, the CPU pipeline's result is returned unchanged, and the reason lands
in `job.warnings` and `job.repaint`. A photo editor that silently drops a
requested step, or that fails the whole job because a helper service is down,
is worse than one that says what happened.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config, repaint
from .store import JobStore, StoreError

import skin_core

try:  # body support degrades to face-only when the module is absent, as documented
    import skin_body
except Exception:  # pragma: no cover - only on a partial checkout
    skin_body = None


def _jsonable(value):
    """Convert numpy scalars, tuples and nested containers into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def core_kwargs(params: dict) -> dict:
    """Translate the public parameter names into `skin_core.skin_retouch` kwargs.

    Two names differ on purpose: the node exposes `face_confidence`, the library
    function calls it `conf`, and the node exposes `body_strength` where 0 means
    "follow strength" while the library wants None for that case. This function
    is the single place that knows about the mismatch.
    """
    body = float(params["body_strength"])
    return {
        "strength": float(params["strength"]),
        "texture_keep": float(params["texture_keep"]),
        "blemish": float(params["blemish"]),
        "tone_even": float(params["tone_even"]),
        "feather": float(params["feather"]),
        "conf": float(params["face_confidence"]),
        "max_faces": int(params["max_faces"]),
        "include_body": bool(params["include_body"]),
        "body_strength": None if body <= 0.0 else body,
        "device": str(params["device"]),
    }


def _encode_png(img, compression=3):
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _encode_jpeg(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _fit(img, max_px):
    """Downscale so the longest edge is at most `max_px`; never upscale."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_px:
        return img, 1.0
    scale = max_px / float(longest)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


class QueueFull(RuntimeError):
    pass


class JobNotFound(KeyError):
    pass


@dataclass
class Job:
    id: str
    params: dict
    source_name: str
    created_at: float
    status: str = "queued"          # queued | running | done | error | canceled
    phase: str = "queued"           # decode | pipeline | repaint | encode | finalize
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    image: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    info: list = field(default_factory=list)
    repaint: dict = field(default_factory=dict)     # stats, or why it was skipped
    artifacts: dict = field(default_factory=dict)   # kind -> stored file name
    bytes_in: int = 0
    sha256: str = ""
    cold_start: bool = False
    cancel_requested: bool = False
    warnings: list = field(default_factory=list)

    def elapsed_ms(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return round((end - self.started_at) * 1000.0, 1)


class JobManager:
    def __init__(self, cfg: config.Config, store: JobStore, log=None):
        self.cfg = cfg
        self.store = store
        self._log = log or (lambda *a: None)
        self._jobs: "collections.OrderedDict[str, Job]" = collections.OrderedDict()
        self._pending: "collections.deque[str]" = collections.deque()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="skin-worker", daemon=True)
        self._worker.start()
        self._sweeper = threading.Thread(target=self._sweep_loop, name="skin-sweeper", daemon=True)
        self._sweeper.start()
        self._processed = 0
        self._warm = False

    # -- public API ------------------------------------------------------

    def submit(self, params: dict, data: bytes, source_name: str = "upload") -> Job:
        with self._cv:
            if len(self._pending) >= self.cfg.queue_limit:
                raise QueueFull("queue is full (%d waiting)" % len(self._pending))
            job_id = uuid.uuid4().hex[:16]
            job = Job(id=job_id, params=dict(params), source_name=source_name,
                      created_at=time.time(), bytes_in=len(data),
                      cold_start=not self._warm)
            name, size = self.store.save_source(job_id, data)
            job.artifacts["source"] = name
            job.bytes_in = size
            self._jobs[job_id] = job
            self._pending.append(job_id)
            self._cv.notify_all()
            self._log("job %s queued (%s, %.1f KB, queue=%d)"
                      % (job_id, source_name, size / 1024.0, len(self._pending)))
        return job

    def get(self, job_id: str) -> Job:
        with self._cv:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def list(self, limit: int = 20) -> list[Job]:
        with self._cv:
            items = list(self._jobs.values())
        return items[-limit:][::-1]

    def cancel(self, job_id: str) -> tuple[bool, str]:
        """Drop a queued job; a running one is only flagged, and cannot be stopped.

        The pipeline is a straight-line numpy/OpenCV call with no interruption
        points, so pretending we can abort it mid-flight would be a lie. The UI
        says so explicitly instead of showing a cancel button that does nothing.
        """
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            if job.status == "queued":
                try:
                    self._pending.remove(job_id)
                except ValueError:
                    pass
                job.status = "canceled"
                job.phase = "canceled"
                job.finished_at = time.time()
                self._log("job %s canceled while queued" % job_id)
                return True, "removed from the queue"
            if job.status == "running":
                job.cancel_requested = True
                return False, "already running; the pipeline has no interruption point"
        return False, "job is %s, nothing to cancel" % job.status

    def snapshot(self, job: Job) -> dict:
        with self._cv:
            pending = list(self._pending)
        position = pending.index(job.id) + 1 if job.id in pending else 0
        return {
            "id": job.id,
            "status": job.status,
            "phase": job.phase,
            "created_at": round(job.created_at, 3),
            "started_at": round(job.started_at, 3) if job.started_at else 0,
            "finished_at": round(job.finished_at, 3) if job.finished_at else 0,
            "elapsed_ms": job.elapsed_ms(),
            "queue_position": position,
            "queue_depth": len(pending),
            "params": job.params,
            "source": {"name": job.source_name, "bytes": job.bytes_in},
            "image": job.image,
            "timing": job.timing,
            "info": job.info,
            "repaint": job.repaint,
            "error": job.error,
            "sha256": job.sha256,
            "cold_start": job.cold_start,
            "warnings": job.warnings,
            "artifacts": sorted(k for k in job.artifacts if k != "source"),
            "source_artifact": job.artifacts.get("source", ""),
        }

    def stats(self) -> dict:
        with self._cv:
            running = sum(1 for j in self._jobs.values() if j.status == "running")
            pending = len(self._pending)
            total = len(self._jobs)
        return {"pending": pending, "running": running, "tracked": total,
                "processed": self._processed}

    def shutdown(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    # -- worker ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._pending and not self._stop.is_set():
                    self._cv.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                job_id = self._pending.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                job.status = "running"
                job.phase = "decode"
                job.started_at = time.time()
            try:
                self._run(job)
            except Exception as exc:  # a worker must never die
                job.status = "error"
                job.error = "%s: %s" % (type(exc).__name__, exc)
                job.finished_at = time.time()
                self._log("job %s failed: %s\n%s"
                          % (job.id, job.error, traceback.format_exc()))
            finally:
                with self._cv:
                    self._processed += 1
                    self._prune()
                self.store.sweep(self.cfg.job_ttl_s, keep=self._live_ids())

    def _run(self, job: Job) -> None:
        clock = time.perf_counter()
        with warnings_capture(job.warnings):
            data = self.store.read(job.id, job.artifacts["source"])
            # OpenCV applies the JPEG EXIF orientation on decode unless told not
            # to, which is what a photographer expects; the output PNG is upright
            # because the rotation is baked into the pixels.
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("the uploaded bytes are not a decodable image")
            h, w = bgr.shape[:2]
            if w * h > self.cfg.max_pixels:
                raise ValueError("image is %dx%d; the limit is %d megapixels"
                                 % (w, h, self.cfg.max_pixels // 1_000_000))
            job.timing["decode_ms"] = round((time.perf_counter() - clock) * 1000.0, 1)
            job.image = {"width": int(w), "height": int(h),
                         "megapixels": round(w * h / 1e6, 2)}

            job.phase = "pipeline"
            with self._cv:
                if job.cancel_requested:
                    job.status = "canceled"
                    job.phase = "canceled"
                    job.finished_at = time.time()
                    return
            clock = time.perf_counter()
            out, mask, info = skin_core.skin_retouch(bgr, **core_kwargs(job.params))
            job.timing["pipeline_ms"] = round((time.perf_counter() - clock) * 1000.0, 1)
            job.info = _jsonable(info)

            if job.params.get("repaint"):
                job.phase = "repaint"
                clock = time.perf_counter()
                out = self._repaint(job, bgr, out, mask, info)
                job.timing["repaint_ms"] = round((time.perf_counter() - clock) * 1000.0, 1)

            job.phase = "encode"
            clock = time.perf_counter()
            self._encode(job, bgr, out, mask)
            job.timing["encode_ms"] = round((time.perf_counter() - clock) * 1000.0, 1)

            job.phase = "finalize"
            job.status = "done"
            job.finished_at = time.time()
            self._warm = True
            meta = self.snapshot(job)
            self.store.write(job.id, "meta.json",
                             json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
            self._log("job %s done in %.0f ms (%s)"
                      % (job.id, job.elapsed_ms(), job.sha256[:12]))

    def _repaint(self, job: Job, bgr, out, mask, info):
        """Run the optional repaint pass; always returns a usable image.

        Failure modes are reported, not raised: a missing weight file, an
        unreachable ComfyUI, a tile that times out. The caller keeps the
        pipeline result and the reason shows up in `job.repaint` and in the
        snapshot's warnings, which the UI surfaces as a toast.
        """
        base = self.cfg.repaint_url or repaint.DEFAULT_URL
        engine = str(job.params.get("repaint_engine") or repaint.DEFAULT_ENGINE).lower()
        if engine not in repaint.ENGINES:
            engine = repaint.DEFAULT_ENGINE

        if engine == "sdxl" and not repaint.probe(base):
            job.repaint = {"available": False, "engine": engine, "url": base,
                           "reason": "no ComfyUI answering on %s" % base}
            job.warnings.append("局部重绘已跳过：%s 上没有可用的 ComfyUI 实例" % base)
            self._log("job %s repaint skipped: %s unreachable" % (job.id, base))
            return out
        if engine != "sdxl" and not repaint.net_ready(engine):
            where = repaint.net_path(engine)
            job.repaint = {"available": False, "engine": engine, "reason": where}
            job.warnings.append(
                "局部重绘已跳过：%s 引擎的权重不在 %s，先跑 tests/fetch_repaint_weights.py"
                % (engine, where))
            self._log("job %s repaint skipped: %s weights missing" % (job.id, engine))
            return out

        try:
            image, stats = repaint.apply(bgr, out, mask, info, job.params, base_url=base)
        except repaint.RepaintError as exc:
            job.repaint = {"available": True, "engine": engine, "reason": str(exc)}
            job.warnings.append("局部重绘失败，已回退到磨皮结果：%s" % (exc,))
            self._log("job %s repaint failed: %s" % (job.id, exc))
            return out
        job.repaint = stats
        self._log("job %s repaint[%s]: %.0f ms, mask %.2f%%"
                  % (job.id, stats.get("engine", engine), stats.get("ms", 0.0),
                     100.0 * stats.get("mask_frac", 0.0)))
        return image

    def _encode(self, job: Job, bgr, out, mask) -> None:
        store = self.store
        max_px = self.cfg.preview_max_px

        full = _encode_png(out, compression=3)
        store.write(job.id, "out.png", full)
        job.artifacts["out"] = "out.png"
        job.sha256 = hashlib.sha256(full).hexdigest()

        store.write(job.id, "mask.png", _encode_png(mask, compression=6))
        job.artifacts["mask"] = "mask.png"

        src_prev, _ = _fit(bgr, max_px)
        out_prev, _ = _fit(out, max_px)
        mask_prev, _ = _fit(mask, max_px)
        store.write(job.id, "source.preview.jpg", _encode_jpeg(src_prev))
        store.write(job.id, "out.preview.jpg", _encode_jpeg(out_prev))
        store.write(job.id, "mask.preview.png", _encode_png(mask_prev, compression=6))
        job.artifacts["preview_source"] = "source.preview.jpg"
        job.artifacts["preview_out"] = "out.preview.jpg"
        job.artifacts["preview_mask"] = "mask.preview.png"

    # -- housekeeping ----------------------------------------------------

    def _live_ids(self) -> set[str]:
        with self._cv:
            return {j.id for j in self._jobs.values()
                    if j.status in ("queued", "running")}

    def _prune(self) -> None:
        """Keep memory bounded; the files on disk are swept by TTL separately."""
        limit = 60
        while len(self._jobs) > limit:
            for job_id, job in self._jobs.items():
                if job.status not in ("queued", "running"):
                    del self._jobs[job_id]
                    break
            else:
                break

    def _sweep_loop(self) -> None:
        interval = 120.0
        while not self._stop.wait(interval):
            try:
                removed = self.store.sweep(self.cfg.job_ttl_s, keep=self._live_ids())
                if removed:
                    self._log("swept %d expired job director%s"
                              % (removed, "y" if removed == 1 else "ies"))
            except Exception as exc:  # never let housekeeping kill the thread
                self._log("sweep failed: %s" % (exc,))


class warnings_capture:
    """Collect RuntimeWarning text emitted by the core during one job.

    `skin_core` warns instead of failing when skin_gpu is unavailable or the body
    path degrades; silently dropping those warnings would leave a user wondering
    why their 20 MP job took 10 s. They are surfaced in the job snapshot.

    `catch_warnings` is process-global, so a warning raised by an unrelated HTTP
    thread inside the window would also be attributed to the job. Nothing in the
    request path raises RuntimeWarnings, and the alternative -- threading a
    warning handler through an untouched pipeline -- is worse.
    """

    def __init__(self, sink: list):
        self.sink = sink
        self._ctx = None
        self._records = []

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._records = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        for item in self._records:
            if issubclass(item.category, RuntimeWarning):
                text = str(item.message)
                if text not in self.sink:
                    self.sink.append(text)
        self._ctx.__exit__(*exc)
        return False
