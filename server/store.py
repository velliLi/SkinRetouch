"""Filesystem layout for uploaded sources and produced results.

Every job owns one directory under `<data>/jobs/<id>/`. Uploads are stored under
a generated name, never under the client-supplied filename: the original name is
kept in memory for display only, so a `../../` in it can never reach the disk.
Job ids are validated against a fixed hex pattern by the API layer before they
get here, and `path()` re-checks the resolved location stays inside the store.

Results are written to a temporary name and then moved into place, so a reader
never sees a half-written PNG while the worker is still encoding it.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import time

from . import config

_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,5}$")


class StoreError(RuntimeError):
    pass


def sniff_ext(data: bytes) -> str:
    """Pick an extension from magic bytes; the client's filename proves nothing.

    Only the formats OpenCV can decode are accepted, and the extension is derived
    from the bytes so a `.jpg` payload can never land on disk as a `.png`.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] == b"BM":
        return ".bmp"
    raise StoreError("unsupported image container (accepted: PNG, JPEG, TIFF, WebP, BMP)")


class JobStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.jobs_dir = os.path.join(self.root, "jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)

    # -- paths -----------------------------------------------------------

    def job_dir(self, job_id: str) -> str:
        if not config.JOB_ID_RE.match(job_id or ""):
            raise StoreError("malformed job id")
        path = os.path.abspath(os.path.join(self.jobs_dir, job_id))
        # Defence in depth: the regex already forbids separators, but resolving
        # catches symlinks and Windows 8.3 aliases as well.
        if os.path.commonpath([path, self.jobs_dir]) != self.jobs_dir:
            raise StoreError("job path escapes the store")
        return path

    def path(self, job_id: str, name: str) -> str:
        if not name or os.path.sep in name or (os.path.altsep and os.path.altsep in name):
            raise StoreError("malformed file name")
        if not _EXT_RE.match(os.path.splitext(name)[1]):
            raise StoreError("malformed file name")
        return os.path.join(self.job_dir(job_id), name)

    # -- writes ----------------------------------------------------------

    def create_job(self, job_id: str) -> str:
        path = self.job_dir(job_id)
        os.makedirs(path, exist_ok=True)
        return path

    def write(self, job_id: str, name: str, data: bytes) -> str:
        final = self.path(job_id, name)
        tmp = final + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, final)
        return final

    def save_source(self, job_id: str, data: bytes) -> tuple[str, int]:
        ext = sniff_ext(data)
        self.create_job(job_id)
        name = "source" + ext
        self.write(job_id, name, data)
        return name, len(data)

    # -- reads -----------------------------------------------------------

    def read(self, job_id: str, name: str) -> bytes:
        path = self.path(job_id, name)
        with open(path, "rb") as fh:
            return fh.read()

    def mtime(self, job_id: str, name: str) -> float:
        try:
            return os.path.getmtime(self.path(job_id, name))
        except OSError:
            return 0.0

    # -- maintenance -----------------------------------------------------

    def remove(self, job_id: str) -> None:
        path = self.job_dir(job_id)
        shutil.rmtree(path, ignore_errors=True)

    def sweep(self, ttl_s: int, keep: set[str] | None = None) -> int:
        """Delete job directories older than `ttl_s`; returns the count removed.

        The cutoff uses the directory's own mtime, which is bumped whenever a
        result file is written, so a long-running job cannot be swept out from
        under itself.
        """
        keep = keep or set()
        cutoff = time.time() - max(0, ttl_s)
        removed = 0
        try:
            entries = os.scandir(self.jobs_dir)
        except OSError:
            return 0
        with entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if entry.name in keep or not config.JOB_ID_RE.match(entry.name):
                    continue
                try:
                    if entry.stat(follow_symlinks=False).st_mtime >= cutoff:
                        continue
                    shutil.rmtree(entry.path, ignore_errors=True)
                    removed += 1
                except OSError as exc:
                    if exc.errno != errno.ENOENT:
                        continue
        return removed
