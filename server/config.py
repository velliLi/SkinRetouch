"""Configuration and the parameter contract shared by the GUI and the HTTP API.

There is exactly one description of every tunable in this module. The web UI
renders itself from `/api/schema`, which is generated from PARAM_SPECS, so the
sliders can never drift away from what `skin_core.skin_retouch` accepts. Adding
a knob means touching this file only.

Defaults follow the ComfyUI node, not the library function: the node ships
`include_body=True` and `device="auto"` because it is the user-facing entry
point. `skin_core.skin_retouch()` keeps the conservative pair for backwards
compatibility. Both statements are documented in HANDOFF.md; keep them true.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass, field

VERSION = "1.0.0"

# Longest edge of the preview images the browser loads. A 20 MP source is
# ~80 MB decoded as RGBA in the page; the compare slider would stall on it.
# 2400 px is still enough to judge pores at 1:1 on a 1440p panel, and the
# full-resolution PNG stays one click away for download.
DEFAULT_PREVIEW_MAX_PX = 2400

PARAM_SPECS = [
    {
        "key": "strength", "label": "总强度", "kind": "float", "group": "核心",
        "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
        "hint": "整体修图量。0 等于原图，1 等于全部按算法走满。",
    },
    {
        "key": "texture_keep", "label": "毛孔保留", "kind": "float", "group": "核心",
        "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
        "hint": "0 = 塑料脸，0.35 = 默认的皮肤质感，0.6 = 几乎没动。",
    },
    {
        "key": "blemish", "label": "痘印清除", "kind": "float", "group": "核心",
        "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
        "hint": "决定去痘的尺度半径，越高清掉的斑点越大片。",
    },
    {
        "key": "tone_even", "label": "色斑均匀", "kind": "float", "group": "核心",
        "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "hint": "泛红与色斑向局部均值收敛，不动肤色均值。",
    },
    {
        "key": "feather", "label": "边缘羽化", "kind": "float", "group": "核心",
        "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "hint": "掩膜边缘软度，按脸的大小缩放，不是固定像素。",
    },
    {
        "key": "max_faces", "label": "最大人脸数", "kind": "int", "group": "检测",
        "default": 4, "min": 1, "max": 16, "step": 1,
        "hint": "只处理最大的 N 张脸，多出来的忽略。",
    },
    {
        "key": "face_confidence", "label": "检测阈值", "kind": "float", "group": "检测",
        "default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05,
        "hint": "侧脸、遮挡、小脸可适当调低；误检背景时调高。",
    },
    {
        "key": "include_body", "label": "处理身体皮肤", "kind": "bool", "group": "身体",
        "default": True,
        "hint": "脖子上、肩、手臂、手、胸口。关掉只修脸，也就不用显卡。",
    },
    {
        "key": "body_strength", "label": "身体强度", "kind": "float", "group": "身体",
        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
        "hint": "0 = 跟随总强度。全身照建议比脸低 0.1~0.2，否则皮肤发虚。",
    },
    {
        "key": "repaint", "label": "局部重绘", "kind": "bool", "group": "重绘",
        "default": False,
        "hint": "用本地 inpaint 模型把痘点真正重画一遍，而不是抹平。每块约 10 秒，需要 ComfyUI 在线。",
    },
    {
        "key": "repaint_denoise", "label": "重绘强度", "kind": "float", "group": "重绘",
        "default": 0.60, "min": 0.30, "max": 1.0, "step": 0.05,
        "hint": "实测 0.35~1.0 差异很小，0.6 是平衡点；调高只增加离开原图的自由度。",
    },
    {
        "key": "repaint_steps", "label": "重绘步数", "kind": "int", "group": "重绘",
        "default": 8, "min": 4, "max": 30, "step": 1,
        "hint": "8 步与 20 步结果几乎相同但快一倍多，只有在细节糊掉时才往上加。",
    },
    {
        "key": "device", "label": "计算设备", "kind": "enum", "group": "性能",
        "default": "auto", "options": ["auto", "cuda", "cpu"],
        "hint": "auto 有 CUDA 就用。20 MP 全身照 GPU 3.6 s / CPU 9.8 s。",
    },
]

PARAM_KEYS = tuple(spec["key"] for spec in PARAM_SPECS)
_PARAM_BY_KEY = {spec["key"]: spec for spec in PARAM_SPECS}

PRESETS = [
    {"key": "light", "label": "轻度", "params": {
        "strength": 0.5, "texture_keep": 0.55, "blemish": 0.5}},
    {"key": "standard", "label": "标准", "params": {
        "strength": 0.75, "texture_keep": 0.35, "blemish": 0.7}},
    {"key": "commercial", "label": "商业精修", "params": {
        "strength": 0.9, "texture_keep": 0.18, "blemish": 0.85, "tone_even": 0.7}},
]

JOB_ID_RE = re.compile(r"^[0-9a-f]{16}$")


class BadParams(ValueError):
    """Raised when a request carries a parameter outside its declared domain."""


def default_params() -> dict:
    return {spec["key"]: spec["default"] for spec in PARAM_SPECS}


def coerce_params(raw: dict, base: dict | None = None) -> dict:
    """Validate a partial parameter dict and merge it onto `base`.

    Rejecting rather than clamping is deliberate: a script that asks for
    strength=75 has a bug, and silently running it at 1.0 would hide that bug
    behind a plausible-looking picture.
    """
    out = default_params() if base is None else dict(base)
    for key, value in (raw or {}).items():
        spec = _PARAM_BY_KEY.get(key)
        if spec is None:
            raise BadParams("unknown parameter: %r" % (key,))
        kind = spec["kind"]
        if kind == "bool":
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("1", "true", "yes", "on"):
                    out[key] = True
                elif low in ("0", "false", "no", "off"):
                    out[key] = False
                else:
                    raise BadParams("%s expects a boolean, got %r" % (key, value))
            elif isinstance(value, bool):
                out[key] = value
            else:
                raise BadParams("%s expects a boolean, got %r" % (key, value))
            continue
        if kind == "enum":
            if value not in spec["options"]:
                raise BadParams("%s must be one of %s, got %r"
                                % (key, "/".join(spec["options"]), value))
            out[key] = value
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise BadParams("%s expects a number, got %r" % (key, value)) from None
        if num != num or num in (float("inf"), float("-inf")):
            raise BadParams("%s must be finite, got %r" % (key, value))
        if kind == "int":
            if num != int(num):
                raise BadParams("%s expects an integer, got %r" % (key, value))
            num = int(num)
        if not (spec["min"] <= num <= spec["max"]):
            raise BadParams("%s must be within [%s, %s], got %r"
                            % (key, spec["min"], spec["max"], value))
        out[key] = num
    return out


def parse_query_params(query: dict) -> dict:
    """Turn a query-string dict into the subset of keys that are real params."""
    return {k: v[0] for k, v in query.items() if k in _PARAM_BY_KEY}


@dataclass(slots=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 8791
    token: str = ""
    # A non-loopback bind is a remote-access surface: no token means anyone on
    # the LAN can drive the GPU. run_server.py refuses that combination instead
    # of quietly starting an open service.
    allow_origin: str = ""
    max_upload_bytes: int = 128 * 1024 * 1024
    max_pixels: int = 240_000_000        # ~20 MP x 12, a generous ceiling
    job_ttl_s: int = 3600
    preview_max_px: int = DEFAULT_PREVIEW_MAX_PX
    queue_limit: int = 8
    data_dir: str = ""
    # Where the optional repaint pass looks for ComfyUI. Empty means "use
    # server.repaint.DEFAULT_URL", which honours SKIN_REPAINT_URL. Only probed
    # when a job actually asks for repaint, never at startup.
    repaint_url: str = ""
    verbose: bool = False
    settings: dict = field(default_factory=dict)

    @property
    def is_loopback(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    def resolved_data_dir(self) -> str:
        if self.data_dir:
            return os.path.abspath(self.data_dir)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def serve_schema(self) -> dict:
        return {
            "version": VERSION,
            "params": PARAM_SPECS,
            "presets": PRESETS,
            "defaults": default_params(),
            "preview_max_px": self.preview_max_px,
            "max_upload_bytes": self.max_upload_bytes,
            "auth_required": bool(self.token),
        }


def new_token() -> str:
    return secrets.token_urlsafe(18)
