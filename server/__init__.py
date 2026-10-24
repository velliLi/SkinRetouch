"""Local/remote web service for Skin Retouch.

The package lives inside the node directory on purpose: `E:\\SkinRetouch` is
junctioned into `ComfyUI/custom_nodes/`, and ComfyUI only imports the top-level
package, so nothing here is loaded by ComfyUI at startup. Starting the service is
always an explicit act (`python run_server.py`).

Importing this package prepends the project root to `sys.path` so `skin_core`,
`skin_body` and `skin_gpu` resolve exactly as they do in the standalone tests.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from .config import VERSION, Config, default_params, coerce_params  # noqa: E402

__all__ = ["VERSION", "Config", "default_params", "coerce_params", "serve"]


def serve(cfg: Config):
    """Start the HTTP service on `cfg`; returns the running server object."""
    from .api import create_server

    return create_server(cfg)
