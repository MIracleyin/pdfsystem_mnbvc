"""Hugging Face Spaces entry point.

HF Spaces looks for ``app.py`` at the repo root. We just import the
actual app from ``demo/`` so the demo code stays tucked away and the
root stays uncluttered.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Add packages to path (needed before importing demo modules)
for pkg in ("pdfsys-core", "pdfsys-router", "pdfsys-parser-mupdf", "pdfsys-bench"):
    src = _REPO_ROOT / "packages" / pkg / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

# Add demo dir to path (needed for pipeline import)
_DEMO_DIR = _REPO_ROOT / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

# Load demo app from demo/app.py using importlib to avoid name collision
_demo_module_path = _DEMO_DIR / "app.py"
spec = importlib.util.spec_from_file_location("demo_app", _demo_module_path)
_demo_module = importlib.util.module_from_spec(spec)
sys.modules["demo_app"] = _demo_module
spec.loader.exec_module(_demo_module)
demo = _demo_module.demo  # noqa: F401 — re-exported for HF Spaces

if __name__ == "__main__":
    import os

    import gradio as gr

    CSS = """
.small-num input { font-weight: 600; font-size: 1.1rem; }
footer { display: none !important; }
"""

    demo.queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CSS,
    )
