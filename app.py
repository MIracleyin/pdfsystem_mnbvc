"""Hugging Face Spaces entry point.

HF Spaces looks for ``app.py`` at the repo root. We just import the
actual app from ``demo/`` so the demo code stays tucked away and the
root stays uncluttered.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent / "demo"
sys.path.insert(0, str(_DEMO_DIR))

from app import demo  # noqa: E402,F401 — re-exported for HF Spaces

if __name__ == "__main__":
    import os

    demo.queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
