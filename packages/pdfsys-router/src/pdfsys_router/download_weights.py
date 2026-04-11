"""Fetch the FinePDFs XGBoost router weights from upstream.

The weights file (``xgb.ubj``, ~257 KB) is not committed to this repo —
it's external IP owned by HuggingFace/FinePDFs and lives on their Git-LFS
bucket. Running this module downloads it once into ``models/xgb_classifier.ubj``
next to this package.

Usage::

    python -m pdfsys_router.download_weights
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# media.githubusercontent.com serves the actual LFS payload directly,
# bypassing the pointer file that raw.githubusercontent.com returns.
WEIGHTS_URL = (
    "https://media.githubusercontent.com/media/huggingface/finepdfs/main/"
    "blocks/predictor/xgb.ubj"
)


def target_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "models" / "xgb_classifier.ubj"


def download(force: bool = False) -> Path:
    dst = target_path()
    if dst.exists() and not force:
        print(f"[download_weights] already present: {dst}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_weights] fetching {WEIGHTS_URL}")
    with urllib.request.urlopen(WEIGHTS_URL) as r:  # noqa: S310 — pinned URL
        data = r.read()
    if len(data) < 10_000:
        raise RuntimeError(
            f"downloaded blob is suspiciously small ({len(data)} bytes) — "
            "likely an LFS pointer, not the binary"
        )
    dst.write_bytes(data)
    print(f"[download_weights] wrote {len(data)} bytes -> {dst}")
    return dst


if __name__ == "__main__":
    force = "--force" in sys.argv
    download(force=force)
