"""LayoutCache — content-addressable on-disk store for LayoutDocument.

Every LayoutDocument produced by ``pdfsys-layout-analyser`` is written here
exactly once, then read by any number of downstream consumers (router
stage-B, parser-pipeline, parser-vlm). The cache is *stateless*: the only
"state" is file existence. There is no manifest, no index, no lock file.

File layout::

    {root}/{sha256[:2]}/{sha256[2:4]}/{sha256}.{layout_model_slug}.json

Two-level sharding keeps any single directory under a few thousand entries
even at billion-PDF scale. The model slug is part of the filename so that
bumping ``LayoutConfig.model_version`` lazily invalidates old entries — old
files stay on disk until pruned, new ones get written under the new slug.

Writes are atomic: the JSON blob is first written to a temp file in the
same directory, then ``os.replace``'d onto the final path. Crash-safe on
any POSIX filesystem.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .layout import LayoutDocument
from .serde import from_dict, to_dict


def _slugify_model(layout_model: str) -> str:
    """Make a layout_model string filesystem-safe. Keeps letters/digits/._-@."""
    safe = []
    for ch in layout_model:
        if ch.isalnum() or ch in "._-@":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


class LayoutCache:
    """A directory full of cached LayoutDocument JSON files.

    Usage::

        cache = LayoutCache("/data/pdfsys/cache/layout")
        if not cache.exists(sha256, "pp-doclayoutv3@1.0"):
            doc = run_layout_model(pdf_path)
            cache.save(doc)
        doc = cache.load(sha256, "pp-doclayoutv3@1.0")
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, sha256: str, layout_model: str) -> Path:
        """Resolve the on-disk path for a given (sha256, layout_model) pair."""
        if len(sha256) < 4:
            raise ValueError(f"sha256 too short to shard: {sha256!r}")
        slug = _slugify_model(layout_model)
        return self.root / sha256[:2] / sha256[2:4] / f"{sha256}.{slug}.json"

    def exists(self, sha256: str, layout_model: str) -> bool:
        return self.path_for(sha256, layout_model).is_file()

    def load(self, sha256: str, layout_model: str) -> LayoutDocument:
        path = self.path_for(sha256, layout_model)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return from_dict(LayoutDocument, data)

    def save(self, doc: LayoutDocument) -> Path:
        """Atomically persist a LayoutDocument. Returns the final path."""
        path = self.path_for(doc.sha256, doc.layout_model)
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{doc.sha256}.",
            suffix=".json.tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(to_dict(doc), f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return path
