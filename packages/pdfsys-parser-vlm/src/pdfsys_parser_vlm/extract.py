"""Mineru-backed VLM parser.

Thin wrapper around ``mineru.cli.common.do_parse(backend="vlm-<engine>")``.
Mineru handles layout analysis + per-region VLM extraction + markdown
assembly end-to-end; this module only marshals input PDFs in and reads
markdown + sidecars out.

See ``docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

import mineru
from mineru.cli.common import aio_do_parse

from pdfsys_core import Backend, ExtractedDoc, VlmConfig

_PARSE_METHOD = "auto"
_LOG = logging.getLogger(__name__)


class VlmParser:
    """Mineru VLM-mode parser. Stateless; mineru manages model caching."""

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig()

    def extract(self, pdf_path: Path) -> ExtractedDoc:
        """Extract markdown from ``pdf_path`` via mineru VLM mode.

        Writes to ``config.output_dir/<sha>/<parse_method>/`` if set,
        otherwise a tmpdir that is cleaned up before returning.
        """
        pdf_path = Path(pdf_path)
        pdf_bytes = pdf_path.read_bytes()
        sha = hashlib.sha256(pdf_bytes).hexdigest()

        if self.config.output_dir is not None:
            output_root = Path(self.config.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            return self._run(sha, pdf_bytes, output_root, persistent=True)

        with tempfile.TemporaryDirectory(prefix="pdfsys-mineru-vlm-") as td:
            return self._run(sha, pdf_bytes, Path(td), persistent=False)

    def _run(
        self,
        sha: str,
        pdf_bytes: bytes,
        output_root: Path,
        *,
        persistent: bool,
    ) -> ExtractedDoc:
        backend = f"vlm-{self.config.engine}"
        # NOTE: aio_do_parse (asyncio) instead of do_parse (sync) — the
        # sync path triggers a multiprocessing.Pool that deadlocks on macOS
        # without CUDA. aio_do_parse uses asyncio.gather internally and
        # bypasses the multiprocess PDF render executor.
        asyncio.run(aio_do_parse(
            output_dir=str(output_root),
            pdf_file_names=[sha],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[self.config.p_lang],
            backend=backend,
            parse_method=_PARSE_METHOD,
            formula_enable=self.config.formula_enable,
            table_enable=self.config.table_enable,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_content_list=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            image_analysis=True,
        ))

        md_dir = output_root / sha / _PARSE_METHOD
        md_path = md_dir / f"{sha}.md"
        if not md_path.exists():
            # Defensive fallback: glob, in case mineru changes layout.
            # Prefer the sha-named file if present among siblings; otherwise
            # take the lexicographically-first match (deterministic across runs).
            candidates = sorted(md_dir.glob("*.md")) if md_dir.exists() else []
            if not candidates:
                raise FileNotFoundError(
                    f"mineru did not produce a markdown file under {md_dir} (sha={sha})"
                )
            md_path = next((c for c in candidates if c.stem == sha), candidates[0])
            _LOG.warning(
                "mineru wrote markdown at unexpected location %s (sha=%s); "
                "this is OK but indicates mineru's output layout has shifted",
                md_path, sha,
            )

        markdown = md_path.read_text(encoding="utf-8")

        stats: dict[str, Any] = {
            "mineru_backend": backend,  # variable per VlmConfig.engine; PipelineParser uses fixed "pipeline"
            "mineru_version": _mineru_version(),
            "middle_json_path": _rel_or_none(
                md_dir / f"{sha}_middle.json", output_root, persistent
            ),
            "content_list_path": _rel_or_none(
                md_dir / f"{sha}_content_list.json", output_root, persistent
            ),
        }

        return ExtractedDoc(
            sha256=sha,
            backend=Backend.VLM,
            segments=(),
            markdown=markdown,
            stats=stats,
        )


def _rel_or_none(path: Path, root: Path, persistent: bool) -> str | None:
    """Return path relative to root if it exists AND output is persistent.

    For tmpdir runs, all paths vanish on cleanup so we record None.
    """
    if not persistent:
        return None
    if not path.exists():
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mineru_version() -> str | None:
    """Return mineru's package version (or None if not exposed)."""
    return getattr(mineru, "__version__", None)
