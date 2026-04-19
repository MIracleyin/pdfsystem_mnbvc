"""Annotation server — serves the browser-based PDF labeling tool.

Launches a local HTTP server that:
1. Serves the annotation UI (index.html) from pdfsys-bench/annotation/
2. Serves PDF files from the bench dataset directories
3. Handles annotation export/import via a simple REST API

The key fix: the server root is set to the pdfsys-bench package directory
so that relative PDF paths (``olmocr_bench_50/pdfs/...``,
``omnidocbench_100/pdfs/...``) resolve correctly.

Usage::

    pdfsys annotate                    # default port 8234
    pdfsys annotate --port 9000
    pdfsys annotate --bench-dir /path/to/pdfsys-bench
    pdfsys annotate --import annotations_2026-04-18.json
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


def _find_bench_dir() -> Path | None:
    """Auto-detect the pdfsys-bench package directory."""
    # Try relative to this file (workspace layout).
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent.parent / "pdfsys-bench",
        Path.cwd() / "packages" / "pdfsys-bench",
    ]
    # Also try via the installed package.
    try:
        import pdfsys_bench  # noqa: PLC0415

        pkg_dir = Path(pdfsys_bench.__file__).resolve().parent.parent.parent
        candidates.insert(0, pkg_dir)
    except ImportError:
        pass

    for c in candidates:
        if (c / "annotation" / "index.html").exists():
            return c
    return None


_layout_analysers: dict[str, Any] = {}  # backend_name → analyser


def _get_analyser(backend: str | None = None) -> Any:
    """Lazy-load a layout analyser by backend name. Cached per backend."""
    key = backend or "default"
    if key not in _layout_analysers:
        from pdfsys_layout_analyser import LayoutAnalyser  # noqa: PLC0415

        if backend == "pp-doclayoutv3":
            _layout_analysers[key] = LayoutAnalyser(
                model_path="PaddlePaddle/PP-DocLayoutV3_safetensors",
                backend="pp-doclayoutv3",
            )
        else:
            _layout_analysers[key] = LayoutAnalyser()
    return _layout_analysers[key]


class AnnotationHandler(SimpleHTTPRequestHandler):
    """HTTP handler that also handles annotation API endpoints."""

    def __init__(self, *args: Any, bench_dir: Path, **kwargs: Any) -> None:
        self.bench_dir = bench_dir
        self.metadata_path = bench_dir / "annotation" / "metadata.json"
        super().__init__(*args, directory=str(bench_dir), **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/api/save-annotations":
            self._handle_save()
        else:
            self.send_error(404, "Not Found")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/api/export-annotations":
            self._handle_export()
        elif parsed.path.startswith("/api/layout/"):
            self._handle_layout(parsed.path)
        elif parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", "/annotation/index.html")
            self.end_headers()
        else:
            super().do_GET()

    def _handle_layout(self, path: str) -> None:
        """Run layout analysis on a PDF and return region bboxes as JSON.

        Optional query param: ``?backend=pp-doclayoutv3`` to use PP-DocLayoutV3.
        Default is DocLayout-YOLO.
        """
        parsed_full = urlparse(self.path)
        qs = parse_qs(parsed_full.query)
        backend = qs.get("backend", [None])[0]

        # Path format: /api/layout/<rel_path_to_pdf>
        rel_path = unquote(path[len("/api/layout/"):])
        pdf_path = self.bench_dir / rel_path

        if not pdf_path.exists():
            self.send_error(404, f"PDF not found: {rel_path}")
            return

        try:
            analyser = _get_analyser(backend)
            layout = analyser.analyse(str(pdf_path))

            # Convert LayoutDocument to JSON-serializable format.
            pages_out = []
            for lp in layout.pages:
                regions_out = []
                for r in lp.regions:
                    regions_out.append({
                        "region_id": r.region_id,
                        "type": r.type.value,
                        "bbox": [r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1],
                        "confidence": round(r.confidence, 3),
                        "reading_order": r.reading_order,
                    })
                pages_out.append({
                    "index": lp.index,
                    "width_pt": lp.page_width_pt,
                    "height_pt": lp.page_height_pt,
                    "regions": regions_out,
                })

            response = json.dumps({
                "sha256": layout.sha256,
                "layout_model": layout.layout_model,
                "has_complex": layout.has_complex_content,
                "page_count": layout.page_count,
                "pages": pages_out,
            }, ensure_ascii=False)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response.encode())
        except Exception as e:
            self.send_error(500, f"Layout analysis failed: {e}")

    def _handle_save(self) -> None:
        """Save annotation data posted from the browser UI."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            annotations = json.loads(body)

            # Merge annotations into metadata.json.
            merged = _merge_annotations(self.metadata_path, annotations)

            # Atomic write.
            tmp = self.metadata_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
            shutil.move(str(tmp), str(self.metadata_path))

            response = json.dumps({
                "status": "ok",
                "saved": len(annotations),
                "total_annotated": sum(
                    1 for p in merged["pdfs"] if p.get("label") is not None
                ),
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.encode())
        except Exception as e:
            self.send_error(500, f"Save failed: {e}")

    def _handle_export(self) -> None:
        """Export current annotations as downloadable JSON."""
        try:
            data = json.loads(self.metadata_path.read_text())
            annotated = [p for p in data["pdfs"] if p.get("label") is not None]

            response = json.dumps({
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total": len(annotated),
                "annotations": annotated,
            }, ensure_ascii=False, indent=2)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="annotations_{datetime.now().strftime("%Y-%m-%d")}.json"',
            )
            self.end_headers()
            self.wfile.write(response.encode())
        except Exception as e:
            self.send_error(500, f"Export failed: {e}")

    def log_message(self, format: str, *args: Any) -> None:
        # Quieter logging — only show non-200 or API calls.
        if args and len(args) >= 2:
            code = args[1]
            path = args[0] if args else ""
            if str(code) != "200" or "/api/" in str(path):
                super().log_message(format, *args)


def _merge_annotations(
    metadata_path: Path, annotations: list[dict[str, Any]] | dict[str, Any]
) -> dict[str, Any]:
    """Merge browser-exported annotations into metadata.json."""
    data = json.loads(metadata_path.read_text())

    # Build lookup by id.
    pdf_by_id = {p["id"]: p for p in data["pdfs"]}

    # Handle both list and dict formats.
    items: list[dict[str, Any]] = []
    if isinstance(annotations, list):
        items = annotations
    elif isinstance(annotations, dict):
        # Might be {id: {label, ...}} or {annotations: [...]}
        if "annotations" in annotations:
            items = annotations["annotations"]
        else:
            items = [{"id": k, **v} for k, v in annotations.items()]

    now = datetime.now(timezone.utc).isoformat()
    merged_count = 0

    for ann in items:
        ann_id = ann.get("id")
        if ann_id and ann_id in pdf_by_id:
            pdf = pdf_by_id[ann_id]
            if ann.get("label") is not None:
                pdf["label"] = ann["label"]
                pdf["ocr_reasons"] = ann.get("ocr_reasons", pdf.get("ocr_reasons", []))
                pdf["n_ocr_pages"] = ann.get("n_ocr_pages", pdf.get("n_ocr_pages"))
                pdf["reason_short"] = ann.get("reason_short", pdf.get("reason_short", ""))
                pdf["custom_tags"] = ann.get("custom_tags", pdf.get("custom_tags", []))
                pdf["annotator"] = ann.get("annotator", pdf.get("annotator"))
                pdf["annotated_at"] = ann.get("annotated_at", now)
                merged_count += 1

    return data


def import_annotations(metadata_path: Path, annotations_file: Path) -> int:
    """Import annotations from an exported JSON file into metadata.json."""
    annotations = json.loads(annotations_file.read_text())
    merged = _merge_annotations(metadata_path, annotations)

    # Atomic write.
    tmp = metadata_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    shutil.move(str(tmp), str(metadata_path))

    annotated = sum(1 for p in merged["pdfs"] if p.get("label") is not None)
    return annotated


def serve(bench_dir: Path, port: int = 8234) -> None:
    """Start the annotation HTTP server."""
    handler = partial(AnnotationHandler, bench_dir=bench_dir)
    server = HTTPServer(("localhost", port), handler)

    print(f"[pdfsys annotate] serving from: {bench_dir}")
    print(f"[pdfsys annotate] annotation UI: http://localhost:{port}/annotation/index.html")
    print(f"[pdfsys annotate] API endpoints:")
    print(f"  POST /api/save-annotations       — save from browser")
    print(f"  GET  /api/export-annotations     — download annotated JSON")
    print(f"  GET  /api/layout/<rel_path>.pdf  — run layout analysis (on demand)")
    print(f"[pdfsys annotate] press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[pdfsys annotate] stopped")
        server.server_close()
