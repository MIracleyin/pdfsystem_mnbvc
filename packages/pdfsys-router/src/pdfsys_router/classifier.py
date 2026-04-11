"""Stage-A classifier: decides text-ok (MUPDF) vs needs-ocr (PIPELINE/VLM).

This is the single public entry point of the router for the MVP. Stage-B
(layout-cache driven pipeline-vs-vlm decision) will be added later; for
now, anything that needs OCR is routed to ``Backend.PIPELINE`` unless the
configured policy says otherwise.

The classifier is deliberately stateless. It loads the XGBoost model once
(lazily) and then exposes ``classify(pdf_path) -> RouterDecision``. No
caching, no I/O side effects — pure in, pure out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from pdfsys_core import Backend, RouterConfig

from .feature_extractor import PDFFeatureExtractor, flatten_per_page_features
from .xgb_model import XgbRouterModel, default_weights_path


@dataclass(slots=True)
class RouterDecision:
    """Result of running the Stage-A classifier on a single PDF."""

    backend: Backend
    ocr_prob: float
    num_pages: int
    is_form: bool
    garbled_text_ratio: float
    is_encrypted: bool
    needs_password: bool
    features: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_record(self) -> dict[str, Any]:
        """Flat dict for JSONL emission."""
        return {
            "backend": self.backend.value,
            "ocr_prob": self.ocr_prob,
            "num_pages": self.num_pages,
            "is_form": bool(self.is_form),
            "garbled_text_ratio": float(self.garbled_text_ratio),
            "is_encrypted": bool(self.is_encrypted),
            "needs_password": bool(self.needs_password),
            "error": self.error,
        }


class Router:
    """Stage-A router: PyMuPDF features → XGBoost → Backend."""

    def __init__(
        self,
        config: RouterConfig | None = None,
        model_path: str | Path | None = None,
        num_pages_to_sample: int = 8,
        ocr_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.config = config or RouterConfig()
        self.num_pages_to_sample = num_pages_to_sample
        self.ocr_threshold = ocr_threshold
        self.seed = seed
        self._extractor = PDFFeatureExtractor(
            num_chunks=1, num_pages_to_sample=num_pages_to_sample
        )
        self._model = XgbRouterModel(model_path or default_weights_path())

    # ------------------------------------------------------------------ api

    def classify(self, pdf_path: str | Path) -> RouterDecision:
        """Classify a PDF file. Never raises — errors are in ``decision.error``."""
        path = Path(pdf_path)
        try:
            doc = pymupdf.open(str(path))
        except Exception as e:  # noqa: BLE001 — we want to capture anything
            return RouterDecision(
                backend=Backend.DEFERRED,
                ocr_prob=float("nan"),
                num_pages=0,
                is_form=False,
                garbled_text_ratio=0.0,
                is_encrypted=False,
                needs_password=False,
                error=f"open_failed: {e}",
            )

        try:
            return self._classify_doc(doc)
        finally:
            try:
                doc.close()
            except Exception:
                pass

    def classify_bytes(self, pdf_bytes: bytes) -> RouterDecision:
        """Same as :meth:`classify`, but from an in-memory buffer."""
        import io

        try:
            doc = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        except Exception as e:  # noqa: BLE001
            return RouterDecision(
                backend=Backend.DEFERRED,
                ocr_prob=float("nan"),
                num_pages=0,
                is_form=False,
                garbled_text_ratio=0.0,
                is_encrypted=False,
                needs_password=False,
                error=f"open_failed: {e}",
            )
        try:
            return self._classify_doc(doc)
        finally:
            try:
                doc.close()
            except Exception:
                pass

    # --------------------------------------------------------------- internal

    def _classify_doc(self, doc: pymupdf.Document) -> RouterDecision:
        # Seed the sampling RNGs so the same PDF always produces the same
        # feature vector — critical for reproducibility and debugging.
        random.seed(self.seed)
        np.random.seed(self.seed)

        try:
            if doc.is_encrypted or doc.needs_pass:
                return RouterDecision(
                    backend=Backend.DEFERRED,
                    ocr_prob=float("nan"),
                    num_pages=len(doc),
                    is_form=False,
                    garbled_text_ratio=0.0,
                    is_encrypted=bool(doc.is_encrypted),
                    needs_password=bool(doc.needs_pass),
                    error="encrypted_or_password_protected",
                )

            raw_chunks = self._extractor.extract_all_features(doc)
            if not raw_chunks:
                return RouterDecision(
                    backend=Backend.DEFERRED,
                    ocr_prob=float("nan"),
                    num_pages=len(doc),
                    is_form=False,
                    garbled_text_ratio=0.0,
                    is_encrypted=False,
                    needs_password=False,
                    error="no_pages_sampled",
                )

            flat = flatten_per_page_features(
                raw_chunks[0], sample_to_k_page_features=self.num_pages_to_sample
            )
            ocr_prob = self._model.predict_proba(flat)

            backend = self._route(ocr_prob)
            return RouterDecision(
                backend=backend,
                ocr_prob=ocr_prob,
                num_pages=len(doc),
                is_form=bool(flat.get("is_form", False)),
                garbled_text_ratio=float(flat.get("garbled_text_ratio", 0.0)),
                is_encrypted=bool(doc.is_encrypted),
                needs_password=bool(doc.needs_pass),
                features=flat,
            )
        except Exception as e:  # noqa: BLE001
            return RouterDecision(
                backend=Backend.DEFERRED,
                ocr_prob=float("nan"),
                num_pages=len(doc) if doc else 0,
                is_form=False,
                garbled_text_ratio=0.0,
                is_encrypted=False,
                needs_password=False,
                error=f"classify_failed: {e}",
            )

    def _route(self, ocr_prob: float) -> Backend:
        """Map XGBoost probability + fleet policy → concrete Backend."""
        if ocr_prob < self.ocr_threshold:
            return Backend.MUPDF
        # OCR needed. Stage-B would check LayoutCache for complex content
        # here. For the MVP we have no layout cache yet, so honour the
        # fleet VLM gate: if VLM is enabled we'd need Stage-B to decide,
        # otherwise pipeline handles everything flagged as scanned.
        if self.config.vlm_enabled:
            return Backend.DEFERRED  # Stage-B will run once layout is cached
        return Backend.PIPELINE
