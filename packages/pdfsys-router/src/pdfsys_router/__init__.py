"""pdfsys-router — two-stage routing for the pdfsys extraction pipeline.

Stage A (cheap): classify text-ok vs needs-ocr from PyMuPDF features, using
a ported FinePDFs XGBoost classifier over 124 hand-crafted features.

Stage B (uses layout cache): for needs-ocr, read the LayoutDocument written
by pdfsys-layout-analyser and decide pipeline vs vlm based on whether
complex regions (tables / formulas) exist.
"""

from __future__ import annotations

from .classifier import Router, RouterDecision
from .decider import StageBDecision, decide, decide_from_cache
from .feature_extractor import PDFFeatureExtractor, flatten_per_page_features
from .xgb_model import XgbRouterModel, default_weights_path

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "Router",
    "RouterDecision",
    "StageBDecision",
    "decide",
    "decide_from_cache",
    "PDFFeatureExtractor",
    "flatten_per_page_features",
    "XgbRouterModel",
    "default_weights_path",
]
