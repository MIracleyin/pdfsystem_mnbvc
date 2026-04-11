"""pdfsys-router — two-stage routing for the pdfsys extraction pipeline.

Stage A (cheap): classify text-ok vs needs-ocr from PyMuPDF features.
Stage B (uses layout cache): for needs-ocr, read the LayoutDocument written
by pdfsys-layout-analyser and decide pipeline vs vlm based on whether
complex regions (tables / formulas) exist.
"""

__version__ = "0.0.1"
