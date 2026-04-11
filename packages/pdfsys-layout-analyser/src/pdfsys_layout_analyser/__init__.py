"""pdfsys-layout-analyser — page layout detection.

Runs exactly once per PDF on the needs-ocr branch; the produced
LayoutDocument is cached and read by both pdfsys-router (stage-B decision)
and the downstream parser backends (pipeline / vlm).

Model candidates: PP-DocLayoutV3, docling-layout-heron (OpenVINO INT8).
"""

__version__ = "0.0.1"
