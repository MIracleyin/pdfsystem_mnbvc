# pdfsys-layout-analyser

Page layout model runner. **Stub only — not yet implemented.**

Will run a layout detection model (PP-DocLayoutV3 / docling-layout-heron) on each page and write a `LayoutDocument` to the `LayoutCache`. This layout is consumed by:

1. **pdfsys-router Stage B** — checks `has_complex_content` to decide pipeline vs VLM.
2. **pdfsys-parser-pipeline** — uses region bboxes to crop and OCR individual regions.
3. **pdfsys-parser-vlm** — sends complex regions to a vision-language model.

Layout inference runs at most once per PDF (keyed by sha256 + model_tag in the cache).
