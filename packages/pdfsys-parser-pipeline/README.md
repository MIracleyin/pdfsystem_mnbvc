# pdfsys-parser-pipeline

Region-level OCR backend for scanned PDFs with simple layouts. **Stub only — not yet implemented.**

Will take a `LayoutDocument` from the cache, crop each region at the configured DPI, and run OCR (RapidOCR / PaddleOCR-classic) on each crop individually. Produces an `ExtractedDoc` following the same schema as parser-mupdf.
