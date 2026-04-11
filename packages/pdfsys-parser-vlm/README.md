# pdfsys-parser-vlm

Vision-language model backend for scanned PDFs with complex content (tables, formulas). **Stub only — not yet implemented.**

Will handle regions flagged as TABLE or FORMULA by the layout analyser, sending them to a VLM (MinerU 2.5 / PaddleOCR-VL) that can produce structured output (HTML tables, LaTeX formulas). Simple text regions in the same document may still be handled by the pipeline backend.
