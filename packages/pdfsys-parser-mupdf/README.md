# pdfsys-parser-mupdf

Text-ok extraction backend. This is the fast path for PDFs that the router classifies as having a clean embedded text layer (i.e. `ocr_prob < threshold`).

## What it does

1. Opens the PDF with PyMuPDF.
2. Iterates every page, calling `page.get_text("blocks", sort=True)`.
3. Filters to text blocks (drops image blocks).
4. Normalizes each block's bbox to [0, 1] coordinates.
5. Produces one `Segment` per block, joined into an `ExtractedDoc` with merged Markdown.

## Usage

```python
from pdfsys_parser_mupdf import extract_doc

doc = extract_doc("path/to/clean.pdf")
print(doc.markdown[:500])
print(f"{doc.segment_count} segments, {doc.char_count} chars")
```

## Scope

This backend intentionally does NOT:
- Run OCR (that's what parser-pipeline and parser-vlm are for)
- Use a layout model (not needed for text-ok PDFs)
- Extract images or tables (image-heavy PDFs should be routed elsewhere)

It is the simplest possible extraction: unwrap PyMuPDF blocks into structured output.
