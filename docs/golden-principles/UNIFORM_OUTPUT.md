# Uniform Parser Output

## Rule
All parser backends MUST emit the same `ExtractedDoc` / `Segment` schema. Downstream stages MUST NOT know which backend produced a document.

## DO

```python
# Good: all three backends return the same type
from pdfsys_core import ExtractedDoc, Segment, Backend, RegionType

def extract(pdf_path) -> ExtractedDoc:
    segments = (
        Segment(index=0, backend=Backend.PIPELINE, page_index=0,
                type=RegionType.TEXT, content="Hello world", bbox=bbox),
    )
    return ExtractedDoc(
        sha256=sha, backend=Backend.PIPELINE,
        segments=segments, markdown="Hello world\n",
    )
```

## DON'T

```python
# Bad: parser returns a custom dict with backend-specific fields
def extract(pdf_path) -> dict:
    return {
        "ocr_engine": "rapidocr",      # backend-specific
        "raw_boxes": [...],             # not in the shared schema
        "text": "Hello world",          # should be ExtractedDoc.markdown
    }
```

## Why
The quality scorer, dedup stage, and output packager consume `ExtractedDoc` without caring whether MuPDF, RapidOCR, or MinerU produced it. Backend-specific metadata goes in `ExtractedDoc.stats` (a free-form dict), not in the schema itself.
