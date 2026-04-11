# pdfsys-router

Two-stage routing for the pdfsys extraction pipeline.

## Stage A (implemented)

XGBoost binary classifier ported from [FinePDFs](https://github.com/huggingface/finepdfs). Given a PDF, it extracts 124 features using PyMuPDF (4 doc-level + 15 page-level × 8 sampled pages) and predicts `P(needs OCR)`.

- `ocr_prob < threshold` → **MUPDF** (text-ok, fast path)
- `ocr_prob >= threshold` → **PIPELINE** (needs OCR)

### Usage

```python
from pdfsys_router import Router

router = Router()  # loads xgb_classifier.ubj lazily
decision = router.classify("path/to/document.pdf")
print(decision.backend, decision.ocr_prob)
```

### Weights

The XGBoost model (`models/xgb_classifier.ubj`, 257 KB) is gitignored. Fetch it once:

```bash
python -m pdfsys_router.download_weights
```

## Stage B (not yet implemented)

For PDFs routed to OCR, Stage B reads the cached `LayoutDocument` and decides:
- No complex content → `PIPELINE` (region-level OCR)
- Tables / formulas present → `VLM` (vision-language model)

## Module layout

| File | Purpose |
|------|---------|
| `feature_extractor.py` | Port of FinePDFs' `PDFFeatureExtractor` — DO NOT modify without retraining |
| `xgb_model.py` | Lazy XGBoost model loader |
| `classifier.py` | `Router.classify()` → `RouterDecision` public API |
| `download_weights.py` | Fetches weights from FinePDFs Git LFS |
| `decider.py` | Stage-B stub |
