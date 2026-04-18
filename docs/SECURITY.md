# Security

## Authentication

Not applicable — pdfsys is a batch processing pipeline, not a service. No user authentication.

## Secrets Management

- **HuggingFace tokens:** Loaded from `HF_TOKEN` environment variable for private model downloads. Not required for public models.
- **Object storage credentials:** Loaded from standard cloud SDK env vars (e.g., AWS/OSS/MinIO configs). Not embedded in code.
- **No secrets in code or config files.** The YAML config (`pdfsys.yaml`) contains only model names and thresholds.

## Data Sensitivity

| Data Type | Sensitivity | Handling |
|-----------|------------|---------|
| Source PDFs (L0) | May contain PII | Never modified; access-controlled at storage layer |
| Extracted text (L1/L2) | May contain PII | PII redaction planned (Stage 4); not yet implemented |
| Model weights | Public | Downloaded from HuggingFace Hub / GitHub LFS |
| Benchmark datasets | Public | OmniDocBench, olmOCR-bench (open datasets) |

## Dependencies

- **Security-critical:** PyMuPDF (PDF parsing — handles untrusted input), torch, transformers
- **Dependency updates:** Manual; no Dependabot configured yet

## Threat Model

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Malicious PDF input | PyMuPDF sandboxed parsing; errors caught, not propagated | In place |
| PII leakage in output | PII redaction stage (regex + NER) | Planned (Stage 4) |
| Supply chain (model weights) | Models downloaded from HuggingFace Hub with hash verification | In place |
| Arbitrary code in PDF JS | PyMuPDF does not execute JavaScript | In place |
