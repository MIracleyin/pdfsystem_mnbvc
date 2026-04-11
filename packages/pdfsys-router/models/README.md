# Router model weights

This directory is where the Stage-A XGBoost classifier weights live on disk.

The file `xgb_classifier.ubj` (≈ 257 KB) is **not committed** — it's the
ported FinePDFs binary classifier weights, owned by HuggingFace. Fetch it
once with:

```bash
python -m pdfsys_router.download_weights
```

The downloader pulls from
`media.githubusercontent.com/media/huggingface/finepdfs/main/blocks/predictor/xgb.ubj`,
which is the actual Git-LFS payload (not the pointer file that plain
`raw.githubusercontent.com` would return).
