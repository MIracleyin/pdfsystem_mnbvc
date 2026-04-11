# bench/ — PDF processing pipeline evaluation set

This directory is the **canonical test set** for evaluating the end-to-end PDF
processing pipeline (layout → OCR → markdown / structured text). It bundles
two complementary, pre-sampled subsets so that runs are reproducible and
cheap to iterate on.

| Subset | PDFs | Source benchmark | Focus |
|---|---:|---|---|
| [`olmocr_bench_50/`](./olmocr_bench_50) | 50 | [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) | Fine-grained unit tests on text presence / absence, reading order, tables, math |
| [`omnidocbench_100/`](./omnidocbench_100) | 100 | [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Holistic document-level eval with layout / language / special-issue coverage |

Total footprint: ~108 MB, 150 PDFs.

## Subset details

### `olmocr_bench_50/`
Stratified sample drawn from the 1,403-PDF olmOCR-bench with the script
`scripts/sample_olmocr_subset.py` (seed `20260411`). Covers all 7 document
sources with a minimum floor of 3 PDFs per category plus largest-remainder
proportional allocation, and diversifies by source document inside each
category (at most one page per arXiv paper / scan ID before any repeat).

```
olmocr_bench_50/
├── pdfs/
│   ├── arxiv_math/         (14)
│   ├── headers_footers/    (8)
│   ├── long_tiny_text/     (4)
│   ├── multi_column/       (8)
│   ├── old_scans/          (5)
│   ├── old_scans_math/     (4)
│   └── tables/             (7)
├── subset_tests.jsonl      # 283 olmOCR-bench unit tests for these 50 PDFs
└── subset_manifest.json    # seed, quotas, selected file list, source bench_dir
```

The `subset_tests.jsonl` file is a filtered copy of the original per-category
`*.jsonl` test files merged into one; each row keeps the exact schema used by
the upstream olmOCR-bench evaluator (`pdf`, `type`, `max_diffs`, `checked`,
and type-specific fields like `math`, `cell`, `before`/`after`, …).

Regenerate or resize:
```bash
python3 scripts/sample_olmocr_subset.py --target 50             # default → bench/olmocr_bench_50
python3 scripts/sample_olmocr_subset.py --target 100 --seed 42  # alt subset
python3 scripts/sample_olmocr_subset.py --dry-run               # plan only
```

### `omnidocbench_100/`
Pre-built 100-PDF subset of OmniDocBench v2 with full stratified coverage
across every categorical axis in the upstream dataset.

```
omnidocbench_100/
├── pdfs/                   # 100 single-page PDFs
├── img/                    # matching rendered JPGs (1 per PDF)
├── subset_100.json         # full OmniDocBench annotations for the 100 samples
├── subset_100_stats.json   # coverage & distribution stats vs. full 981-doc set
├── subset_100_pdfs.txt     # flat list of selected PDF filenames
└── subset_100_images.txt   # flat list of selected image filenames
```

Coverage (from `subset_100_stats.json`) — every bucket of every axis is hit:
- **data_source** 9/9 · **language** 3/3 · **layout** 5/5
- **special_issue** 13/13 · **stratum** 67/67

## Using the bench

These two subsets are intended to be run as a pair — olmOCR-bench gives you
sharp per-feature pass/fail signals and OmniDocBench gives you an aggregate
quality score across real-world document types. For each new pipeline
version, run both subsets, record per-subset metrics, and diff against the
previous run.

Common entry points (to be wired up by the pipeline evaluator):

```text
bench/olmocr_bench_50/pdfs/**/*.pdf      # inputs
bench/olmocr_bench_50/subset_tests.jsonl # ground truth unit tests

bench/omnidocbench_100/pdfs/*.pdf        # inputs
bench/omnidocbench_100/subset_100.json   # ground truth annotations
```

Do **not** manually edit files under `bench/`. Regenerate with the sampling
script (for olmocr) or re-export from the upstream builder (for omnidoc) so
results stay reproducible.
