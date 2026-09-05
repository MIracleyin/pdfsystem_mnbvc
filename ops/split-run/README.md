# Split-run scripts

Batch-run the pipeline across a CPU box and a GPU box that share no disk.
Nothing here is specific to one site — the machines, paths and knobs come from
a **site config**, and the same file is read by both boxes.

Why the whole procedure is a file and not a wiki page: the two settings that
matter most (`OCR_THRESHOLD`, the code revision) are catastrophic when they
differ between the boxes and produce no error when they do. A shared file and
a preflight that compares it are cheaper than remembering.

## Setting up a new site

```bash
cp sites/example.sh sites/my-corpus.sh
$EDITOR sites/my-corpus.sh          # machines, paths, threshold
export PDFSYS_SITE=my-corpus
```

Put that file where **both** boxes can read it at the same path — commit it
here, or copy it to both. `preflight.sh` compares its checksum across the two
and refuses when they diverge.

`sites/cmn-hani.sh` is a real site kept as a worked example; the numbers it
was measured against are in
[`docs/deployment/cmn-hani-batch.md`](../../docs/deployment/cmn-hani-batch.md).

## Running

```bash
# on BOTH boxes
./00-deploy.sh          # clone/update, deps, router weights, smoke x2

# on the CPU box
./preflight.sh          # both boxes, both services, both configs. read-only.
./01-inventory.sh       # find every PDF, cut it into buckets
./02-cpu-lane.sh        # classify + extract the text-layer documents
./status.sh             # progress, rate, errors — repeatable
./03-handoff.sh         # derive each lane's worklist
./04-transfer.sh        # ship the OCR-bound PDFs

# on the GPU box
./05-gpu-lane.sh        # MinerU

# each box scores and packages its own lane, into one dataset directory
./06-score.sh   cpu|gpu
./07-package.sh cpu|gpu /path/to/dataset
```

Each step refuses to run on the wrong box (`hostname` against `CPU_HOST` /
`GPU_HOST`), so `05` on the CPU box stops instead of quietly doing nothing.

Override a knob for one run without editing the site:

```bash
WORKERS=16 ./02-cpu-lane.sh
IMAGES=pages ./07-package.sh gpu /data/dataset/v2
```

## What each guard is for

| Guard | Without it |
|---|---|
| `preflight.sh` compares site-config checksums | The boxes disagree on `OCR_THRESHOLD`; a document is handed off by one and skipped by the other, so it is in **no** lane. Silent. |
| `preflight.sh` compares git revisions | Lane semantics live in the code. Two versions is two pipelines. |
| `preflight.sh` checks the served model name | Two lanes scored by two models put two scales in one column, and nothing in the data says so. |
| `_require_host` in every step | A step run on the wrong box reads empty directories and reports success. |
| `03-handoff.sh` compares processed rows to the inventory | "No workers running" is not "finished". A killed run yields a short worklist, the next steps process the subset happily, and the corpus is quietly missing a slice. |
| `pdfsys run` refuses without router weights | `classify()` never raises, so missing weights route the whole corpus to `deferred` and exit 0. |

## Notes that have cost time here

- **`pgrep -f pdfsys` matches your own ssh command line.** Every guard here
  matches the workers' `--pdf-list` argument instead.
- **`pgrep -c` prints `0` *and* exits nonzero** when it finds nothing, so
  `|| echo 0` prints it twice.
- **`split -n l/N` is GNU-only.** macOS `split` has no `-n`.
- **Each worker needs its own `--out-dir`.** `results.jsonl` is append-only;
  two writers in one directory interleave into a file neither can resume.
  `--markdown-dir` is safe to share — files are named `<sha256>.md`.
