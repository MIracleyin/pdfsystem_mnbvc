#!/bin/bash
# Shared settings for the cmn_Hani batch. Sourced by every step.
#
# Everything that is a site fact rather than a decision lives here, so the
# numbered scripts read as the procedure and not as a pile of paths.

# --- where the code is (NOT /root/pdfsys — that is an older copy without
#     --extract-backends, --pdf-list, --resume, score or smoke) -------------
PDFSYS=${PDFSYS:-/root/pdfsys-main}

# --- the corpus, on xsy-01 ------------------------------------------------
CORPUS=${CORPUS:-/hdd_common/xiaoxin/data/cmn_Hani}

# --- working directories --------------------------------------------------
RUN=${RUN:-/hdd_common/pdfsys-run}          # xsy-01, CPU lane
LANE=${LANE:-/hdd_common/pdfsys-lane}       # xsy-02, GPU lane

# --- the other box, over bond0. NOT 10.253.253.2: that is the EasyTier
#     tunnel, which routes through a laptop. bond0 measured at 441 MB/s. ----
GPUBOX=${GPUBOX:-10.0.49.102}
MINERU_URL=${MINERU_URL:-http://10.0.49.102:8000}
QUALITY_URL_=${QUALITY_URL_:-http://10.0.49.102:8765}
QUALITY_MODEL=${QUALITY_MODEL:-miracleyin/mnbvc-pdf-quality-scorer-modernbert}

# --- knobs ----------------------------------------------------------------
# 32 workers on a 64-core box: measured 15 docs/s wall, ~4 h for the corpus.
# The box also runs QuestDB, a k8s control plane and a colleague's demo, so
# taking half of it is the polite maximum, not a tuning result.
WORKERS=${WORKERS:-32}

# Must be IDENTICAL on both boxes. A document routed differently on the two
# sides ends up in NO lane: the CPU box hands it away, the GPU box decides it
# did not need OCR and skips it.
OCR_THRESHOLD=${OCR_THRESHOLD:-0.05}

# How many pdfsys processes talk to mineru-api at once. Start low: one server,
# one GPU. Raise it while watching `queued_tasks` on /health.
GPU_WORKERS=${GPU_WORKERS:-4}

set -o pipefail
