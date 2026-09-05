#!/bin/bash
# cmn_Hani on xsy-01 / xsy-02 — a real site, kept as a worked example.
# Measured numbers are in docs/deployment/cmn-hani-batch.md.

CPU_HOST=xsy-01
GPU_HOST=xsy-02

# bond0, measured at 441 MB/s. NOT 10.253.253.2 — that is the EasyTier
# tunnel, which routes this traffic through a laptop.
GPUBOX=10.0.49.102

# NOT /root/pdfsys: that is an older copy with none of --extract-backends,
# --pdf-list, --resume, score or smoke.
PDFSYS=/root/pdfsys-main

CORPUS=/hdd_common/xiaoxin/data/cmn_Hani
RUN=/hdd_common/pdfsys-run
LANE=/hdd_common/pdfsys-lane

MINERU_URL=http://10.0.49.102:8000
QUALITY_URL_=http://10.0.49.102:8765
QUALITY_MODEL=miracleyin/mnbvc-pdf-quality-scorer-modernbert

# 32 of 64 cores: the box also runs QuestDB, a k8s control plane and a
# colleague's demo, so half is the polite maximum rather than a tuning result.
# Measured 15 docs/s wall, ~4 h for 217,997 documents.
WORKERS=32
GPU_WORKERS=4

OCR_THRESHOLD=0.05

# 36% of these documents need OCR but they are 72% of the bytes, and `pages`
# would add ~1 TB of rasters on top. The disk does not have it.
IMAGES=none

DATASET=/hdd_common/dataset/v2
