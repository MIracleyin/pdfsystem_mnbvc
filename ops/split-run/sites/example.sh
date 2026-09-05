#!/bin/bash
# Copy this to sites/<your-name>.sh, edit, then: export PDFSYS_SITE=<your-name>
#
# BOTH boxes read this same file. Put it somewhere both can see it, or copy it
# to both — preflight.sh compares its checksum across the two and refuses when
# they differ, because the one value that must not disagree is OCR_THRESHOLD.

# ── the two machines ──────────────────────────────────────────────────────
# CPU_HOST/GPU_HOST are matched against `hostname` so a step run on the wrong
# box stops immediately. They are patterns, so "gpu-*" works for a fleet.
CPU_HOST=cpu-01           # where the corpus is; runs routing + mupdf
GPU_HOST=gpu-01           # runs mineru-api and the quality server

# How the CPU box reaches the GPU box for rsync and HTTP. Use the fastest
# path between them — a datacentre LAN address, not a VPN or tunnel address
# that routes through somewhere else.
GPUBOX=gpu-01.internal

# ── paths ────────────────────────────────────────────────────────────────
PDFSYS=/opt/pdfsys                    # the checkout, on BOTH boxes
CORPUS=/data/corpus                   # the PDFs, on the CPU box
RUN=/data/pdfsys-run                  # CPU-lane working dir
LANE=/data/pdfsys-lane                # GPU-lane working dir, on the GPU box

# ── services on the GPU box ──────────────────────────────────────────────
# Defaults are http://$GPU_HOST:8000 and :8765 — set these when the ports or
# the address the CPU box must dial differ.
MINERU_URL=http://gpu-01.internal:8000
QUALITY_URL_=http://gpu-01.internal:8765
QUALITY_MODEL=miracleyin/mnbvc-pdf-quality-scorer-modernbert

# ── knobs ────────────────────────────────────────────────────────────────
# Defaults: WORKERS = half the CPU box's cores, GPU_WORKERS = 4.
# WORKERS=32

# THE value that must be identical on both boxes. A document routed one way
# here and the other way there is handed off by one box and skipped by the
# other: it ends up in NO lane. This is why both boxes read one file.
OCR_THRESHOLD=0.05

# How many pdfsys processes talk to mineru-api at once. Watch queued_tasks on
# its /health: if it stays 0, the server is idle and this is too low.
# GPU_WORKERS=4

# crops | pages | none. `pages` rasterises every page at 200 dpi (~311 KiB
# per page) — check the arithmetic against your disk before choosing it.
# `none` leaves mnbvc-export's 图片 column null.
IMAGES=none
