#!/bin/bash
# Config resolution. Sourced by every numbered script; not run directly.
#
# A site config is one file that BOTH boxes read. That is deliberate: the CPU
# box and the GPU box must agree on --ocr-threshold, and a value written twice
# is a value that can differ. preflight.sh compares this file's checksum
# across the two boxes and refuses when they diverge.
#
# Pick a site with either:
#     export PDFSYS_SITE=my-corpus          # -> ops/split-run/sites/my-corpus.sh
#     export PDFSYS_SITE=/abs/path/to.sh    # -> that file
#
# Start from sites/example.sh. sites/cmn-hani.sh is a real one, kept as a
# worked example with measured numbers.

set -o pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${PDFSYS_SITE:-}" ]; then
  echo "PDFSYS_SITE is not set." >&2
  echo "  available sites: $(ls "$_here/sites" | sed 's/\.sh$//' | tr '\n' ' ')" >&2
  echo "  export PDFSYS_SITE=<name>, or copy sites/example.sh and edit it." >&2
  exit 1
fi

case "$PDFSYS_SITE" in
  /*) SITE_FILE="$PDFSYS_SITE" ;;
  *)  SITE_FILE="$_here/sites/$PDFSYS_SITE.sh" ;;
esac

[ -f "$SITE_FILE" ] || { echo "no such site config: $SITE_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$SITE_FILE"

# The checksum both boxes compare. md5sum on Linux, md5 on macOS.
SITE_SUM=$(md5sum "$SITE_FILE" 2>/dev/null | cut -d' ' -f1 \
        || md5 -q "$SITE_FILE" 2>/dev/null)

# --- defaults for anything the site did not set --------------------------
: "${WORKERS:=$(( $(nproc 2>/dev/null || echo 8) / 2 ))}"
: "${GPU_WORKERS:=4}"
: "${OCR_THRESHOLD:=0.05}"
: "${IMAGES:=none}"
: "${DATASET:=$RUN/dataset}"
: "${QUALITY_MODEL:=miracleyin/mnbvc-pdf-quality-scorer-modernbert}"
: "${MINERU_URL:=http://$GPU_HOST:8000}"
: "${QUALITY_URL_:=http://$GPU_HOST:8765}"

# --- every site must set these -------------------------------------------
_missing=()
for v in PDFSYS CORPUS RUN LANE CPU_HOST GPU_HOST; do
  [ -n "${!v:-}" ] || _missing+=("$v")
done
if [ ${#_missing[@]} -gt 0 ]; then
  echo "site $SITE_FILE does not set: ${_missing[*]}" >&2
  exit 1
fi

# --- refuse to run a box's script on the wrong box ------------------------
# Costs nothing and rules out the whole class of "I ran 05 on the CPU box".
_require_host() {
  local want="$1" me
  me=$(hostname)
  case "$me" in
    $want) return 0 ;;
    *) echo "this step runs on $want, but this is $me" >&2; exit 1 ;;
  esac
}
