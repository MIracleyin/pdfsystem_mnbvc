#!/bin/bash
# Step 0 — put working code on a box, and prove it works. Run on xsy-01.
#
# The check at the end is the whole point. It costs three seconds and it is
# the difference between finding out now and finding out four hours in.
set -e
cd "$(dirname "$0")" && source ./config.sh

if [ ! -d "$PDFSYS" ]; then
  git clone https://github.com/MIracleyin/pdfsystem_mnbvc.git "$PDFSYS"
  git -C "$PDFSYS" submodule update --init --recursive --depth 1
fi
cd "$PDFSYS"
git pull --ff-only
uv sync --locked

# models/ is gitignored — the XGBoost weights are FinePDFs' to distribute, so
# every fresh checkout starts without them. Since v0.6 the run refuses to
# start rather than routing the whole corpus to `deferred` and exiting 0.
uv run python -m pdfsys_router.download_weights

# Eight tiny PDFs through all four phases against in-process stubs.
uv run pdfsys smoke

# And against the real services, which is what actually gets used.
uv run pdfsys smoke --mineru-url "$MINERU_URL" --quality-url "$QUALITY_URL_"
