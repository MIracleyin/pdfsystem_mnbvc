#!/bin/bash
# Step 0 — put working code on a box, and prove it works. Run on BOTH boxes.
#
# The smoke checks at the end are the point. They cost seconds and they are
# the difference between finding out now and finding out four hours in.
set -e
cd "$(dirname "$0")" && source ./config.sh

REPO=${REPO:-https://github.com/MIracleyin/pdfsystem_mnbvc.git}

# Boxes in CN often carry a global `insteadOf` rule pointing GitHub at a
# mirror, and those mirrors go away. xsy-02 had one for ghfast.top that had
# stopped resolving, which makes `git clone` fail with an SSL timeout against
# a URL nobody typed. Probe the rewrite and step around it rather than
# editing someone else's global config.
GIT="git"
if ! timeout 25 git ls-remote "$REPO" HEAD >/dev/null 2>&1; then
  rewrite=$(git config --global --get-regexp 'url\..*\.insteadof' 2>/dev/null | head -1)
  if [ -n "$rewrite" ] && \
     timeout 25 env GIT_CONFIG_GLOBAL=/dev/null git ls-remote "$REPO" HEAD >/dev/null 2>&1; then
    echo "  note: a global git url rewrite is broken on this box, bypassing it"
    echo "        ($rewrite)"
    GIT="env GIT_CONFIG_GLOBAL=/dev/null git"
  else
    echo "cannot reach $REPO from this box." >&2
    [ -n "$rewrite" ] && echo "  a global rewrite is active: $rewrite" >&2
    exit 1
  fi
fi

if [ ! -d "$PDFSYS" ]; then
  $GIT clone "$REPO" "$PDFSYS"
fi
cd "$PDFSYS"
$GIT pull --ff-only
$GIT submodule update --init --recursive --depth 1

uv sync --locked

# models/ is gitignored — the XGBoost weights are FinePDFs' to distribute, so
# every fresh checkout starts without them. The run refuses to start rather
# than routing the whole corpus to `deferred` and exiting 0, so this is not
# optional.
uv run python -m pdfsys_router.download_weights

# Eight tiny PDFs through all four phases against in-process stubs: no GPU,
# no network, no model weights beyond the router's.
uv run pdfsys smoke

# And against the real services, which is what actually gets used. Skipped
# when they are not reachable from this box — the GPU box may be the only one
# that can see them, and that is not an error on the CPU box.
if curl -s -m 8 --noproxy '*' "$MINERU_URL/health" >/dev/null 2>&1; then
  uv run pdfsys smoke --mineru-url "$MINERU_URL" --quality-url "$QUALITY_URL_"
else
  echo "  note: $MINERU_URL not reachable from here; skipped the live smoke."
  echo "        preflight.sh checks the services from the box that needs them."
fi
