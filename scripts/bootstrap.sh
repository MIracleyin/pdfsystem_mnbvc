#!/usr/bin/env bash
# Initialize a freshly-cloned pdfsys checkout.
#
# Idempotent — safe to re-run. Steps:
#   1. git submodule update --init --recursive
#   2. sanity-check external/parsers/packages exists (unless still in-tree)
#   3. uv sync
#   4. uv run pdfsys release verify
#
# Usage:
#   git clone --recurse-submodules https://github.com/MIracleyin/pdfsystem_mnbvc.git
#   cd pdfsystem_mnbvc
#   bash scripts/bootstrap.sh
#
# Exit codes:
#   0 — environment ready
#   1 — bootstrap failed (one of the steps above produced an error)

set -euo pipefail

# Resolve repo root from script location (works whether invoked directly
# or sourced, from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

cd "$REPO_ROOT"

# Step 1: initialize submodules
echo "[bootstrap] git submodule update --init --recursive..."
git submodule update --init --recursive

# Step 2: sanity-check the parsers submodule landed.
#
# If system_release.toml still uses `repo = "in-tree:"` for ALL components
# that would live under external/parsers, the submodule has not been extracted
# yet (Phase 2 pending). In that case the missing directory is expected — exit 0.
#
# If any component points at a real repo URL, the missing path is a hard failure.
_check_parsers_dir() {
    if [ -d "external/parsers/packages" ]; then
        return 0  # present — all good
    fi

    # Directory is missing. Check whether we're still pre-extraction.
    local all_intree=1
    if [ -f "system_release.toml" ]; then
        # Look for any repo line that is NOT an in-tree: pin.
        # A real submodule URL would be http(s):// or git@.
        if grep -qE '^repo\s*=\s*"(https?://|git@)' system_release.toml 2>/dev/null; then
            all_intree=0
        fi
    fi

    if [ "$all_intree" -eq 1 ]; then
        echo "[bootstrap] external/parsers/packages not found — system_release.toml" >&2
        echo "[bootstrap] still has in-tree pins; submodule extraction (Phase 2)" >&2
        echo "[bootstrap] has not been done yet. Continuing." >&2
        return 0  # pre-extraction state is OK
    else
        echo "error: external/parsers/packages not found." >&2
        echo "  system_release.toml references real repo URLs, so the parsers" >&2
        echo "  submodule should have been cloned but is missing." >&2
        echo "  Try: git submodule update --init --recursive" >&2
        return 1
    fi
}

_check_parsers_dir

# Step 3: sync uv workspace
echo "[bootstrap] uv sync..."
uv sync

# Step 4: verify release manifest matches submodule HEADs
echo "[bootstrap] uv run pdfsys release verify..."
uv run pdfsys release verify

echo "[bootstrap] ready"
