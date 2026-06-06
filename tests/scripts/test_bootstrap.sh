#!/usr/bin/env bash
# Smoke test for scripts/bootstrap.sh — gated by CI=1.
#
# Verifies: bootstrap.sh runs without error in a fresh clone of the
# current worktree (using a tmp dir + git clone --local).
#
# Skipped by default in dev — CI runners set CI=1 to enable.

set -euo pipefail

if [ "${CI:-}" != "1" ]; then
    echo "[test] skipped (CI=1 not set)"
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[test] cloning $REPO_ROOT to $TMP_DIR/clone..."
git clone --local "$REPO_ROOT" "$TMP_DIR/clone"
cd "$TMP_DIR/clone"

echo "[test] running scripts/bootstrap.sh..."
bash scripts/bootstrap.sh

echo "[test] PASS"
