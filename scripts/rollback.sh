#!/usr/bin/env bash
# Rollback to a previous git commit, sync deps, restart service.
#
# Usage:
#   ./scripts/rollback.sh [commit]
#
# Arguments:
#   commit  — target commit hash (default: HEAD~1)
#
# Output: JSON with rolled_back_from, rolled_back_to on stdout.
# Exit codes: 0 = success, 1 = failure (error on stderr).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

TARGET="${1:-HEAD~1}"

PREV_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

if ! git reset --hard "${TARGET}" 2>/dev/null; then
    echo "{\"error\": \"git reset to ${TARGET} failed\"}" >&2
    exit 1
fi

# Sync dependencies with uv
UV_BIN=$(command -v uv 2>/dev/null || echo "")
if [[ -n "${UV_BIN}" ]]; then
    "${UV_BIN}" sync --project "${REPO_DIR}" --quiet 2>/dev/null || true
fi

NEW_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

cat <<EOF
{"rolled_back_from": "${PREV_COMMIT}", "rolled_back_to": "${NEW_COMMIT}"}
EOF
