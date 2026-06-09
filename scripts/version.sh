#!/usr/bin/env bash
# Report current git version info.
#
# Usage:
#   ./scripts/version.sh
#
# Output: JSON with commit, branch, last_commit_message, dirty on stdout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
MESSAGE=$(git log -1 --format='%s' 2>/dev/null || echo "")
DATE=$(git log -1 --format='%ci' 2>/dev/null || echo "")
DIRTY=$(git status --porcelain 2>/dev/null)

if [[ -n "${DIRTY}" ]]; then
    DIRTY_FLAG="true"
else
    DIRTY_FLAG="false"
fi

cat <<EOF
{"commit": "${COMMIT}", "branch": "${BRANCH}", "last_commit_message": "${MESSAGE}", "last_commit_date": "${DATE}", "dirty": ${DIRTY_FLAG}}
EOF
