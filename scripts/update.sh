#!/usr/bin/env bash
# Git-pull latest code, sync deps, restart service.
#
# Usage:
#   ./scripts/update.sh [branch] [commit]
#
# Arguments:
#   branch  — git branch to pull (default: from .env GIT_BRANCH or "dev")
#   commit  — specific commit hash to checkout (optional)
#
# Output: JSON with previous_commit, new_commit, branch on stdout.
# Exit codes: 0 = success, 1 = failure (error on stderr).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

# Read defaults from .env if present
if [[ -f .env ]]; then
    GIT_REMOTE=$(grep -E '^GIT_REMOTE=' .env | cut -d= -f2 || echo "origin")
    GIT_BRANCH_DEFAULT=$(grep -E '^GIT_BRANCH=' .env | cut -d= -f2 || echo "dev")
fi
GIT_REMOTE="${GIT_REMOTE:-origin}"

BRANCH="${1:-${GIT_BRANCH_DEFAULT:-dev}}"
COMMIT="${2:-}"

# Save current commit for rollback reference
PREV_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Fetch latest
if ! git fetch "${GIT_REMOTE}" 2>/dev/null; then
    echo '{"error": "git fetch failed"}' >&2
    exit 1
fi

# Checkout specific commit or reset to remote branch head
if [[ -n "${COMMIT}" ]]; then
    if ! git checkout "${COMMIT}" 2>/dev/null; then
        echo "{\"error\": \"git checkout ${COMMIT} failed\"}" >&2
        exit 1
    fi
else
    if ! git reset --hard "${GIT_REMOTE}/${BRANCH}" 2>/dev/null; then
        echo "{\"error\": \"git reset to ${GIT_REMOTE}/${BRANCH} failed\"}" >&2
        exit 1
    fi
fi

# Sync dependencies with uv
UV_BIN=$(command -v uv 2>/dev/null || echo "")
if [[ -n "${UV_BIN}" ]]; then
    "${UV_BIN}" sync --project "${REPO_DIR}" --quiet 2>/dev/null || true
fi

NEW_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Output result as JSON
cat <<EOF
{"previous_commit": "${PREV_COMMIT}", "new_commit": "${NEW_COMMIT}", "branch": "${BRANCH}"}
EOF
