#!/usr/bin/env bash
# Install / refresh AI Parking BE systemd services for this device.
#
# Renders unit templates from ./systemd/ with device-specific values
# (install dir, run user, uv path) and installs them into
# /etc/systemd/system/.
#
# Usage:
#   sudo ./install.sh
#
# Overrides:
#   RUN_USER=<user>   service runtime user (defaults to invoking user)
#   UV_BIN=<path>     path to the uv binary (defaults to autodetect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}"
SYSTEMD_DIR="/etc/systemd/system"
TEMPLATE_DIR="${SCRIPT_DIR}/systemd"

if [[ "$EUID" -ne 0 ]]; then
    echo "This script must be run as root (use sudo)." >&2
    exit 1
fi

RUN_USER="${RUN_USER:-${SUDO_USER:-}}"
if [[ -z "${RUN_USER}" || "${RUN_USER}" == "root" ]]; then
    echo "Cannot determine a non-root run user. Invoke via 'sudo ./install.sh' from your user shell, or set RUN_USER=<user>." >&2
    exit 1
fi
RUN_GROUP="$(id -gn "${RUN_USER}")"

UV_BIN="${UV_BIN:-$(sudo -u "${RUN_USER}" bash -lc 'command -v uv || true')}"
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
    echo "uv not found for user ${RUN_USER}. Install it (https://docs.astral.sh/uv/) or set UV_BIN=<path>." >&2
    exit 1
fi

echo ">> install_dir = ${INSTALL_DIR}"
echo ">> run_user    = ${RUN_USER} (${RUN_GROUP})"
echo ">> uv          = ${UV_BIN}"

echo ">> uv sync (idempotent)"
sudo -u "${RUN_USER}" "${UV_BIN}" sync --project "${INSTALL_DIR}"

render() {
    local src="$1" dst="$2"
    sed \
        -e "s|__INSTALL_DIR__|${INSTALL_DIR}|g" \
        -e "s|__RUN_USER__|${RUN_USER}|g" \
        -e "s|__RUN_GROUP__|${RUN_GROUP}|g" \
        -e "s|__UV__|${UV_BIN}|g" \
        "${src}" > "${dst}"
    chmod 0644 "${dst}"
}

echo ">> rendering units to ${SYSTEMD_DIR}"
render "${TEMPLATE_DIR}/ai-parking-be.service"     "${SYSTEMD_DIR}/ai-parking-be.service"
render "${TEMPLATE_DIR}/ai-parking-client.service" "${SYSTEMD_DIR}/ai-parking-client.service"

systemctl daemon-reload

# ai-parking-be: long-running, enable on boot + (re)start now.
systemctl enable ai-parking-be.service
systemctl restart ai-parking-be.service

# ai-parking-client: oneshot reboot helper, install only — do not enable.
systemctl --quiet is-enabled ai-parking-client.service && systemctl disable ai-parking-client.service || true

sleep 2
systemctl status --no-pager ai-parking-be.service | head -12
echo ">> done"
