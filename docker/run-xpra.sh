#!/usr/bin/env bash
# Launch the OpenPTV2 GUI in a browser via Xpra — no X server on the host
# needed, unlike run-gui.sh. For native Windows (PowerShell, not WSL/Git
# Bash), use run-xpra.ps1 instead — same behavior, Windows path syntax.
#
#   ./docker/run-xpra.sh                     # mount $PWD at /data, port 9876
#   ./docker/run-xpra.sh /path/to/data        # mount a specific folder
#   OPENPTV2_XPRA_PASSWORD=secret ./docker/run-xpra.sh   # password-protect it
#
# Then open http://localhost:9876 in any browser. The baked demo lives at
# /demo/test_cavity inside the container, copied to /data automatically if
# nothing is mounted there, so the very first run has something to open.
#
# No OPENPTV2_XPRA_PASSWORD set? Binds to 127.0.0.1 only (safe for local
# use) and runs with no xpra auth. Do NOT change PORT_BIND below to 0.0.0.0
# without also setting a password — see docs/browser-gui.md.
#
# Mounts a single folder as /data, same as run-gui.sh — pass a common
# parent (e.g. your whole experiments folder) rather than one experiment's
# folder if you want to switch between experiments from pyptv_gui's file
# picker without restarting the container. Docker needs an *absolute* host
# path for a bind mount, so a relative arg is resolved below; that
# resolution needs the directory to already exist.
#
# Platform notes (see docs/browser-gui.md "Cross-platform local filesystem
# access" for the full version):
#   - macOS: the folder must be under Docker Desktop's allowed file-sharing
#     paths (Settings -> Resources -> File Sharing) or the mount silently
#     fails.
#   - Linux: files created in the container appear on the host owned by
#     UID 1000 (the container's non-root user); harmless, but `sudo chown`
#     if you need another user to edit them directly.
#   - Windows: run this script from WSL2 or Git Bash; native PowerShell
#     users should use run-xpra.ps1 instead.
set -euo pipefail

RAW_DIR="${1:-$PWD}"
if [ ! -d "$RAW_DIR" ]; then
  echo "Not a directory: $RAW_DIR" >&2
  exit 1
fi
# Portable absolute-path resolution (works on macOS's bash 3.2 / no GNU
# coreutils too, unlike `realpath`/`readlink -f`).
DATA_DIR="$(cd "$RAW_DIR" && pwd)"
IMAGE="${OPENPTV2_IMAGE:-openptv2-xpra}"
PORT_BIND="127.0.0.1:9876:9876"

ENV_ARGS=(-e QT_X11_NO_MITSHM=1)
if [ -n "${OPENPTV2_XPRA_PASSWORD:-}" ]; then
  ENV_ARGS+=(-e "OPENPTV2_XPRA_PASSWORD=${OPENPTV2_XPRA_PASSWORD}")
else
  ENV_ARGS+=(-e OPENPTV2_XPRA_ALLOW_INSECURE=1)
  echo "No OPENPTV2_XPRA_PASSWORD set — running with no auth, bound to" >&2
  echo "127.0.0.1 only. Set OPENPTV2_XPRA_PASSWORD to add a login prompt." >&2
fi

TTY_ARGS=(-it)
if [ ! -t 0 ]; then
  TTY_ARGS=(-i)
fi

docker run --rm "${TTY_ARGS[@]}" \
  -p "$PORT_BIND" \
  "${ENV_ARGS[@]}" \
  -v "$DATA_DIR:/data" \
  "$IMAGE" "${@:2}"
