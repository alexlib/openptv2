#!/usr/bin/env bash
# Launch the OpenPTV2 GUI in a browser via Xpra — no X server on the host
# needed, unlike run-gui.sh. Works the same on Linux/macOS/Windows (WSL).
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
set -euo pipefail

DATA_DIR="${1:-$PWD}"
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

docker run --rm -it \
  -p "$PORT_BIND" \
  "${ENV_ARGS[@]}" \
  -v "$DATA_DIR:/data" \
  "$IMAGE"
