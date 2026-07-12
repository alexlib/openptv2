#!/usr/bin/env bash
# Launch the OpenPTV2 GUI from the Docker image, rendering on the host's X
# display. Linux/X11 only.
#
#   ./docker/run-gui.sh                 # mount $PWD at /data, open the GUI
#   ./docker/run-gui.sh /path/to/data   # mount a specific folder at /data
#   ./docker/run-gui.sh /path/to/data openptv2-batch /data/exp/parameters.yaml 10001 10004
#
# The baked demo lives at /demo/test_cavity inside the container, so on a first
# run you can open /demo/test_cavity even with nothing mounted.
set -euo pipefail

DATA_DIR="${1:-$PWD}"
IMAGE="${OPENPTV2_IMAGE:-openptv2}"

if [ -z "${DISPLAY:-}" ]; then
  echo "DISPLAY is not set — the GUI needs an X server on the host." >&2
  exit 1
fi

# Let the container's X client reach the host X server, then revoke on exit.
xhost +local:root >/dev/null 2>&1 || true
trap 'xhost -local:root >/dev/null 2>&1 || true' EXIT

docker run --rm -it \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$DATA_DIR:/data" \
  "$IMAGE" "${@:2}"
