#!/bin/bash
# Starts xpra with pyptv_gui as its child process, HTML5 client on :9876.
#
# Pattern taken from CyVerse VICE's app entrypoints (orange/entry.sh,
# cellprofiler-desktop/entry.sh - github.com/cyverse-vice/xpra), which serve
# scientific desktop GUI apps to browsers the same way. The load-bearing
# flags:
#
#   --start-child=<cmd>       the app to launch inside the virtual display
#   --exit-with-children=no   keep xpra (and the container) alive if the
#                              app window is closed, so it can be relaunched
#                              from a terminal without the container dying
#   --html=on                 serve the HTML5 client, no separate webserver
#   --daemon=no                run in the foreground (required for `docker
#                              run` to track the process / for signals to
#                              reach it)
set -euo pipefail

# --- auth ------------------------------------------------------------------
# xpra has NO authentication by default. CyVerse VICE gets away without
# setting any here because their own platform puts an authenticated gateway
# in front of every session; this image has no such gateway built in, so it
# must not be exposed with no auth of its own. Set OPENPTV2_XPRA_PASSWORD to
# require a password in the HTML5 login page. Running with neither that env
# var nor OPENPTV2_XPRA_ALLOW_INSECURE=1 refuses to start, so an insecure
# deployment has to be requested explicitly rather than happen by accident.
# See docs/browser-gui.md for the recommended GCP setup (Cloud Run + IAP,
# which handles auth in front of this instead).
AUTH_ARGS=()
if [ -n "${OPENPTV2_XPRA_PASSWORD:-}" ]; then
    AUTH_ARGS=(--tcp-auth=password:value="${OPENPTV2_XPRA_PASSWORD}" --auth=password:value="${OPENPTV2_XPRA_PASSWORD}")
elif [ "${OPENPTV2_XPRA_ALLOW_INSECURE:-0}" = "1" ]; then
    echo "WARNING: starting with no xpra authentication (OPENPTV2_XPRA_ALLOW_INSECURE=1)." >&2
    echo "         Anyone who can reach this port can open the GUI. Fine for" >&2
    echo "         127.0.0.1-only local testing; not fine on a public IP." >&2
else
    cat >&2 <<'EOF'
Refusing to start: no xpra authentication configured.

Set one of:
  OPENPTV2_XPRA_PASSWORD=<a password>   require it in the HTML5 login page
  OPENPTV2_XPRA_ALLOW_INSECURE=1        explicitly opt into no auth
                                        (only for 127.0.0.1-only local use)

See docs/browser-gui.md - on GCP, prefer Cloud Run + Identity-Aware Proxy
over relying on this password.
EOF
    exit 1
fi

# --- data ---------------------------------------------------------------
# If a working directory with an experiment isn't mounted at /data, default
# to the baked demo so the very first run has something to open.
if [ -z "$(ls -A /data 2>/dev/null)" ]; then
    cp -r /demo/test_cavity/* /data/ 2>/dev/null || true
fi

# --- permissions check ----------------------------------------------------
chmod 700 "${XDG_RUNTIME_DIR:-/run/user/1000}" "${XDG_RUNTIME_DIR:-/run/user/1000}/xpra" 2>/dev/null || true

# --- child command --------------------------------------------------------
APP_CMD="openptv2-gui"
if [ $# -gt 0 ]; then
    APP_CMD="$*"
fi

exec xpra start \
    --bind-tcp=0.0.0.0:9876 \
    --html=on \
    "${AUTH_ARGS[@]}" \
    --start="${APP_CMD}" \
    --exit-with-children=no \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 8192x4096x24+32 -nolisten tcp -noreset" \
    --audio=no \
    --pulseaudio=no \
    --mdns=no \
    --webcam=no \
    --ssh=no \
    --printing=no \
    --notifications=no \
    --bell=no \
    :100
