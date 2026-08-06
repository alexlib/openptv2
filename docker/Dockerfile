# OpenPTV2 single-shot image: desktop GUI (PySide6/chaco) + headless batch.
# GUI renders on the host via X11; mount your data at /data.
#
#   Build:  docker build -t openptv2 .
#   GUI:    ./docker/run-gui.sh                 (opens the GUI, X11 -> host)
#   Batch:  docker run --rm -v "$PWD:/data" openptv2 \
#             openptv2-batch /data/<exp>/parameters_Run1.yaml <first> <last>
#
# A trimmed copy of test_data/test_cavity is baked in at /demo/test_cavity so
# the very first run works with no data mounted.
FROM python:3.12-slim

# --- system deps -----------------------------------------------------------
# build-essential: compile the Cython algorithms/ extensions.
# The libxcb*/xkb/GL/fontconfig set is what Qt6 (PySide6) needs to open a
# window over X11 — without these the GUI aborts with "xcb plugin" errors.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 libegl1 libglib2.0-0 libdbus-1-3 libfontconfig1 \
    libxkbcommon0 libxkbcommon-x11-0 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
    libxcb-util1 libxcb-xkb1 libxrender1 libxi6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# --- python package --------------------------------------------------------
# [gui] = the default headless/batch runtime + the desktop GUI stack, so this
# one image serves both GUI and batch. Build deps come from pyproject
# build-system. For a lean GIL-free cloud batch image, see Dockerfile.cloud.
RUN pip install --no-cache-dir -e ".[gui]"

# --- baked first-run demo --------------------------------------------------
RUN mkdir -p /demo && cp -r /app/test_data/test_cavity /demo/test_cavity

# Qt/X11 defaults; DISPLAY is supplied at `docker run` time.
ENV QT_X11_NO_MITSHM=1 \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
WORKDIR /data

CMD ["openptv2-gui"]
