# Cloud-like batch deployment (headless, no GUI)

For running the tracking/sequence pipeline on a server or in a container —
fast to deploy, no Qt, no display. The engine is the compiled Cython 3 runtime
with OpenMP kernels; that is where the performance comes from.

## Install (one command)

The cloud profile is the **default** install (no GUI extras). Use a
non-editable install so the compiled kernels land in `site-packages` and the
source tree is not needed at runtime:

```bash
uv venv --python 3.14t /opt/venv          # or 3.11–3.13; 3.14t = free-threaded
VIRTUAL_ENV=/opt/venv uv pip install .
```

This pulls **18 packages** (numpy, scipy, scikit-image, pydantic, pyyaml,
imageio, traits, cython + deps) and compiles the extensions in ~7 s. No Qt,
chaco, PySide6, or matplotlib. Verify:

```bash
/opt/venv/bin/python -c "import openptv2; print(openptv2.get_runtime_info())"
# {"engine": "cython3-pure-python", "compiled": true, "package": "openptv2"}
```

## Run a batch

`openptv2-batch` takes a YAML parameter file (frames default to its `sequence`
section) or an experiment directory:

```bash
openptv2-batch /data/exp1/parameters_Run1.yaml            # frames from YAML
openptv2-batch /data/exp1/parameters_Run1.yaml 10001 10004  # explicit range
openptv2-batch /data/exp1 --mode both                     # dir; both=seq+track
```

`--mode {both,sequence,tracking}` selects the steps; `--track3d` uses 3D segment
tracking. Output (`ptv_is.*`, `rt_is.*`, `added.*`) is written to the
experiment's `res/` directory. The run is verbose by default — per-frame
correspondences and per-step tracking links/lost are printed to stdout.

With no plugin flags, the runner uses the `plugins.selected_*` selection
saved in the experiment YAML by the GUI — the YAML alone fully describes
the run, so a GUI-tuned splitter experiment needs nothing extra:

```bash
openptv2-batch /data/exp1/parameters_Run1.yaml 1000001 1000002
```

`--sequence-plugin`/`--tracking-plugin` override that selection — for
example, an image-splitter dataset that multiplexes four views onto one
sensor:

```bash
openptv2-batch /data/exp1/parameters_Run1.yaml 1000001 1000002 \
  --sequence-plugin splitter_sequence --tracking-plugin splitter_tracking
```

In splitter mode the multiplexed frame is split into per-camera views in
memory (nothing intermediate is written to disk) and detection + stereo
matching run in the same process — this also holds per worker in the
parallel runner (`python -m openptv2.batch.pyptv_batch_parallel <yaml>
<first> <last> <n_processes>`), which splits the frame range into chunks.

`default` and every other plugin name go through the same
`openptv2.plugins` loader — see `src/openptv2/plugins/` for the built-ins
(`splitter_sequence`/`splitter_tracking`, `contour_sequence`,
`rembg_sequence`, `rembg_contour_sequence`) and how to add your own.

## Free-threading (3.14t): measured, not assumed

The `test_cavity` demo (4 frames, sequence + tracking) on free-threaded
3.14.3, compiled Cython + OpenMP:

| Mode | Wall clock | CPU | Tracking result |
|------|-----------|-----|-----------------|
| GIL **on** (default) | **4.6 s** | 127% | particles 1195.0, links 772.3, lost 422.7 |
| GIL **off** (`PYTHON_GIL=0`) | 5.9 s | 200% | *identical* |

**Free-threading does not speed up this batch — it is ~25% slower.** The
kernels are OpenMP-parallel Cython driven by a single Python process; there is
no Python-level thread pool for a disabled GIL to unlock, so `PYTHON_GIL=0`
only adds interpreter overhead. Two consequences:

- The Cython modules **re-enable the GIL on import** (they don't declare
  `Py_mod_gil` free-thread safety), printing a `RuntimeWarning`. That is
  expected; the batch runs correctly.
- To scale across cores, run **independent experiments/frame-ranges as separate
  processes** (multiprocessing / one container task per range). Separate
  processes don't share a GIL, so free-threading buys nothing there either.

Use the standard interpreter (or 3.14t at its default GIL-on behavior). Only
force `PYTHON_GIL=0` to experiment; revisit if the algorithms modules ever
declare free-threading safety.

## Docker

`Dockerfile.cloud` bakes exactly the install above (free-threaded 3.14t,
`uv pip install .`, no GUI) plus two demos: `test_cavity` at
`/demo/test_cavity` (default pipeline) and `test_splitter` at
`/demo/test_splitter` (exercises the built-in splitter plugins, proving
plugin support ships in this image with no extra data):

```bash
docker build -f Dockerfile.cloud -t openptv2-cloud .

# zero-data smoke test on the baked demo (default pipeline):
docker run --rm openptv2-cloud sh -c \
  "cp -r /demo/test_cavity /tmp/c && cd /tmp/c && \
   openptv2-batch parameters_Run1.yaml 10001 10004"

# zero-data smoke test of a non-default plugin (image-splitter dataset):
docker run --rm openptv2-cloud sh -c \
  "cp -r /demo/test_splitter /tmp/s && cd /tmp/s && \
   openptv2-batch parameters_Run1.yaml 1000001 1000002 \
   --sequence-plugin splitter_sequence --tracking-plugin splitter_tracking"

# your own data (mounted at /data):
docker run --rm -v "$PWD:/data" openptv2-cloud \
  openptv2-batch /data/<exp>/parameters_Run1.yaml <first> <last>
```

The image leaves the GIL enabled by default. Add `-e PYTHON_GIL=0` to force it
off (see the table above first).
