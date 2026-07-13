# Tutorial: Sequence & Tracking Plugins

openptv2's detection/correspondence step ("sequence") and its tracking step
each run through a **plugin**: a small class that implements one method.
The core algorithm is itself a plugin named `default` — there is no separate
code path for "the real pipeline" versus "a plugin". Selecting a different
sequence or tracking plugin swaps out the processing strategy while
everything else (parameters, calibration, output files) stays the same.

This is useful for dataset-specific preprocessing — an image-splitter
camera, background removal, contour masking — without forking the core
pipeline.

---

## Built-in plugins

Shipped in `src/openptv2/plugins/`:

| Name | Kind | What it does |
| --- | --- | --- |
| `default` | sequence + tracking | The core pipeline. Always available, cannot be shadowed. |
| `splitter_sequence` / `splitter_tracking` | sequence + tracking | Four-view image-splitter cameras: one physical sensor split into per-view images before detection. See the [4-view splitter tutorial](four_view_splitter/README.md). |
| `contour_sequence` | sequence | Masks each frame to its largest smooth contour before detection. |
| `rembg_sequence` | sequence | Removes the background with [rembg](https://github.com/danielgatis/rembg) before detection. Requires `pip install openptv2[rembg]`. |
| `rembg_contour_sequence` | sequence | Like `rembg_sequence`, but also tracks the mask area per frame and writes `res/mask_areas.csv`. Requires `openptv2[rembg]`. |

## Selecting a plugin

**Batch / cloud** (`openptv2-batch`):

```bash
openptv2-batch parameters_Run1.yaml 1000001 1000002 \
  --sequence-plugin splitter_sequence --tracking-plugin splitter_tracking
```

Both flags default to `default`. See
[Command-Line Batch Processing](batch_processing.md) for the full CLI, and
[Cloud-like batch deployment](../cloud-batch.md) for the Docker image (which
ships every built-in plugin — no extra data-folder setup needed).

**GUI** (`openptv2-gui`): open the experiment, then **Plugins → Select
plugin**. The dialog always shows the full, current list — built-ins plus
anything in the experiment's `plugins/` folder — rescanned fresh each time
it opens, so a file you just dropped into `plugins/` shows up immediately.
Your choice is saved into the experiment's YAML `plugins:` section the next
time you save parameters.

**YAML** (read by both GUI and batch): the `plugins:` section records the
current selection —

```yaml
plugins:
  selected_sequence: splitter_sequence
  selected_tracking: splitter_tracking
```

`available_sequence`/`available_tracking` in that same section are
informational (populated by the last scan) — the loader itself doesn't read
them; edit `selected_sequence`/`selected_tracking` directly if hand-editing
YAML.

---

## Writing your own plugin

A plugin module defines a `Sequence` class, a `Tracking` class, or both:

```python
class Sequence:
    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv    # the openptv2.gui.ptv module, injected by the loader
        self.exp = exp    # the active experiment (has .cpar, .spar, .vpar, .tpar, .cals, ...)

    def do_sequence(self) -> None:
        ...  # detect targets, write res/rt_is.<frame>


class Tracking:
    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        ...  # link targets across frames, write res/ptv_is.<frame>
```

Both classes are instantiated as `Cls(ptv=<module>, exp=<experiment>)` and
called with no arguments — never construct `ptv` yourself. See
`src/openptv2/plugins/default_sequence.py` and `default_tracking.py` for the
minimal reference implementation, or `splitter_sequence.py` for a fuller
example (masking, per-camera image splitting, error handling).

Plugins are deliberately **pure Python, not Cython** — they're I/O-and-glue
code around the compiled `algorithms/` kernels, and users need to be able to
read and drop in a `.py` file without a build step.

### Where to put it

Three ways a plugin becomes resolvable, in this order:

1. **Built-in** — add a module under `src/openptv2/plugins/` and register it
   in `BUILTIN_SEQUENCE_PLUGINS`/`BUILTIN_TRACKING_PLUGINS` in
   `src/openptv2/plugins/loader.py`. For plugins meant to ship with
   openptv2 itself.
2. **Third-party package** — register an
   [entry point](https://packaging.python.org/en/latest/specifications/entry-points/)
   in the `openptv2.plugins` group, mapping a plugin name to your module.
   For a plugin distributed as its own installable package.
3. **Experiment-local** — drop a `.py` file into a `plugins/` folder next to
   the experiment's YAML file, named after the plugin (e.g.
   `plugins/my_sequence.py` for a plugin selected as `my_sequence`). No
   packaging needed — the loader imports it directly. This is the right
   place for a one-off, dataset-specific script; it's tried last, so it
   can't accidentally override a built-in of the same name.

`default` is always a built-in, so it can never be shadowed by an
experiment-local file — selecting `"default"` always runs the core
pipeline.

### Failure handling

If a selected plugin can't be resolved (unknown name) or fails to import
(e.g. `rembg_sequence` without the `rembg` extra installed), the loader
raises `openptv2.plugins.PluginError` (or the underlying `ImportError`).
Batch propagates this as a non-zero exit; the GUI shows it in a dialog
instead of crashing.
