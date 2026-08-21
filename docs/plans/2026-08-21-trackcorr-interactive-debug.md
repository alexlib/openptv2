# Interactive trackcorr candidate/parameter debugging

Branch: `feature/trackcorr-interactive-debug`

## Goal

Let a user tune trackcorr's tracking parameters (`dvxmin/max`, `dvymin/max`,
`dvzmin/max`, `dacc`, `dangle`) by eye: pick a tracer particle in a frame,
see the real search volume trackcorr computes for it, every candidate it
considered in the next frame (with tracer IDs), which one it picked (and
why others lost), and how that changes live as parameters are adjusted.

Scope: **trackcorr only** (`track_mode=0`, `Tracker.step_forward()` /
`trackcorr_c_loop`). Not track3d/priority_segment_3d or any other tracker —
those have a different candidate-search structure and are out of scope
here.

Delivery: a marimo notebook (matplotlib overlays + `mo.ui` sliders), not
the Tkinter/Enable GUI — matches the existing `visualize_calibration_nb.py`
pattern, and live-updating overlays are far cheaper to build against
matplotlib than against Chaco/Enable's plot objects.

## What already exists (and doesn't work)

`src/openptv2/gui/tracking_viz_panel.py`'s `TrackingDebugPanel` +
`docs/tutorials/tracking_debug_visualization.md` describe exactly this
feature (click a particle, see search volumes/candidates/epipolar lines),
wired into the main GUI's "Tracking → Debugging with display" menu. It does
not work:

- `_visualize_click()` calls `self._compute_search_volumes(pos_3d,
  velocity)` — **this method is not defined anywhere in the codebase.**
  Clicking a particle raises `AttributeError` immediately.
- `_triangulate_target()` (meant to get a candidate's 3D position) computes
  `flat_x`/`flat_y` and discards them, unconditionally returning a
  hardcoded `(0.0, 0.0, 100.0)`.
- The candidate-accept check compares a 3D distance (mm) against `dvxmax`
  (mm/frame, a velocity bound) — a unit mismatch even setting the above
  aside.
- `openptv2.algorithms.tracking_frame_buf.Pathinfo.decis`/`.linkdecis` (a
  per-candidate cost/target-index array, `register_link_candidate()`) looks
  purpose-built to record exactly the candidate list we want, but
  `register_link_candidate()` is **never called** anywhere in the
  codebase — dead instrumentation, not populated by a real run.

Conclusion: build the candidate reconstruction fresh, do not try to reuse
`TrackingDebugPanel`'s logic. The one thing that *is* real and worth
reusing: `openptv2.tracker.Tracker` (`step_forward()` → `trackcorr_c_loop`)
for actually stepping real tracking data forward.

## Design

### Backend (plain Python, testable independent of the notebook)

New module, e.g. `src/openptv2/gui/trackcorr_debug.py`:

1. **Load an experiment**: parameters YAML → `cpar`/`spar`/`vpar`/`tpar` +
   calibrations (reuse `openptv2.gui.ptv.py_start_proc_c` or
   `autocalibration`'s loaders — do not reinvent parameter loading).
2. **Step trackcorr forward** over a frame range using the real `Tracker`,
   same as `TrackingDebugPanel._initialize_tracker` already does correctly
   (just drop the `track_mode == 1` branch — trackcorr only).
3. **Reconstruct one particle's real candidate search**, for a given frame
   `t` and particle index, by calling `trackcorr_c_loop`'s own constituent
   functions from `src/openptv2/algorithms/track.py` (all public, already
   individually unit-tested in `tests/unit/test_track.py`) directly:
   - `predict()` / `search_volume_center_moving()` — the predicted position
     trackcorr searches around.
   - `searchquader()` — the real 3D search box, then per-camera projected
     bounds (this *is* the "search volume" the broken panel tried to draw).
   - `candsearch_in_pix()` / `candsearch_in_pix_rest()` — the actual
     per-camera 2D candidate targets within the projected box, by tracer ID
     (`pnr`/`tnr`), not an approximation.
   - `angle_acc()` / `pos3d_in_bounds()` — the acceleration/angle gate that
     accepts or rejects each 3D candidate — this is what actually
     determines trackcorr's winner, not raw distance.
   - `sort_candidates_by_freq()` — cross-camera consistency ranking, same
     as the real loop.

   Output: a plain data structure (dataclass or dict) — predicted position,
   per-camera search box corners, per-camera candidate list (tracer ID,
   pixel position, in/out of box), the ranked 3D candidates with their
   angle/acc costs, and the winner's tracer ID — everything the overlay and
   a text report need, with no drawing-backend coupling.
4. Parameters (`dvxmin/max`, ..., `dacc`, `dangle`) are plain function
   arguments to step 3, not mutated tracker state — recomputing for new
   parameter values must not require re-running the whole tracked sequence,
   only the search+gate for the one particle. This is what makes live
   parameter tuning cheap.

### Notebook (`src/openptv2/gui/trackcorr_debug_nb.py`)

- `mo.cli_args()` for experiment path + frame range, editable fallback UI
  like `visualize_calibration_nb.py`.
- Frame slider, tracer-ID dropdown (populated from that frame's real
  particles) or click-to-pick if marimo's mpl click events are usable
  in-app mode.
- Parameter sliders (`dvxmin/max`, `dvymin/max`, `dvzmin/max`, `dacc`,
  `dangle`) wired to recompute step 3 above on change.
- One matplotlib subplot per camera: raw detections as small dots, the
  clicked tracer highlighted, the projected search box outline, candidates
  colored by accepted/rejected-and-why (out of box vs. failed angle/acc
  gate), the winning candidate labeled with its tracer ID.
- A text/table panel: predicted position, per-candidate costs, which one
  won and why the others lost — the numeric detail a plot can't show
  precisely.

## Phased build

1. Backend module + a couple of direct unit tests against a small fixture
   (e.g. `test_data/track` or `test_data/synthetic`, whichever already has
   trackcorr-friendly frame-to-frame data) confirming the reconstructed
   candidate list/winner matches what a real `trackcorr_c_loop` step
   actually links — this is the correctness bar, not just "it runs."
2. Static notebook: load one experiment, step trackcorr, plot one
   frame/particle's search box + candidates non-interactively.
3. Add the interactive controls (frame/tracer pick, parameter sliders).
4. Polish: text report panel, `--extra viz` wiring / run instructions
   matching `visualize_calibration_nb.py`'s doc header conventions, a
   `docs/tutorials/` page once it's real.

## Explicitly out of scope for this branch

- track3d / priority_segment_3d / any tracker other than trackcorr.
- Fixing or removing the broken `TrackingDebugPanel` (separate cleanup;
  note it here so it isn't mistaken for prior art again).
- Wiring this into the Tkinter/Enable GUI.
