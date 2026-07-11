# Plan: migrate the GUI from TraitsUI/Chaco (Qt) to Tkinter/ttk + embedded matplotlib

## ⚖️ Feasibility verdict (read this first — it is the decision gate)

**Motivation (assumed, must be confirmed):** the current stack — `traits` + `traitsui.qt` + `chaco` + `enable` + `pyface` + Qt — lags free-threaded Python (3.14t/3.15). Tkinter ships with CPython and tracks the interpreter, so a Tk GUI would unblock free-threaded development. *(Verify before committing: does `chaco`/`enable` publish 3.14/3.14t wheels? If they already do, the whole migration may be unnecessary.)*

**Can every part of the present system be implemented in Tkinter? → YES, no hard blocker.**
Every current capability maps to a Tk equivalent, with **matplotlib embedded via `FigureCanvasTkAgg`** doing the heavy lifting for all image/plot/overlay/3D views (it is the standard Chaco replacement, and the 3D view is *already* matplotlib). See the component inventory below.

**Can I do it in "complete auto mode" as requested? → NO — not honestly.**
This is a **~13,700-LOC view-layer rewrite** with several components whose correctness is **visual/interactive and not machine-verifiable**:
- **Calibration point-picking** (`ClickerTool`): clicking an image must return the correct *image* pixel under zoom/pan/aspect. The coordinate transform is unit-testable, but "did the click land on the right dot" needs eyes.
- **Overlay alignment**: crosses, quiver residuals, epipolar lines must sit pixel-accurately on the image.
- **Live multi-camera updates** during detection/sequence/tracking.
- **3D trajectory interaction** (rotate/zoom).

The business logic (`ptv.py`, `tracker`, `algorithms/*`, `parameter_manager`, `parameter_models`) is already framework-free and tested, which shrinks the risk to the view layer — but the view layer's interactive fidelity to "**the same exact functionality**" cannot be self-certified.

**Recommendation given the user's gate ("complete auto mode, otherwise stay with the system I like"):**
Do **not** attempt a fully-autonomous, all-at-once rewrite. Two acceptable paths:
1. **Stay** with the present GUI until `chaco`/`enable` free-threaded wheels are confirmed unavailable *and* you are ready to invest human review time. (Lowest cost; matches "I like the present system".)
2. **Checkpointed migration** (this plan): a throwaway **pilot** proves the two hardest primitives under free-threaded Python; you visually approve it; only then proceed slice-by-slice with a **human visual review at each checkpoint**. Abandon cheaply if the pilot feels wrong. This is explicitly **not** complete-auto — it has ~4 human gates.

The rest of this document is path #2. If you require #1's "complete auto" bar, the answer is **stay**.

---

## Architecture decisions

- **Toolkit:** `tkinter` + `ttk` for structure/controls; **CustomTkinter** optional, applied last as a theme only (keep core on plain ttk so nothing depends on a third-party widget set).
- **All scientific views:** matplotlib `Figure` + `FigureCanvasTkAgg` + `NavigationToolbar2Tk`. Replaces Chaco `Plot`, `img_plot`, `quiverplot`, overlays, and the click tool (`fig.canvas.mpl_connect('button_press_event', …)`), with zoom/pan for free.
- **No business-logic changes.** The new GUI calls the exact same `ptv.py` / `Tracker` / `parameter_manager` APIs. This is the single most important constraint — it keeps the migration a view-layer swap.
- **Parameter forms:** a generic ttk form builder driven by the existing Pydantic models (`parameter_models.py`) / dataclasses, replacing the hand-authored TraitsUI `View`s.
- **Coexistence:** build under `src/openptv2/gui_tk/` with a new entry point (`openptv2-gui-tk`); the Traits GUI stays fully working until the Tk one reaches parity. No big-bang cutover.

## Component feasibility inventory (every current feature → Tk plan + how it's verified)

| Current (Traits/Chaco) | Tk replacement | Verification | Auto? |
|---|---|---|---|
| Main window, menus, buttons, tabs (`pyptv_gui`) | `ttk.Notebook`, `ttk.Button`, `Menu` | Xvfb: instantiate, invoke callbacks, assert `ptv.*` called | ✅ auto |
| Camera image view + zoom/pan (`plot.img_plot` + tools) | mpl `imshow` + `NavigationToolbar2Tk` | figure has 1 image artist; toolbar present | ✅ auto |
| **Calibration click-pick (`ClickerTool`)** | `mpl_connect('button_press_event')` → `event.xdata/ydata` | unit-test the px↔data transform; **human: click lands on dot** | ⚠️ **human** |
| Cross markers / detected points (`drawcross`) | mpl `scatter` | artist count == n points | ✅ auto |
| Residual quiver (`drawquiver`, `quiverplot.py`) | mpl `quiver` | artist present, N vectors | ✅ auto |
| Epipolar lines / text overlays (`text_box_overlay`) | mpl `plot` / `annotate` | artist counts | ✅ auto |
| **Overlay-on-image alignment** | shared mpl axes | **human: overlay sits on image** | ⚠️ **human** |
| 3D positions (`plot_3d_positions`, already mpl) | mpl `Axes3D` in `FigureCanvasTkAgg` | renders, N points; **human: rotate** | ⚠️ mostly auto |
| Parameter dialogs (`parameter_gui`, 13 `View`s) | generic ttk form from Pydantic schema | every model field has a widget; round-trip load/save equals YAML | ✅ auto |
| Code/text/file editors (`code_editor`, editors) | `tk.Text`/`ScrolledText`, `filedialog` | load→edit→save round-trip | ✅ auto |
| Detection / mask / dumbbell panels | ttk panels calling same `ptv.*` | callback wiring under Xvfb | ✅ auto |
| Live updates during sequence/tracking | `canvas.draw_idle()` after each frame | frames advance; **human: looks live** | ⚠️ human |
| Plugins, batch, experiment mgmt | unchanged (framework-free) | existing tests | ✅ auto |

**Net:** ~80% is machine-verifiable (wiring, forms, artist presence, round-trips under Xvfb); ~20% (interactive pick, overlay/px alignment, live-feel, 3D rotate) needs human eyes — concentrated in the calibration and live-view slices.

---

## Phase 0 — PILOT (throwaway, proves the risk before any commitment)

Goal: de-risk the two things that could kill the migration, under free-threaded Python.

### Task 0.1: Free-threaded reality check
**Acceptance:** document whether `chaco`/`enable` have 3.14/3.14t wheels (if yes → recommend NOT migrating) and that `tkinter` + `matplotlib` TkAgg import and render under a `python3.14t` build.
**Verify:** `python3.14t -c "import tkinter, matplotlib; matplotlib.use('TkAgg')"` in a container; note results. **Dependencies:** None. **Scope:** S. **Gate:** human reads result.

### Task 0.2: Pilot — one camera image + click-pick + overlay
**Acceptance:** a standalone `gui_tk_pilot.py` loads a real cal image (test_cavity cam1), shows it with zoom/pan, lets the user click 4 points, prints image-pixel coords, and overlays crosses + a quiver of reprojected residuals using the *existing* `full_calibration` output.
**Verify:** unit test the px↔data transform against known values; **human: click the 4 body corners, confirm coords + overlay alignment look right.**
**Dependencies:** 0.1. **Scope:** M. **Gate:** 🚦 **human visual review — GO/NO-GO for the whole migration.**

### 🚦 Checkpoint P0 (HUMAN, mandatory)
- Pilot pick coordinates are correct and overlays align.
- If it feels worse than Chaco or the effort/quality is unacceptable → **stop, stay with the present GUI.**

---

## Phase 1 — Foundation (only if P0 passes)

### Task 1.1: `gui_tk/` skeleton + entry point
Main window, `ttk.Notebook` with 4 camera tabs + control column, `openptv2-gui-tk` script, loads a YAML experiment via existing `ParameterManager`. **Verify (Xvfb):** app starts, tabs exist, experiment loads. **Scope:** M.

### Task 1.2: Reusable `MplImageView` widget
Encapsulates `FigureCanvasTkAgg` + toolbar + `imshow` + `set_overlay(crosses, quiver, lines)` + `on_click` callback. Every camera view uses it. **Verify:** unit-test transform; Xvfb artist counts. **Scope:** M.

### Task 1.3: Generic `ParamForm` from Pydantic models
Build ttk forms from `parameter_models.py`; load/save through `ParameterManager`. **Verify:** every field present; YAML round-trip equals input for all sections. **Scope:** M.

### 🚦 Checkpoint C1 (HUMAN): skeleton opens a real experiment, one image renders, one param form round-trips.

## Phase 2 — Vertical slices (each = one working workflow, human-reviewed)

- **Task 2.1 Detection slice:** detection button → `ptv.py_detection_proc_c` → crosses overlay. (auto wiring + human overlay check)
- **Task 2.2 Calibration slice:** manual-orient click-pick, raw/fine orientation, residual quiver (reuses PR #19 logic). (🚦 human — the riskiest slice)
- **Task 2.3 Correspondence/3D slice:** run correspondences → 3D positions view (mpl Axes3D). (human rotate)
- **Task 2.4 Sequence + tracking slice:** run sequence/tracking, live per-frame updates, trajectory overlay. (human live-feel)
- **Task 2.5 Parameter editor slice:** all 13 param groups via `ParamForm`. (auto round-trip)
- **Task 2.6 Aux panels:** mask, dumbbell, editors, plugin selection. (auto wiring)

Each slice: 🚦 human checkpoint before the next.

## Phase 3 — Parity, polish, cutover

- **Task 3.1 Parity audit:** feature-by-feature checklist Tk vs Traits, both driven on test_cavity; diff outputs (rt_is/ptv_is/ori identical since business logic is shared).
- **Task 3.2 CustomTkinter theming** (optional, last).
- **Task 3.3 Cutover:** flip default entry point once parity is signed off; keep Traits GUI importable for one release.

### 🚦 Checkpoint C-final (HUMAN): full parity sign-off on a real dataset.

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Interactive fidelity can't be self-verified | High | Human gates at P0 and every slice; unit-test all coordinate transforms |
| chaco/enable actually already support 3.14t | High (wasted work) | Task 0.1 confirms the premise *before* any build |
| Scope underestimate (13.7k LOC) | High | Pilot-first; abandon cheaply; slices independently shippable |
| Overlay/px alignment drift | Med | Single shared mpl axes for image+overlay; visual checkpoint |
| Matplotlib TkAgg interactivity feels sluggish on big images | Med | `blit`/`draw_idle`, downsample display; evaluate in pilot |
| "Same exact functionality" is subjective | Med | Written parity checklist in Task 3.1, signed off by human |

## Open questions (need human input)
1. Is "complete auto mode" a hard requirement? If yes → **stay** (this plan needs human gates).
2. Confirm the free-threaded premise (Task 0.1) before investing.
3. Is a "slightly different look" acceptable enough that CustomTkinter theming is worth it, or is plain ttk fine?
4. Must the Traits GUI keep working during migration (coexistence), or is a clean break acceptable?

---

## Architecture v2 (expanded per user spec, 2026-07-11)

Target is a full multi-window application, not a tabbed single window:

- **Multi-panel camera grid**: all N cameras visible at once (2×2 grid of
  `MplImageView`s), not tabs. Each panel = image + overlays + zoom/pan + click.
- **Cross-view click propagation**: clicking in one camera publishes an event on
  a central **EventBus**; other views react (e.g. draw the epipolar line via the
  existing `epi.py`, highlight candidates). Decoupled pub/sub, no view-to-view
  refs.
- **Detachable / additional windows**: any panel (a camera, the 3D view, a plot)
  can pop out into its own `Toplevel`; "open additional window" from the menu.
- **Parameter tree**: `ttk.Treeview` mirroring the YAML sections (ptv, criteria,
  sequence, track, detect_plate, …). **Right-click a node → context menu →**
  opens a `ParamForm` editor `Toplevel` for that section (from the Pydantic
  models); Save writes through `ParameterManager`.
- **Menus & submenus**: `tk.Menu` menubar (File / View / Run / Calibration /
  Windows / Help) with cascades.
- **Calibration as a separate app**: its own `Toplevel` (or `openptv2-cal-tk`
  entry) with the full button set (Detection, Manual orient, Sortgrid, Raw/Fine
  orientation, Orient-from-dumbbell, edit ori/addpar, orientation-with-particles)
  and its own sub-windows.
- **Embedded matplotlib for debugging + 3D**: tracking/calibration debug views
  and a `Axes3D` 3-D positions view embedded in-app (`FigureCanvasTkAgg`).

### Foundation modules (Phase 1, revised)
- `gui_tk/events.py` — `EventBus` pub/sub (pure Python; fully unit-tested).
- `gui_tk/widgets.py` — `MplImageView` (image+overlays+click→bus), detachable `PanelFrame`.
- `gui_tk/paramform.py` — build a ttk form from a Pydantic model/section; load/save via `ParameterManager`.
- `gui_tk/paramtree.py` — `ttk.Treeview` + right-click context menu → `ParamForm`.
- `gui_tk/app.py` — `MainWindow`: menubar, param tree, N-camera grid, detach, bus-wired click propagation.
- `gui_tk/calibration.py` — calibration `Toplevel` app (buttons + sub-windows).
- `gui_tk/view3d.py` — embedded `Axes3D` positions view.
- Entry points: `openptv2-gui-tk`, `openptv2-cal-tk`.

### Verification upgrade
Xvfb is available → the whole Tk app is **headlessly testable**: instantiate under
`xvfb-run`, invoke menu/tree/context callbacks, fire a click through the bus and
assert other views received it, open/save a `ParamForm`. Only visual look/feel &
overlay-alignment remain human checkpoints.
