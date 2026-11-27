# Branch audit, 2026-11-27

We had 26 local branches, 21 remote-only branches, 4 stashes and 2 worktrees, and we had lost track of them.
Every one was read: its commits, its files, and whether `main` already has the same thing. This page records what was
decided, so that the question "what happened to branch X?" has an answer.

"In main" is checked by commit identity *and* by patch, so a squash-merge is also found.

## 1. Already in main (nothing to do)

`chore/adopt-agent-tests`, `chore/commit-uv-lock`, `feat/illmenau-bubble-detection`, `feat/ilp-temporal-smoothing`,
`feat/ilp-temporal-smoothing-impl`, `feat/selfcal-scale-anchor`, `feat/tracker-improvements`, `feat/unified-tracker`,
`feat/zarr-only-remove-legacy`, `feature/optimize-two-phase-bidirectional`, `feature/tree-forest-tracking`,
`fix/brown-affine-scx-x-only`, `fix/correspondence-candidate-buffer`, `plan/what-we-missed`, `review/point-position-methods`,
and the remote `ci/fix-macos-openmp-wheels` (#11).

## 2. Brought into main by the integration pull request

| Branch | What it is | How it came in |
|---|---|---|
| `feat/calibration-consistency` (#53) | calibration consistency check, tracer re-calibration, explicit alignment, truth rig | merged |
| `docs/course-rewrite` (#54) | the beginner course and the recipes, tested end to end | merged |
| `feature/roi-polygon-mask` | per-camera image mask and a 3D trajectory ROI | merged; the tests and the store code were ported to the Zarr store; the per-point `rcm` array is now filtered with the points (a test guards it) |
| `feature/dynamic-tracking-params` | opt-in per-step tracking parameters and `tune-dynamic` | merged; `tune-dynamic` now reads `res/run.zarr`; **works with the classic `trackcorr` tracker only**, not with `two_phase` (see `docs/tracking_guide.md`) |
| `feature/trackcorr-interactive-debug` | trackcorr candidate viewer (notebook, probe, GUI menu) | merged; the tracking kernel gained arguments since, the viewer was updated, and it now works on a scratch copy of the run store |
| `feat/two-phase-bootstrap-affine-consensus` (#43) | spatial consensus ("VFC") for `two_phase`; tracer shaking with ghost filtering | taken commit by commit (see below) |
| `feat/two-phase-spatial-consensus-vfc` | the first VFC commit | contained in #43 |
| `feat/tracking-brain-viz` | click-a-tracer replay notebook and a link lab | merged; **it could only be compiled here, not run interactively** |
| `worktree-marimo-gui-brainstorm` | doc and import fixes | **only 5 files taken** (the GUI is TraitsUI/Chaco, not Tkinter; one notebook import). Its 13 file deletions are not taken: `image_inspector.py`, `draw_3d_target.py`, `gui/cli.py` and `gui/__main__.py` are still used |

#43 was not merged as a whole. Four of its late commits (`.ori` line 2 twice, a cavity recalibration, the triangulation
choice in the dialog) are already in `main` in another form. Its VFC commits and its shaking commits were taken one by one.
The VFC tools no longer have the author's machine paths as defaults (`--zarr` is now required; the notebook folder starts empty).

### Removed from the merges on purpose

* `[tool.uv.sources] flowtracks = { path = "../postptv" }` from the ROI branch: a path on one machine, it breaks every other install.
* `pyvista` as a core dependency (the same branch): it would put the VTK stack in the headless install. `uv.lock` is unchanged.

## 3. Not merged

| Branch | Why | Restore |
|---|---|---|
| `feat/merged-blob-split`, `feat/unified+merged-blob` | our own commit says the split plugin "does not improve metrics". The merged-blob work that *did* help is in `main` (`blob_model`, `blob_hypotheses`, `tracking_shared`, the holistic loop) | tags `archive/merged-blob-split`, `archive/unified-merged-blob` |
| `feat/calibration-lab` (local only) | an earlier integration of #51 and an early draft of #53. Every function and test it has is in the integration branch in a newer form | tag `archive/calibration-lab` |
| `review/pr51-align4cam` | a copy of #51 | none needed (#51 is open) |
| `python_integration_into_pyptv` | June migration branch. Its goals (no `exec()`, parity tests) are done in `main`; the rest is file moves in an old layout | tag `archive/python-integration-into-pyptv` |
| `alexlib-fix-cibuildwheel-all-platforms`, `copilot/fix-lint-ruff-failure` | closed pull requests (#9, #32) | on GitHub |

## 4. Waiting for a decision

| Branch | Question |
|---|---|
| `feat/zero-drift-4cam-alignment` (#51) | **It changes results by design.** With it, the default run of `test_cavity` gives 152 trajectories instead of 153, and the dense-seeding recipe gives 154 instead of 170 (the merged-blob loop gains almost nothing on that 5-frame example). The correction it applies is the one our "Repair B" recipe teaches by hand. The output of course page 9 and of three recipes (camera-set-changes, dense-seeding, water-cell-two-sides) changes, and the camera-set recipe needs a new story. Decide whether the new default is wanted, then merge #51 together with refreshed docs |
| `feature/python314-freethreaded` | changes `requires-python` and the wheel build for free-threaded 3.13t/3.14t. A release decision |
| `plan/tkinter-gui-migration` | a parked pilot of a second GUI (`gui_tk/`). Build it out, or archive |
| `feature/pyvista-3d-viz` | marked WIP in its last commit ("fails with pyvista on windows"); conflicts with the current plot code |
| `feature/differentiable-ptv-autoresearch` | a large experiment (differentiable pipeline, optimiser agent, 24 files) |
| `feat/detection-prototypes-dask-trackpy-skimage-proptv` | detector prototypes and a benchmark; needs optional libraries and adds a module to the compiled `algorithms/` |

## 5. Stashes

| Stash | Content | Decision |
|---|---|---|
| `stash@{0}` | local test settings in `test_cavity/parameters_Run1.yaml` (`two_phase`, `align_4cam`, ...) | not a change to keep (it would change a shared test fixture); tag `archive/stash-test-cavity-settings` |
| `stash@{1}` | ILP work in progress on `layered_tracking.py`, `track_assisted.py` | **already in main** (the files differ by one blank line); tag `archive/stash-ilp-layered-wip` |
| `stash@{2}` | `scripts/epipolar_bootstrap_calibrate.py` (calibrates two poor cameras from two good ones) | not merged: cameras 3 and 4 are hard-coded and it calls an API that changed; tag `archive/stash-epipolar-bootstrap` |
| `stash@{3}` | August generated outputs in `test_data/test_cavity/` | nothing to keep |

## 6. Known problems found on the way

* `tests/unit/test_sequence_triangulation_method.py::test_lsq_triangulation_reprojects_better_than_pairs` passes only where an
  ignored `test_data/test_cavity/run.zarr` exists. In a clean clone it fails (also on `main`).
* About 35 scripts and a few modules still carry machine paths (for example `notebooks/marimo_epipolar_tool.py`,
  `src/openptv2/plate_calibration.py`).
