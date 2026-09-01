# Plan: Refactor Burgers 5-frame synthetic tests → on-demand test_cavity-calibrated synthetic factory

Date: 2026-09-02
Status: draft — for review before implementation
Authors: openptv2 team

## 1. Context

`test_data/burgers` (`README.md:1` — `PTV_SYN` / Ruiz, Liberzon, Bhattacharya 2023) is a **fixed 5-frame fixture**: `img/cam{1..4}.1000{1..5}` (20 TIFs) + `img_orig/*_targets` + `res_orig/rt_is.1000{1..6}` (6th is a generator leftover). It was valuable as a smoke test when `src/openptv2/` had no synthetic generator, but it now hides the real problems:

* **Fixed size cannot probe the lever:** `gap_relinking` needs `max_gap*4` frames, turbulence needs `12+` frames, SNR sweep needs varying `spacing_mm` vs `motion_mm`. 5 frames cannot test `relink_trajectory_gaps(max_gap=2)` vs `forward_backward` double-claim (`docs/plans/2026-08-27-backward-postprocess-double-claim-bug-plan.md` 185 double-claims) properly.
* **Clean, no gaps / no accel / no pixel noise:** `dv 0.5 mm`, `dacc 0.1 mm` pass on 5 particles/frame but fail on `test_cavity_like` (`n=80, spacing 3.8 mm, motion 0.3 mm, noise 1.0 px`). The smoke test never drives `relink_trajectory_gaps`, `enforce_reciprocity`, or `dacc/angle` gates.
* **Ideal calibration:** `test_data/burgers/cal/*.ori` is pinhole-perfect, not the `test_cavity` `cc 8.5 mm + k1/k2 + S` that gives `1.5 mm` reprojection and `0.58 %` RCM. Position noise in pixel space via real `K` is not tested.
* **Committed `run.zarr` churn:** `test_data/burgers/run.zarr` was tracked (`git ls-files`) and showed as `modified` after every `pytest`/`openptv2-batch` run, even though it is *output* (fixed in `a565df81` by untracking `**/run.zarr`). Keeping a 5-frame fixture as ground truth invites the same.

`tests/batch/test_burgers_synthetic.py:192/246/304` + `tests/unit/test_track.py:580` + `tests/unit/test_track3d.py:327/457` all hard-code `first 10001 last 10005` and `BURGERS_DIR = TEST_DATA_ROOT / "burgers"`.

## 2. Goals

* **Remove the fixed 5-frame dependency:** tests create *as many frames as that specific test needs* at the start of the test, in `tmp_path`, via a helper.
* **Add realistic, controllable degradations:** `gap_prob/gap_len` (drop 1-2 frames), `accel_sigma` (random-walk acceleration), `turb_sigma` (OU turbulence), `pixel_noise` (px, via `test_cavity` `K,R,t` projection) so tests probe **real SNR**, not `5` clean frames.
* **Use realistic calibration:** `test_data/test_cavity/cal/cam{1..4}.tif{.ori,.addpar}` as the default optics for projection — non-ideal, measured, with distortion. Keeps `test_data/burgers/cal` only as a legacy smoke.
* **Keep `test_data/burgers` but deprecate:** `test_data/burgers/README.md` gets a banner “legacy 5-frame smoke, use synthetic factory for new tests”; no new test may `git ls-files test_data/burgers/img`.

## 3. Non-goals

* Not deleting `test_data/burgers/` from the repo in this plan (docs still reference it as the Ruiz et al. dataset). Just deprecating for tests.
* Not changing `src/openptv2/` tracker logic — only test harness.
* Not adding a new external dependency (uses `numpy`, `scipy.signal.savgol_filter` already in `dev`, `Calibration`, `ControlPar`, `RunStore`).

## 4. Design

### 4.1 New helper — `tests/helpers/synthetic_scene.py`

Single factory, no new package dependency, imports only `src/openptv2/`:

```py
# tests/helpers/synthetic_scene.py:20
from pathlib import Path
import numpy as np
from openptv2.storage import RunStore
from openptv2.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar


def load_cavity_calibration(
    root="test_data/test_cavity",
) -> tuple[ControlPar, list[Calibration]]:
    """Load test_cavity K,R,t + ControlPar (real cc, k1/k2, pix_x/y)."""


def make_cavity_scene(
    tmp_path: Path,
    n_frames: int,
    n_particles: int = 80,
    calib_root: str | Path = "test_data/test_cavity",
    *,
    spacing_mm: float = 4.0,  # controls density vs motion
    motion_mm: float = 0.3,  # mean displacement / frame
    gap_prob: float = 0.0,
    gap_len: tuple[int, int] = (1, 2),  # drop 1–2 consecutive frames
    accel_sigma: float = 0.0,  # mm/frame², random-walk accel
    turb_sigma: float = 0.0,  # mm, OU turbulence
    pixel_noise: float = 0.0,  # px, added after projection via K
    seed: int = 0,
    store_path: Path | None = None,  # default tmp_path / "scene" / "res" / "run.zarr"
) -> Path:
    """Generate smooth Lagrangian tracks in world mm, project via test_cavity K,R,t,
    inject gaps/noise, write targets (x,y) + correspondences (X,Y,Z) to
    RunStore at store_path, return scene root for Tracker/runner."""
```

* **Trajectory generation:** start from `benchmarking/scenario.py` + `generate_burgers_smooth_gif.py:14` helical vortex model, but in world mm: `r~4-18`, `vt_factor=(1-exp(-r²/r_core²))/r`, `omega=0.18+0.35*vt`, `v_z` axial jet, `z0` uniform, then **random-walk accel** (`vel += accel_sigma * N(0,1)`, clipped by `dv`) + **OU turbulence** (`turb += -turb/tau + turb_sigma*N`). Smooth with `savgol_filter(window 9, poly 3)` — same as hero GIF, now in test helper.
* **Projection:** `Xw -> Xc = R*Xw + t -> x = K*Xc / Z -> u = x/pix + imx/2` via `Calibration` + `ControlPar` (real `cc`, `k1/k2`, `S`). This makes `pixel_noise` physically meaningful (vs world mm noise).
* **Gaps:** for each particle, with `gap_prob` per frame, drop `L ~ Uniform(gap_len)` consecutive frames — tests `relink_trajectory_gaps` with `max_gap=2` vs `forward_backward` double-claim.
* **Writing:** `RunStore(store_path, mode="w")` then `write_targets(cam, frame, xy)` + `write_correspondences(frame, pos_3d, cam_ids)` — no TIFF rendering needed (faster, no `MAXCAND` threshold). For image-based tests, optionally call `generate_synthetic_images_from_targets.py:16 --data tmp_path/scene` to rasterize `img/cam*.tif`.

### 4.2 Test migration matrix

| Current test (5-frame) | New test (on-demand) | n_frames / n_particles | Degradations | What it actually probes (SNR lever) |
|---|---|---|---|---|
| `tests/batch/test_burgers_synthetic.py:192` `test_burgers_detection_roundtrip` | `tests/unit/test_cavity_synthetic_detection.py:20` | `6, 80` | `pixel_noise 0.5, turb 0.6` | `targ_rec` SNR vs `eps0`/`corrmin` trade-off, not just `MAXCAND` |
| `tests/batch/test_burgers_synthetic.py:246` `test_burgers_3d_trajectory_vs_res_orig` | `tests/unit/test_cavity_synthetic_correspondence.py:20` | `8, 120` | `accel 0.4, pixel_noise 0.25` | `correspondences` quad-uniqueness vs `dacc/angle` |
| `tests/batch/test_burgers_synthetic.py:304` `test_burgers_image_space_add_particle` | `tests/unit/test_cavity_synthetic_gap_relink.py:20` | `12, 80` (`max_gap*4`) | `gap_prob 0.15, gap_len 1-2` | `relink_trajectory_gaps` gap relinking vs `forward_backward` double-claim |
| `tests/unit/test_track.py:580` `test_burgers` | `tests/unit/test_track_cavity_synthetic.py:20` | `12, 80` | `turb 0.6, pixel_noise 0.5` | `track3d` SNR (`spacing vs motion` 3.8 vs 0.3) |
| `tests/unit/test_track3d.py:327/457` parity with Cython | keep but add `pytest.mark.parametrize("n_frames",[6,12])` via factory | `6,12` | `pixel_noise 0.25` | parity at realistic calibration, not ideal pinhole |

### 4.3 Keeping `test_data/burgers` as deprecated smoke

* Add banner to `test_data/burgers/README.md:1`: “Legacy 5-frame smoke for `docs/algorithms/burgers_gap_relinking_case_study.md` — new tests must use `tests/helpers/synthetic_scene.py:make_cavity_scene`.”
* No new test may `import` or `Path("test_data/burgers")` — enforced via `ruff` per-file-ignores or `tests/conftest.py` helper.

## 5. Implementation phases

**Phase 0 — Scaffolding (0.5 day):**
* Create `tests/helpers/__init__.py` + `tests/helpers/synthetic_scene.py` with `load_cavity_calibration` + `make_cavity_scene` skeleton (no rendering yet, just `RunStore` write). Add `tests/helpers/test_synthetic_scene_smoke.py` (1 test: `make_cavity_scene(tmp_path, 4, 10)` → `RunStore.frames() == 4`).

**Phase 1 — Port `test_burgers_detection_roundtrip` as example (0.5 day):**
* Implement `make_cavity_scene` with `pixel_noise` + `savgol` smoothing, port `tests/batch/test_burgers_synthetic.py:192` to `tests/unit/test_cavity_synthetic_detection.py` with `n_frames=6, pixel_noise=0.5` and `pytest.mark.parametrize("pixel_noise",[0.2,0.5,1.0])` SNR sweep. Verify `uv run pytest tests/unit/test_cavity_synthetic_detection.py -v` passes.

**Phase 2 — Gaps + turbulence + accel (1 day):**
* Add `gap_prob/gap_len`, `accel_sigma`, `turb_sigma` to factory, add `tests/unit/test_cavity_synthetic_gap_relink.py` (12 frames, `gap_prob 0.15`) asserting `relink_trajectory_gaps` bridges `>= 80%` of injected gaps and `forward_backward` does not double-claim (regression for `2026-08-27-backward-postprocess-double-claim-bug-plan.md` 185). Add `tests/unit/test_cavity_synthetic_turbulence.py` (12 frames, `turb 0.6`) asserting `mean_track_length` vs `kalman` etc.

**Phase 3 — Migrate remaining Burgers tests (0.5 day):**
* Port `test_burgers_3d_trajectory_vs_res_orig` → `test_cavity_synthetic_correspondence.py` (8 frames), `test_burgers` → `test_track_cavity_synthetic.py` (12 frames). Keep original `test_burgers` files but mark `@pytest.mark.legacy` + `skip` if `helpers` available, so `CI` (`-m "not slow"` + `-m ci`) runs only new.

**Phase 4 — Deprecate `test_data/burgers` (0.25 day):**
* Update `test_data/burgers/README.md` banner, add `ruff` check `tests/**` must not import `test_data/burgers` (or `grep` in `pre-commit`), update `docs/algorithms/burgers_gap_relinking_case_study.md:1` to note new factory.

## 6. Verification

* `uv run pytest tests/helpers/test_synthetic_scene_smoke.py -v` → `1 passed`
* `uv run pytest tests/unit/test_cavity_synthetic_*.py -v` → all pass at `n_frames` requested (6,8,12)
* `uv run pytest -m ci -q` → `45 passed` (previously `38 passed, 7 skipped`) — new tests are `ci`-marked, legacy 5-frame hidden
* `uv run pytest tests/batch/test_burgers_synthetic.py -v` still passes (legacy, now `skip` unless `--run-legacy-burgers`)
* Manual: `uv run python -c "from tests.helpers.synthetic_scene import make_cavity_scene; p=make_cavity_scene(...); print(list((p/'res/run.zarr').rglob('*')))"` → `store_path` exists and `zarr` groups `targets/cam_*/frame_*` + `correspondences/frame_*` have `n_frames` entries.

## 7. Risks & mitigations

* **Risk:** `test_cavity` calibration has distortion → projection is non-linear, `pixel_noise` must be in px not mm. Mitigation: use `Calibration` + `ControlPar` for projection, not world mm noise.
* **Risk:** `savgol` window must be odd and `< T`. Mitigation: `win = 9 if T>=9 else T//2*2+1` with `mode="interp"`.
* **Risk:** `RunStore` writing without TIFFs bypasses `MAXCAND` threshold. Mitigation: keep one image-based test via `generate_synthetic_images_from_targets.py:16` for `targ_rec` threshold path.
* **Risk:** flaky gaps (random). Mitigation: `seed` param, `pytest.mark.parametrize` with fixed seeds, `np.random.default_rng(seed)`.

## 8. Acceptance criteria

* No new test does `Path("test_data/burgers")` or `first 10001 last 10005` hard-coded.
* `test_data/burgers/README.md` banner present.
* `uv run pytest -m ci --tb=short -q` shows `0 failed` and `n_frames` varies per test (not always 5).
* `git ls-files test_data/burgers` unchanged (still `img`, `cal`, `res_orig`) but `**/run.zarr` remains untracked (`a565df81`).

## 9. Effort

~2.5 days total (0.5+0.5+1+0.5+0.25), can start with Phase 0-1 as the first example for review.

## 10. Appendix — Decisive tracker benchmark: which tracker is best / fastest / most robust (added 2026-09-02)

The 5-frame smoke can only say “it links”; it cannot decide the real trade-off
`accuracy (P/R/F1, ghost, fragmented)` vs `speed (ms/frame)` vs
`robustness (slope vs noise)`. This harness builds on the same
`make_cavity_scene` factory to make the decision decisive, at controlled SNR,
with `test_cavity` optics as the truth.

**Noise models (one knob at a time, so the lever is visible):**

* **Calibration:** `dR ~ N(0,0.1°)`, `dt ~ N(0,0.1 mm)`, `dcc ±1%`, `k1/k2` jitter — perturb `Calibration` before projection (tests `epipolar` vs `3D` sensitivity).
* **Gaussian pixel noise:** `0.0, 0.2, 0.5, 1.0, 1.5 px` after `img_coord → pixel` (`ControlPar` `pix_x/y`).
* **Electronics (sensor):** `read ~ N(0,2 ADU) + Poisson(sqrt(I)) + quant 8-bit` on synthetic `img` before `targ_rec` (`image_scaling.py:to_uint8` `fixed` vs `stretch`).
* **Gaps:** `gap_prob 0, 0.05, 0.10, 0.15` × `len 1-2` (tests `relink_trajectory_gaps` vs `forward_backward` double-claim).
* **Turbulence:** OU `tau=3` frames, `turb_sigma 0, 0.3, 0.6, 1.0 mm`.
* **Ghosts:** spurious detections at `spacing_mm` vs `motion_mm` (see `verified-pipeline` 64/38/16% baseline).

**Harness — `tests/benchmarks/test_tracker_comprehensive.py` (`@pytest.mark.ci`):**

```py
from openptv2.benchmarking.runner import run_tracker
for tr in BUILTIN_TRACKING_PLUGINS:  # fast_3d, cython_3d, myptv, proptv, priority_segment_3d, trackcorr …
  for sigma in [0.2,0.5,1.0]:
    scene = make_cavity_scene(tmp_path/f"{tr}-{sigma}", 12, 120,
              spacing_mm=3.8, motion_mm=0.3, pixel_noise=sigma, gap_prob=0.07, seed=s)
      t0 = time.perf_counter(); run_tracker(yaml, tr, store); dt=(time.perf_counter()-t0)/12
      # accuracy via benchmarking/metrics.py: GhostRealMetrics
      P,R,F1,ghost,frag,miss = compute_identity_metrics(pred, gt, eps=0.5)
```

**Metrics that decide “best”:**

* **Accuracy:** `P, R, F1, ghost_capture_rate, fragmented/incomplete/missed, track_lifetime` distribution.
* **Speed:** `ms/frame` and `ms/particle`, slope `log(dt) vs log(density)` for `n=40,80,120`.
* **Robustness:** `slope ΔF1/Δsigma` per noise type + variance over `seed 0..4` (flat + tight wins).

**Decision — Pareto, not single number:**

* `F1 vs ms/frame` at `pixel_noise 0.5, gap 0.07` (typical `test_cavity_like`) — frontier wins.
* `robustness = F1(1.0 px)/F1(0.2 px)` — closest to `1.0` wins.
* Rank table `tracker | best @ clean (P/R) | best @ noisy (P/R) | speed | robustness` — “fastest” (usually `cython_3d`), “most accurate” (`priority_segment_3d`/`proptv` at `turb`), “most robust” (`trackcorr` image-space).

This will be implemented as Phase 5 after the 4 migration phases, reusing the same factory and `RunStore` groups `targets/cam_*/frame_*` + `correspondences/frame_*`.

