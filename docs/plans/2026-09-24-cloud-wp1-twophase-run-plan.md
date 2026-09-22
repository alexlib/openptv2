# Plan: Cloud Run of wp1 with Bidirectional Two-Phase Tracker & Postprocessing

**Date:** 2026-09-24  
**Target Dataset:** `C:\Users\alex\Downloads\HiDImaging\CompleteTest\wp1\test`  
**GCP Environment:** Project `iucc-alex-liberzon`, Region `europe-west3`, Buckets `openptv-uploads` / `openptv-results`  
**Objective:** Re-run tracking on the full 5,005 frames directly in the cloud using the new bidirectional Two-Phase tracker, execute flowtracks repair and smoothing, generate Eulerian and phase-averaged Zarr datasets, and fetch trajectories and results locally for analysis against `res_orig` (3dptv.exe `xuap`).

---

## Background & Rationale

1. **Why rerun in Cloud:**
   The full correspondences and target intermediate files for the 5,005 frames already reside in `openptv-results` on GCS (`CompleteTest-track-20260915-065041/wp1/test/res.tar`). Running tracking on Cloud Run directly where the correspondence data is located avoids pulling tens of gigabytes across the internet and executes in parallel.

2. **Why the new Two-Phase Bidirectional Tracker:**
   - Previous runs suffered from overly conservative kinematic limits (`dacc: 0.4`, `dvxmax: 1.4`), which gate-dropped fast systolic bursts in the aorta.
   - The new Two-Phase tracker with **Forward-First Bidirectional Tracking** (`bidirectional: true`):
     - Forward pass with $v_{\max} = 2.2\text{ mm}$ establishes the clean diastolic core.
     - Backward pass with wider reach ($v_{\max}^{\text{bwd}} = 3.5\text{ mm}$) recovers high-velocity systolic bursts without risking swaps in dense regions.
     - Closed 75% of the link gap to 4-frame trackcorr on wp1 (reaching 97.48% exact recall in 3.0 s vs 15.0 s).
     - Incorporates canonical `TargetArray.sort_y()` and auto-grid safety.

---

## Step-by-Step Execution Plan

### Step 1: Merge PR #39 in `openptv2`
- **Current Status:** PR #39 (`refactor/dedup-tracking-kernels` -> `main`) is open, mergeable, and all 8 GitHub Actions CI checks are **GREEN** (cross-platform wheels on Linux/macOS/Windows, ruff linting, multi-Python tests).
- **Action:**
  ```bash
  gh pr merge 39 --merge
  git switch main
  git pull --ff-only
  ```

### Step 2: Build & Push Docker Image in `openptv-cloud`
- Update `openptv-cloud` dependency reference to include the merged `openptv2` `main`.
- Build the container image and push to Google Artifact Registry:
  ```bash
  cd C:\Users\alex\projects\openptv-cloud
  uv run openptv-cloud build
  ```
- Image target:
  `europe-west3-docker.pkg.dev/iucc-alex-liberzon/openptv/openptv-cloud-job:latest`

### Step 3: Verify & Configure `parameters_twophase.yaml`
- Parameter file already prepared at:
  `C:\Users\alex\Downloads\HiDImaging\CompleteTest\wp1\test\parameters_twophase.yaml`
- Config details:
  ```yaml
  plugins:
    selected_sequence: default
    selected_tracking: two_phase

  track:
    selected_tracking: two_phase
    preset: two_phase
    use_velocity: true
    cost_mode: projected
    leaf_weight: 1.0
    v_max: 2.2            # Diastolic core search radius
    bidirectional: true   # Forward-first locked bidirectional tracking
    bwd_v_max: 3.5        # Systolic burst recovery radius
    max_gap: 2            # Missing-frame bridge capacity
    allow_shared: true    # Occlusion clustering
    share_tol: 1.0
    confirm_tol: 1.9      # Kinematic acceleration gate matching dacc=1.9
    confirm_ends: false
    max_group_size: 128
    postprocess: false

  sequence:
    first: 1
    last: 5005
  ```

### Step 4: Configure `experiment.yaml` for wp1 Single-Folder Run
In `C:\Users\alex\Downloads\HiDImaging\CompleteTest\experiment.yaml`:
- Scope run folders strictly to `wp1/test`:
  ```yaml
  runs:
    folders: [wp1/test]

  frames:
    first: 1
    last: 5005

  trajectories:
    repair: true          # flowtracks.repair (cut bad links, join pieces across gaps)
    smoothing_window: 7   # Savitzky-Golay filter
    smoothing_order: 2
  ```

### Step 5: Launch Tracking on GCP Cloud Run
- Run the cloud pipeline:
  ```bash
  uv run openptv-cloud run C:\Users\alex\Downloads\HiDImaging\CompleteTest
  ```
- Monitor progress:
  ```bash
  uv run openptv-cloud status C:\Users\alex\Downloads\HiDImaging\CompleteTest --watch
  ```

### Step 6: Fetch Trajectories & Postprocess Locally
- Download the resulting trajectory table and phase-binned Zarr datasets:
  ```bash
  uv run openptv-cloud fetch C:\Users\alex\Downloads\HiDImaging\CompleteTest
  ```
- Confirm output files in `C:\Users\alex\Downloads\HiDImaging\CompleteTest\wp1\test\res\`:
  - `run.zarr/trajectories` (repaired & smoothed positions, velocities, accelerations)
  - `run.zarr/eulerian` (Eulerian velocity, TKE, MKE fields)
  - `phase_binned.zarr` (24 cardiac phase bins)

### Step 7: Validation against `res_orig/xuap`
- Run comparison script between newly generated trajectories in `res/run.zarr` vs ground-truth `res_orig`:
  - Track length distribution (mean length $\ge 14$ frames).
  - Trajectory coverage and sample counts.
  - Phase-averaged velocity profiles in peak systole vs diastole.
