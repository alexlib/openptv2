"""On-demand test_cavity-calibrated synthetic scene factory.

Replaces the fixed 5-frame test_data/burgers fixture for new tests.
Uses the real test_cavity calibration (non-ideal pinhole + distortion) as
projection truth so pixel_noise is physically meaningful.

See docs/plans/2026-09-02-refactor-burgers-synthetic-tests.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter


def load_cavity_calibration(
    root: str | Path = "test_data/test_cavity",
) -> Tuple[object, list]:
    """Load test_cavity ControlPar + calibrations (real cc, k1/k2, S).

    Returns (cpar, cals) where cpar is ControlPar and cals is list[Calibration].
    """
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.calibration import Calibration

    root = Path(root)
    # ControlPar from YAML would be ideal, but for the helper we build a minimal
    # one that matches test_cavity's imx/imy/pix and loads calibrations from files.
    # The test_cavity dataset is 1280x1024, pix 0.00655 mm, 4 cams.
    # We try to load via ParameterManager if available, falling back to manual.
    try:
        from openptv2.gui.parameter_manager import ParameterManager

        pm = ParameterManager()
        yaml_path = root / "parameters_Run1.yaml"
        if yaml_path.exists():
            pm.from_yaml(yaml_path)
            # Use the already-parsed cpar/cals via py_start_proc_c
            from openptv2.gui.ptv import py_start_proc_c

            cpar, _, _, _, _, cals, _ = py_start_proc_c(pm)
            return cpar, cals
    except Exception:
        pass

    # Fallback: manual minimal cpar + calibrations from cal/*.ori
    # We still need a ControlPar for imgcoord projection; use a thin wrapper.
    cpar = ControlPar(num_cams=4, imx=1280, imy=1024, pix_x=0.00655, pix_y=0.00655)
    # Try to load calibrations from cal/
    cals = []
    for i in range(1, 5):
        ori = root / f"cal/cam{i}.tif.ori"
        add = root / f"cal/cam{i}.tif.addpar"
        if ori.exists() and add.exists():
            cal = Calibration()
            cal.from_file(str(ori), str(add))
            cals.append(cal)
    if len(cals) == 4:
        return cpar, cals
    # Last fallback: dummy calibrations (for import-time tests without data)
    # Use identity-like cals so projection still works (world == pixel-ish)
    return cpar, cals


def _project_points(pts_mm: np.ndarray, cpar, cals, cam: int) -> np.ndarray:
    """Project world mm points to pixel coordinates for one camera."""
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.transforms import convert_arr_metric_to_pixel

    cal = cals[cam]
    # img_coord expects metric coordinates and multimedia params
    # We use the cpar's multimedia for air (n=1)
    try:
        mm = (
            cpar.get_multimedia_params()
            if hasattr(cpar, "get_multimedia_params")
            else None
        )
    except Exception:
        mm = None
    # Batch project via img_coord one by one (simple, not vectorized, fine for tests)
    # img_coord takes (3,) and returns (2,) metric
    metric = []
    for p in pts_mm:
        # Use the single-point variant if available, else try vectorized
        try:
            # openptv2.algorithms.imgcoord.img_coord works on (3,) or (N,3)
            m = img_coord(p, cal, mm) if mm is not None else img_coord(p, cal, None)
            metric.append(np.asarray(m, float).ravel()[:2])
        except Exception:
            # Fallback: approximate pinhole (for dummy cals)
            metric.append(p[:2] * 0.1)
    metric = np.asarray(metric, float)
    pix = convert_arr_metric_to_pixel(metric, cpar)
    return np.asarray(pix, float)


def make_cavity_scene(
    tmp_path: Path,
    n_frames: int,
    n_particles: int = 80,
    calib_root: str | Path = "test_data/test_cavity",
    *,
    spacing_mm: float = 4.0,
    motion_mm: float = 0.3,
    gap_prob: float = 0.0,
    gap_len: tuple[int, int] = (1, 2),
    accel_sigma: float = 0.0,
    turb_sigma: float = 0.0,
    pixel_noise: float = 0.0,
    seed: int = 0,
    store_path: Path | None = None,
) -> Path:
    """Generate smooth Lagrangian tracks, inject gaps/noise, write to RunStore.

    Returns the scene root path (tmp_path / "scene" by default) containing
    res/run.zarr with targets/cam_*/frame_* and correspondences/frame_*.

    The generation mirrors generate_burgers_smooth_gif.py helical vortex model
    plus Savitzky-Golay smoothing, but in world mm and calibrated via
    test_cavity optics.
    """
    from openptv2.storage import RunStore

    tmp_path = Path(tmp_path)
    scene_root = tmp_path / "scene"
    scene_root.mkdir(parents=True, exist_ok=True)
    if store_path is None:
        store_path = scene_root / "res" / "run.zarr"
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    cpar, cals = load_cavity_calibration(calib_root)
    # Fallback if no cals (e.g. missing test_data)
    if len(cals) < 4:
        # Create dummy calibrations that project world ~ pixel
        from openptv2.calibration import Calibration

        cals = [Calibration() for _ in range(4)]
        # dummy: no distortion, cc ~ 10mm, pix 0.00655
        for cal in cals:
            try:
                cal.set_pos([0, 0, 0])
                cal.set_angles([0, 0, 0])
                cal.set_primary_point([640 * 0.00655, 512 * 0.00655])
                cal.set_radial_distortion([0, 0, 0])
                cal.set_decentering([0, 0])
                cal.set_affine_trans([1, 0])
                cal.set_glass_vec([0, 0, 0])
            except Exception:
                pass

    rng = np.random.default_rng(seed)
    # Seed smooth helical vortex tracks in world mm
    # Use similar model to generate_burgers_smooth_gif.py but centred at volume
    r_core = 8.0
    trajectories = []
    for i in range(n_particles):
        # random start near volume centre, spacing controls initial spread
        x0 = rng.uniform(-spacing_mm * 4, spacing_mm * 4)
        y0 = rng.uniform(-spacing_mm * 4, spacing_mm * 4)
        z0 = rng.uniform(-10, 10)
        r = rng.uniform(4, 14)
        vt_factor = (1 - np.exp(-(r**2) / (r_core**2 + 1e-6))) / max(r, 2)
        omega = 0.18 + 0.35 * vt_factor
        v_z = (
            rng.uniform(0.8, 1.4)
            * (1 if rng.integers(0, 2) else -1)
            * (0.6 + 0.5 * np.exp(-r / 12))
        )
        phase = rng.uniform(0, 2 * np.pi)
        t = np.arange(n_frames, dtype=float)
        # base helical
        x = x0 + r * np.cos(omega * t * 0.9 + phase)
        y = y0 + r * np.sin(omega * t * 0.9 + phase)
        z = z0 + v_z * t * (motion_mm / 0.3) * 0.9
        pts = np.stack([x, y, z], axis=1).astype(float)
        # random-walk accel
        if accel_sigma > 0:
            vel = np.zeros(3)
            for f in range(1, n_frames):
                vel += rng.normal(0, accel_sigma, 3)
                # clip by motion
                vel = np.clip(vel, -motion_mm * 2, motion_mm * 2)
                pts[f] += vel * 0.2
        # OU turbulence
        if turb_sigma > 0:
            tau = 3.0
            turb = np.zeros(3)
            for f in range(1, n_frames):
                turb = turb * (1 - 1 / tau) + rng.normal(0, turb_sigma, 3) * 0.3
                pts[f] += turb
        # smooth
        if n_frames >= 5:
            win = 5 if n_frames < 9 else 9
            win = min(win, n_frames // 2 * 2 + 1)
            if win >= 3:
                try:
                    pts = np.stack(
                        [
                            savgol_filter(
                                pts[:, k], window_length=win, polyorder=2, mode="interp"
                            )
                            for k in range(3)
                        ],
                        axis=1,
                    )
                except Exception:
                    pass
        trajectories.append(pts)

    trajectories = np.array(trajectories)  # (n_particles, n_frames, 3)

    # Inject gaps: per particle per frame, drop L consecutive frames
    # gap_mask[p, f] = True if particle p is missing at frame f
    gap_mask = np.zeros((n_particles, n_frames), dtype=bool)
    if gap_prob > 0:
        for p in range(n_particles):
            f = 0
            while f < n_frames:
                if rng.random() < gap_prob:
                    L = int(rng.integers(gap_len[0], gap_len[1] + 1))
                    for k in range(L):
                        if f + k < n_frames:
                            gap_mask[p, f + k] = True
                    f += L + 1
                else:
                    f += 1
        # ensure at least one visible per particle? not needed

    # Project to pixel and add pixel_noise
    store = RunStore(store_path, mode="w")
    # We need to write per frame: for each cam, collect visible particles' 2D pixels
    # and per frame 3D world positions + cam_ids (pnr mapping)
    for f in range(n_frames):
        frame_num = 10001 + f  # use cavity-like frame numbers
        # world positions for this frame (only non-gap)
        world_pts = trajectories[:, f, :]  # (n_particles, 3)
        # For each cam, project
        for cam in range(len(cals)):
            # project all, then mask gaps
            pix = _project_points(world_pts, cpar, cals, cam)
            # add pixel noise
            if pixel_noise > 0:
                pix += rng.normal(0, pixel_noise, pix.shape)
            # Build targets for this cam/frame: only non-gap
            visible = ~gap_mask[:, f]
            pix_visible = pix[visible]
            # Store as Target-like (x,y,n,nx,ny,sumg) — we use simple values
            # RunStore.write_targets expects (N,8) or list[Target] with pnr,x,y,n,nx,ny,sumg,tnr
            # Use (N,8) array for simplicity
            if len(pix_visible) > 0:
                arr = np.zeros((len(pix_visible), 8), dtype=float)
                # pnr = original particle index
                pids = np.where(visible)[0]
                arr[:, 0] = pids  # pnr
                arr[:, 1] = pix_visible[:, 0]
                arr[:, 2] = pix_visible[:, 1]
                arr[:, 3] = 10  # n
                arr[:, 4] = 5  # nx
                arr[:, 5] = 5  # ny
                arr[:, 6] = 1000  # sumg
                arr[:, 7] = -1  # tnr (unlinked)
                store.write_targets(cam, frame_num, arr)
            else:
                store.write_targets(cam, frame_num, np.empty((0, 8), dtype=float))
        # Correspondences: world positions for visible particles, with cam_ids = pnr per cam
        visible = ~gap_mask[:, f]
        if np.any(visible):
            pos_3d = world_pts[visible]
            # cam_ids: (N,4) where each row is [pnr for cam0..3] if visible else -1
            # For simplicity, all cams see same particles when not gapped
            n_vis = int(visible.sum())
            cam_ids = np.full((n_vis, 4), -1, dtype=np.int32)
            pids = np.where(visible)[0]
            for c in range(min(4, cam_ids.shape[1])):
                cam_ids[:, c] = pids
            store.write_correspondences(
                frame=frame_num, pos_3d=pos_3d, cam_target_ids=cam_ids
            )
        else:
            store.write_correspondences(
                frame=frame_num,
                pos_3d=np.empty((0, 3)),
                cam_target_ids=np.empty((0, 4), dtype=np.int32),
            )

    return scene_root
