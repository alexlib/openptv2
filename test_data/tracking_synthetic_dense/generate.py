"""Parametrized dense/ambiguous synthetic ground-truth tracking fixture.

test_data/tracking_synthetic (12 particles, well-separated, motion ~0.2x
spacing) is deliberately the EASY case: no candidate ambiguity, so it verifies
correctness but can't expose a tracker's behavior under the density/noise
regime real experiments actually have. This generator controls particle
DENSITY (spacing) and DETECTION NOISE independently, so a test can sweep
toward test_cavity's measured regime (spacing ~3.8mm, true motion ~0.3mm/
frame, ~1-2px detection noise) and beyond, and count correct/wrong/lost links
against KNOWN ground truth (particle p is rt_is row p in every frame).

Correspondence is written directly from the known particle identity (every
particle seen by all 4 cameras, no epipolar ambiguity/ghost matching) -- this
isolates TRACKING robustness from CORRESPONDENCE robustness, which is a
separate, already-diagnosed problem (see docs/plans/two-subrig-calibration.md
for the ghost-rate measurements). 3D positions are reconstructed via real
multi-camera triangulation from NOISY 2D targets (not written directly), so
noise propagates through this rig's real, measured anisotropic z-sensitivity
(~7x worse along z than in-plane -- verified against test_cavity's own
calibration) rather than a synthetic isotropic jitter.

Reuses test_data/tracking_synthetic's own calibration (already has the same
anisotropic z-sensitivity as test_cavity's, verified: ~10-11 px/mm in-plane,
~1.1-2.0 px/mm along z).
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_FIXTURE = os.path.join(os.path.dirname(HERE), "tracking_synthetic")
NCAM = 4


def _poisson_box_size(n_particles: int, target_spacing_mm: float) -> float:
    """Cube side length so n uniformly-random particles have the given
    median nearest-neighbor spacing (closed form for a 3D Poisson process:
    P(NN > r) = exp(-density * 4/3 * pi * r^3) = 0.5 at the median)."""
    density = np.log(2) / ((4.0 / 3.0) * np.pi * target_spacing_mm**3)
    volume = n_particles / density
    return volume ** (1.0 / 3.0)


def build_scene(n_particles, spacing_mm, motion_mm, n_frames, seed):
    """Ground-truth (noise-free) 3D positions: dict frame_index -> (N,3) mm.

    Particles start at random positions (median nearest-neighbor spacing ~=
    spacing_mm) and drift in independent random straight-line directions at
    motion_mm per frame -- controls the R/S ratio (true motion / spacing)
    that this whole investigation identified as the key conditioning number,
    independent of noise.
    """
    rng = np.random.default_rng(seed)
    L = _poisson_box_size(n_particles, spacing_mm)
    pos0 = rng.uniform(-L / 2, L / 2, (n_particles, 3))

    dirs = rng.normal(size=(n_particles, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    velocity = dirs * motion_mm

    frames = {}
    for t in range(n_frames):
        frames[10001 + t] = pos0 + velocity * t
    return frames


def build_fixture(
    outdir,
    n_particles=40,
    spacing_mm=4.0,
    motion_mm=0.3,
    noise_px=1.0,
    n_frames=5,
    seed=0,
):
    """Write a full img_orig/_targets + res_orig/rt_is + cal/ fixture to
    outdir, in the same format tests/unit/test_tracking_synthetic.py's
    fixture uses. Returns the ground-truth ideal (noise-free) positions dict
    for reference."""
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.orientation import point_position
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.algorithms.trafo import metric_to_pixel, pixel_to_metric

    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "cal"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "img_orig"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "res_orig"), exist_ok=True)

    for c in range(NCAM):
        for ext in (".ori", ".addpar"):
            src = os.path.join(BASE_FIXTURE, "cal", f"cam{c + 1}.tif{ext}")
            dst = os.path.join(outdir, "cal", f"cam{c + 1}.tif{ext}")
            with open(src) as f_in, open(dst, "w") as f_out:
                f_out.write(f_in.read())
    with open(os.path.join(BASE_FIXTURE, "parameters_Run1.yaml")) as f_in, open(
        os.path.join(outdir, "parameters_Run1.yaml"), "w"
    ) as f_out:
        f_out.write(f_in.read())

    cwd = os.getcwd()
    try:
        os.chdir(outdir)
        cpar = ControlPar.from_yaml(os.path.join(BASE_FIXTURE, "parameters_Run1.yaml"))
    finally:
        os.chdir(cwd)
    mm = cpar.mm
    cals = []
    for c in range(NCAM):
        cal = Calibration()
        cal.from_file(
            os.path.join(outdir, "cal", f"cam{c + 1}.tif.ori"),
            os.path.join(outdir, "cal", f"cam{c + 1}.tif.addpar"),
        )
        cals.append(cal)

    rng = np.random.default_rng(seed + 1)
    frames = build_scene(n_particles, spacing_mm, motion_mm, n_frames, seed)
    n = n_particles

    for fr, P in frames.items():
        pix = np.zeros((NCAM, n, 2))
        for c in range(NCAM):
            for p in range(n):
                mx, my = img_coord(P[p], cals[c], mm)
                px, py = metric_to_pixel(mx, my, cpar)
                pix[c, p] = (px, py)
        pix += rng.normal(0.0, noise_px, pix.shape)

        corres_p = np.full((n, NCAM), -1, dtype=int)
        for c in range(NCAM):
            order = np.argsort(pix[c, :, 1], kind="stable")
            with open(f"{outdir}/img_orig/cam{c + 1}.{fr}_targets", "w") as f:
                f.write(f"{n}\n")
                for pnr, p in enumerate(order):
                    x, y = pix[c, p]
                    f.write(
                        "%4d %9.4f %9.4f %5d %5d %5d %5d %5d\n"
                        % (pnr, x, y, 100, 10, 10, 1000, p)
                    )
                    corres_p[p, c] = pnr

        # Reconstruct 3D from the NOISY 2D targets via real multi-camera
        # triangulation, so z-noise inherits this rig's true anisotropic
        # sensitivity instead of being injected as an isotropic 3D jitter.
        recon = np.empty((n, 3))
        for p in range(n):
            targets_metric = np.array(
                [pixel_to_metric(*pix[c, p], cpar) for c in range(NCAM)]
            )
            recon[p], _ = point_position(targets_metric, NCAM, mm, cals)

        with open(f"{outdir}/res_orig/rt_is.{fr}", "w") as f:
            f.write(f"{n}\n")
            for p in range(n):
                f.write(
                    "%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n"
                    % (p + 1, recon[p, 0], recon[p, 1], recon[p, 2], *corres_p[p])
                )

    return frames


def build_fixture_with_correspondence(
    outdir,
    n_particles=40,
    spacing_mm=4.0,
    motion_mm=0.3,
    noise_px=1.0,
    n_frames=5,
    seed=0,
):
    """Like build_fixture, but rt_is is derived from the REAL combinatorial
    correspondence matcher (openptv2.algorithms.correspondences) run on the
    noisy 2D targets, instead of being written directly from known particle
    identity.

    build_fixture's rt_is is ghost-free by construction: correspondence
    indices come straight from the known particle-to-target mapping, so
    every row is a correct quad. Real experiments are not that lucky --
    epipolar/volume ambiguity at real density produces mismatched
    quads/triplets/pairs (ghosts) purely from geometry, with no injected
    false detections needed (see docs/plans/two-subrig-calibration.md's
    test_cavity measurements: pairs 64% ghost, triplets 38%, quads 16%).
    This is Stage 0.5 of docs/plans/2026-08-15-tracking-quality-overhaul.md:
    a primary tracking benchmark that reproduces that failure mode instead
    of missing it.

    Returns (frames, row_gt): ``frames`` is build_scene's noise-free
    ground truth (frame -> (N,3) true positions, unchanged meaning).
    ``row_gt`` maps frame_num -> list of true particle id per rt_is row
    (-1 for a ghost/mixed-identity correspondence) -- row index is no
    longer guaranteed to equal particle id once real correspondence runs,
    so downstream identity metrics (openptv2.benchmarking.metrics) must
    key off row_gt, not row index.
    """
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.constants import COORD_UNUSED, TR_UNUSED
    from openptv2.algorithms.correspondences import correct_frame
    from openptv2.algorithms.correspondences import correspondences as _correspondences
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.orientation import point_position
    from openptv2.algorithms.parameters import ControlPar, VolumePar
    from openptv2.algorithms.tracking_frame_buf import Frame, Target
    from openptv2.algorithms.trafo import metric_to_pixel

    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "cal"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "img_orig"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "res_orig"), exist_ok=True)

    for c in range(NCAM):
        for ext in (".ori", ".addpar"):
            src = os.path.join(BASE_FIXTURE, "cal", f"cam{c + 1}.tif{ext}")
            dst = os.path.join(outdir, "cal", f"cam{c + 1}.tif{ext}")
            with open(src) as f_in, open(dst, "w") as f_out:
                f_out.write(f_in.read())
    with open(os.path.join(BASE_FIXTURE, "parameters_Run1.yaml")) as f_in, open(
        os.path.join(outdir, "parameters_Run1.yaml"), "w"
    ) as f_out:
        f_out.write(f_in.read())

    cwd = os.getcwd()
    try:
        os.chdir(outdir)
        cpar = ControlPar.from_yaml(os.path.join(BASE_FIXTURE, "parameters_Run1.yaml"))
        vpar = VolumePar.from_yaml(os.path.join(BASE_FIXTURE, "parameters_Run1.yaml"))
    finally:
        os.chdir(cwd)
    mm = cpar.mm
    cals = []
    for c in range(NCAM):
        cal = Calibration()
        cal.from_file(
            os.path.join(outdir, "cal", f"cam{c + 1}.tif.ori"),
            os.path.join(outdir, "cal", f"cam{c + 1}.tif.addpar"),
        )
        cals.append(cal)

    rng = np.random.default_rng(seed + 1)
    frames = build_scene(n_particles, spacing_mm, motion_mm, n_frames, seed)
    n = n_particles
    row_gt: dict[int, list[int]] = {}

    for fr, P in frames.items():
        pix = np.zeros((NCAM, n, 2))
        for c in range(NCAM):
            for p in range(n):
                mx, my = img_coord(P[p], cals[c], mm)
                px, py = metric_to_pixel(mx, my, cpar)
                pix[c, p] = (px, py)
        pix += rng.normal(0.0, noise_px, pix.shape)

        # Two DIFFERENT pnr numberings are needed here, matching two
        # different consumers:
        #   - Target FILES on disk (read later by the tracker) must be
        #     y-sorted (candsearch_in_pix does a binary search + early
        #     termination on y -- see test_synthetic_tracking.py).
        #   - The Frame used HERE, only in memory, to run the real
        #     correspondence matcher must have each camera's pnr assigned
        #     in X-SORTED rank order. algorithms.correspondences'
        #     four_camera_matching/three_camera_matching cross-reference
        #     one camera pair's adjacency table using another pair's
        #     candidate pnr AS an array index (e.g. p2_arr[1, 2, p2] where
        #     p2 came from the (0,1) pair's find_candidate output) -- that
        #     table's row dimension is actually built from the x-sorted
        #     position in corrected[cam], not pnr. The two only coincide
        #     when pnr happens to already be x-sorted-rank; any other pnr
        #     order (e.g. the y-sorted order the tracker needs) makes
        #     nearly every camera pair's cross-check compare unrelated
        #     targets, collapsing correspondence to near-total ghosts
        #     regardless of true particle separation (verified: even 2
        #     widely-separated random particles fail to match). This is a
        #     real property of the current matcher, not modeled physical
        #     ambiguity -- flagged for a follow-up fix in
        #     algorithms/correspondences.py; this generator works around it
        #     by matching on an x-sorted-pnr Frame and translating results
        #     back to the on-disk y-sorted pnr afterward.
        frm_match = Frame(num_cams=NCAM, max_targets=n)
        pid_to_disk_pnr = [dict() for _ in range(NCAM)]
        pnr_to_pid_match = [dict() for _ in range(NCAM)]
        for c in range(NCAM):
            y_order = np.argsort(pix[c, :, 1], kind="stable")
            x_order = np.argsort(pix[c, :, 0], kind="stable")
            frm_match.num_targets[c] = n
            with open(f"{outdir}/img_orig/cam{c + 1}.{fr}_targets", "w") as f:
                f.write(f"{n}\n")
                for pnr, p in enumerate(y_order):
                    x, y = pix[c, p]
                    pid_to_disk_pnr[c][int(p)] = pnr
                    f.write(
                        "%4d %9.4f %9.4f %5d %5d %5d %5d %5d\n"
                        % (pnr, x, y, 100, 10, 10, 1000, TR_UNUSED)
                    )
            for pnr, p in enumerate(x_order):
                x, y = pix[c, p]
                frm_match.targets[c][pnr] = Target(
                    pnr=pnr, x=x, y=y, n=100, nx=10, ny=10, sumg=1000, tnr=TR_UNUSED
                )
                pnr_to_pid_match[c][pnr] = int(p)

        corrected = correct_frame(frm_match, cals, cpar, 0.0001)
        by_pnr = [{c2d.pnr: (c2d.x, c2d.y) for c2d in corrected[c]} for c in range(NCAM)]
        con, _match_counts = _correspondences(frm_match, corrected, vpar, cpar, cals)

        rows_pid = []
        rows_pos = []
        rows_p = []
        for tup in con:
            # p[0] is an index into corrected[0] (the x-sorted list for
            # camera 0), not a pnr -- unlike p[1..3], which genuinely are
            # pnr values written by find_candidate.
            p = list(tup.p)
            if p[0] >= 0:
                p[0] = corrected[0][p[0]].pnr

            cam_pids = [pnr_to_pid_match[c].get(p[c]) for c in range(NCAM) if p[c] >= 0]
            true_pid = cam_pids[0] if cam_pids and all(x == cam_pids[0] for x in cam_pids) else -1

            targets_metric = np.full((NCAM, 2), COORD_UNUSED)
            for c in range(NCAM):
                if p[c] >= 0:
                    targets_metric[c] = by_pnr[c][p[c]]
            pos, _dist = point_position(targets_metric, NCAM, mm, cals)

            # Translate each camera's match-space pnr to the on-disk
            # (y-sorted) pnr the target files and the tracker actually use --
            # by looking up which true particle it was (per-camera, so a
            # ghost row with mismatched cams still gets a valid disk pnr per
            # camera, just not a shared one).
            disk_p = []
            for c in range(NCAM):
                if p[c] < 0:
                    disk_p.append(-1)
                    continue
                pid_c = pnr_to_pid_match[c][p[c]]
                disk_p.append(pid_to_disk_pnr[c].get(pid_c, -1))

            rows_pid.append(true_pid)
            rows_pos.append(pos)
            rows_p.append(disk_p)

        row_gt[fr] = rows_pid
        with open(f"{outdir}/res_orig/rt_is.{fr}", "w") as f:
            f.write(f"{len(con)}\n")
            for i, p in enumerate(rows_p):
                pos = rows_pos[i]
                f.write(
                    "%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n"
                    % (i + 1, pos[0], pos[1], pos[2], *p)
                )

    return frames, row_gt


if __name__ == "__main__":
    import shutil
    import sys

    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_demo")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    build_fixture(dest)
    print(f"generated demo fixture in {dest}")
