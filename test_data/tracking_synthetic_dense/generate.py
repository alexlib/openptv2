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


if __name__ == "__main__":
    import shutil
    import sys

    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_demo")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    build_fixture(dest)
    print(f"generated demo fixture in {dest}")
