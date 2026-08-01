"""Generate a synthetic 4-camera ground-truth tracking fixture.

The scene is fully known: 12 particles, 5 frames (10001-10005). Particle ``p``
occupies rt_is row ``p`` in EVERY frame, so the correct forward link is the
identity  ``next[p] == p``  (and ``prev[p] == p``). This lets a test assert
exact ground truth and probe how each tracking parameter (dvxmax, dacc, dangle,
add) gates real links.

Motion is designed so different particles stress different gates:
  - p0        FAST      : extra x-velocity (~4 mm/frame)  -> gated by dvxmax
  - p1        ACCEL     : constant acceleration ~1.5 mm/frame^2 -> gated by dacc
  - p2        TURN      : ~90 deg direction change each step -> gated by dangle
  - p3..p11   CALM      : slow, straight, well separated -> always linkable

Files written (committed as ground truth; image bases use %d notation):
  img_orig/camC.FFFFF_targets  detected 2D targets per camera/frame (tnr = particle id)
  res_orig/rt_is.FFFFF         3D positions + per-camera correspondence indices

Regenerate with:  uv run python test_data/tracking_synthetic/generate.py
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST, LAST = 10001, 10005
NCAM = 4
IMX, IMY = 1280, 1024


def trajectories():
    """Return dict frame_index -> (N,3) array of 3D positions (mm).

    12 particles on a coarse 4x3 grid (spacing 20 x 14 mm) so that no particle
    ever comes near another: the ground-truth assignment is unambiguous and the
    default parameters link everything correctly. Motions are small (<= ~4 mm/
    frame) and probed by TIGHTENING gates below a particle's signature:
      p0 FAST : extra +2.5 mm/frame in x  (x-disp ~4)   -> dvxmax
      p1 ACCEL: constant a = 1.5 mm/frame^2 in x         -> dacc
      p2 TURN : +/-2 mm zig-zag (~90 deg direction flip) -> dangle
    """
    xs = [-30.0, -10.0, 10.0, 30.0]
    ys = [-14.0, 0.0, 14.0]
    base = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)  # 12 pts
    N = len(base)
    calm_v = np.array([1.5, 0.8, 0.3])  # gentle uniform drift, well within gates

    frames = {}
    for t in range(LAST - FIRST + 1):
        P = base + calm_v * t
        # p0 FAST: extra x-velocity -> larger straight-line displacement
        P[0] = base[0] + np.array([4.0, 0.8, 0.3]) * t
        # p1 ACCEL: x = x0 + v t + 0.5 a t^2  (between-frame accel ~1.5)
        accel = 0.5 * np.array([1.5, 0.0, 0.0]) * t * t
        P[1] = base[1] + np.array([1.5, 0.8, 0.3]) * t + accel
        # p2 TURN: zig-zag ~90 deg direction change each step (on top of drift)
        turn = base[2] + calm_v * t
        for s in range(t):
            zig = np.array([2.0, 0.0, 0.0]) if s % 2 == 0 else np.array([0.0, 2.0, 0.0])
            turn = turn + zig
        P[2] = turn
        frames[FIRST + t] = P
    return frames, N


def main():
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.algorithms.trafo import metric_to_pixel

    os.chdir(HERE)
    cpar = ControlPar.from_yaml("parameters_Run1.yaml")
    mm = cpar.mm
    cals = []
    for c in range(NCAM):
        cal = Calibration()
        cal.from_file(f"cal/cam{c + 1}.tif.ori", f"cal/cam{c + 1}.tif.addpar")
        cals.append(cal)

    frames, N = trajectories()
    os.makedirs("img_orig", exist_ok=True)
    os.makedirs("res_orig", exist_ok=True)

    for fr, P in frames.items():
        # project every particle to every camera
        pix = np.zeros((NCAM, N, 2))
        for c in range(NCAM):
            for p in range(N):
                mx, my = img_coord(P[p], cals[c], mm)
                px, py = metric_to_pixel(mx, my, cpar)
                pix[c, p] = (px, py)

        corres_p = np.full((N, NCAM), -1, dtype=int)
        for c in range(NCAM):
            order = np.argsort(pix[c, :, 1], kind="stable")  # sort targets by y
            with open(f"img_orig/cam{c + 1}.{fr}_targets", "w") as f:
                f.write(f"{N}\n")
                for pnr, p in enumerate(order):
                    x, y = pix[c, p]
                    # tnr = particle id (== rt_is row), the correspondence back-ref
                    f.write(
                        "%4d %9.4f %9.4f %5d %5d %5d %5d %5d\n"
                        % (pnr, x, y, 100, 10, 10, 1000, p)
                    )
                    corres_p[p, c] = pnr

        with open(f"res_orig/rt_is.{fr}", "w") as f:
            f.write(f"{N}\n")
            for p in range(N):
                f.write(
                    "%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n"
                    % (p + 1, P[p, 0], P[p, 1], P[p, 2], *corres_p[p])
                )

    print(f"generated {N} particles x {LAST - FIRST + 1} frames in {HERE}")


if __name__ == "__main__":
    main()
