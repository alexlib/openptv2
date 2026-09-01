"""Can the current benchmark data evaluate Lagrangian turbulence quality?

Checks the GROUND TRUTH itself against the physics the guide says matters:
acceleration intermittency (K_a), velocity autocorrelation / integral time,
and trajectory span in units of that time. If the ground truth has none of
the structure, no tracker can be scored on preserving it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import benchmark_utils as bu  # noqa: E402

for src, label, first, n in [
    (Path("test_data/synthetic_turbulent"), "synthetic_turbulent", 10001, 30),
    (Path("test_data/synthetic_turbulent_1k"), "synthetic_turbulent_1k", 10001, 30),
]:
    tt = bu.build_true_tracks(bu.read_gt_frames(src, first, n), first)
    lens = np.array([len(v) for v in tt.values()])

    # Component-wise velocity and acceleration by finite difference on the
    # TRUE positions (contiguous runs only, so gaps don't fake a jump).
    vels, accs = [], []
    for pts in tt.values():
        pts = sorted(pts)
        f = np.array([p[0] for p in pts])
        x = np.array([[p[1], p[2], p[3]] for p in pts])
        # split at gaps
        brk = np.where(np.diff(f) != 1)[0] + 1
        for seg in np.split(x, brk):
            if len(seg) >= 2:
                vels.append(np.diff(seg, axis=0))
            if len(seg) >= 3:
                accs.append(np.diff(seg, 2, axis=0))
    v = np.concatenate(vels).ravel() if vels else np.array([])
    a = np.concatenate(accs).ravel() if accs else np.array([])

    def kurt(z):
        if z.size < 4:
            return float("nan")
        z = z - z.mean()
        return float(np.mean(z**4) / (np.mean(z**2) ** 2))

    # Lagrangian velocity autocorrelation of the component series, lag 1..5
    rho = []
    for lag in range(1, 6):
        num = den = 0.0
        cnt = 0
        for pts in tt.values():
            pts = sorted(pts)
            f = np.array([p[0] for p in pts])
            x = np.array([[p[1], p[2], p[3]] for p in pts])
            brk = np.where(np.diff(f) != 1)[0] + 1
            for seg in np.split(x, brk):
                if len(seg) < lag + 2:
                    continue
                d = np.diff(seg, axis=0)
                d = d - d.mean(axis=0)
                num += float(np.sum(d[:-lag] * d[lag:]))
                den += float(np.sum(d * d))
                cnt += 1
        rho.append(num / den if den else float("nan"))

    print(f"\n=== {label} ===")
    print(f"  true tracks              {len(tt)}")
    print(f"  track length  mean/max   {lens.mean():.1f} / {lens.max()}")
    print(f"  velocity component rms   {v.std():.3f} mm/frame")
    print(f"  accel    component rms   {a.std():.3f} mm/frame^2")
    print(f"  velocity kurtosis        {kurt(v):.2f}   (Gaussian = 3)")
    print(
        f"  ACCEL    kurtosis K_a    {kurt(a):.2f}   "
        f"(Gaussian = 3; real turbulence at high Re ~ 10-60)"
    )
    print(f"  vel autocorr lag 1..5    {' '.join(f'{r:+.3f}' for r in rho)}")
