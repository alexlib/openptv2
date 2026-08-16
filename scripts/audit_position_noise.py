"""How much position noise does the stereo reconstruction carry, and what
does it do to a finite-difference acceleration?

Acceleration by 3-point second difference amplifies position noise by
sqrt(6)/dt^2. If sigma_a_noise is comparable to the true a_rms, acceleration
is not measurable from raw positions at all -- no tracker can fix that, and
any benchmark that scores "acceleration accuracy" without accounting for it
is measuring the reconstruction, not the tracker.
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
    gt = bu.read_gt_frames(src, first, n)
    resid = []
    n_rec = n_true = 0
    for fn, rows in gt.items():
        truth = np.array([[r[1], r[2], r[3]] for r in rows if r[0] >= 0])
        p = src / "res" / f"rt_is.{fn}"
        if not p.exists():
            continue
        lines = p.read_text().strip().splitlines()
        rec = []
        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) >= 4:
                rec.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not rec or not len(truth):
            continue
        rec = np.array(rec)
        n_rec += len(rec)
        n_true += len(truth)
        # nearest true particle for each reconstructed point
        d = np.linalg.norm(rec[:, None, :] - truth[None, :, :], axis=2)
        j = np.argmin(d, axis=1)
        dist = d[np.arange(len(rec)), j]
        keep = dist < 1.0          # matched, not a ghost
        resid.append(rec[keep] - truth[j[keep]])

    r = np.concatenate(resid) if resid else np.zeros((0, 3))
    sigma = r.std(axis=0)
    sigma_c = float(np.sqrt((sigma**2).mean()))   # per-component rms

    # True acceleration scale, from the truth trajectories.
    tt = bu.build_true_tracks(gt, first)
    accs = []
    for pts in tt.values():
        pts = sorted(pts)
        f = np.array([p[0] for p in pts])
        x = np.array([[p[1], p[2], p[3]] for p in pts])
        for seg in np.split(x, np.where(np.diff(f) != 1)[0] + 1):
            if len(seg) >= 3:
                accs.append(np.diff(seg, 2, axis=0))
    a_rms = float(np.concatenate(accs).std()) if accs else float("nan")

    sigma_a = sigma_c * np.sqrt(6.0)   # dt = 1 frame

    print(f"\n=== {label} ===")
    print(f"  reconstructed pts / true pts   {n_rec} / {n_true}")
    print(f"  position noise per component   {sigma_c:.4f} mm  "
          f"(x/y/z: {sigma[0]:.4f} {sigma[1]:.4f} {sigma[2]:.4f})")
    print(f"  true accel rms                 {a_rms:.4f} mm/frame^2")
    print(f"  noise-induced accel rms        {sigma_a:.4f} mm/frame^2  "
          f"(= sqrt(6) * sigma_x, dt=1)")
    print(f"  ==> noise / signal in accel    {sigma_a / a_rms:.2f}")
