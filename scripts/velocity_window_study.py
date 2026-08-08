"""How reliable is a Savitzky-Golay velocity estimate vs window length,
and how much does position noise eat into it?

Uses the synthetic_turbulent GT tracks (true positions) and adds synthetic
triangulation noise sigma to simulate a "true experimental" dataset, then
measures the noise-amplification (RMSE of velocity estimate vs the clean
fixed-signal) as a function of SG window length.

Theory for the answer:
  - A single-frame difference v = (x_{t+1}-x_t)/dt has noise amp sqrt(2).
  - SG derivative smoothing replaces it; noise scales as ||c||/dt where c are
    the filter coefficients, falling roughly like 1/sqrt(effective N).
  - Tradeoff vs window W: lag/bias for accelerating flows grows ~ a*(W/2),
    so the optimum W is where position noise == smoothing truncation error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_utils import build_true_tracks, read_gt_frames


def main():
    tt = build_true_tracks(read_gt_frames())
    tracks = [np.array([p[1:] for p in pt]) for pt in tt.values()]
    tracks = [t for t in tracks if len(t) >= 5]
    print(f"tracks used: {len(tracks)} (len>=5)")

    sigmas = [0.0, 0.05, 0.1, 0.2, 0.5]
    windows = [3, 5, 7, 9, 11, 13]

    # "True" velocity per track: SG derivative (window 13, poly 3) on clean pos.
    trues = []
    for t in tracks:
        v = np.zeros_like(t)
        for d in range(3):
            v[:, d] = signal.savgol_filter(t[:, d], 13, 3, deriv=1, mode="interp")
        trues.append(v)

    # Signals: |v| magnitude, per-frame velocity-change (approx accel), so we
    # can judge error levels against physical scale.
    mags = [np.linalg.norm(np.diff(t, axis=0), axis=1) for t in tracks]
    dvels = [np.linalg.norm(np.diff(np.diff(t, axis=0), axis=0), axis=1) for t in tracks]
    print(f"signal scale: mean |v| = {np.mean([m.mean() for m in mags]):.3f} mm/frame, "
          f"mean per-frame |dv| = {np.mean([d.mean() for d in dvels]):.3f}")

    print("\nNOISE AMPLIFICATION = RMS(v_est(noisy,w) - v_est(clean,w))")
    print(f"{'sigma':>6} | " + " | ".join(f"w={w:>2}" for w in windows))
    rng = np.random.default_rng(42)
    for sigma in sigmas:
        cells = []
        for w in windows:
            rms_sq_num = 0.0
            rms_sq_den = 0
            for tr in tracks:
                noise = rng.normal(0, sigma, size=tr.shape)
                xn = tr + noise
                v = np.zeros_like(xn)
                vc = np.zeros_like(tr)
                for d in range(3):
                    p = min(3, w - 1)
                    v[:, d] = signal.savgol_filter(xn[:, d], w, p, deriv=1, mode="interp")
                    vc[:, d] = signal.savgol_filter(tr[:, d], w, p, deriv=1, mode="interp")
                err = v - vc
                rms_sq_num += float(np.sum(err ** 2))
                rms_sq_den += err.size
            cells.append(
                f"{np.sqrt(rms_sq_num / max(rms_sq_den, 1)):6.3f}"
            )
        print(f"{sigma:6.2} | " + " | ".join(cells))

    # Bias (smoothing truncation) on the clean signal vs wide-window reference.
    print("\nBIAS - clean signal estimator vs w=13 reference:")
    cells = []
    for w in windows:
        rms_num = 0.0
        rms_den = 0
        for tr, vtrue in zip(tracks, trues):
            v = np.zeros_like(tr)
            for d in range(3):
                p = min(3, w - 1)
                v[:, d] = signal.savgol_filter(tr[:, d], w, p, deriv=1, mode="interp")
            err = v - vtrue
            rms_num += float(np.sum(err ** 2))
            rms_den += err.size
        cells.append(f"{np.sqrt(rms_num / max(rms_den, 1)):6.3f}")
    print(f"{'0.0':>6} | " + " | ".join(cells))

    # Reference: single-frame central difference (no smoothing), sigma=0.1.
    print("\nsingle-frame velocity |dx| estimate noise (sigma=0.1):")
    rms_num = rms_den = 0
    for tr in tracks:
        noise = rng.normal(0, 0.1, size=tr.shape)
        xn = tr + noise
        d = np.diff(xn, axis=0)
        err = d - np.diff(tr, axis=0)
        rms_num += np.sum(err ** 2)
        rms_den += err.size
    print(f"  RMS 1-frame velocity error = {np.sqrt(rms_num/max(rms_den,1)):.3f}")


if __name__ == "__main__":
    main()
