# ruff: noqa
"""Does the blob detector return the true sub-pixel centre? (the step between an image and every calibration)

    uv run python bench/calibration_lab/detection_bias.py

Isolated Gaussian spots at random sub-pixel centres (convention of the library: pixel (i, j) is centred at i + 0.5, so a spot
drawn at index position c is expected at c + 0.5), uint8 images with Gaussian read noise, detected with ``targ_rec`` and the
syn1 detection settings. Per spot size, peak height and noise: mean bias (px), rms error, and the pixel-locking signature
(the amplitude of the bias as a function of the fractional part of the position: a smooth detector gives ~0).
"""
import numpy as np
from openptv2.algorithms.segmentation import targ_rec

PAR = dict(gvthres=10, discont=20, nnmin=2, nnmax=200, nxmin=1, nxmax=15, nymin=2, nymax=15, sumg_min=20)
rng = np.random.default_rng(0)


def render(centres, sigma, amp, noise, shape=(512, 512)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.zeros(shape)
    for cx, cy in centres:                         # continuous coordinate; pixel i covers [i, i+1), centre i + 0.5
        x0, y0 = int(cx) - 8, int(cy) - 8
        sl = (slice(max(y0, 0), y0 + 17), slice(max(x0, 0), x0 + 17))
        px, py = xx[sl] + 0.5, yy[sl] + 0.5
        img[sl] += amp * np.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2 * sigma**2))
    img += rng.normal(0, noise, shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


print(f"{'sigma':>5} {'peak':>5} {'noise':>5} {'n':>5} {'found':>6} | {'bias x':>8} {'bias y':>8} {'rms err':>8} | {'locking amp (px)':>17}")
for sigma in (0.8, 1.0, 1.5, 2.0, 3.0):
    for amp in (60, 255):
        for noise in (0.0, 3.0):
            gx, gy = np.meshgrid(np.arange(20, 492, 24), np.arange(20, 492, 24))
            base = np.c_[gx.ravel(), gy.ravel()].astype(float)
            err_l, fr_l = [], []
            found = n = 0
            for rep in range(4):
                c = base + rng.uniform(0.0, 1.0, base.shape)
                img = render(c, sigma, amp, noise)
                t = targ_rec(np.ascontiguousarray(img), PAR["gvthres"], PAR["discont"], PAR["nnmin"], PAR["nnmax"], PAR["nxmin"], PAR["nxmax"], PAR["nymin"], PAR["nymax"], PAR["sumg_min"])
                n += len(c)
                if not len(t):
                    continue
                d = np.array([[tt.x(), tt.y()] for tt in t])
                from scipy.spatial import cKDTree
                dist, j = cKDTree(c).query(d)
                ok = dist < 1.5
                found += int(ok.sum())
                err_l.append(d[ok] - c[j[ok]]); fr_l.append(c[j[ok]] % 1.0)
            if not err_l:
                print(f"{sigma:>5} {amp:>5} {noise:>5} {n:>5} {0:>6} | not detected")
                continue
            e, fr = np.concatenate(err_l), np.concatenate(fr_l)
            # pixel locking: first harmonic of the bias against the fractional position, per axis
            lock = []
            for k in (0, 1):
                ph = 2 * np.pi * fr[:, k]
                lock.append(np.hypot(2 * np.mean(e[:, k] * np.cos(ph)), 2 * np.mean(e[:, k] * np.sin(ph))))
            print(f"{sigma:>5} {amp:>5} {noise:>5} {n:>5} {found:>6} | {e[:, 0].mean():>8.4f} {e[:, 1].mean():>8.4f} {np.sqrt(np.mean(np.sum(e**2, 1))):>8.4f} | {max(lock):>17.4f}")
