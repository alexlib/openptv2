"""Minimal controlled synthetic test: two 2D Gaussian PSF blobs, known ground
truth (always 2 distinct particles), swept across separation distance and
brightness ratio. Compares openptv2's compiled targ_rec against a literal
Python transliteration of 3dptv.exe's actual targ_rec C source, to find
exactly where the two diverge on "is this 1 blob or 2".

See docs/plans/2026-08-27-verified-pipeline-ghost-particle-study-plan.md,
Phase 1 step 4 (revised approach).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from literal_3dptv_targ_rec import literal_3dptv_targ_rec  # noqa: E402

from openptv2.algorithms.segmentation import targ_rec  # noqa: E402

IMSIZE = 64
PARAMS = dict(
    gvthres=10,
    discont=20,
    nnmin=2,
    nnmax=200,
    nxmin=1,
    nxmax=15,
    nymin=2,
    nymax=15,
    sumg_min=20,
)


def make_two_blob_image(sep_px, sigma, amp1, amp2, size=IMSIZE):
    yy, xx = np.mgrid[0:size, 0:size]
    cx1, cy = size / 2 - sep_px / 2, size / 2
    cx2 = size / 2 + sep_px / 2
    img = amp1 * np.exp(
        -((xx - cx1) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
    ) + amp2 * np.exp(-((xx - cx2) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return np.clip(img, 0, 255).astype(np.uint8)


def count_ours(img):
    ts = targ_rec(img, **PARAMS)
    if len(ts) == 1 and ts[0].pnr == 1 and ts[0].x == 1 and ts[0].y == 1:
        return 0  # empty-image sentinel, see targ_rec docstring
    return len(ts)


def count_3dptv(img):
    p = dict(PARAMS)
    p["disco"] = p.pop("discont")
    ts = literal_3dptv_targ_rec(img, **p)
    return len(ts)


print("Sweep A: separation distance (fixed sigma=2.0px, equal brightness amp=200)")
print(f"{'sep_px':>8} {'ours_n':>7} {'3dptv_n':>8}  divergence")
for sep in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    img = make_two_blob_image(sep, sigma=2.0, amp1=200, amp2=200)
    n_ours = count_ours(img)
    n_3dptv = count_3dptv(img)
    flag = "  <-- DIVERGES" if n_ours != n_3dptv else ""
    print(f"{sep:8d} {n_ours:7d} {n_3dptv:8d}{flag}")

print("\nSweep B: brightness ratio (fixed sep=6px, sigma=2.0px, amp1=200 fixed)")
print(f"{'amp2':>8} {'ours_n':>7} {'3dptv_n':>8}  divergence")
for amp2 in [200, 150, 100, 60, 40, 25, 15]:
    img = make_two_blob_image(6, sigma=2.0, amp1=200, amp2=amp2)
    n_ours = count_ours(img)
    n_3dptv = count_3dptv(img)
    flag = "  <-- DIVERGES" if n_ours != n_3dptv else ""
    print(f"{amp2:8d} {n_ours:7d} {n_3dptv:8d}{flag}")

print("\nSweep C: PSF width sigma (fixed sep=6px, equal brightness amp=200)")
print(f"{'sigma':>8} {'ours_n':>7} {'3dptv_n':>8}  divergence")
for sigma in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    img = make_two_blob_image(6, sigma=sigma, amp1=200, amp2=200)
    n_ours = count_ours(img)
    n_3dptv = count_3dptv(img)
    flag = "  <-- DIVERGES" if n_ours != n_3dptv else ""
    print(f"{sigma:8.1f} {n_ours:7d} {n_3dptv:8d}{flag}")
