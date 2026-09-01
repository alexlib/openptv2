"""Triangulate EVERY calibration frame with the final .ori/.addpar and separate
CALIBRATION quality from LABELLING quality.

The reference frame is the one the world is anchored to, so it is expected to be
near-perfect; the value of this check is the other 47 frames, which the
calibration never saw as a rigid target.

The discriminator is the per-dot ray-convergence miss (RCM) returned by
multi_cam_point_positions: the closest-approach distance between the sight lines
of the cameras that saw the dot.  A mislabelled dot pairs ray A of one physical
dot with ray B of a different one, so the rays do not meet and RCM explodes.  A
correctly labelled dot has RCM at the calibration's noise floor whatever the
plate's depth.  RCM uses no plate model at all, so it cannot be fooled by the
grid, and it is the only per-dot signal that tells the two failures apart.

Reported per frame: all dots, then the RCM-inlier subset.  If the inlier
planarity stays sub-millimetre at every plate distance, the calibration model
holds over the whole volume and the wide-spread numbers are the labeller.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np

from openptv2.algorithms.orientation import COORD_UNUSED
from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
from openptv2.orientation import multi_cam_point_positions

out = CFG.DIR
PITCH, NX, REF = CFG.PITCH, CFG.NX, CFG.REF
RCM_TOL = 1.0  # [mm] a dot whose sight lines miss by more than this is not one dot

cpar = CFG.control_par()
cals = CFG.load_calibrations()

views = CFG.load_views()


def triangulate(fr):
    """-> positions, point ids, per-dot ray-convergence miss [mm], n cameras."""
    per = {
        ci: dict(zip(views[(ci, fr)][0].tolist(), views[(ci, fr)][1].tolist()))
        for ci in range(CFG.NCAM)
        if (ci, fr) in views
    }
    ids = [
        i
        for i in sorted({i for m in per.values() for i in m})
        if sum(i in m for m in per.values()) >= 2
    ]
    if len(ids) < 12:
        return None, None, None, 0
    t = np.full((len(ids), CFG.NCAM, 2), COORD_UNUSED)
    for k, pid in enumerate(ids):
        for ci, m in per.items():
            if pid in m:
                mx, my = pixel_to_metric(m[pid][0], m[pid][1], cpar)
                a = cals[ci].added_par
                t[k, ci] = dist_to_flat(
                    mx,
                    my,
                    cals[ci].int_par.xh,
                    cals[ci].int_par.yh,
                    a.k1,
                    a.k2,
                    a.k3,
                    a.p1,
                    a.p2,
                    a.scx,
                    a.she,
                )
    pos, rcm = multi_cam_point_positions(t, cpar, cals)
    ok = np.isfinite(pos).all(1) & (np.abs(pos) < 1e5).all(1) & np.isfinite(rcm)
    return pos[ok], [p for p, k in zip(ids, ok) if k], np.asarray(rcm)[ok], len(per)


def plane_and_pitch(pos, ids):
    """Planarity RMS about the best-fit plane, plus median X and Y pitch."""
    c = pos.mean(0)
    try:
        n = np.linalg.svd(pos - c)[2][2]
    except np.linalg.LinAlgError:
        return None
    resid = (pos - c) @ n
    idx = {p: k for k, p in enumerate(ids)}
    dx = [
        np.linalg.norm(pos[idx[p]] - pos[idx[p + 1]])
        for p in ids
        if p + 1 in idx and ((p - 1) % NX) < NX - 1
    ]
    dy = [np.linalg.norm(pos[idx[p]] - pos[idx[p + NX]]) for p in ids if p + NX in idx]
    if not dx or not dy:
        return None
    return (
        float(np.sqrt(np.mean(resid**2))),
        float(np.abs(resid).max()),
        float(np.median(dx)),
        float(np.median(dy)),
        float(np.linalg.norm(c)),
    )


print(
    "                 ALL dots                |        RCM-inlier dots (rays converge < "
    f"{RCM_TOL} mm)\nframe       n  plane RMS  pitchX  pitchY  |  n   plane RMS / max   pitchX   "
    "pitchY   RCM med  dist from ref\n" + "-" * 118
)
rows = []
for fr in sorted({f for _, f in views}):
    pos, ids, rcm, ncam = triangulate(fr)
    if pos is None or ncam < 2:
        continue
    allf = plane_and_pitch(pos, ids)
    keep = rcm < RCM_TOL
    inl = (
        plane_and_pitch(pos[keep], [p for p, k in zip(ids, keep) if k])
        if keep.sum() >= 8
        else None
    )
    if allf is None or inl is None:
        print(f"{fr}  {len(ids):3d}  -- too few converging dots ({int(keep.sum())}) --")
        continue
    rows.append((fr, len(ids), int(keep.sum()), *inl, float(np.median(rcm))))
    print(
        f"{fr}  {len(ids):3d}  {allf[0]:8.3f} {allf[2]:7.2f} {allf[3]:7.2f}  | "
        f"{int(keep.sum()):3d}  {inl[0]:7.3f} /{inl[1]:7.3f}  {inl[2]:8.3f} {inl[3]:8.3f}"
        f"  {np.median(rcm):8.3f}  {inl[4]:9.1f}"
    )

r = np.array([(x[2], x[3], x[5], x[6], x[7], x[8]) for x in rows])
n_in, rms_in, px_in, py_in, dist, rcm_med = r.T
print(
    f"\n{len(rows)} frames, {int(n_in.sum())}/{sum(x[1] for x in rows)} dots kept by the "
    f"RCM<{RCM_TOL} mm gate"
)
print(
    f"inlier planarity RMS : median {np.median(rms_in):.3f} mm, p90 "
    f"{np.percentile(rms_in, 90):.3f} mm, max {rms_in.max():.3f} mm"
)
print(
    f"inlier pitch         : X {np.median(px_in):.3f} mm ({100 * (np.median(px_in) / PITCH - 1):+.2f} %), "
    f"Y {np.median(py_in):.3f} mm ({100 * (np.median(py_in) / PITCH - 1):+.2f} %)"
)
print(f"per-dot RCM          : median of frame medians {np.median(rcm_med):.3f} mm")

print(
    "\nis the calibration depth-dependent?  inliers only, binned by plate distance"
    "\n  distance from ref plane   frames   dots   planarity RMS med / max   pitchX   RCM med"
)
edges = [0, 1000, 2000, 3000, 4000, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dist >= lo) & (dist < hi)
    if not m.any():
        continue
    print(
        f"  {lo:5.0f} - {hi if hi < 1e8 else 99999:5.0f} mm       {m.sum():4d}   {int(n_in[m].sum()):4d}"
        f"   {np.median(rms_in[m]):8.3f} /{rms_in[m].max():7.3f}  {np.median(px_in[m]):8.3f}"
        f"  {np.median(rcm_med[m]):7.3f}"
    )
