"""Determine the ONE shared cc from multi-plane consistency.

For a trial cc: fit each camera's pose on the reference frame, then for every
other frame ask each camera separately where the plate is.  If cc is right the
four answers coincide
if cc is wrong each camera's world frame sits at the
wrong distance and the answers spread apart, the more so the further the plate
is from the reference plane.  Minimise that spread.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import cv2
import numpy as np

out = CFG.DIR
PITCH, NX, PIX, IMX, IMY, REF = (CFG.PITCH, CFG.NX, CFG.PIX, CFG.IMX, CFG.IMY, CFG.REF)
views = CFG.load_views()

obj_of = CFG.obj_of


def pose(K, ids, px):
    o = obj_of(ids)
    if len(o) < 6:
        return None
    ok, rv, tv = cv2.solvePnP(o, px.astype(np.float64), K, np.zeros(5))
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(o, px.astype(np.float64), K, np.zeros(5), rv, tv)
    rep, _ = cv2.projectPoints(o, rv, tv, K, np.zeros(5))
    return rv, tv, float(np.sqrt(np.mean(np.sum((rep.reshape(-1, 2) - px) ** 2, 1))))


frames = [
    f
    for f in sorted({f for _, f in views})
    if all((ci, f) in views and len(views[(ci, f)][0]) >= 12 for ci in range(CFG.NCAM))
]


def spread(cc, detail=False):
    K = np.array([[cc / PIX, 0, IMX / 2], [0, cc / PIX, IMY / 2], [0, 0, 1.0]])
    ref = {}
    for ci in range(CFG.NCAM):
        p = pose(K, *views[(ci, REF)])
        if p is None:
            return np.inf, []
        ref[ci] = p
    rows = []
    for fr in frames:
        if fr == REF:
            continue
        ts = []
        for ci in range(CFG.NCAM):
            p = pose(K, *views[(ci, fr)])
            if p is None or p[2] > 1.5:
                ts = None
                break
            R0, _ = cv2.Rodrigues(ref[ci][0])
            R1, _ = cv2.Rodrigues(p[0])
            ts.append((R0.T @ (p[1] - ref[ci][1])).ravel())
        if ts is None or len(ts) < 4:
            continue
        ts = np.array(ts)
        rows.append(
            (
                fr,
                np.linalg.norm(ts.mean(0)),
                float(np.max(np.linalg.norm(ts - ts.mean(0), axis=1))),
            )
        )
    if not rows:
        return np.inf, []
    return float(np.median([r[2] for r in rows])), rows


print("  cc [mm]   frames   median cross-camera spread of the plate centre [mm]")
best = None
for cc in np.arange(7.6, 11.61, 0.20):
    s, rows = spread(float(cc))
    print(f"   {cc:5.2f}     {len(rows):3d}      {s:10.2f}")
    if best is None or s < best[0]:
        best = (s, float(cc))
lo, hi = best[1] - 0.20, best[1] + 0.20
for _ in range(30):
    # golden-ish bisection on the 1-D curve
    m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
    if spread(m1)[0] < spread(m2)[0]:
        hi = m2
    else:
        lo = m1
cc = 0.5 * (lo + hi)
s, rows = spread(cc, True)
print(
    f"\nbest shared cc = {cc:.4f} mm   median spread {s:.2f} mm "
    f"(was {spread(9.44)[0]:.2f} mm at cc=9.44)"
)
np.save(out / "cal" / "fitted_cc.npy", cc)
print(
    "\nframe      plate distance from reference plane [mm]   cross-camera spread [mm]"
)
for fr, dist, sp in sorted(rows, key=lambda r: r[1])[:14]:
    print(f"{fr}          {dist:10.1f}                        {sp:8.2f}")
