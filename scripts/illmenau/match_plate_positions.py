"""Pair two camera groups' frames by WHERE the plate was, not by file number.

    !! ON THE ILLMENAU DATASET THIS CANNOT SUCCEED, AND THAT IS THE POINT. !!

The two groups measure DIFFERENT HALVES of the circular test section, in
separate acquisition sessions, so the plate never visited the same position
twice.  There is no correspondence to find.  Keep this script as the
demonstration of why, because the failure is instructive and very convincing:

* the position spreads match closely (front [5365, 2443, 3111] mm vs back
  [5399, 2422, 3093]), which looks like the same positions but is only the same
  PLACEMENT PROCEDURE repeated in each half;
* signature matching then returns a strongly structured pairing -- consecutive
  runs mapping to reversed consecutive runs -- that is the mirror symmetry of
  that procedure, not real point pairs;
* it is exposed only by the residual: 700-2500 mm, and 1633 mm of it ACROSS the
  plate normal while the along-normal part is already small.

Two further traps this script measures, worth knowing before trusting any
rig-to-rig registration:

* the front and back datum dots are NOT co-located -- there is an in-plane
  offset between the two printed patterns -- so anchoring on them is not
  anchoring on the same physical thing;
* the plate is always held vertical, so every plate rotation is a yaw about the
  same axis and the hand-eye translation is rank-deficient along it.  Measured:
  all 40 rotation axes within 7.1 deg of each other, system singular values
  6.62 / 2.05 / 0.05, worst-determined direction exactly vertical.

If two groups genuinely share a volume, tie them through the lab frame or a
target both can see -- not through plate positions.

    ILLMENAU_RAW=... python match_plate_positions.py [--max-pair-mm 40]

Writes <RAW>/plate_position_pairs.csv and prints the recovered transform.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from openptv2.algorithms.calibration import Calibration  # noqa: E402
from openptv2.algorithms.orientation import COORD_UNUSED  # noqa: E402
from openptv2.algorithms.parameters import ControlPar, MmNp  # noqa: E402
from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric  # noqa: E402
from openptv2.orientation import multi_cam_point_positions  # noqa: E402

RAW = Path(os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau"))
PITCH, NX, NY, DATUM = 120.0, 6, 7, (2, 3)
THICKNESS_MM = 6.0
MAX_PAIR_MM = 40.0
for _i, _a in enumerate(sys.argv):
    if _a == "--max-pair-mm":
        MAX_PAIR_MM = float(sys.argv[_i + 1])

RIGS = {
    "front": (RAW / "openptv_illmenau_4cam", [1, 2, 3, 4]),
    "back": (RAW / "openptv_illmenau_5678", [5, 6, 7, 8]),
}


def ideal_grid(ids):
    ids = np.asarray(ids)
    ix, iy = (ids - 1) % NX, (ids - 1) // NX
    return np.stack([(ix - DATUM[0]) * PITCH, (iy - DATUM[1]) * PITCH,
                     np.zeros(len(ids))], 1).astype(float)


def kabsch(A, B):
    """Rigid transform taking A onto B; returns (R, t) with B ~ A@R.T + t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = (U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    return R, cb - R @ ca


def datum_positions(folder, cams):
    """{frame: (datum xyz, plate normal, n dots, grid-fit residual)} for one rig."""
    cpar = ControlPar(num_cams=len(cams), imx=2560, imy=2048, pix_x=0.005,
                      pix_y=0.005, mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
                      chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
                      img_base_name=[""] * len(cams),
                      cal_img_base_name=[""] * len(cams))
    cals = []
    for n in cams:
        c = Calibration()
        c.from_file(str(folder / f"cal/cam{n}.tif.ori"),
                    str(folder / f"cal/cam{n}.tif.addpar"))
        cals.append(c)
    d = np.load(folder / "cal" / "labelled_all_frames.npz")
    views = {}
    for k in d.files:
        if k.endswith("_ids"):
            ci, fr, _ = k.split("_")
            views[(int(ci[1:]), fr)] = (d[k], d[f"{ci}_{fr}_px"])

    out = {}
    for fr in sorted({f for _, f in views}):
        per = {ci: dict(zip(views[(ci, fr)][0].tolist(), views[(ci, fr)][1].tolist()))
               for ci in range(len(cams)) if (ci, fr) in views}
        ids = [i for i in sorted({i for m in per.values() for i in m})
               if sum(i in m for m in per.values()) >= 2]
        if len(ids) < 8:
            continue
        t = np.full((len(ids), len(cams), 2), COORD_UNUSED)
        for k, pid in enumerate(ids):
            for ci, m in per.items():
                if pid in m:
                    mx, my = pixel_to_metric(m[pid][0], m[pid][1], cpar)
                    a = cals[ci].added_par
                    t[k, ci] = dist_to_flat(mx, my, cals[ci].int_par.xh,
                                            cals[ci].int_par.yh, a.k1, a.k2, a.k3,
                                            a.p1, a.p2, a.scx, a.she)
        pos, _ = multi_cam_point_positions(t, cpar, cals)
        ok = np.isfinite(pos).all(1) & (np.abs(pos) < 1e5).all(1)
        pos, ids = pos[ok], np.array(ids)[ok]
        if len(pos) < 8:
            continue
        # rigid ideal grid fitted to the dots -> the datum is the fitted origin
        G = ideal_grid(ids)
        R, tr = kabsch(G, pos)
        resid = float(np.sqrt(np.mean(np.sum((G @ R.T + tr - pos) ** 2, 1))))
        out[fr] = (tr, R[:, 2], len(pos), resid)      # grid origin, plate normal
    return out


def signature(P, k):
    """Sorted distances from each point to the k nearest others -- pose-invariant."""
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    S = np.sort(D, axis=1)[:, 1:k + 1]
    return S


print("triangulating the datum dot of every frame in both rigs ...")
data = {}
for tag, (folder, cams) in RIGS.items():
    data[tag] = datum_positions(folder, cams)
    print(f"  {tag:5s}  {len(data[tag])} frames from {folder.name}")

fa = sorted(data["front"])
fb = sorted(data["back"])
PA = np.array([data["front"][f][0] for f in fa])
PB = np.array([data["back"][f][0] for f in fb])
k = min(len(fa), len(fb)) - 1

print(f"\nspread of the plate positions: front {np.ptp(PA, axis=0).round(0)} mm, "
      f"back {np.ptp(PB, axis=0).round(0)} mm")

# ---- correspondence from pose-invariant distance signatures
SA, SB = signature(PA, k), signature(PB, k)
cost = np.linalg.norm(SA[:, None, :] - SB[None, :, :], axis=2)
ri, ci = linear_sum_assignment(cost)
print(f"signature matching: {len(ri)} candidate pairs, cost median "
      f"{np.median(cost[ri, ci]):.1f} mm, max {cost[ri, ci].max():.1f} mm")

# ---- transform from the matched points, then refine by rejecting outliers
keep = np.ones(len(ri), bool)
for it in range(5):
    R, t = kabsch(PB[ci[keep]], PA[ri[keep]])
    err = np.linalg.norm(PB[ci] @ R.T + t - PA[ri], axis=1)
    nxt = err < max(MAX_PAIR_MM, 3 * np.median(err[keep]))
    print(f"  round {it}: {keep.sum():3d} pairs, residual median "
          f"{np.median(err[keep]):7.2f} mm, max {err[keep].max():8.2f} mm")
    if nxt.sum() == keep.sum():
        keep = nxt
        break
    keep = nxt

R, t = kabsch(PB[ci[keep]], PA[ri[keep]])
err = np.linalg.norm(PB[ci] @ R.T + t - PA[ri], axis=1)
ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
axis = axis / max(np.linalg.norm(axis), 1e-12)

print(f"\n{keep.sum()}/{len(ri)} pairs accepted (residual < "
      f"{max(MAX_PAIR_MM, 3*np.median(err[keep])):.1f} mm)")
print("\nS = (front world <- back world)")
print(f"  rotation {ang:.3f} deg about ({axis[0]:+.4f},{axis[1]:+.4f},{axis[2]:+.4f})")
print(f"  translation ({t[0]:+9.2f},{t[1]:+9.2f},{t[2]:+9.2f}) mm")
print(f"  residual over accepted pairs: median {np.median(err[keep]):.2f} mm, "
      f"max {err[keep].max():.2f} mm")

# the front and back datum dots sit on the same normal, one plate thickness apart,
# so the residual should be about THICKNESS_MM and should lie along the normal
nrm = np.array([data["front"][fa[i]][1] for i in ri[keep]])
delta = (PB[ci[keep]] @ R.T + t) - PA[ri[keep]]
along = np.abs(np.sum(delta * nrm, axis=1))
across = np.linalg.norm(delta - nrm * np.sum(delta * nrm, axis=1)[:, None], axis=1)
print(f"\nresidual split about the plate normal (expected ~{THICKNESS_MM:.0f} mm along, "
      "~0 across):")
print(f"  along  the normal: median {np.median(along):6.2f} mm   "
      f"[{along.min():.2f} .. {along.max():.2f}]")
print(f"  across the normal: median {np.median(across):6.2f} mm   "
      f"[{across.min():.2f} .. {across.max():.2f}]")

dst = RAW / "plate_position_pairs.csv"
with dst.open("w") as fh:
    fh.write("front_frame,back_frame,accepted,residual_mm,along_normal_mm,"
             "front_x,front_y,front_z,back_x,back_y,back_z\n")
    j = 0
    for m in range(len(ri)):
        a, b = fa[ri[m]], fb[ci[m]]
        pa_, pb_ = PA[ri[m]], PB[ci[m]]
        al = ""
        if keep[m]:
            al = f"{along[j]:.3f}"
            j += 1
        fh.write(f"{a},{b},{int(keep[m])},{err[m]:.3f},{al},"
                 + ",".join(f"{v:.2f}" for v in (*pa_, *pb_)) + "\n")
print(f"\nwrote {dst}")

same = sum(1 for m in range(len(ri)) if fa[ri[m]] == fb[ci[m]])
print(f"\n{same}/{len(ri)} pairs have the SAME frame number in both rigs "
      f"-- the file numbering is {'consistent' if same == len(ri) else 'NOT a valid pairing'}")
for m in range(min(len(ri), 60)):
    if fa[ri[m]] != fb[ci[m]] and keep[m]:
        print(f"    front {fa[ri[m]]}  <->  back {fb[ci[m]]}   ({err[m]:.1f} mm)")
