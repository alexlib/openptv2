"""Measure the tracker's actual speed ceiling against its configured one.

Synthetic particles move in straight lines at a sweep of known speeds. Every
particle is isolated (no two are close enough to contest a candidate), so the
ONLY thing that can break a link is the search/gate chain -- not conflict
resolution. We then ask, per speed: did the tracker link this particle all the
way through?

The configured ceiling is track.par's dvxmax (mm/frame). If measured links die
well below it, the search chain is losing fast particles and the gap is a bug,
not a parameter choice.

Run from repo root:
    uv run python scripts/diag_speed_ceiling.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.algorithms.parameters import ControlPar

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "test_data" / "track"
NCAMS = 2
FIRST = 10001


def write_targets(path: Path, rows) -> None:
    lines = [str(len(rows))]
    for t in rows:
        lines.append(
            f"{t[0]:4d}  {t[1]:10.4f}  {t[2]:10.4f}  "
            f"{t[3]:6d}  {t[4]:4d}  {t[5]:4d}  {t[6]:6d}  {t[7]:4d}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_rt_is(path: Path, rows) -> None:
    lines = [str(len(rows))]
    for r in rows:
        lines.append(
            f"{r[0]:4d}  {r[1]:12.6f}  {r[2]:12.6f}  {r[3]:12.6f}  "
            f"{r[4]:5d}  {r[5]:5d}  {r[6]:5d}  {r[7]:5d}"
        )
    path.write_text("\n".join(lines) + "\n")


def build_scene(work: Path, speeds, n_frames, jitter=0.0, seed=42):
    """Write targets + rt_is for straight-line particles at the given speeds.

    Returns {particle_index: {frame: (x, y, z)}} for the particles that stayed
    visible in every frame (the only ones we score).
    """
    rng = np.random.default_rng(seed)
    cpar = ControlPar.from_file(str(SRC / "parameters" / "ptv.par"))
    cals = [
        Calibration.from_file(
            str(SRC / "cal" / f"cam{c}.tif.ori"),
            str(SRC / "cal" / f"cam{c}.tif.addpar"),
        )
        for c in range(1, NCAMS + 1)
    ]

    img = work / "img"
    res = work / "res"
    img.mkdir(exist_ok=True)
    res.mkdir(exist_ok=True)

    # Spread particles along y so they never compete for the same candidate.
    # Start x low enough that even the fastest stays in the measured volume.
    truth = {
        i: {
            FIRST + f: (10.0 + v * f, -18.0 + 3.0 * i, 40.0)
            for f in range(n_frames)
        }
        for i, v in enumerate(speeds)
    }

    visible_everywhere = set(truth)
    projected = {}

    for f in range(n_frames):
        frame = FIRST + f
        per_cam = {c: [] for c in range(NCAMS)}
        corres = []
        for pid in sorted(truth):
            x, y, z = truth[pid][frame]
            pnrs = []
            for ci in range(NCAMS):
                xy = img_coord_batch(
                    np.array([[x, y, z]], dtype=np.float64), cals[ci], cpar.mm
                )[0]
                px = xy[0] / cpar.pix_x + cpar.imx / 2
                py = cpar.imy / 2 - xy[1] / cpar.pix_y
                if jitter:
                    px += rng.normal(0.0, jitter)
                    py += rng.normal(0.0, jitter)
                if not (0 <= px < cpar.imx and 0 <= py < cpar.imy):
                    pnrs.append(-1)
                    continue
                # tnr must equal the particle index for candidate sorting.
                per_cam[ci].append((pid, px, py, 50, 5, 5, 1000, pid))
                pnrs.append(pid)
            while len(pnrs) < 4:
                pnrs.append(-1)
            if sum(1 for p in pnrs[:NCAMS] if p >= 0) < 2:
                visible_everywhere.discard(pid)
                continue
            corres.append(pnrs)
        projected[frame] = corres

    # rt_is rows must be renumbered per frame AND the per-camera target index
    # must point at the row's position in that camera's target list -- rebuild
    # both now that we know which particles survived.
    for f in range(n_frames):
        frame = FIRST + f
        per_cam = {c: [] for c in range(NCAMS)}
        rows = []
        for pid in sorted(visible_everywhere):
            x, y, z = truth[pid][frame]
            pnrs = []
            for ci in range(NCAMS):
                xy = img_coord_batch(
                    np.array([[x, y, z]], dtype=np.float64), cals[ci], cpar.mm
                )[0]
                px = xy[0] / cpar.pix_x + cpar.imx / 2
                py = cpar.imy / 2 - xy[1] / cpar.pix_y
                if jitter:
                    px += rng.normal(0.0, jitter)
                    py += rng.normal(0.0, jitter)
                idx = len(per_cam[ci])
                per_cam[ci].append((idx, px, py, 50, 5, 5, 1000, len(rows)))
                pnrs.append(idx)
            while len(pnrs) < 4:
                pnrs.append(-1)
            rows.append((len(rows) + 1, x, y, z, *pnrs))

        write_rt_is(res / f"rt_is.{frame}", rows)
        for ci in range(NCAMS):
            write_targets(img / f"cam{ci + 1}.{frame}_targets", per_cam[ci])

    return {p: truth[p] for p in sorted(visible_everywhere)}


def read_ptv_is(path: Path):
    """Return the `prev` link column: prev[i] is row i's parent in frame-1."""
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    if not lines:
        return []
    n = int(lines[0])
    return [int(line.split()[0]) for line in lines[1 : n + 1]]


def run_tracker(work: Path):
    import os

    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / "parameters_Run1.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        tracker = Tracker(cpar, vpar, track_par, spar, cals, default_naming)
        tracker.full_forward()
        return tracker.npart, tracker.nlinks
    finally:
        os.chdir(old)


def score(work: Path, truth, n_frames):
    """Per particle: how many of the n_frames-1 possible links it actually got.

    Rows in our synthetic rt_is are in particle order and stable across frames,
    so row i in every frame is the same particle -- a correct link at frame f is
    prev[i] == i.
    """
    pids = sorted(truth)
    got = dict.fromkeys(pids, 0)
    for f in range(1, n_frames):
        prev = read_ptv_is(work / "res" / f"ptv_is.{FIRST + f}")
        for i, p in enumerate(pids):
            if i < len(prev) and prev[i] == i:
                got[p] += 1
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--vmax", type=float, default=20.0)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--jitter", type=float, default=0.0, help="pixel noise sigma")
    ap.add_argument("--out", type=Path, default=REPO / "scratch" / "speed_ceiling.png")
    args = ap.parse_args()

    work = REPO / "scratch" / "_speed_ceiling_run"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copytree(SRC / "cal", work / "cal")
    shutil.copytree(SRC / "parameters", work / "parameters")
    shutil.copy(SRC / "parameters_Run1.yaml", work / "parameters_Run1.yaml")

    speeds = np.linspace(0.5, args.vmax, args.n)
    truth = build_scene(work, speeds, args.frames, jitter=args.jitter)
    print(f"{len(truth)}/{len(speeds)} particles visible in all frames")

    # Point the sequence at our synthetic frames.
    import yaml

    ycfg = yaml.safe_load((work / "parameters_Run1.yaml").read_text())
    for key in ("sequence", "Sequence", "sequence_par"):
        if key in ycfg:
            ycfg[key]["first"] = FIRST
            ycfg[key]["last"] = FIRST + args.frames - 1
    (work / "parameters_Run1.yaml").write_text(yaml.safe_dump(ycfg))

    npart, nlinks = run_tracker(work)
    print(f"tracker: npart={npart:.1f} nlinks={nlinks:.1f}")

    got = score(work, truth, args.frames)
    possible = args.frames - 1
    dvxmax = float((SRC / "parameters" / "track.par").read_text().split()[1])

    print(f"\n  speed(mm/frame)   links {possible} possible   configured max={dvxmax}")
    xs, ys = [], []
    for pid in sorted(truth):
        v = speeds[pid]
        frac = got[pid] / possible
        xs.append(v)
        ys.append(frac)
        bar = "#" * int(round(frac * 30))
        print(f"  {v:8.2f}        {got[pid]:2d}/{possible}  {bar:<30} {frac:5.0%}")

    plot(xs, ys, dvxmax, args.out)
    print(f"\nwrote {args.out}")


def plot(xs, ys, dvxmax, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#2a6ebb", label="measured link fraction")
    ax.axvline(dvxmax, color="#c0392b", ls="--", label=f"configured dvxmax = {dvxmax}")
    ax.set_xlabel("particle speed (mm/frame)")
    ax.set_ylabel("fraction of links made")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Tracker speed ceiling: measured vs configured")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
