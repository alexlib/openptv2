"""A/B the Phase-3 "loser retry" link resolution, with statistics.

Phase 3 (track_kernels_corr.py) lets a particle that lost a contested candidate
claim its next-best free candidate. The original 3dptv track.c drops that
particle instead. This script runs BOTH on identical input and asks whether the
difference is real and which way it points.

Two modes:

  synthetic  curved trajectories + 3D position noise + ground truth.
             Noise is the point: real PTV positions jitter either side of the
             smooth path, so per-frame angle/acceleration are noise-dominated
             and the acc/angle gate fires on noise, not physics. Straight clean
             lines hide exactly the regime we care about.
             Verdict metric: link precision / recall vs truth.

  folder     a real working folder (res/rt_is.*, cal/, parameters_Run1.yaml).
             No ground truth, so "better" is not directly observable -- we
             report paired per-frame differences in link count and in the
             acceleration tail (a wrong link injects a spurious large
             acceleration, so a growing tail is evidence of bad links).

Run from repo root:
    uv run python scripts/ab_loser_retry.py synthetic
    uv run python scripts/ab_loser_retry.py folder --path <working-folder>
"""

from __future__ import annotations

import argparse
import os
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


# ----------------------------------------------------------------- synthetic


def make_tracks(n, n_frames, noise_mm, seed, speed=4.0, curve=0.35):
    """Curved trajectories + isotropic 3D position noise.

    Returns (truth_clean, observed) as {pid: {frame: (x,y,z)}}. `observed` is
    what goes into rt_is -- the noisy positions a real triangulation produces.
    """
    rng = np.random.default_rng(seed)
    truth, obs = {}, {}
    for pid in range(n):
        # Start on a loose grid so search volumes overlap and contests happen.
        x0 = 10.0 + 14.0 * (pid % 5)
        y0 = -16.0 + 7.0 * (pid // 5)
        z0 = 35.0 + 4.0 * ((pid * 7) % 3)
        phase = rng.uniform(0, 2 * np.pi)
        truth[pid], obs[pid] = {}, {}
        for f in range(n_frames):
            t = float(f)
            x = x0 + speed * t
            y = y0 + curve * speed * np.sin(0.45 * t + phase) * 3.0
            z = z0 + curve * speed * np.cos(0.30 * t + phase) * 2.0
            truth[pid][FIRST + f] = (x, y, z)
            obs[pid][FIRST + f] = (
                x + rng.normal(0, noise_mm),
                y + rng.normal(0, noise_mm),
                z + rng.normal(0, noise_mm),
            )
    return truth, obs


def write_scene(work: Path, obs, n_frames):
    """Project observed positions to targets + rt_is. Row order == pid order."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diag_speed_ceiling import write_rt_is, write_targets

    cpar = ControlPar.from_file(str(SRC / "parameters" / "ptv.par"))
    cals = [
        Calibration.from_file(
            str(SRC / "cal" / f"cam{c}.tif.ori"),
            str(SRC / "cal" / f"cam{c}.tif.addpar"),
        )
        for c in range(1, NCAMS + 1)
    ]
    img, res = work / "img", work / "res"
    img.mkdir(exist_ok=True)
    res.mkdir(exist_ok=True)

    pids = sorted(obs)
    for f in range(n_frames):
        frame = FIRST + f
        per_cam = {c: [] for c in range(NCAMS)}
        rows = []
        for pid in pids:
            x, y, z = obs[pid][frame]
            pnrs = []
            for ci in range(NCAMS):
                xy = img_coord_batch(
                    np.array([[x, y, z]], dtype=np.float64), cals[ci], cpar.mm
                )[0]
                px = xy[0] / cpar.pix_x + cpar.imx / 2
                py = cpar.imy / 2 - xy[1] / cpar.pix_y
                idx = len(per_cam[ci])
                per_cam[ci].append((idx, px, py, 50, 5, 5, 1000, len(rows)))
                pnrs.append(idx)
            while len(pnrs) < 4:
                pnrs.append(-1)
            rows.append((len(rows) + 1, x, y, z, *pnrs))
        write_rt_is(res / f"rt_is.{frame}", rows)
        for ci in range(NCAMS):
            write_targets(img / f"cam{ci + 1}.{frame}_targets", per_cam[ci])
    return pids


# ---------------------------------------------------------------- run / read


def run(work: Path, loser_retry: int, n_frames: int):
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
        t = Tracker(
            cpar, vpar, track_par, spar, cals, default_naming,
            loser_retry=loser_retry,
        )
        t.full_forward()
    finally:
        os.chdir(old)


def read_links(res: Path, frame: int):
    """prev column of ptv_is.<frame>: prev[i] is row i's parent in frame-1."""
    p = res / f"ptv_is.{frame}"
    if not p.exists():
        return []
    lines = p.read_text().strip().splitlines()
    if not lines:
        return []
    n = int(lines[0])
    return [int(line.split()[0]) for line in lines[1 : n + 1]]


def read_xyz(res: Path, frame: int):
    p = res / f"rt_is.{frame}"
    lines = p.read_text().strip().splitlines()
    n = int(lines[0])
    return np.array(
        [[float(v) for v in line.split()[1:4]] for line in lines[1 : n + 1]]
    )


# ------------------------------------------------------------------ scoring


def score_truth(res: Path, n_frames):
    """Link precision/recall. Row i is pid i in every frame by construction."""
    tp = fp = 0
    for f in range(1, n_frames):
        prev = read_links(res, FIRST + f)
        for i, p in enumerate(prev):
            if p < 0:
                continue
            if p == i:
                tp += 1
            else:
                fp += 1
    return tp, fp


def per_frame_stats(res: Path, n_frames):
    """Per frame: link count, mean step, 95th-pct acceleration magnitude."""
    nlink, mstep, a95 = [], [], []
    for f in range(1, n_frames):
        prev = read_links(res, FIRST + f)
        cur = read_xyz(res, FIRST + f)
        old = read_xyz(res, FIRST + f - 1)
        steps = [
            np.linalg.norm(cur[i] - old[p])
            for i, p in enumerate(prev)
            if 0 <= p < len(old) and i < len(cur)
        ]
        nlink.append(len(steps))
        mstep.append(float(np.mean(steps)) if steps else 0.0)

        accs = []
        if f >= 2:
            prev2 = read_links(res, FIRST + f - 1)
            older = read_xyz(res, FIRST + f - 2)
            for i, p in enumerate(prev):
                if not (0 <= p < len(old) and i < len(cur)):
                    continue
                pp = prev2[p] if p < len(prev2) else -1
                if 0 <= pp < len(older):
                    accs.append(
                        float(np.linalg.norm(cur[i] - 2 * old[p] + older[pp]))
                    )
        a95.append(float(np.percentile(accs, 95)) if accs else 0.0)
    return np.array(nlink), np.array(mstep), np.array(a95)


def paired(name, a, b, unit=""):
    """Paired difference B-A with a bootstrap CI and a Wilcoxon p."""
    from scipy import stats

    d = b - a
    if np.allclose(d, 0):
        print(f"  {name:<28} identical")
        return
    rng = np.random.default_rng(0)
    boots = [
        np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(4000)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    sig = "significant" if p < 0.05 else "n.s."
    print(
        f"  {name:<28} retry-ON minus OFF = {np.mean(d):+8.4f}{unit}  "
        f"95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4g} ({sig})"
    )


# --------------------------------------------------------------------- main


def prepare(work: Path):
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copytree(SRC / "cal", work / "cal")
    shutil.copytree(SRC / "parameters", work / "parameters")
    shutil.copy(SRC / "parameters_Run1.yaml", work / "parameters_Run1.yaml")


def set_frames(work: Path, n_frames):
    import yaml

    y = yaml.safe_load((work / "parameters_Run1.yaml").read_text())
    for key in ("sequence", "Sequence", "sequence_par"):
        if key in y:
            y[key]["first"] = FIRST
            y[key]["last"] = FIRST + n_frames - 1
    (work / "parameters_Run1.yaml").write_text(yaml.safe_dump(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synthetic", "folder", "gate"])
    ap.add_argument("--path", type=Path, help="working folder for mode=folder")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--n", type=int, default=25, help="particles (synthetic)")
    ap.add_argument("--noise", type=float, default=0.25, help="mm, 3D position")
    ap.add_argument("--reps", type=int, default=8, help="noise seeds (synthetic)")
    args = ap.parse_args()

    if args.mode == "gate":
        # How much position noise can the per-frame acc gate survive?
        #
        # acc is |x[n+1] - 2x[n] + x[n-1]|. With independent noise sigma on
        # each position, each component of that combination has variance
        # (1+4+1)*sigma^2 = 6 sigma^2, so the 3D magnitude of the PURE NOISE
        # acceleration is about sqrt(3*6)*sigma = 4.24*sigma -- before the
        # particle has actually accelerated at all. Once 4.24*sigma reaches
        # dacc, the gate is rejecting noise, not physics.
        dacc = float((SRC / "parameters" / "track.par").read_text().split()[7])
        predicted = dacc / np.sqrt(18.0)
        levels = [0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5]
        print(f"\ndacc = {dacc}   predicted noise ceiling = dacc/sqrt(18) = "
              f"{predicted:.3f} mm")
        print(f"\n{'noise sigma':>12}{'4.24*sigma':>12}{'recall':>9}"
              f"{'precision':>11}   (vs dacc={dacc})")
        xs, ys = [], []
        for nz in levels:
            recs, precs = [], []
            for rep in range(args.reps):
                work = REPO / "scratch" / f"_gate_{rep}"
                prepare(work)
                _, obs = make_tracks(args.n, args.frames, nz, seed=200 + rep)
                write_scene(work, obs, args.frames)
                set_frames(work, args.frames)
                run(work, 1, args.frames)
                tp, fp = score_truth(work / "res", args.frames)
                recs.append(tp / (args.n * (args.frames - 1)))
                precs.append(tp / (tp + fp) if tp + fp else 0.0)
            r, p = float(np.mean(recs)), float(np.mean(precs))
            xs.append(nz)
            ys.append(r)
            flag = "  <-- gate is eating noise" if 4.24 * nz > dacc else ""
            print(f"{nz:>12.2f}{4.24 * nz:>12.2f}{r:>9.3f}{p:>11.3f}{flag}")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, "o-", color="#2a6ebb", label="link recall")
        ax.axvline(
            predicted, color="#c0392b", ls="--",
            label=f"dacc/$\\sqrt{{18}}$ = {predicted:.2f} mm",
        )
        ax.set_xlabel("3D position noise $\\sigma$ (mm)")
        ax.set_ylabel("fraction of true links found")
        ax.set_title("Per-frame acceleration gate vs position noise")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = REPO / "scratch" / "gate_noise_ceiling.png"
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
        return

    if args.mode == "synthetic":
        rows = []
        for rep in range(args.reps):
            work = REPO / "scratch" / f"_ab_syn_{rep}"
            prepare(work)
            _, obs = make_tracks(args.n, args.frames, args.noise, seed=100 + rep)
            write_scene(work, obs, args.frames)
            set_frames(work, args.frames)

            out = {}
            for retry in (0, 1):
                run(work, retry, args.frames)
                tp, fp = score_truth(work / "res", args.frames)
                out[retry] = (tp, fp)
            rows.append(out)

        poss = args.n * (args.frames - 1)
        print(f"\nsynthetic: {args.n} particles, {args.frames} frames, "
              f"{args.noise} mm position noise, {args.reps} seeds")
        print(f"{'':10}{'OFF (track.c)':>28}{'ON (openptv2)':>28}")
        print(f"{'seed':10}{'TP':>7}{'FP':>7}{'prec':>7}{'rec':>7}"
              f"{'TP':>7}{'FP':>7}{'prec':>7}{'rec':>7}")
        pa, pb, ra, rb = [], [], [], []
        for i, o in enumerate(rows):
            line = f"{i:<10}"
            for retry in (0, 1):
                tp, fp = o[retry]
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / poss
                line += f"{tp:>7}{fp:>7}{prec:>7.3f}{rec:>7.3f}"
                (pa if retry == 0 else pb).append(prec)
                (ra if retry == 0 else rb).append(rec)
            print(line)

        print("\npaired over seeds:")
        paired("link precision", np.array(pa), np.array(pb))
        paired("link recall", np.array(ra), np.array(rb))
        return

    work = args.path.resolve()
    if not (work / "res").is_dir():
        raise SystemExit(f"no res/ in {work}")
    import yaml

    y = yaml.safe_load((work / "parameters_Run1.yaml").read_text())
    seq = next(y[k] for k in ("sequence", "Sequence", "sequence_par") if k in y)
    first, last = int(seq["first"]), int(seq["last"])
    n_frames = last - first + 1

    global FIRST
    FIRST = first

    stats_by = {}
    for retry in (0, 1):
        run(work, retry, n_frames)
        keep = work / f"res_retry{retry}"
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(work / "res", keep)
        stats_by[retry] = per_frame_stats(keep, n_frames)

    print(f"\nreal folder: {work}  frames {first}..{last}")
    names = ["links per frame", "mean step (mm)", "95th-pct |accel| (mm)"]
    units = ["", " mm", " mm"]
    for k, (nm, un) in enumerate(zip(names, units)):
        paired(nm, stats_by[0][k], stats_by[1][k], un)
    print(
        "\nNo ground truth here, so this shows DIFFERENT and which way -- not "
        "better.\nA growing acceleration tail with retry ON is evidence the "
        "extra links are wrong."
    )


if __name__ == "__main__":
    main()
