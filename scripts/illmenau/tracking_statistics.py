"""Tracking statistics + figures for the Ilmenau dual-rig dataset.

Mirrors the content of Full_measurement_with_8_cameras (trajectory counts,
length histograms, longest tracks, velocity stats, reconstruction quality),
computed on the FINAL postptv stores (and raw stores for detection-level
numbers) over frames 10001-10500.

Usage:
    uv run --with matplotlib,numpy,zarr python \
        scripts/illmenau/tracking_statistics.py [--outdir DIR]
Outputs stats_*.png figures and prints markdown-ready numbers to stdout.
"""
import argparse
from pathlib import Path

import numpy as np

BASE = Path("/Users/alex/Downloads/Ilmenau")
RIGS = [
    ("1-4", BASE / "openptv_illmenau_4cam" / "res" / "run_postptv_clean.zarr",
     BASE / "openptv_illmenau_4cam" / "res" / "run.zarr"),
    ("5-8", BASE / "openptv_illmenau_5678" / "res" / "run_postptv_clean.zarr",
     BASE / "openptv_illmenau_5678" / "res" / "run.zarr"),
]
FMIN, FMAX = 10001, 10500
FRAMES = np.arange(FMIN, FMAX + 1)


def load_traj(store):
    import zarr
    g = zarr.open(str(store), mode="r")["trajectories"]
    tj, tm, pos, vel = g["trajid"][:], g["time"][:], g["pos"][:], g["vel"][:]
    m = (tm >= FMIN) & (tm <= FMAX)
    return tj[m], tm[m], pos[m], vel[m]


def per_frame_counts(tj, tm):
    n = np.zeros(len(FRAMES))
    u, c = np.unique(tm, return_counts=True)
    n[u - FMIN] = c
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=BASE / "Full_measurement_with_8_cameras")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.outdir
    data = {}
    for name, clean, raw in RIGS:
        tj, tm, pos, vel = load_traj(clean)
        ids, counts = np.unique(tj, return_counts=True)
        sp = np.linalg.norm(vel, axis=1)
        vy = vel[:, 1]
        r = np.hypot(pos[:, 0], pos[:, 2])
        # roughness on tracks >= 7
        rough = []
        for tid in ids[counts >= 7]:
            k = tj == tid
            t, pp = tm[k], pos[k]
            o = np.argsort(t)
            t, pp = t[o], pp[o]
            c = np.diff(t) == 1
            dd = pp[2:] - 2 * pp[1:-1] + pp[:-2]
            ok = c[:-1] & c[1:]
            if ok.sum() >= 3:
                rough.append(np.sqrt((dd[ok] ** 2).sum(1)).mean() * 1000)
        rough = np.array(rough)
        # detection/correspondence level from raw store
        import zarr
        root = zarr.open_group(str(raw), mode="r")
        corr = root["correspondences"]
        ckeys = sorted(k for k in corr.keys()
                       if FMIN <= int(k.split("_")[1]) <= FMAX)
        corr_n = np.array([corr[k].shape[0] for k in ckeys])
        tg = root["targets"]
        det_n = np.array([[tg[c][k].shape[0] for k in
                           sorted(tg[c].keys()) if FMIN <= int(k.split("_")[1]) <= FMAX]
                          for c in sorted(tg.keys())])
        data[name] = dict(tj=tj, tm=tm, pos=pos, vel=vel, ids=ids, counts=counts,
                          sp=sp, vy=vy, r=r, rough=rough,
                          corr_n=corr_n, det_med=np.median(det_n, axis=1))
        print(f"== rig {name}: rows={len(tj)} ntraj={len(ids)} "
              f"len med/max={np.median(counts):.0f}/{counts.max()} "
              f"spd med/p95={np.median(sp):.3f}/{np.percentile(sp, 95):.3f} "
              f"vy med/p5/p95={np.median(vy):+.3f}/{np.percentile(vy, 5):+.3f}/{np.percentile(vy, 95):+.3f} "
              f"rough med/p95={np.median(rough):.2f}/{np.percentile(rough, 95):.2f}mm "
              f"y<0={(pos[:, 1] < 0).mean() * 100:.1f}% r>3.6={(r > 3.6).mean() * 100:.1f}% "
              f"corr/frame med={np.median(corr_n):.0f} det/cam med={np.median(det_n, axis=1).round(0)}")

    # Fig 1: trajectory length histograms (log-y) + longest-20 printed
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for i, name in enumerate(["1-4", "5-8"]):
        c = data[name]["counts"]
        bins = np.unique(np.logspace(0, np.log10(c.max()), 40).astype(int))
        ax[i].hist(c, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_title(f"rig {name}: {len(c)} trajectories")
        ax[i].set_xlabel("track length [frames]")
        ax[i].grid(True, alpha=0.3)
    ax[0].set_ylabel("count")
    fig.suptitle("Trajectory length distributions (frames 10001-10500, final postptv stores)")
    fig.tight_layout()
    fig.savefig(out / "stats_trajlen_hist.png", dpi=110)
    print("wrote stats_trajlen_hist.png")

    # Fig 2: pipeline funnel per frame (detections -> correspondences -> linked)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, name in enumerate(["1-4", "5-8"]):
        d = data[name]
        linked = per_frame_counts(d["tj"], d["tm"])
        fr = np.arange(FMIN, FMAX + 1)
        det_tot = d["det_med"].sum()
        ax[i].plot(fr, np.full_like(fr, det_tot, dtype=float), label="detections/frame (all cams, med)", color="gray")
        ax[i].plot(fr, np.interp(fr, np.arange(FMIN, FMIN + len(d["corr_n"])), d["corr_n"]),
                   label="correspondences/frame", color="darkorange")
        ax[i].plot(fr, linked, label="linked pts/frame (final tracks)", color="steelblue")
        ax[i].set_title(f"rig {name}")
        ax[i].set_ylabel("points / frame (log scale)")
        ax[i].set_yscale("log")
        ax[i].legend(fontsize=8, loc="upper right")
        ax[i].grid(True, alpha=0.3, which="both")
    ax[1].set_xlabel("frame")
    fig.suptitle("Reconstruction pipeline per frame: detection -> triangulation -> linked tracks")
    fig.tight_layout()
    fig.savefig(out / "stats_funnel_perframe.png", dpi=110)
    print("wrote stats_funnel_perframe.png")

    # Fig 3: speed + vy distributions
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, col in [("1-4", "firebrick"), ("5-8", "steelblue")]:
        ax[0].hist(data[name]["sp"], bins=np.linspace(0, 6, 120), histtype="step",
                   density=True, label=f"rig {name}", color=col, linewidth=1.5)
        ax[1].hist(data[name]["vy"], bins=np.linspace(-3, 3, 120), histtype="step",
                   density=True, label=f"rig {name}", color=col, linewidth=1.5)
    ax[0].set_xlabel("speed [m/s]")
    ax[0].set_ylabel("density")
    ax[0].legend()
    ax[0].set_title("speed distribution")
    ax[1].set_xlabel("vy (vertical velocity) [m/s]")
    ax[1].legend()
    ax[1].set_title("vertical-velocity distribution (red=up, blue=down)")
    ax[1].axvline(0, color="k", linewidth=0.8)
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle("Velocity statistics (final postptv stores)")
    fig.tight_layout()
    fig.savefig(out / "stats_velocity_hist.png", dpi=110)
    print("wrote stats_velocity_hist.png")

    # Fig 4: spatial coverage (side x-z + top-down x-y density)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for i, name in enumerate(["1-4", "5-8"]):
        p = data[name]["pos"]
        sel = np.random.default_rng(0).choice(len(p), size=min(60000, len(p)), replace=False)
        ax[i, 0].scatter(p[sel][:, 0], p[sel][:, 2], s=0.3, alpha=0.4)
        ax[i, 0].set_title(f"rig {name}: side view (x-z)")
        ax[i, 0].set_xlabel("x [m]")
        ax[i, 0].set_ylabel("z [m]")
        ax[i, 0].set_aspect("equal")
        ax[i, 0].grid(True, alpha=0.3)
        ax[i, 1].scatter(p[sel][:, 0], p[sel][:, 1], s=0.3, alpha=0.4)
        ax[i, 1].set_title(f"rig {name}: front view (x-y, y up)")
        ax[i, 1].set_xlabel("x [m]")
        ax[i, 1].set_ylabel("y [m]")
        ax[i, 1].set_aspect("equal")
        ax[i, 1].grid(True, alpha=0.3)
    fig.suptitle("Spatial coverage (60k-point samples, final tracks)")
    fig.tight_layout()
    fig.savefig(out / "stats_coverage.png", dpi=110)
    print("wrote stats_coverage.png")

    # longest-20 table rows for the report
    for name in ["1-4", "5-8"]:
        d = data[name]
        top = np.argsort(d["counts"])[::-1][:20]
        print(f"== longest {name}: " + ", ".join(str(int(d['counts'][i])) for i in top))


if __name__ == "__main__":
    main()
