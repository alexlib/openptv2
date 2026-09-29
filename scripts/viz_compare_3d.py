"""One-to-one 3D comparison of trackers on a synthetic-benchmark case.

For every long TRUE trajectory the script finds, per tracker, the output
trajectory pieces (after postptv repair) that follow it, scores how much of
it the best single piece recovers, and draws hand-picked examples: the true
path (black), each tracker's pieces (one colour per piece), and the points
where a piece leaves the true particle (red x = wrong link / ghost).

Usage (repo root):
    uv run python scripts/viz_compare_3d.py --case L1_d2_c0.3_k4_n150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_bench as sb  # noqa: E402
from flowtracks.repair import repair_arrays  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

TRACKERS = {
    "trackcorr (original gates)": "trackcorr",
    "trackcorr, dacc 0.25": "trackcorr+dacc=0.25",
    "trackcorr, dacc 0.25 + confirm": "trackcorr+dacc=0.25+postconfirm_tol=0.3+postconfirm_ends=true",
    "two_phase": "two_phase",
}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]


def load(case: str, label: str, by_frame, max_gap=3):
    p = np.load(sb.WORK / "runs" / case / label / "pred.npz")
    frame, pos = p["frame"], p["pos"]
    g0, _, off = sb._match(by_frame, frame, pos, 0.6)
    bias = np.median(off[g0 >= 0], axis=0)
    pos = pos - bias
    tid, _ = repair_arrays(p["trajid"], frame, pos, max_gap=max_gap)[0], None
    gid, dist, _ = sb._match(by_frame, frame, pos, 0.4)
    return {"tid": tid, "frame": frame, "pos": pos, "gid": gid}


def coverage(out, true_id, n_true_frames):
    """(best single-piece fraction of the true trajectory, number of pieces >=3 pts)."""
    m = out["gid"] == true_id
    if not m.any():
        return 0.0, 0
    t, c = np.unique(out["tid"][m], return_counts=True)
    return float(c.max() / n_true_frames), int((c >= 3).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="L1_d2_c0.3_k4_n150")
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()
    case = args.case
    cd = sb.WORK / "cases" / case
    tr = np.load(cd / "truth.npz")
    t_id, t_fr, t_pos = tr["particle_id"], tr["frame"], tr["pos_mm"]
    by_frame = {}
    for f in np.unique(t_fr):
        m = t_fr == f
        by_frame[int(f)] = (t_pos[m], t_id[m], cKDTree(t_pos[m]))
    outs = {name: load(case, lab, by_frame) for name, lab in TRACKERS.items()}

    length = np.bincount(t_id)
    ids = np.flatnonzero(length >= args.min_len)
    cov = {n: np.zeros(len(ids)) for n in outs}
    pcs = {n: np.zeros(len(ids), int) for n in outs}
    for n, o in outs.items():
        for k, i in enumerate(ids):
            cov[n][k], pcs[n][k] = coverage(o, i, length[i])
    a, b, c = (cov[n] for n in ("trackcorr (original gates)", "trackcorr, dacc 0.25 + confirm", "two_phase"))
    print(f"{len(ids)} true trajectories with >= {args.min_len} frames")
    for n in TRACKERS:
        print(f"  {n:22s} mean best-piece coverage {cov[n].mean():.3f}  "
              f"whole (>=90%) {np.mean(cov[n] >= .9):.3f}  pieces/track {pcs[n].mean():.2f}")

    picks = []
    d1 = np.argsort(-(c - a) - 1e-3 * length[ids])  # two_phase >> trackcorr as is
    for k in d1:
        if c[k] >= 0.9 and a[k] <= 0.5 and length[ids[k]] >= 80:
            picks.append(("two_phase whole, trackcorr broken", k))
        if len(picks) == 2:
            break
    d2 = np.argsort(-(b - c))
    for k in d2:
        if b[k] >= 0.9 and c[k] <= 0.6:
            picks.append(("trackcorr tuned+confirm whole, two_phase broken", k))
            break
    d3 = np.argsort(-length[ids])
    for k in d3:
        if min(a[k], b[k], c[k]) >= 0.95:
            picks.append(("all three recover it whole", k))
            break
    picks = picks[: args.rows]
    for title, k in picks:
        print(f"  pick true#{ids[k]} len {length[ids[k]]}: coverage as-is {a[k]:.2f} "
              f"+confirm {b[k]:.2f} two_phase {c[k]:.2f}  [{title}]")

    # ---- geometry per (row, tracker): truth path + the pieces that follow it
    scenes = []
    for title, k in picks:
        tid_true = ids[k]
        m = t_id == tid_true
        o = np.argsort(t_fr[m])
        tp, tf = t_pos[m][o], t_fr[m][o]
        row = {"title": title, "true_id": int(tid_true), "truth": tp, "truth_f": tf, "cols": []}
        for n, out in outs.items():
            mine = out["gid"] == tid_true
            cnt = {p: int((out["tid"][mine] == p).sum()) for p in np.unique(out["tid"][mine])}
            pieces = []
            for p in sorted((p for p in cnt if cnt[p] >= 3), key=lambda p: -cnt[p])[:7]:
                s_ = out["tid"] == p
                oo = np.argsort(out["frame"][s_])
                pieces.append((out["pos"][s_][oo], out["frame"][s_][oo],
                               out["gid"][s_][oo] != tid_true))
            row["cols"].append({"name": n, "cov": float(cov[n][k]), "npieces": int(pcs[n][k]),
                                "pieces": pieces})
        pad = 0.6
        row["rng"] = [(tp[:, d].min() - pad, tp[:, d].max() + pad) for d in range(3)]
        scenes.append(row)
    out_dir = sb.WORK / "viz"
    out_dir.mkdir(exist_ok=True)
    nr, ntr = len(scenes), len(TRACKERS)

    # ---- static PNGs (matplotlib): one file per example, trackers stacked vertically
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r, row in enumerate(scenes, start=1):
        fig_m = plt.figure(figsize=(7.5, 4.6 * ntr))
        for ci, col in enumerate(row["cols"], start=1):
            ax = fig_m.add_subplot(ntr, 1, ci, projection="3d")
            tp = row["truth"]
            ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color="black", lw=4, alpha=0.55)
            for j, (pp, ff, bad) in enumerate(col["pieces"]):
                ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], "-o", ms=2.5, lw=1.6, color=PALETTE[j % 7])
                if bad.any():
                    ax.plot(pp[bad, 0], pp[bad, 1], pp[bad, 2], "x", color="red", ms=6, mew=2)
            ax.set_xlim(*row["rng"][0])
            ax.set_ylim(*row["rng"][1])
            ax.set_zlim(*row["rng"][2])
            ax.set_box_aspect([r_[1] - r_[0] for r_ in row["rng"]])
            head_txt = f"[{row['title']}]  true #{row['true_id']}\n" if ci == 1 else ""
            ax.set_title(f"{head_txt}{col['name']}: best piece {col['cov']:.0%}, "
                         f"{col['npieces']} piece(s)", fontsize=11)
            ax.view_init(elev=22, azim=-55)
            ax.tick_params(labelsize=7)
        fig_m.tight_layout()
        png = out_dir / f"compare_{case}_ex{r}.png"
        fig_m.savefig(png, dpi=75)
        plt.close(fig_m)
        print("wrote", png)

    # ---- interactive HTML (plotly): a single column, scroll down
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    total = nr * ntr
    titles = [(f"<b>[{row['title']}] &mdash; true #{row['true_id']}</b><br>" if i == 0 else "")
              + f"<b>{c['name']}</b>: best piece {c['cov']:.0%}, {c['npieces']} piece(s)"
              for row in scenes for i, c in enumerate(row["cols"])]
    fig = make_subplots(rows=total, cols=1, specs=[[{"type": "scene"}]] * total,
                        subplot_titles=titles, vertical_spacing=0.012)
    for r, row in enumerate(scenes):
        for ci, col in enumerate(row["cols"]):
            k = r * ntr + ci + 1  # subplot number == row
            tp = row["truth"]
            fig.add_trace(go.Scatter3d(
                x=tp[:, 0], y=tp[:, 1], z=tp[:, 2], mode="lines",
                line=dict(color="black", width=7), name="truth", legendgroup="truth",
                showlegend=(k == 1), text=row["truth_f"],
                hovertemplate="truth, frame %{text}<extra></extra>"), row=k, col=1)
            for j, (pp, ff, bad) in enumerate(col["pieces"]):
                fig.add_trace(go.Scatter3d(
                    x=pp[:, 0], y=pp[:, 1], z=pp[:, 2], mode="lines+markers",
                    line=dict(color=PALETTE[j % 7], width=3),
                    marker=dict(size=2.5, color=PALETTE[j % 7]), showlegend=False,
                    text=ff, hovertemplate="frame %{text}<extra></extra>"), row=k, col=1)
                if bad.any():
                    fig.add_trace(go.Scatter3d(
                        x=pp[bad, 0], y=pp[bad, 1], z=pp[bad, 2], mode="markers",
                        marker=dict(size=5, color="red", symbol="x"),
                        name="off the true particle", legendgroup="bad",
                        showlegend=(k == 1), text=ff[bad],
                        hovertemplate="wrong particle, frame %{text}<extra></extra>"),
                        row=k, col=1)
            name = "scene" if k == 1 else f"scene{k}"
            fig.layout[name].update(
                xaxis=dict(range=row["rng"][0], title="x mm"),
                yaxis=dict(range=row["rng"][1], title="y mm"),
                zaxis=dict(range=row["rng"][2], title="z mm"), aspectmode="data",
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)))
    summary = "  |  ".join(f"{n}: whole {np.mean(cov[n] >= .9):.0%}, {pcs[n].mean():.2f} pieces"
                           for n in TRACKERS)
    fig.update_layout(
        height=520 * total, width=900, template="plotly_white",
        title=f"{case}: {len(ids)} true trajectories of 50+ frames<br><sub>{summary}</sub>",
        legend=dict(orientation="h", y=1.0))
    html = out_dir / f"compare_{case}.html"
    fig.write_html(html, include_plotlyjs="cdn")
    print("wrote", html)
    json.dump({n: {"whole": float(np.mean(cov[n] >= .9)), "coverage": float(cov[n].mean()),
                   "pieces": float(pcs[n].mean())} for n in TRACKERS},
              open(out_dir / f"compare_{case}.json", "w"), indent=1)



if __name__ == "__main__":
    main()
