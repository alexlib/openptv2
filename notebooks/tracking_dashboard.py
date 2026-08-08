# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "matplotlib>=3.7.0",
#     "pyyaml>=6.0",
#     "pandas>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pickle
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # make scripts/ helpers importable
    _scripts = Path("scripts").absolute()
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))

    return mo, np, pd, plt, mpl, Path, pickle, sys


@app.cell
def _(mo):
    mo.md(
        """
        # OpenPTV2 Tracking Comparison Dashboard

        Interactive 3D view of **ground-truth vs tracker trajectories** on the
        **synthetic_turbulent** benchmark case (30 frames, turbulent / DNS-like
        flow). Explore *how* each tracker behaves — fragmentation, wrong links,
        entering/leaving particles — and tune parameters.

        **Trackers:** fast_3d, myptv_3d_tracking, proptv_tracking
        """
    )
    return


@app.cell
def _():
    import benchmark_utils  # noqa: F401

    return (benchmark_utils,)


@app.cell
def _(mo):
    run_btn = mo.ui.button(label="Rerun trackers (slow, ~60s)", kind="warn")
    run_btn
    return (run_btn,)


@app.cell
def _(CACHE, Path, benchmark_utils, pickle, run_btn):
    """Run all trackers, cached on disk so reloads are instant.

    Press 'Rerun trackers' (run_btn) to invalidate the cache and recompute.
    """
    if CACHE.exists() and not run_btn.value:
        with open(CACHE, "rb") as fh:
            stored = pickle.load(fh)
            if stored.get("ok"):
                results = stored["results"]
            else:
                results = None
    else:
        results = benchmark_utils.run_all_trackers(silent=True)
        with open(CACHE, "wb") as fh:
            pickle.dump({"results": results, "ok": True}, fh)

    return results


@app.cell
def _(Path):
    CACHE = Path("notebooks/_tracking_cache.pkl")
    return (CACHE,)


@app.cell
def _(benchmark_utils):
    gt_tracks = benchmark_utils.build_true_tracks(
        benchmark_utils.read_gt_frames()
    )
    return (gt_tracks,)


@app.cell
def _(results, mo):
    # summary metrics table for all trackers
    rows = []
    for name, r in results.items():
        m = r.get("metrics")
        if m is None:
            rows.append({"tracker": name, "pmt": None, "purity": None,
                         "frag": None, "comp": None, "#tracks": None})
            continue
        rows.append({
            "tracker": name,
            "pmt": round(m.pmt, 1),
            "purity": round(m.purity, 3),
            "fragmentation": round(m.fragmentation, 2),
            "completeness": round(m.completeness, 3),
            "#tracks": m.n_reconstructed,
        })
    df_metrics = pd.DataFrame(rows)
    mo.ui.table(df_metrics, page_size=10)
    return df_metrics


@app.cell
def _(mo, df_metrics):
    tracker_sel = mo.ui.dropdown(
        options={k: k for k in df_metrics["tracker"]},
        value=df_metrics["tracker"].iloc[0],
        label="Tracker",
    )
    n_tracks = mo.ui.slider(1, 40, value=12, label="# longest tracks")
    show_gt = mo.ui.checkbox(value=True, label="Overlay ground truth")
    frame_slider = mo.ui.slider(0, 29, value=29, label="Max frame")
    tracker_sel, n_tracks, show_gt, frame_slider
    return tracker_sel, n_tracks, show_gt, frame_slider


@app.cell
def _(tracker_sel, n_tracks, show_gt, frame_slider):
    view = dict(
        tracker=tracker_sel.value,
        n=n_tracks.value,
        show_gt=show_gt.value,
        max_frame=frame_slider.value,
    )
    return (view,)


@app.cell
def _(results, gt_tracks, view, np, plt, mo):
    """Render 3D trajectories for the selected tracker vs ground truth."""
    sel = view["tracker"]
    n = view["n"]
    maxf = view["max_frame"]
    showgt = view["show_gt"]

    tracks = results[sel]["tracks"]

    # pick the n longest tracks
    lengths = [(tid, len(pts)) for tid, pts in tracks.items()]
    lengths.sort(key=lambda t: t[1], reverse=True)
    top_ids = [tid for tid, _ in lengths[:n]]

    fig = plt.figure(figsize=(11, 9))
    ax3 = fig.add_subplot(111, projection="3d")

    # colormap for tracker tracks
    cmap = plt.get_cmap("viridis")
    seg = np.linspace(0, 1, max(1, len(top_ids)))
    for i, tid in enumerate(top_ids):
        pts = np.array(tracks[tid], dtype=float)
        pts = pts[pts[:, 0] <= maxf]
        if len(pts) < 2:
            continue
        ax3.plot(pts[:, 1], pts[:, 2], pts[:, 3],
                 lw=0.8, alpha=0.7, color=cmap(seg[i]))

    if showgt:
        # overlay a subset of ground-truth tracks faintly
        gt_ids = sorted(gt_tracks.keys())
        for tid in gt_ids[: n * 4]:
            pts = np.array(gt_tracks[tid], dtype=float)
            pts = pts[pts[:, 0] <= maxf]
            if len(pts) < 2:
                continue
            ax3.plot(pts[:, 1], pts[:, 2], pts[:, 3],
                     lw=0.6, alpha=0.15, color="gray")

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title(f"Tracker: {sel}  ({len(top_ids)} longest tracks)")

    mo.mpl.interactive(plt.gcf())
    return (fig,)


@app.cell
def _(results, gt_tracks, np, plt, mo):
    """Track-length histograms: GT vs each selected tracker (all tracks)."""
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    # ground truth length distribution
    gt_lens = [len(pts) for pts in gt_tracks.values()]
    ax2.hist(gt_lens, bins=30, alpha=0.4, label="Ground truth", color="black")
    for _name, _r in results.items():
        lens = [len(pts) for pts in _r["tracks"].values()]
        ax2.hist(lens, bins=30, alpha=0.4, label=_name)
    ax2.set_xlabel("track length (frames)")
    ax2.set_ylabel("count")
    ax2.legend()
    fig2
    return (fig2,)

