import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import zarr

    return go, mo, np, zarr


@app.cell
def _(mo):
    mo.md(
        """
    # VFC comparison — Lagrangian trajectories, side by side

    Three runs of the **same** data (the CompleteTest wp1 recording):

    * **existing** — the result that is already in the run folder,
    * **off** — the `two_phase` tracker without the spatial consensus,
    * **on** — the same tracker with the spatial consensus (VFC), the
      cold-start bootstrap and the affine consensus.

    The runs differ only in the `vfc_*` keys of the `track:` section of the
    YAML.  Every other parameter is identical.
    """
    )
    return


@app.cell
def _(mo):
    base = mo.ui.text(
        value="/Users/alex/Downloads/HiDImaging/CompleteTest-5000",
        label="Experiment folder",
    )
    run_paths = {
        "existing": "wp1/test/res/run.zarr",
        "off": "vfc_compare_2026-10-20/off/test/res/run.zarr",
        "on": "vfc_compare_2026-10-20/on/test/res/run.zarr",
    }
    mo.vstack(
        [
            base,
            mo.md("**Runs** (folder under the experiment folder)"),
            mo.hstack(
                [mo.ui.text(value=v, label=k) for k, v in run_paths.items()], gap=2
            ),
        ]
    )
    return base, run_paths


@app.cell
def _(base, mo, np, run_paths, zarr):
    import os

    def load_store(path):
        """Trajectory index and positions of a sealed run store."""
        if not path or not os.path.exists(path):
            return None
        z = zarr.open(path, mode="r")
        if "traj" not in z:
            return None
        idx = z["traj"]
        return {
            "trajid": np.asarray(idx["trajid"]),
            "length": np.asarray(idx["length"]),
            "first": np.asarray(idx["first"]),
            "last": np.asarray(idx["last"]),
            "first_row": np.asarray(idx["first_row"]),
            "pos": z["trajectories"]["pos"],
            "time": z["trajectories"]["time"],
        }

    data = {
        key: load_store(os.path.join(base.value, rel)) for key, rel in run_paths.items()
    }
    missing = [k for k, v in data.items() if v is None]
    mo.md(
        f"**Missing or not sealed:** {', '.join(missing)}"
        if missing
        else "All three runs are loaded."
    )
    return data


@app.cell
def _(data, mo, np):
    def run_stats(entry):
        length = entry["length"]
        span = int(entry["last"].max() - entry["first"].min() + 1) if len(length) else 0
        return {
            "trajectories": int(len(length)),
            "points": int(length.sum()),
            "mean length": float(length.mean()) if len(length) else 0.0,
            "median length": float(np.median(length)) if len(length) else 0.0,
            "long (>= 20)": int((length >= 20).sum()),
            "short (< 5)": int((length < 5).sum()),
            "points / frame": length.sum() / span if span else 0.0,
        }

    rows = [
        {"run": key, **run_stats(entry)}
        for key, entry in data.items()
        if entry is not None
    ]
    if rows:
        mo.ui.table(rows)
    return (rows,)


@app.cell
def _(mo):
    n_traj = mo.ui.slider(1, 400, value=120, step=1, label="Trajectories per run")
    f0 = mo.ui.number(1, 5005, value=1, label="First frame")
    f1 = mo.ui.number(1, 5005, value=5005, label="Last frame")
    pick = mo.ui.dropdown(
        options={
            "typical (median length)": "typical",
            "longest": "longest",
            "shortest": "shortest",
            "random": "random",
        },
        value="typical (median length)",
        label="Which trajectories",
    )
    mo.vstack([mo.hstack([n_traj, f0, f1, pick], gap=2)])
    return f0, f1, n_traj, pick


@app.cell
def _(data, f0, f1, mo, n_traj, np, pick):
    def collect(entry, n_show, frame_lo, frame_hi, mode):
        """n_show trajectories of the requested kind inside the frame range."""
        if entry is None:
            return [], []
        length = entry["length"]
        first = entry["first"]
        last = entry["last"]
        first_row = entry["first_row"]
        keep = np.where((last >= frame_lo) & (first <= frame_hi))[0]
        if len(keep) == 0:
            return [], []
        if mode == "longest":
            order = keep[np.argsort(length[keep])[::-1]][:n_show]
        elif mode == "shortest":
            order = keep[np.argsort(length[keep])][:n_show]
        elif mode == "random":
            rng = np.random.default_rng(0)
            order = keep[rng.permutation(len(keep))[:n_show]]
        else:  # typical: the tracks whose length is around the median
            ordered = keep[np.argsort(length[keep])]
            mid = len(ordered) // 2
            half = n_show // 2
            lo = max(0, mid - half)
            order = ordered[lo : lo + n_show]
        out_paths, out_speeds = [], []
        for i in order:
            lo = int(first_row[i])
            hi = lo + int(length[i])
            p = np.asarray(entry["pos"][lo:hi]) * 1000.0  # m -> mm
            t = np.asarray(entry["time"][lo:hi]) * 1.0  # frame numbers
            sel = (t >= frame_lo) & (t <= frame_hi)
            p, t = p[sel], t[sel]
            if len(p) < 2:
                continue
            out_paths.append((t, p))
            out_speeds.append(
                np.linalg.norm(np.diff(p, axis=0), axis=1)
                / np.diff(t)[:, None]
                * 1000.0
            )
        return out_paths, out_speeds

    series = {
        key: collect(entry, n_traj.value, f0.value, f1.value, pick.value)
        for key, entry in data.items()
    }
    for key, value in series.items():
        mo.md(f"**{key}**: {len(value[0])} trajectories drawn")
    return (series,)


@app.cell
def _(go, mo, np, series):
    def trace_figure(paths, title, height=620):
        fig = go.Figure()
        for _t, p in paths:
            fig.add_trace(
                go.Scatter3d(
                    x=p[:, 0],
                    y=p[:, 1],
                    z=p[:, 2],
                    mode="lines",
                    line=dict(width=2, color="#1f77b4"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        if paths:
            allp = np.vstack([p for _t, p in paths])
            mid = (allp.max(0) + allp.min(0)) / 2
            span = float((allp.max(0) - allp.min(0)).max()) / 2 or 1.0
        else:
            mid, span = np.zeros(3), 1.0
        fig.update_layout(
            title=title,
            width=560,
            height=height,
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(title="x [mm]", range=[mid[0] - span, mid[0] + span]),
                yaxis=dict(title="y [mm]", range=[mid[1] - span, mid[1] + span]),
                zaxis=dict(title="z [mm]", range=[mid[2] - span, mid[2] + span]),
                aspectmode="data",
            ),
        )
        return fig

    run_order = [k for k in ("existing", "off", "on") if k in series]
    figs = [trace_figure(series[k][0], k) for k in run_order]
    mo.hstack(figs, gap=1, justify="center")
    return (figs,)


@app.cell
def _(go, mo, np, series):
    def speed_figure(paths, speeds, title, height=380):
        fig = go.Figure()
        for (_t, p), s in zip(paths, speeds):
            if len(s) != len(p) - 1:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=p[:-1, 0],
                    y=p[:-1, 1],
                    z=p[:-1, 2],
                    mode="markers",
                    marker=dict(size=2, color=s, colorscale="Viridis", showscale=False),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.update_layout(
            title=title,
            width=560,
            height=height,
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(aspectmode="data"),
        )
        return fig

    speed_order = [k for k in ("existing", "off", "on") if k in series]
    speed_figs = [
        speed_figure(series[k][0], series[k][1], f"{k} — speed") for k in speed_order
    ]
    mo.hstack(speed_figs, gap=1, justify="center")
    return (speed_figs,)


if __name__ == "__main__":
    app.run()
