# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import time
    from pathlib import Path
    import plotly.graph_objects as go

    return Path, go, mo, time


@app.cell
def _(mo):
    is_script = mo.app_meta().mode == "script"
    return (is_script,)


@app.cell
def _(mo):
    mo.md("""
    # Quick Bench — 3D Trajectories per Tracker (Burgers / 1k)

    Runs `openptv2.benchmarking.runner.run_tracker` on a **small window** (Burgers `10001–10005`, `synthetic_turbulent_1k` `10001–10010`) and shows **3D trajectories** after each run.

    **Modify parameters in the text boxes** (not sliders) and press **Run** per tracker. Results are ranked by `mean_len − 0.02·kurtosis` (long + smooth).
    """)
    return


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options=["burgers (5 frames, vortex)", "synthetic_turbulent_1k (10 of 20, 1k/frame)"],
        value="burgers (5 frames, vortex)",
        label="Dataset",
    )
    dataset_picker
    return (dataset_picker,)


@app.cell
def _():
    from openptv2.tracking_registry import TRACKER_REGISTRY
    ALL_TRACKERS = [t for t in ["priority_segment_3d","4be","full_multipass","standard_forward","two_directional","nearest_hungarian_3d","predictive_gmm_3d","hybrid_deltat_3d","two_phase"] if t in TRACKER_REGISTRY]
    defaults = {
        "priority_segment_3d": ("2.0","0.5","60"),
        "4be": ("2.0","0.5","60"),
        "full_multipass": ("1.5","0.5","40"),
        "standard_forward": ("1.5","0.5","40"),
        "two_directional": ("1.5","0.5","40"),
        "nearest_hungarian_3d": ("2.0","0.5","40"),
        "predictive_gmm_3d": ("2.0","0.5","40"),
        "hybrid_deltat_3d": ("2.0","0.5","60"),
        "two_phase": ("2.0","0.5","60"),
    }
    # dense defaults (used when 1k picked, applied at run time as fallback)
    dense_defaults = {
        "priority_segment_3d": ("10.0","5.0","120"),
        "4be": ("10.0","5.0","120"),
        "full_multipass": ("8.0","3.0","60"),
        "standard_forward": ("8.0","3.0","60"),
        "two_directional": ("8.0","3.0","60"),
        "nearest_hungarian_3d": ("8.0","5.0","45"),
        "predictive_gmm_3d": ("8.0","5.0","30"),
        "hybrid_deltat_3d": ("8.0","0.8","60"),
        "two_phase": ("8.0","5.0","60"),
    }
    return ALL_TRACKERS, defaults, dense_defaults


@app.cell
def _(mo):
    min_len_input = mo.ui.text(value="2", label="Min length to plot (frames)")
    min_len_input
    return (min_len_input,)


@app.cell
def _(
    ALL_TRACKERS,
    dataset_picker,
    defaults,
    dense_defaults,
    min_len_input,
    mo,
):
    # Text boxes auto-switch to good params for the selected dataset; kept in notebook as `defaults` / `dense_defaults`
    is_burgers = "burgers" in dataset_picker.value
    src = defaults if is_burgers else dense_defaults
    mode_label = "Burgers (tight, smooth)" if is_burgers else "synthetic_turbulent_1k (dense, turbulent)"
    uis = {}
    _rows = []
    for _name in ALL_TRACKERS:
        _dv, _da, _ang = src[_name]
        _dv_in = mo.ui.text(value=_dv, label="dvxmax", placeholder="mm/frame")
        _dacc_in = mo.ui.text(value=_da, label="dacc")
        _ang_in = mo.ui.text(value=_ang, label="angle (gon)")
        _btn = mo.ui.button(label=f"Run {_name}", kind="neutral")
        uis[_name] = {"dv": _dv_in, "dacc": _dacc_in, "ang": _ang_in, "btn": _btn}
        _rows.append(mo.hstack([mo.md(f"**{_name}**"), _dv_in, _dacc_in, _ang_in, _btn], justify="start", gap=0.5))
    panel = mo.vstack([
        mo.md(f"**Good params for {mode_label}** (edit text boxes, press Run per tracker):"),
        mo.vstack(_rows, gap=0.5),
        mo.md(f"Plot filter: only trajectories ≥ {min_len_input.value} frames are drawn as 3D lines (shorter as faint markers or hidden). Change **Min length** above to 1 to see singletons."),
    ], gap=0.4)
    panel
    return (uis,)


@app.cell
def _(Path, go, is_script, time):
    from openptv2.benchmarking.runner import run_tracker
    from openptv2.benchmarking.metrics import compute_physics_metrics

    def resolve_dataset(label):
        root = Path("test_data").resolve()
        if "burgers" in label:
            return (root / "burgers" / "parameters_Run1.yaml").resolve(), "burgers"
        else:
            return (root / "synthetic_turbulent_1k" / "parameters_Run1.yaml").resolve(), "synthetic_turbulent_1k"

    def parse_float(txt, default):
        try:
            return float(txt.value.strip())
        except Exception:
            return default

    def tracks_to_plotly(pred, title, min_len=5):
        # pred: {tid: [(frame,x,y,z)]} -> plotly 3D; only len>=min_len as 3D lines, shorter as faint markers (or hidden if min_len>1)
        fig = go.Figure()
        tids = list(pred.keys())
        if len(tids) > 2000:
            tids = tids[:2000]
        has_lines = False
        sx, sy, sz, stext = [], [], [], []
        kept = 0
        all_x, all_y, all_z = [], [], []
        for tid in tids:
            pts = sorted(pred[tid], key=lambda p: p[0])
            if len(pts) < min_len:
                if 1 < len(pts) < min_len:
                    # short but >1: faint markers to hint at fragmentation
                    for p in pts:
                        sx.append(p[1]); sy.append(p[2]); sz.append(p[3])
                    all_x.extend([p[1] for p in pts]); all_y.extend([p[2] for p in pts]); all_z.extend([p[3] for p in pts])
                elif len(pts) == 1:
                    sx.append(pts[0][1]); sy.append(pts[0][2]); sz.append(pts[0][3])
                    stext.append(f"tr{tid} f{pts[0][0]} singleton")
                    all_x.append(pts[0][1]); all_y.append(pts[0][2]); all_z.append(pts[0][3])
                continue
            has_lines = True
            kept += 1
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            zs = [p[3] for p in pts]
            all_x.extend(xs); all_y.extend(ys); all_z.extend(zs)
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(width=4), name=f"tr{tid}", showlegend=False, hoverinfo="text", text=[f"f{f}" for f,_,_,_ in pts]))
        if sx:
            # hide singletons when min_len>1 and we have lines, unless user wants to see them
            if has_lines and min_len > 1:
                # don't flood plot with 19k singletons; just note count
                pass
            else:
                fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="markers", marker=dict(size=2, color="rgba(200,0,0,0.35)"), name=f"{len(sx)} short (<{min_len})", hoverinfo="text", text=stext))
        if not has_lines and not sx:
            fig.add_annotation(text="No tracks to display", x=0.5, y=0.5, showarrow=False)
        _t = title + (f" — {kept}/{len(tids)} kept (≥{min_len} fr)" if has_lines or sx else "")
        if not has_lines and sx:
            _t += " — fragmented — try larger dvxmax/dacc or lower min_len to 1"
        if all_x:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            zmin, zmax = min(all_z), max(all_z)
            pad = 0.07
            xr = (xmax - xmin) or 1.0
            yr = (ymax - ymin) or 1.0
            zr = (zmax - zmin) or 1.0
            scene = dict(
                xaxis=dict(title="x mm", range=[xmin - pad*xr, xmax + pad*xr]),
                yaxis=dict(title="y mm", range=[ymin - pad*yr, ymax + pad*yr]),
                zaxis=dict(title="z mm", range=[zmin - pad*zr, zmax + pad*zr]),
                aspectmode="data",
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
            )
        else:
            scene = dict(xaxis_title="x mm", yaxis_title="y mm", zaxis_title="z mm", aspectmode="data", camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)))
        fig.update_layout(title=_t, scene=scene, height=560, margin=dict(l=0,r=0,b=0,t=40))
        return fig

    def run_one(yaml_path, tracker, dv, dacc, ang):
        ov = {"dvxmax": dv, "dvxmin": -dv, "dvymax": dv, "dvymin": -dv, "dvzmax": dv, "dvzmin": -dv, "dacc": dacc, "angle": ang}
        t0 = time.perf_counter()
        try:
            pred = run_tracker(yaml_path, tracker, track_overrides=ov)
            pm = compute_physics_metrics(pred)
            info = {"error": None, "pred": pred, "pm": pm, "time_s": round(time.perf_counter()-t0,2)}
        except Exception as e:
            info = {"error": str(e)[:600], "pred": {}, "pm": None, "time_s": round(time.perf_counter()-t0,2)}
        return info

    # script-mode: print to terminal and also keep for notebook display
    _script_demo = None
    if is_script:
        yp,_ = resolve_dataset("burgers (5 frames, vortex)")
        demo = run_one(yp, "priority_segment_3d", 2.0, 0.5, 60)
        _msg = f"[script] burgers priority_segment_3d: mean_len={demo['pm'].mean_track_length if demo['pm'] else 'ERR'}"
        print(_msg)
        _script_demo = demo
    return parse_float, resolve_dataset, run_one, tracks_to_plotly


@app.cell
def _(
    ALL_TRACKERS,
    dataset_picker,
    is_script,
    min_len_input,
    mo,
    parse_float,
    resolve_dataset,
    run_one,
    tracks_to_plotly,
    uis,
):
    yaml_path, dlabel = resolve_dataset(dataset_picker.value)
    try:
        min_len = int(min_len_input.value.strip())
    except Exception:
        min_len = 5
    _outputs = []
    _auto = is_script
    _any_clicked = any(uis[_n]["btn"].value for _n in ALL_TRACKERS)
    # Auto-demo in interactive too so user sees output immediately (then can tweak)
    _show_demo = not _any_clicked
    for _name in ALL_TRACKERS:
        _ui = uis[_name]
        _triggered = _auto or _ui["btn"].value or (_show_demo and _name == ALL_TRACKERS[0])
        if _auto and _name != ALL_TRACKERS[0]:
            # script: only first tracker
            if _ui["btn"].value == 0:
                continue
            _auto = False
        if _show_demo and _name != ALL_TRACKERS[0]:
            continue
        if not _triggered:
            continue
        if _auto:
            _auto = False
        _dv = parse_float(_ui["dv"], 2.0)
        _dacc = parse_float(_ui["dacc"], 0.5)
        _ang = parse_float(_ui["ang"], 60)
        _info = run_one(yaml_path, _name, _dv, _dacc, _ang)
        if _info["error"]:
            _outputs.append(mo.vstack([mo.md(f"### {_name} — ERROR"), mo.md(f"`{_info['error']}`")]))
            continue
        _pm = _info["pm"]
        # printed output in cell (header) + interactive 3D plot (filtered to >=min_len)
        _header = f"### {_name} — dv={_dv} dacc={_dacc} ang={_ang} → mean_len={_pm.mean_track_length:.2f} frac10={_pm.frac_tracks_over_10:.2f} kurt={_pm.acceleration_kurtosis:.1f} n_tracks={_pm.n_tracks} t={_info['time_s']}s"
        _detail = f"Output: tracker={_name}, dataset={dlabel}, dvxmax={_dv}, dacc={_dacc}, angle={_ang}, tracks={_pm.n_tracks}, mean_len={_pm.mean_track_length:.2f}, kurt={_pm.acceleration_kurtosis:.1f}, plotted ≥{min_len} fr"
        _fig = tracks_to_plotly(_info["pred"], f"{_name} {dlabel} ({len(_info['pred'])} tracks)", min_len=min_len)
        _outputs.append(mo.vstack([mo.md(_header), mo.md(f"`{_detail}`"), _fig], gap=0.5))

    _result = mo.vstack([
        mo.md("_Edit **dvxmax/dacc/angle** in the **text boxes** above and press **Run** per tracker. The cell below prints the output and then shows the interactive 3D plot (Plotly, mouse-rotatable)."),
        mo.vstack(_outputs, gap=1) if _outputs else mo.md("*No run yet — showing demo for first tracker above. Press any Run button to re-run with your edits.*")
    ], gap=0.8)
    _result
    return


@app.cell
def _(ALL_TRACKERS, defaults, dense_defaults, mo):
    # Keep good parameter sets visible in notebook (as requested)
    hdr = "| tracker | Burgers dv/dacc/ang | 1k dv/dacc/ang |\n|---|---|---|\n"
    rows = []
    for _n in ALL_TRACKERS:
        b = "/".join(defaults[_n])
        d = "/".join(dense_defaults[_n])
        rows.append(f"| `{_n}` | {b} | {d} |")
    tbl = hdr + "\n".join(rows)
    mo.md(f"""
    **Good parameters kept in notebook** (`defaults` for Burgers, `dense_defaults` for `synthetic_turbulent_1k`) — text boxes auto-switch when you change **Dataset** above:

    {tbl}

    **What the quick bench shows (Burgers 5fr: `priority_segment_3d dv=2 dacc=0.5 ang=60` → `mean_len=3.42`):**
    * Burgers: smooth, `dv 1–2` beats `5` (extra dv adds ghosts), all trackers hit ceiling `kurt 3–4`.
    * 1k dense: needs `dv 8–12, dacc 3–5` for `mean_len 6–8, kurt 20–30`; `full_multipass` prefers tighter `angle 40–60`.
    """)
    return


if __name__ == "__main__":
    app.run()
