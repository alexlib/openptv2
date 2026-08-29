# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
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
    return mo, np, pd, time, Path


@app.cell
def _(mo):
    is_script = mo.app_meta().mode == "script"
    return (is_script,)


@app.cell
def _(mo):
    mo.md("""
    # Dense Bench — synthetic_turbulent_1k + proptv_500_25

    Dense, turbulent, many ghosts. Text boxes per tracker + Run. Plot only ≥ min_len.
    """)
    return


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options=["synthetic_turbulent_1k (10 of 20, 1k/frame)", "proptv_500_25 (25 fr, 500)"],
        value="synthetic_turbulent_1k (10 of 20, 1k/frame)",
        label="Dense dataset",
    )
    dataset_picker
    return (dataset_picker,)


@app.cell
def _():
    from openptv2.tracking_registry import TRACKER_REGISTRY
    ALL_TRACKERS = [t for t in ["priority_segment_3d","4be","full_multipass","standard_forward","two_directional","nearest_hungarian_3d","predictive_gmm_3d","hybrid_deltat_3d","two_phase"] if t in TRACKER_REGISTRY]
    defaults_1k = {
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
    defaults_proptv = {
        "priority_segment_3d": ("8.0","5.0","120"),
        "4be": ("8.0","5.0","120"),
        "full_multipass": ("6.0","3.0","60"),
        "standard_forward": ("6.0","3.0","60"),
        "two_directional": ("6.0","3.0","60"),
        "nearest_hungarian_3d": ("6.0","5.0","45"),
        "predictive_gmm_3d": ("6.0","5.0","30"),
        "hybrid_deltat_3d": ("6.0","0.8","60"),
        "two_phase": ("6.0","5.0","60"),
    }
    return ALL_TRACKERS, defaults_1k, defaults_proptv


@app.cell
def _(mo):
    last_clicked, set_last_clicked = mo.state((None, 0))
    return last_clicked, set_last_clicked


@app.cell
def _(mo):
    min_len_input = mo.ui.text(value="2", label="Min length to plot (frames)")
    min_len_input
    return (min_len_input,)


@app.cell
def _(
    ALL_TRACKERS,
    dataset_picker,
    defaults_1k,
    defaults_proptv,
    min_len_input,
    mo,
):
    is_dense1k = "synthetic" in dataset_picker.value
    src = defaults_1k if is_dense1k else defaults_proptv
    label = "synthetic_turbulent_1k (1k, turbulent)" if is_dense1k else "proptv_500_25 (500, 25fr)"
    uis = {}
    _rows=[]
    for _name in ALL_TRACKERS:
        _dv,_da,_ang = src[_name]
        _dv_in=mo.ui.text(value=_dv,label="dvxmax")
        _dacc_in=mo.ui.text(value=_da,label="dacc")
        _ang_in=mo.ui.text(value=_ang,label="angle")
        _btn=mo.ui.button(value=0, on_click=lambda v, n=_name: (set_last_clicked((n, v+1)), v+1)[1], label=f"Run {_name}")
        uis[_name]={"dv":_dv_in,"dacc":_dacc_in,"ang":_ang_in,"btn":_btn}
        _rows.append(mo.hstack([mo.md(f"**{_name}**"),_dv_in,_dacc_in,_ang_in,_btn],justify="start",gap=0.5))
    panel=mo.vstack([mo.md(f"**Good params for {label}**:"),mo.vstack(_rows,gap=0.5),mo.md(f"Filter ≥ {min_len_input.value} fr")],gap=0.4)
    panel
    return (uis,)


@app.cell
def _(Path, is_script, time):
    from openptv2.benchmarking.runner import run_tracker
    from openptv2.benchmarking.metrics import compute_physics_metrics
    def resolve_dataset(label):
        root=Path("test_data").resolve()
        if "synthetic" in label:
            return (root/"synthetic_turbulent_1k"/"parameters_Run1.yaml").resolve(),"synthetic_turbulent_1k"
        else:
            return (root/"proptv_500_25"/"parameters_Run1.yaml").resolve(),"proptv_500_25"
    def parse_float(txt, default):
        try:
            return float(txt.value.strip())
        except Exception:
            return default
    def tracks_to_plotly(pred, title, min_len=2):
        # Use Visualize 3D trajectories (openptv2.gui.plot_3d_trajectories:39) – matplotlib Figure, tight to cloud
        import numpy as np
        from openptv2.gui.plot_3d_trajectories import build_3d_trajectories_figure
        tids = list(pred.keys())
        filt = [pts for pts in pred.values() if len(pts) >= min_len]
        if not filt:
            filt = [pts for pts in pred.values() if pts]
        if len(filt) > 800:
            filt = filt[:800]
        trajs = [np.array([[x, y, z] for _, x, y, z in sorted(pts, key=lambda p: p[0])]) for pts in filt]
        fig = build_3d_trajectories_figure(trajs)
        fig.suptitle(title + f" — {len(filt)}/{len(tids)} kept (≥{min_len}fr)", fontsize=10)
        return fig
    def run_one(yaml_path,tracker,dv,dacc,ang):
        ov={"dvxmax":dv,"dvxmin":-dv,"dvymax":dv,"dvymin":-dv,"dvzmax":dv,"dvzmin":-dv,"dacc":dacc,"angle":ang}
        t0=time.perf_counter()
        try:
            pred=run_tracker(yaml_path,tracker,track_overrides=ov)
            pm=compute_physics_metrics(pred)
            return {"error":None,"pred":pred,"pm":pm,"time_s":round(time.perf_counter()-t0,2)}
        except Exception as e:
            return {"error":str(e)[:600],"pred":{},"pm":None,"time_s":round(time.perf_counter()-t0,2)}
    if is_script:
        yp,_=resolve_dataset("synthetic_turbulent_1k (10 of 20, 1k/frame)")
        demo=run_one(yp,"priority_segment_3d",10,5,120)
        print(f"[script] dense {demo['pm'].mean_track_length if demo['pm'] else 'ERR'}")
    return parse_float, resolve_dataset, run_one, tracks_to_plotly


@app.cell
def _(
    ALL_TRACKERS,
    dataset_picker,
    is_script,
    last_clicked,
    min_len_input,
    mo,
    parse_float,
    resolve_dataset,
    run_one,
    tracks_to_plotly,
    uis,
):
    yaml_path,dlabel=resolve_dataset(dataset_picker.value)
    try:
        min_len=int(min_len_input.value.strip())
    except Exception:
        min_len=5
    _outputs=[]
    _auto=is_script
    _any_clicked=any(uis[_n]["btn"].value for _n in ALL_TRACKERS)
    _show_demo=not _any_clicked
    for _name in ALL_TRACKERS:
        _ui=uis[_name]
        _triggered=_auto or _ui["btn"].value or (_show_demo and _name==ALL_TRACKERS[0])
        if _auto and _name!=ALL_TRACKERS[0]:
            if _ui["btn"].value==0:
                continue
            _auto=False
        if _show_demo and _name!=ALL_TRACKERS[0]:
            continue
        if not _triggered:
            continue
        if _auto:
            _auto=False
        _dv=parse_float(_ui["dv"],2.0); _dacc=parse_float(_ui["dacc"],0.5); _ang=parse_float(_ui["ang"],60)
        _info=run_one(yaml_path,_name,_dv,_dacc,_ang)
        if _info["error"]:
            _outputs.append(mo.vstack([mo.md(f"### {_name} — ERROR"),mo.md(f"`{_info['error']}`")]))
            continue
        _pm=_info["pm"]
        _fig=tracks_to_plotly(_info["pred"],f"{_name} {dlabel} ({len(_info['pred'])} tracks)",min_len=min_len)
        _header=f"### {_name} — dv={_dv} dacc={_dacc} ang={_ang} → mean={_pm.mean_track_length:.2f} frac10={_pm.frac_tracks_over_10:.2f} kurt={_pm.acceleration_kurtosis:.1f} n={_pm.n_tracks}"
        _outputs.append(mo.vstack([mo.md(_header),_fig],gap=0.5))
    _result=mo.vstack([mo.md("_Edit text boxes, Run per tracker._"), mo.vstack(_outputs,gap=1) if _outputs else mo.md("*Press Run*")],gap=0.8)
    _result
    return


@app.cell
def _(ALL_TRACKERS, defaults_1k, defaults_proptv, mo):
    hdr="| tracker | 1k dv/dacc/ang | proptv dv/dacc/ang |\n|---|---|---|\n"
    rows=[]
    for _n in ALL_TRACKERS:
        rows.append(f"| `{_n}` | {'/'.join(defaults_1k[_n])} | {'/'.join(defaults_proptv[_n])} |")
    mo.md("**Good params kept in notebook** (auto-switch on Dataset):\n\n"+hdr+"\n".join(rows))
    return


if __name__ == "__main__":
    app.run()
