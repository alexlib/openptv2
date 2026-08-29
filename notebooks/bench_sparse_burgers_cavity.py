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
    # Sparse Bench — Burgers + test_cavity

    Tight, smooth, low-density. Text boxes (not sliders) per tracker + **Run** per tracker. Plot shows **only ≥ min_len** 3D trajectories with tight limits.
    """)
    return


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options=["burgers (5 fr, vortex)", "test_cavity (4 fr, sparse)"],
        value="burgers (5 fr, vortex)",
        label="Sparse dataset",
    )
    dataset_picker
    return (dataset_picker,)


@app.cell
def _():
    from openptv2.tracking_registry import TRACKER_REGISTRY
    ALL_TRACKERS = [t for t in ["priority_segment_3d","4be","full_multipass","standard_forward","two_directional","nearest_hungarian_3d","predictive_gmm_3d","hybrid_deltat_3d","two_phase"] if t in TRACKER_REGISTRY]
    defaults_burgers = {
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
    defaults_cavity = {
        "priority_segment_3d": ("2.0","0.8","60"),
        "4be": ("2.0","0.8","60"),
        "full_multipass": ("1.5","0.8","60"),
        "standard_forward": ("1.5","0.8","60"),
        "two_directional": ("1.5","0.8","60"),
        "nearest_hungarian_3d": ("2.0","0.8","40"),
        "predictive_gmm_3d": ("2.0","0.8","40"),
        "hybrid_deltat_3d": ("2.0","0.8","60"),
        "two_phase": ("2.0","0.8","60"),
    }
    return ALL_TRACKERS, defaults_burgers, defaults_cavity


@app.cell
def _(mo):
    min_len_input = mo.ui.text(value="5", label="Min length to plot (frames)")
    min_len_input
    return (min_len_input,)


@app.cell
def _(
    ALL_TRACKERS,
    dataset_picker,
    defaults_burgers,
    defaults_cavity,
    min_len_input,
    mo,
):
    is_burgers = "burgers" in dataset_picker.value
    src = defaults_burgers if is_burgers else defaults_cavity
    mode_label = "Burgers (vortex)" if is_burgers else "test_cavity (sparse)"
    uis = {}
    _rows = []
    for _name in ALL_TRACKERS:
        _dv, _da, _ang = src[_name]
        _dv_in = mo.ui.text(value=_dv, label="dvxmax")
        _dacc_in = mo.ui.text(value=_da, label="dacc")
        _ang_in = mo.ui.text(value=_ang, label="angle")
        _btn = mo.ui.button(label=f"Run {_name}")
        uis[_name] = {"dv": _dv_in, "dacc": _dacc_in, "ang": _ang_in, "btn": _btn}
        _rows.append(mo.hstack([mo.md(f"**{_name}**"), _dv_in, _dacc_in, _ang_in, _btn], justify="start", gap=0.5))
    panel = mo.vstack([mo.md(f"**Good params for {mode_label}** (edit, Run per tracker):"), mo.vstack(_rows, gap=0.5), mo.md(f"Filter ≥ {min_len_input.value} fr")], gap=0.4)
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
            return (root / "test_cavity" / "parameters_Run1.yaml").resolve(), "test_cavity"
    def parse_float(txt, default):
        try:
            return float(txt.value.strip())
        except Exception:
            return default
    def tracks_to_plotly(pred, title, min_len=5):
        fig = go.Figure()
        tids = list(pred.keys())
        if len(tids) > 2000:
            tids = tids[:2000]
        has_lines=False; sx,sy,sz,stext=[],[],[],[]; kept=0; all_x,all_y,all_z=[],[],[]
        for tid in tids:
            pts = sorted(pred[tid], key=lambda p: p[0])
            if len(pts) < min_len:
                if 1 < len(pts) < min_len:
                    for p in pts:
                        sx.append(p[1]); sy.append(p[2]); sz.append(p[3])
                    all_x.extend([p[1] for p in pts]); all_y.extend([p[2] for p in pts]); all_z.extend([p[3] for p in pts])
                elif len(pts)==1:
                    sx.append(pts[0][1]); sy.append(pts[0][2]); sz.append(pts[0][3]); stext.append(f"tr{tid} singleton"); all_x.append(pts[0][1]); all_y.append(pts[0][2]); all_z.append(pts[0][3])
                continue
            has_lines=True; kept+=1
            xs,ys,zs = [p[1] for p in pts],[p[2] for p in pts],[p[3] for p in pts]
            all_x.extend(xs); all_y.extend(ys); all_z.extend(zs)
            fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode="lines",line=dict(width=4),showlegend=False,hoverinfo="text",text=[f"f{f}" for f,_,_,_ in pts]))
        if sx and not (has_lines and min_len>1):
            fig.add_trace(go.Scatter3d(x=sx,y=sy,z=sz,mode="markers",marker=dict(size=2,color="rgba(200,0,0,0.35)"),name=f"{len(sx)} short"))
        _t = title + (f" — {kept}/{len(tids)} kept (≥{min_len}fr)" if has_lines or sx else "")
        if all_x:
            xmin,xmax=min(all_x),max(all_x); ymin,ymax=min(all_y),max(all_y); zmin,zmax=min(all_z),max(all_z)
            pad=0.07; xr=(xmax-xmin) or 1; yr=(ymax-ymin) or 1; zr=(zmax-zmin) or 1
            scene=dict(xaxis=dict(title="x mm",range=[xmin-pad*xr,xmax+pad*xr]),yaxis=dict(title="y mm",range=[ymin-pad*yr,ymax+pad*yr]),zaxis=dict(title="z mm",range=[zmin-pad*zr,zmax+pad*zr]),aspectmode="data",camera=dict(eye=dict(x=1.6,y=1.6,z=0.9)))
        else:
            scene=dict(xaxis_title="x mm",yaxis_title="y mm",zaxis_title="z mm",aspectmode="data",camera=dict(eye=dict(x=1.6,y=1.6,z=0.9)))
        fig.update_layout(title=_t,scene=scene,height=560,margin=dict(l=0,r=0,b=0,t=40))
        return fig
    def run_one(yaml_path, tracker, dv, dacc, ang):
        ov={"dvxmax":dv,"dvxmin":-dv,"dvymax":dv,"dvymin":-dv,"dvzmax":dv,"dvzmin":-dv,"dacc":dacc,"angle":ang}
        t0=time.perf_counter()
        try:
            pred=run_tracker(yaml_path,tracker,track_overrides=ov)
            pm=compute_physics_metrics(pred)
            return {"error":None,"pred":pred,"pm":pm,"time_s":round(time.perf_counter()-t0,2)}
        except Exception as e:
            return {"error":str(e)[:600],"pred":{},"pm":None,"time_s":round(time.perf_counter()-t0,2)}
    _script_demo=None
    if is_script:
        yp,_=resolve_dataset("burgers (5 frames, vortex)")
        demo=run_one(yp,"priority_segment_3d",2.0,0.5,60)
        print(f"[script] burgers {demo['pm'].mean_track_length if demo['pm'] else 'ERR'}")
        _script_demo=demo
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
    yaml_path,dlabel = resolve_dataset(dataset_picker.value)
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
        _header=f"### {_name} — dv={_dv} dacc={_dacc} ang={_ang} → mean_len={_pm.mean_track_length:.2f} frac10={_pm.frac_tracks_over_10:.2f} kurt={_pm.acceleration_kurtosis:.1f} n={_pm.n_tracks} t={_info['time_s']}s"
        _detail=f"Output: {dlabel} dv={_dv} dacc={_dacc} ang={_ang} tracks={_pm.n_tracks} mean={_pm.mean_track_length:.2f} plotted ≥{min_len}fr"
        _fig=tracks_to_plotly(_info["pred"],f"{_name} {dlabel} ({len(_info['pred'])} tracks)",min_len=min_len)
        _outputs.append(mo.vstack([mo.md(_header),mo.md(f"`{_detail}`"),_fig],gap=0.5))
    _result=mo.vstack([mo.md("_Edit text boxes, Run per tracker — 3D tight to cloud._"), mo.vstack(_outputs,gap=1) if _outputs else mo.md("*Press Run*")],gap=0.8)
    _result
    return


@app.cell
def _(ALL_TRACKERS, defaults_burgers, defaults_cavity, mo):
    hdr="| tracker | Burgers dv/dacc/ang | test_cavity dv/dacc/ang |\n|---|---|---|\n"
    rows=[]
    for _n in ALL_TRACKERS:
        rows.append(f"| `{_n}` | {'/'.join(defaults_burgers[_n])} | {'/'.join(defaults_cavity[_n])} |")
    mo.md("**Good params kept in notebook** (auto-switch on Dataset):\n\n"+hdr+"\n".join(rows))
    return


if __name__ == "__main__":
    app.run()
