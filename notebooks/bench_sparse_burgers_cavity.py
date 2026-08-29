# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
#     "openptv2==0.5.6",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from pathlib import Path

    import marimo as mo

    from openptv2.benchmarking.metrics import compute_physics_metrics
    from openptv2.benchmarking.param_search import find_smooth_params
    from openptv2.benchmarking.runner import run_tracker
    from openptv2.gui.plot_3d_trajectories import build_3d_trajectories_figure


    return (
        Path,
        build_3d_trajectories_figure,
        compute_physics_metrics,
        find_smooth_params,
        mo,
        run_tracker,
        time,
    )


@app.cell
def _(mo):
    is_script = mo.app_meta().mode == "script"
    return (is_script,)


@app.cell
def _(mo):
    mo.md("""
    # Tracker Bench — Burgers / test_cavity / synthetic_turbulent_1k

    Pick a **dataset**, then a **tracker**, then tune **dv / dacc / angle** and press **Run**.
    Plot shows **only ≥ min_len** 3D trajectories with tight limits.

    `proptv_500_25`/`_30` are **not** in this list: they have no calibration in
    this repo and aren't run through `run_tracker` at all — they're consumed
    directly by `openptv2.plugins.proptv_tracking.ProPTVTracker` from
    pre-triangulated 3D positions (`origin_*.txt`), a different tracker with a
    different parameter surface (`Vmin`/`Vmax`/`maxvel`/... not dv/dacc/angle).
    """)
    return


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options=[
            "burgers (5 fr, vortex)",
            "test_cavity (4 fr, sparse)",
            "synthetic_turbulent_1k (10 fr, 1k/frame)",
        ],
        value="burgers (5 fr, vortex)",
        label="Dataset",
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
    return ALL_TRACKERS, defaults_1k, defaults_burgers, defaults_cavity


@app.cell
def _(mo):
    min_len_input = mo.ui.text(value="5", label="Min length to plot (frames)")
    min_len_input
    return (min_len_input,)


@app.cell
def _(ALL_TRACKERS, dataset_picker, mo):
    if "burgers" in dataset_picker.value:
        dataset_key = "burgers"
        mode_label = "Burgers (vortex)"
    elif "test_cavity" in dataset_picker.value:
        dataset_key = "test_cavity"
        mode_label = "test_cavity (sparse)"
    else:
        dataset_key = "synthetic_1k"
        mode_label = "synthetic_turbulent_1k (dense)"

    # Select the tracker first (own cell: its .value can't be read here)
    tracker_dropdown = mo.ui.dropdown(
        options=ALL_TRACKERS,
        value=ALL_TRACKERS[0] if ALL_TRACKERS else "priority_segment_3d",
        label="Select Tracker",
    )
    mo.vstack([mo.md(f"### {mode_label}"), tracker_dropdown], gap=0.5)
    return dataset_key, tracker_dropdown


@app.cell
def _(
    dataset_key,
    defaults_1k,
    defaults_burgers,
    defaults_cavity,
    min_len_input,
    mo,
    tracker_dropdown,
):
    src = {"burgers": defaults_burgers, "test_cavity": defaults_cavity, "synthetic_1k": defaults_1k}[dataset_key]

    # Params for the selected tracker
    selected_tracker = tracker_dropdown.value
    _dv_def, _da_def, _ang_def = src.get(selected_tracker, ("2.0", "0.5", "60"))

    dv_in = mo.ui.text(value=_dv_def, label="dvxmax")
    dacc_in = mo.ui.text(value=_da_def, label="dacc")
    ang_in = mo.ui.text(value=_ang_def, label="angle")

    btn = mo.ui.run_button(label=f"Run {selected_tracker}", kind="success")

    panel = mo.vstack([
        mo.hstack([dv_in, dacc_in, ang_in, btn], justify="start", align="center", gap=1.0),
        mo.md(f"Filter ≥ {min_len_input.value} fr")
    ], gap=0.5)

    panel
    return ang_in, btn, dacc_in, dv_in, selected_tracker


@app.cell
def _(mo):
    mo.md("""
    #### How to tune dv / dacc / angle (manually, or via **Auto-tune** below)

    There's no gradient here — the tracker is a discrete, non-differentiable
    black box, so nudging a parameter can jump the result discontinuously.
    Search it like you'd tune any noisy black box: **start small, grow until
    it stops helping, then back off a step or two.**

    1. Start with **dv small** (tighter than you think you need) and
       **dacc, angle also small**.
    2. Increase **dv** a step at a time (e.g. ×1.5 each time). Watch
       `mean_len` and `kurt` in the result header. `mean_len` should climb;
       once it stops climbing (or `kurt` — acceleration kurtosis, ~3 is
       Gaussian/smooth — starts climbing steeply instead) you've hit
       saturation: more `dv` is now buying you spurious long-jump links, not
       real ones.
    3. **Stop and back up 1–2 steps** from where it saturated. That's your
       working `dv`.
    4. Repeat the same climb-then-back-off for **dacc**, then **angle**,
       holding the others fixed at what you just picked.
    5. Note: `angle` is a no-op for `priority_segment_3d`/`4be`/`two_phase`
       (the 3D-only search box has no angle gate) — only the `dv*` box and,
       via postprocess gap-relinking, `dacc` matter for those. It's fully
       active for `standard_forward`/`two_directional`/`full_multipass`.

    **Auto-tune** runs exactly this procedure for you (`dv` → `dacc` →
    `angle`, each grown ×1.6 per step, 2-step patience, 1-step back-off),
    scoring by `mean_track_length` penalized for acceleration kurtosis above
    ~3 — i.e. it favors long trajectories that don't secretly contain
    physically-impossible mid-track jumps.
    """)
    return


@app.cell
def _(mo):
    autotune_btn = mo.ui.run_button(label="Auto-tune dv / dacc / angle", kind="warn")
    autotune_btn
    return (autotune_btn,)


@app.cell
def _(
    autotune_btn,
    dataset_picker,
    find_smooth_params,
    resolve_dataset,
    selected_tracker,
    mo,
):
    if autotune_btn.value:
        _yaml_path, _dlabel = resolve_dataset(dataset_picker.value)
        _result = find_smooth_params(_yaml_path, selected_tracker)
        _rows = "\n".join(
            f"| {_s.param} | {_s.value:.3g} | {_s.score:.3g} |" for _s in _result.history
        )
        _out = mo.vstack([
            mo.md(
                f"### Auto-tuned {selected_tracker} on {_dlabel} → "
                f"**dv={_result.dv:.3g} dacc={_result.dacc:.3g} angle={_result.angle:.3g}** "
                f"(score={_result.score:.3g})"
            ),
            mo.md("Copy these into the dv / dacc / angle boxes above, then press Run to plot them."),
            mo.md("| param | value tried | score |\n|---|---|---|\n" + _rows),
        ], gap=0.4)
    else:
        _out = mo.md("_Press **Auto-tune** to search dv/dacc/angle for the selected dataset + tracker._")
    _out
    return


@app.cell
def _(
    ang_in,
    btn,
    dacc_in,
    dataset_picker,
    dv_in,
    min_len_input,
    mo,
    parse_float,
    resolve_dataset,
    run_one,
    selected_tracker,
    tracks_to_plotly,
):
    yaml_path, dlabel = resolve_dataset(dataset_picker.value)
    try:
        min_len = int(min_len_input.value.strip())
    except Exception:
        min_len = 5

    if btn.value:
        _dv = parse_float(dv_in, 2.0)
        _dacc = parse_float(dacc_in, 0.5)
        _ang = parse_float(ang_in, 60.0)

        _info = run_one(yaml_path, selected_tracker, _dv, _dacc, _ang)

        if _info["error"]:
            _result = mo.vstack([
                mo.md(f"### {selected_tracker} — ERROR"),
                mo.md(f"`{_info['error']}`")
            ])
        else:
            _pm = _info["pm"]
            _header = f"### {selected_tracker} — dv={_dv} dacc={_dacc} ang={_ang} → mean_len={_pm.mean_track_length:.2f} frac10={_pm.frac_tracks_over_10:.2f} kurt={_pm.acceleration_kurtosis:.1f} n={_pm.n_tracks} t={_info['time_s']}s"
            _detail = f"Output: {dlabel} dv={_dv} dacc={_dacc} ang={_ang} tracks={_pm.n_tracks} mean={_pm.mean_track_length:.2f} plotted ≥{min_len}fr"
            _fig = tracks_to_plotly(_info["pred"], f"{selected_tracker} {dlabel} ({len(_info['pred'])} tracks)", min_len=min_len)
            _result = mo.vstack([
                mo.md(_header),
                mo.md(f"`{_detail}`"),
                mo.mpl.interactive(_fig)
            ], gap=0.5)
    else:
        _result = mo.md("_Select a tracker from the dropdown list, adjust parameters, and click **Run**._")

    _result
    return


@app.cell
def _(
    Path,
    build_3d_trajectories_figure,
    compute_physics_metrics,
    is_script,
    run_tracker,
    time,
):
    def resolve_dataset(label):
        root = Path("test_data").resolve()
        if "burgers" in label:
            name = "burgers"
        elif "test_cavity" in label:
            name = "test_cavity"
        else:
            name = "synthetic_turbulent_1k"
        return (root / name / "parameters_Run1.yaml").resolve(), name

    def parse_float(txt, default):
        try:
            return float(txt.value.strip())
        except Exception:
            return default

    def tracks_to_plotly(pred, title, min_len=2):
        # Use Visualize 3D trajectories (openptv2.gui.plot_3d_trajectories:39) – matplotlib Figure, tight to cloud
        import numpy as np
        tids = list(pred.keys())
        # filter to >=min_len, but show singletons as well if nothing else (was empty 2D)
        filt = [pts for pts in pred.values() if len(pts) >= min_len]
        if not filt:
            filt = [pts for pts in pred.values() if pts]
        # cap for perf
        if len(filt) > 800:
            filt = filt[:800]
        trajs = [np.array([[x, y, z] for _, x, y, z in sorted(pts, key=lambda p: p[0])]) for pts in filt]
        # build_3d_trajectories_figure expects (N,3) arrays in mm
        fig = build_3d_trajectories_figure(trajs)
        # title + tight 3D view (burgers already tight)
        fig.suptitle(title + f" — {len(filt)}/{len(tids)} kept (≥{min_len}fr)", fontsize=10)
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
def _(ALL_TRACKERS, defaults_1k, defaults_burgers, defaults_cavity, mo):
    hdr="| tracker | Burgers dv/dacc/ang | test_cavity dv/dacc/ang | synthetic_1k dv/dacc/ang |\n|---|---|---|---|\n"
    rows=[]
    for _n in ALL_TRACKERS:
        rows.append(f"| `{_n}` | {'/'.join(defaults_burgers[_n])} | {'/'.join(defaults_cavity[_n])} | {'/'.join(defaults_1k[_n])} |")
    mo.md("**Good params kept in notebook** (auto-switch on Dataset):\n\n"+hdr+"\n".join(rows))
    return


if __name__ == "__main__":
    app.run()
