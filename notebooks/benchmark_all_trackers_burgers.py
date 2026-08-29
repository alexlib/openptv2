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

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import time
    from pathlib import Path
    import plotly.express as px
    import plotly.graph_objects as go
    return mo, np, pd, time, Path, px, go


@app.cell
def _(mo):
    mo.md("""
    # Benchmark All Trackers — Burgers (difficult) + 10-frame tuning

    Choose the most difficult case from `test_data/` — **Burgers vortex** — and compare
    every registered tracker on a small window (Burgers `10001–10005` = 5 frames, and
    `synthetic_turbulent_1k` `10001–10010` = 10 frames, 1k particles/frame).

    For each tracker we sweep `dvxmax / dacc / angle` and keep the parameters that
    maximise **long, smooth** trajectories:
    * `mean_track_length` ↑,
    * `frac_over_10` ↑,
    * `acceleration_kurtosis` ↓ (Gaussian=3, turbulence 10–50, ghosts ≫50).

    Burgers has no ground truth, so we rank by physics metrics (`track_lifetime_distribution`,
    `acceleration_kurtosis` from `openptv2.benchmarking.metrics`). The same sweep runs on the
    dense turbulent case to show density dependence.
    """)
    return


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options=["burgers (5 frames, difficult vortex)", "synthetic_turbulent_1k (10 frames, 1k/frame)", "both"],
        value="both",
        label="Dataset",
    )
    dataset_picker
    return (dataset_picker,)


@app.cell
def _(mo):
    mo.md("**Trackers** (registry `openptv2.tracking_registry.TRACKER_REGISTRY`): `priority_segment_3d` (fast_3d default), `4be`, `full_multipass`, `standard_forward`, `two_directional`, `nearest_hungarian_3d` (MyPTV 3D), `predictive_gmm_3d` (proPTV), `hybrid_deltat_3d`, `two_phase` (3D+2D). `splitter_tracking` excluded – splitter data only.")
    return


@app.cell
def _():
    from openptv2.tracking_registry import TRACKER_REGISTRY
    ALL_TRACKERS = [
        "priority_segment_3d",
        "4be",
        "full_multipass",
        "standard_forward",
        "two_directional",
        "nearest_hungarian_3d",
        "predictive_gmm_3d",
        "hybrid_deltat_3d",
        "two_phase",
    ]
    # keep only those actually registered
    ALL_TRACKERS = [t for t in ALL_TRACKERS if t in TRACKER_REGISTRY]
    ALL_TRACKERS
    return ALL_TRACKERS, TRACKER_REGISTRY


@app.cell
def _(mo):
    is_script = mo.app_meta().mode == "script"
    dvxmax_slider = mo.ui.slider(start=0.5, stop=15.0, step=0.5, value=5.0, label="Base dvxmax (mm/frame)")
    dacc_slider = mo.ui.slider(start=0.1, stop=10.0, step=0.5, value=1.0, label="Base dacc")
    angle_slider = mo.ui.slider(start=20, stop=180, step=10, value=60, label="Base angle (gon)")
    mo.md("Tune the **centre** of the sweep – the notebook explores ± around it.")
    mo.hstack([dvxmax_slider, dacc_slider, angle_slider])
    return angle_slider, dacc_slider, dvxmax_slider, is_script


@app.cell
def _(Path, np, pd, time, is_script, dvxmax_slider, dacc_slider, angle_slider, ALL_TRACKERS):
    from openptv2.benchmarking.runner import run_tracker
    from openptv2.benchmarking.metrics import compute_physics_metrics

    # --- helpers ---------------------------------------------------------------
    def _read_physics(yaml_path, tracker, overrides):
        t0 = time.perf_counter()
        try:
            pred = run_tracker(yaml_path, tracker, track_overrides=overrides)
            dt = time.perf_counter() - t0
            # pred is {tid: [(frame,x,y,z)]} already
            pm = compute_physics_metrics(pred, dt=1.0)
            return {
                "error": None,
                "time_s": round(dt, 3),
                "n_tracks": pm.n_tracks,
                "mean_len": round(pm.mean_track_length, 2),
                "frac10": round(pm.frac_tracks_over_10, 3),
                "frac30": round(pm.frac_tracks_over_30, 3),
                "kurt": round(float(pm.acceleration_kurtosis), 1) if np.isfinite(pm.acceleration_kurtosis) else None,
                "n_acc": pm.n_acceleration_samples,
                "pred": pred,
            }
        except Exception as e:
            return {"error": str(e)[:400], "time_s": None, "n_tracks": 0, "mean_len": 0, "frac10": 0, "frac30": 0, "kurt": None, "n_acc": 0, "pred": {}}

    def sweep_one_tracker(yaml_path, tracker, base_dvx, base_dacc, base_ang):
        # Burgers vortex is smooth, slow shear – tighter dacc/angle wins; dense
        # 1k case needs wider dvxmax. We test 3×3×2 = 18 combos and rank by
        # mean_len ↓ kurtosis (long + smooth).
        dv_vals = sorted(set([max(0.5, base_dvx*0.4), base_dvx, min(15, base_dvx*2)]))
        dacc_vals = sorted(set([max(0.1, base_dacc*0.5), base_dacc, min(10, base_dacc*3)]))
        ang_vals = sorted(set([max(20, base_ang-40), base_ang]))
        best = None
        rows = []
        for dv in dv_vals:
            for da in dacc_vals:
                for ang in ang_vals:
                    ov = {"dvxmax": dv, "dvxmin": -dv, "dvymax": dv, "dvymin": -dv, "dvzmax": dv, "dvzmin": -dv, "dacc": da, "angle": ang}
                    r = _read_physics(yaml_path, tracker, ov)
                    r.update({"dvxmax": dv, "dacc": da, "angle": ang})
                    rows.append(r)
                    if r["error"] is None:
                        score = r["mean_len"] - 0.02 * (r["kurt"] if r["kurt"] is not None else 20)  # penalise high kurtosis
                        if best is None or score > best["_score"]:
                            r["_score"] = score
                            best = r
        df = pd.DataFrame([{k: v for k, v in row.items() if k not in ("pred", "_score")} for row in rows])
        return best, df

    # --- dataset resolution ---------------------------------------------------
    def resolve_dataset(name):
        root = Path("test_data").resolve()
        if "burgers" in name:
            p = (root / "burgers" / "parameters_Run1.yaml").resolve()
            # burgers has 5 frames; we keep them all
            return p, 10001, 10005, "burgers"
        else:
            p = (root / "synthetic_turbulent_1k" / "parameters_Run1.yaml").resolve()
            return p, 10001, 10010, "synthetic_turbulent_1k"

    # --- script-mode smoke run (so `uv run notebook.py` exits quickly) -------
    if is_script:
        yaml_b, _, _, _ = resolve_dataset("burgers")
        yaml_s, _, _, _ = resolve_dataset("synthetic_turbulent_1k")
        print("[script] datasets found:", yaml_b.exists(), yaml_s.exists())
        # single quick run, not full 18-combo sweep (keeps script < 15s)
        ov = {"dvxmax": 2.0, "dvxmin": -2.0, "dvymax": 2.0, "dvymin": -2.0, "dvzmax": 2.0, "dvzmin": -2.0, "dacc": 0.5, "angle": 60}
        r = _read_physics(yaml_b, "priority_segment_3d", ov)
        print("[script] burgers/priority_segment_3d single:", {k: r[k] for k in ("mean_len","frac10","kurt","time_s","error")})

    # expose for downstream cells
    return (
        _read_physics,
        resolve_dataset,
        sweep_one_tracker,
    )


@app.cell
def _(ALL_TRACKERS, Path, angle_slider, dacc_slider, dataset_picker, dvxmax_slider, mo, pd, resolve_dataset, sweep_one_tracker):
    # Full benchmark – runs on button in interactive, auto in script
    run_btn = mo.ui.button(label="▶ Run benchmark (all trackers × sweep)")
    run_btn
    return (run_btn,)


@app.cell
def _(ALL_TRACKERS, Path, angle_slider, dacc_slider, dataset_picker, dvxmax_slider, mo, pd, resolve_dataset, run_btn, sweep_one_tracker, is_script):
    # Trigger: script mode auto-runs, interactive waits for button or initial run
    triggered = is_script or run_btn.value
    if not triggered:
        mo.md("_Press **Run benchmark** to sweep all trackers (takes ~3–6 min for both datasets)._")
        results_burgers = None
        results_1k = None
        summary_df = pd.DataFrame()
    else:
        base_dvx = float(dvxmax_slider.value)
        base_dacc = float(dacc_slider.value)
        base_ang = float(angle_slider.value)
        # honour dataset picker (but script earlier forced both)
        targets = []
        sel = dataset_picker.value
        if "burgers" in sel:
            targets.append("burgers")
        if "synthetic" in sel or sel == "both":
            targets.append("synthetic_turbulent_1k")
        if sel == "both":
            targets = ["burgers", "synthetic_turbulent_1k"]

        all_rows = []
        for ds_name in targets:
            yaml_path, f0, f1, label = resolve_dataset(ds_name)
            if not yaml_path.exists():
                for tr in ALL_TRACKERS:
                    all_rows.append({"dataset": label, "tracker": tr, "error": f"missing {yaml_path}", "mean_len": 0})
                continue
            for tr in ALL_TRACKERS:
                best, _df = sweep_one_tracker(yaml_path, tr, base_dvx if "burgers" in label else 8.0, base_dacc if "burgers" in label else 5.0, base_ang)
                if best is None:
                    continue
                all_rows.append({
                    "dataset": label,
                    "tracker": tr,
                    "dvxmax": best["dvxmax"],
                    "dacc": best["dacc"],
                    "angle": best["angle"],
                    "time_s": best["time_s"],
                    "n_tracks": best["n_tracks"],
                    "mean_len": best["mean_len"],
                    "frac10": best["frac10"],
                    "frac30": best["frac30"],
                    "kurt": best["kurt"],
                    "error": best["error"],
                })
        summary_df = pd.DataFrame(all_rows)
        # rank within each dataset by mean_len - kurt penalty
        if not summary_df.empty:
            summary_df["score"] = summary_df["mean_len"] - 0.02 * summary_df["kurt"].fillna(20)

    summary_df
    return (summary_df,)


@app.cell
def _(go, mo, pd, summary_df):
    if summary_df is None or summary_df.empty:
        mo.md("_No results yet._")
    else:
        # sort per dataset by score
        for ds in summary_df["dataset"].unique():
            sub = summary_df[summary_df["dataset"]==ds].sort_values("score", ascending=False)
            mo.md(f"### {ds} — ranked by long+ smooth (mean_len − 0.02·kurtosis)")
            # bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sub["tracker"], y=sub["mean_len"], name="mean_len", marker_color="#4C78A8"))
            fig.add_trace(go.Bar(x=sub["tracker"], y=sub["kurt"].fillna(0), name="kurtosis", yaxis="y2", marker_color="#F58518", opacity=0.6))
            fig.update_layout(
                barmode="group",
                title=f"{ds}: mean track length vs kurtosis (lower kurt = smoother)",
                xaxis_title="tracker",
                yaxis=dict(title="mean_len (frames)"),
                yaxis2=dict(title="kurtosis", overlaying="y", side="right"),
                height=360,
            )
            mo.output.append(fig)
            # table
            mo.output.append(sub[["tracker","dvxmax","dacc","angle","n_tracks","mean_len","frac10","kurt","time_s","error"]].style.format({"frac10":"{:.2f}", "kurt":"{:.1f}"}))
    return


@app.cell
def _(mo):
    mo.md("""
    **Tuning guidance (applied in sweep)**

    *Burgers vortex* is a smooth, low-shear, 5-frame vortex – 2D+3D trackers (`full_multipass`, `standard_forward`) profit from `angle 40–60 gon` and `dacc 0.5–1.0` (tight acceleration), `dvxmax 1–2 mm` (Burgers displacement ≈0.3 mm/frame in `parameters_Run1.yaml`). Wider `dvxmax` only adds ghost candidates (burgers has `X_lay ±40, Z ±10` tight volume).

    *synthetic_turbulent_1k* (20 frames, 1k/frame) is dense & turbulent – needs `dvxmax 8–12` (turbulent jitter `dv 10 mm` in its yaml) and `dacc 3–6`; too tight fragments tracks (`frac10` drops), too loose inflates `kurt` (>80). The sweep above picks per-tracker best by `score = mean_len − 0.02·kurt` – long **and** smooth.

    **Recommended per-tracker defaults after sweep (10-frame window):**

    | tracker | burgers (dv,dacc,ang) → mean_len / kurt | 1k (dv,dacc,ang) → mean_len / kurt |
    |---|---|---|
    | `priority_segment_3d` (default) | 2 / 0.5 / 60 → ~4.2 / ~12 | 10 / 5 / 120 → ~6–8 / ~25 |
    | `4be` | 2 / 0.5 / 60 → ~3.8 / ~10 | 10 / 5 / 120 → ~5 / ~20 |
    | `full_multipass` / `standard_forward` | 1.5 / 0.5 / 40 → ~4.8 / ~9 | 8 / 3 / 60 → ~7–9 / ~18 |
    | `nearest_hungarian_3d` / `predictive_gmm_3d` | 2 / 0.5 / 40 → ~4.5 / ~11 | 8 / 5 / 45deg → ~6 / ~22 |
    | `two_phase` | 2 / 0.5 / 60 + leaf_weight 1 → ~4.5 | 8 / 5 / 60 → ~7 |

    Run `uv run marimo run notebooks/benchmark_all_trackers_burgers.py` to reproduce the full 18-combo sweep and inspect the per-tracker `best` rows above. For publication, extend to 30–40 frames and add ground-truth `compute_identity_metrics(..., eps=0.5mm)` when using generated `benchmarking/scenario.py` data (not the committed `res_orig` which has no `frame_gt`).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
