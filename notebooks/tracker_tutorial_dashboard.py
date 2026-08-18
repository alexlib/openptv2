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

"""Interactive companion to docs/tracker-tutorials.md.

Runs the actual openptv2 tracker engines (not a simulation of them) on the
clean and realistic-noise proPTV benchmark case, and lets you change each
tracker's parameters live to see the effect on the recovered turbulence
statistics (K_a, a_rms error) immediately -- the same metrics and the same
scripts/bench_with_without_noise.py machinery the static tutorial's numbers
came from, just reactive.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    _scripts = Path("scripts").absolute()
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))

    import adapt_proptv_dataset as apd
    import benchmark_utils as bu
    from bench_proptv_kinematics import kinematics, stats
    from openptv2.tracking_registry import TRACKER_REGISTRY

    return Path, TRACKER_REGISTRY, apd, bu, kinematics, mo, np, pd, stats


@app.cell
def _(mo):
    mo.md(r"""
    # Tracker Tutorial Dashboard

    Companion to `docs/tracker-tutorials.md`. Same 30-frame, 500-particle
    proPTV-derived case, same two datasets ("clean" = ground-truth
    correspondences injected directly, "realistic" = the real
    detection→correspondence→triangulation error chain), same five
    survivor trackers. **This notebook runs the real engines** — every
    number you see is a live measurement, not a lookup table.

    Ground truth on both datasets: `a_rms=0.01101`, `K_a=19.80`
    (acceleration kurtosis — Gaussian is 3, real turbulence intermittency
    is 10-60; a false trajectory doesn't just add noise, it injects the
    wrong kinematics into this number, which link-count metrics like
    precision/yield cannot see at all — see
    `docs/lagrangian_turbulence_quality_guide.md`).

    **Workflow**: 1) generate a dataset below, 2) explore one tracker's
    parameters interactively in §2, 3) run the full 5-tracker comparison
    in §3.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Generate the dataset
    """)
    return


@app.cell
def _(mo):
    severity_sel = mo.ui.dropdown(
        options={"clean (no noise)": "clean", "realistic: mild": "mild",
                 "realistic: moderate": "moderate", "realistic: severe": "severe"},
        value="realistic: mild",
        label="Dataset",
    )
    seed_num = mo.ui.number(0, 100, value=0, step=1, label="Seed (realistic only)")
    gen_btn = mo.ui.run_button(label="Generate dataset")
    mo.hstack([severity_sel, seed_num, gen_btn])
    return gen_btn, seed_num, severity_sel


@app.cell
def _(Path, apd, gen_btn, mo, seed_num, severity_sel):
    SRC = Path("test_data/proptv_500_30")
    PROPTV_ROOT = Path(r"C:/Users/alex/Github/proPTV/data")
    FIRST, N = 10001, 30

    mo.stop(not gen_btn.value, mo.md("*Click **Generate dataset** to build it (10-20s).*"))

    if severity_sel.value == "clean":
        apd.convert(PROPTV_ROOT / "500_30", Path("test_data/synthetic_turbulent"), SRC)
        dataset_label = "clean"
    else:
        apd.convert_realistic(
            PROPTV_ROOT / "500_30", Path("test_data/synthetic_turbulent"), SRC,
            seed=int(seed_num.value), **apd.SEVERITY_PRESETS[severity_sel.value],
        )
        dataset_label = f"realistic ({severity_sel.value})"

    mo.md(f"✅ Dataset ready: **{dataset_label}** -> `{SRC}`")
    return FIRST, N, SRC, dataset_label


@app.cell
def _(FIRST, N, SRC, bu, dataset_label, kinematics, mo, stats):
    mo.stop(dataset_label is None)
    _frames = bu.read_gt_frames(SRC, FIRST, N)
    tt = bu.build_true_tracks(_frames, FIRST)
    _v_t, _a_t = kinematics(tt)
    a_rms_truth, K_a_truth = stats(_a_t)
    a_t_arr = _a_t
    mo.md(f"Ground truth on this dataset: `a_rms={a_rms_truth:.5f}`  `K_a={K_a_truth:.2f}`")
    return K_a_truth, a_rms_truth, a_t_arr, tt


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One tracker, live parameters

    The shared parameter surface every tracker is actually driven by:
    `dvxmax` (velocity search box, mm/frame, same on all 3 axes here for
    simplicity), `dacc` (seeded-step search radius, mm — ignored by
    `4be`'s own cost, kept for API parity), `angle` (max angular
    deviation, **gon**, 400 gon = 360°). Sliders start at the
    auto-tuned recommended value for the current dataset
    (`benchmark_utils.per_tracker_overrides`) — move them to see what
    happens when you get it wrong.
    """)
    return


@app.cell
def _(TRACKER_REGISTRY, mo):
    TRACKERS = ["priority_segment_3d", "trackcorr", "4be", "myptv_3d_tracking", "proptv_tracking"]
    tracker_sel = mo.ui.dropdown(
        options={f"{t} — {TRACKER_REGISTRY[t].short_description}": t for t in TRACKERS},
        value=f"priority_segment_3d — {TRACKER_REGISTRY['priority_segment_3d'].short_description}",
        label="Tracker",
    )
    tracker_sel
    return TRACKERS, tracker_sel


@app.cell
def _(FIRST, N, SRC, bu, mo, tracker_sel):
    _auto = bu.per_tracker_overrides([tracker_sel.value], src=SRC, first=FIRST, n_frames=N)[tracker_sel.value]
    auto_dvxmax = float(_auto.get("dvxmax", 6.0))
    auto_dacc = float(_auto.get("dacc", 6.0))
    auto_angle = float(_auto.get("angle", 120.0))
    mo.md(
        f"Auto-tuned for **{tracker_sel.value}** on this dataset: "
        f"`dvxmax={auto_dvxmax:.4f}`  `dacc={auto_dacc:.4f}`  `angle={auto_angle:.1f}` gon "
        "(sliders below start here)."
    )
    return auto_angle, auto_dacc, auto_dvxmax


@app.cell
def _(auto_angle, auto_dacc, auto_dvxmax, mo):
    dvxmax_slider = mo.ui.slider(
        0.05, max(auto_dvxmax * 5, 5.0), value=auto_dvxmax, step=0.01,
        label="dvxmax (mm/frame)", show_value=True,
    )
    dacc_slider = mo.ui.slider(
        0.01, max(auto_dacc * 5, 5.0), value=auto_dacc, step=0.01,
        label="dacc (mm)", show_value=True,
    )
    angle_slider = mo.ui.slider(
        1.0, 400.0, value=auto_angle, step=1.0,
        label="angle (gon)", show_value=True,
    )
    run_one_btn = mo.ui.run_button(label="Run this tracker with these parameters")
    mo.vstack([dvxmax_slider, dacc_slider, angle_slider, run_one_btn])
    return angle_slider, dacc_slider, dvxmax_slider, run_one_btn


@app.cell
def _(
    FIRST,
    K_a_truth,
    SRC,
    a_rms_truth,
    a_t_arr,
    angle_slider,
    bu,
    dacc_slider,
    dvxmax_slider,
    kinematics,
    mo,
    np,
    run_one_btn,
    stats,
    tracker_sel,
    tt,
):
    mo.stop(not run_one_btn.value, mo.md("*Set your parameters and click **Run this tracker**.*"))

    _ov = dict(
        dvxmax=dvxmax_slider.value, dvxmin=-dvxmax_slider.value,
        dvymax=dvxmax_slider.value, dvymin=-dvxmax_slider.value,
        dvzmax=dvxmax_slider.value, dvzmin=-dvxmax_slider.value,
        dacc=dacc_slider.value, angle=angle_slider.value,
    )
    _tracks, _elapsed = bu.run_single_tracker(
        tracker_sel.value, track_overrides=_ov, src=SRC, first=FIRST,
    )
    _m = bu.combined_metrics(tt, _tracks, eps=1.0)
    _v_p, _a_p = kinematics(_tracks)
    _a_rms, _a_k = stats(_a_p)
    _lens = np.array([len(v) for v in _tracks.values()]) if _tracks else np.zeros(1)
    _outl = 100 * np.mean(np.abs(_a_p - a_t_arr.mean()) > 5 * a_rms_truth) if _a_p.size else float("nan")

    mo.md(
        f"""
        ### Result: `{tracker_sel.value}`, dvxmax={dvxmax_slider.value:.3f}, dacc={dacc_slider.value:.3f}, angle={angle_slider.value:.0f} gon

        | metric | value | truth |
        |---|---|---|
        | a_rms error | {100*(_a_rms/a_rms_truth-1):+.1f}% | 0% |
        | **K_a** | **{_a_k:.2f}** | {K_a_truth:.2f} |
        | outlier rate (>5σ) | {_outl:.3f}% | — |
        | mean track length | {_lens.mean():.2f} frames | 30 |
        | precision | {_m['precision']:.4f} | 1.0 |
        | yield | {_m['yield_recall']:.4f} | 1.0 |
        | wall time | {_elapsed:.2f}s | — |

        {"⚠️ **K_a far above truth** — likely contamination (wrong links injecting fake jump-accelerations); try tightening `dacc`/`dvxmax`." if _a_k > K_a_truth * 1.5 else ""}
        {"⚠️ **K_a below truth** — likely survivorship bias (real fast/erratic particles being dropped or fragmented); try loosening `dvxmax`/`angle`." if _a_k < K_a_truth * 0.7 else ""}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Full 5-tracker comparison

    Reproduces `scripts/bench_with_without_noise.py`'s table on the
    currently-generated dataset, at each tracker's own auto-tuned
    parameters (§2's sliders don't affect this — it always uses the
    recommended value per tracker).
    """)
    return


@app.cell
def _(mo):
    run_all_btn = mo.ui.run_button(label="Run all 5 trackers (slow, ~1-2 min)", kind="warn")
    run_all_btn
    return (run_all_btn,)


@app.cell
def _(
    FIRST,
    K_a_truth,
    N,
    SRC,
    TRACKERS,
    a_rms_truth,
    a_t_arr,
    bu,
    kinematics,
    mo,
    np,
    pd,
    run_all_btn,
    stats,
    tt,
):
    mo.stop(not run_all_btn.value, mo.md("*Click above to run the full comparison.*"))

    _overrides = bu.per_tracker_overrides(TRACKERS, src=SRC, first=FIRST, n_frames=N)
    _rows = []
    for _tr in TRACKERS:
        _ov = _overrides[_tr]
        _tracks, _elapsed = bu.run_single_tracker(_tr, track_overrides=_ov, src=SRC, first=FIRST)
        _m = bu.combined_metrics(tt, _tracks, eps=1.0)
        _v_p, _a_p = kinematics(_tracks)
        _a_rms, _a_k = stats(_a_p)
        _lens = np.array([len(v) for v in _tracks.values()]) if _tracks else np.zeros(1)
        _outl = 100 * np.mean(np.abs(_a_p - a_t_arr.mean()) > 5 * a_rms_truth) if _a_p.size else float("nan")
        _rows.append({
            "tracker": _tr, "a_err_%": round(100 * (_a_rms / a_rms_truth - 1), 1),
            "K_a": round(_a_k, 2), "outlier_%": round(_outl, 3),
            "meanlen": round(_lens.mean(), 2), "precision": round(_m["precision"], 4),
            "yield": round(_m["yield_recall"], 4), "time_s": round(_elapsed, 2),
        })
    df_all = pd.DataFrame(_rows).sort_values("K_a")
    mo.vstack([
        mo.md(f"Truth: `K_a={K_a_truth:.2f}`. Sorted by K_a (closer to truth = better recovered physics)."),
        mo.ui.table(df_all, page_size=10),
    ])
    return (df_all,)


@app.cell
def _(K_a_truth, df_all, mo, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _colors = ["#2a9d8f" if abs(k - K_a_truth) < 10 else "#e76f51" for k in df_all["K_a"]]
    _ax.barh(df_all["tracker"], df_all["K_a"], color=_colors)
    _ax.axvline(K_a_truth, color="black", ls="--", label=f"truth K_a={K_a_truth:.1f}")
    _ax.set_xlabel("K_a (acceleration kurtosis)")
    _ax.set_title("Lower is better here (closer to the dashed truth line)")
    _ax.legend()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Per-tracker reference (from `tracking_registry.py`, always in sync with the code)
    """)
    return


@app.cell
def _(TRACKER_REGISTRY, mo, tracker_sel):
    _info = TRACKER_REGISTRY[tracker_sel.value]
    _param_rows = "\n".join(
        f"| `{p.name}` | {p.default} | {p.unit} | {p.description} | {p.how_to_choose} |"
        for p in _info.parameters
    ) or "| *(uses only the shared surface in §2 — no tracker-specific parameters)* | | | | |"
    mo.md(
        f"""
        ### {_info.display_name}

        {_info.algorithm_summary}

        {_info.algorithm_detail}

        **Best for**: {_info.best_for}
        **Avoid when**: {_info.avoid_when}

        | parameter | default | unit | description | how to choose |
        |---|---|---|---|---|
        {_param_rows}
        """
    )
    return


if __name__ == "__main__":
    app.run()
