"""Interactive trackcorr candidate viewer (marimo notebook), Phase 2 (static).

See docs/plans/2026-08-21-trackcorr-interactive-debug.md. Loads a trackcorr
tracking run, steps it forward over a frame range, and for one chosen
frame/particle shows every candidate trackcorr's real search actually
considered (from openptv2.gui.trackcorr_debug -- these are the real
per-step results, not a reimplementation of the search), overlaid on the
camera images: all next-frame detections as small dots, considered
candidates ranked/labeled by cost, the winning link highlighted.

trackcorr only (track_mode=0). No interactivity yet (Phase 3) beyond
editable frame/particle fields -- parameter sliders that live-recompute the
search come next.

Run as an app (read-only, no visible code):
    uv run marimo run src/openptv2/gui/trackcorr_debug_nb.py -- \\
        --dataset test_data/test_cavity --first 10001 --last 10002 --particle 0

Edit interactively (shows code):
    uv run marimo edit src/openptv2/gui/trackcorr_debug_nb.py

`--dataset` is a directory containing parameters.yaml, cal/, img_orig/ (or
img/), and res_orig/ (or res/) -- the same layout test_track.py's own
test_cavity fixture uses. `--first`/`--last` bound the frame range to step
trackcorr over (last is exclusive, i.e. last-1 -> last is the final
transition); `--particle` is the particle index (row in that frame's
tracked-particle array) to inspect in the first stepped frame.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # trackcorr candidate viewer

    Every candidate trackcorr's real search considered for one particle,
    overlaid on the camera images: small gray dots are all detections in the
    next frame, colored circles are the candidates trackcorr actually
    evaluated (ranked/labeled by cost -- lower is better), and the winner
    (the link trackcorr made) is outlined in red.
    """)
    return


@app.cell
def _(mo):
    args = mo.cli_args()
    dataset_arg = args.get("dataset") or ""
    first_arg = args.get("first")
    last_arg = args.get("last")
    particle_arg = args.get("particle")
    return dataset_arg, first_arg, last_arg, particle_arg


@app.cell
def _(dataset_arg, first_arg, last_arg, mo, particle_arg):
    # Always-visible editable fields (CLI args pre-fill them); no separate
    # "no CLI arg" branch -- fill in and re-run if the defaults don't apply.
    dataset_text = mo.ui.text(
        value=dataset_arg or "test_data/test_cavity",
        label="dataset directory (parameters.yaml, cal/, img_orig/, res_orig/)",
        full_width=True,
    )
    first_number = mo.ui.number(value=int(first_arg or 10001), label="first frame")
    last_number = mo.ui.number(value=int(last_arg or 10003), label="last frame (exclusive)")
    particle_number = mo.ui.number(value=int(particle_arg or 0), label="particle index")
    mo.vstack([dataset_text, mo.hstack([first_number, last_number, particle_number])])
    return dataset_text, first_number, last_number, particle_number


@app.cell
def _(Path, dataset_text):
    dataset_dir = Path(dataset_text.value).expanduser().resolve()
    return (dataset_dir,)


@app.cell
def _():
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.constants import NEXT_NONE
    from openptv2.algorithms.parameters import (
        ControlPar,
        SequencePar,
        TrackPar,
        VolumePar,
    )
    from openptv2.gui.trackcorr_debug import (
        candidates_for_particle,
        load_run,
        probe_particle,
        step_and_capture,
    )

    return (
        Calibration,
        ControlPar,
        NEXT_NONE,
        Path,
        SequencePar,
        TrackPar,
        VolumePar,
        candidates_for_particle,
        imread,
        load_run,
        np,
        os,
        plt,
        probe_particle,
        step_and_capture,
    )


@app.cell
def _(Calibration, ControlPar, SequencePar, TrackPar, VolumePar, dataset_dir, load_run, os):
    # trackcorr_debug.load_run() reads res/rt_is.*/ptv_is.*/added.* (the
    # standard file bases) relative to the current directory -- chdir into
    # the dataset for the duration of loading+stepping, same as
    # test_track.py's own test_cavity fixtures do.
    _cwd = os.getcwd()
    os.chdir(dataset_dir)
    try:
        cpar = ControlPar.from_yaml("parameters.yaml")
        spar = SequencePar.from_yaml("parameters.yaml")
        vpar = VolumePar.from_yaml("parameters.yaml")
        tpar = TrackPar.from_yaml("parameters.yaml")
        cals = [
            Calibration.from_file(f"cal/cam{c + 1}.tif.ori", f"cal/cam{c + 1}.tif.addpar")
            for c in range(cpar.num_cams)
        ]
        run = load_run(cpar, spar, vpar, tpar, cals)
    finally:
        os.chdir(_cwd)
    return cals, cpar, run, spar, tpar, vpar


@app.cell
def _(first_number, last_number, run, step_and_capture):
    snapshots = step_and_capture(run, int(first_number.value), int(last_number.value))
    first_step = min(snapshots)
    snapshot = snapshots[first_step]
    return (snapshot,)


@app.cell
def _(candidates_for_particle, particle_number, snapshot):
    result = candidates_for_particle(snapshot, int(particle_number.value))
    return (result,)


@app.cell
def _(NEXT_NONE, active_result, mo):
    _rows = ["| rank | cost | row | cams | winner |", "|---|---|---|---|---|"]
    for _c in active_result.candidates:
        _is_winner = "**yes**" if _c.row == active_result.winner_row else ""
        _cams = ",".join(str(k) for k in sorted(_c.cameras))
        _rows.append(f"| {_c.rank} | {_c.cost:.4g} | {_c.row} | {_cams} | {_is_winner} |")

    _winner_label = "best-by-cost (isolated probe)" if active_result.is_isolated else "linked to"
    _winner_note = (
        "no candidate found (winner_row == NEXT_NONE)"
        if active_result.winner_row == NEXT_NONE
        else f"{_winner_label} row {active_result.winner_row} in the next frame"
    )
    mo.md(
        f"""
        **Step {active_result.step}, particle {active_result.particle_index}** at
        `{tuple(round(v, 2) for v in active_result.pos_3d)}` mm -- {_winner_note}.

        {len(active_result.candidates)} candidate(s) trackcorr's real search
        accepted (passed the angle/acceleration gate):

        {mo.md(chr(10).join(_rows))}
        """
    )
    return


@app.cell
def _(mo, tpar):
    # "Tune then press Run", not live-recompute-on-drag: probe_particle() is
    # ~200x faster than a real full-frame step (~25-40ms vs ~7.5s for a
    # ~700-particle frame), but still one recompute per interaction is
    # nicer as an explicit action than a reactive drag.
    dvxmin_slider = mo.ui.slider(-50.0, 50.0, 0.5, value=tpar.dvxmin, label="dvxmin")
    dvxmax_slider = mo.ui.slider(-50.0, 50.0, 0.5, value=tpar.dvxmax, label="dvxmax")
    dvymin_slider = mo.ui.slider(-50.0, 50.0, 0.5, value=tpar.dvymin, label="dvymin")
    dvymax_slider = mo.ui.slider(-50.0, 50.0, 0.5, value=tpar.dvymax, label="dvymax")
    dvzmin_slider = mo.ui.slider(-50.0, 50.0, 0.5, value=tpar.dvzmin, label="dvzmin")
    dvzmax_slider = mo.ui.slider(-50.0, 50.0, 0.5, value=tpar.dvzmax, label="dvzmax")
    dacc_slider = mo.ui.slider(0.1, 50.0, 0.1, value=tpar.dacc, label="dacc")
    dangle_slider = mo.ui.slider(1.0, 200.0, 1.0, value=tpar.dangle, label="dangle")
    run_button = mo.ui.run_button(label="Run with these parameters")
    mo.vstack([
        mo.hstack([dvxmin_slider, dvxmax_slider]),
        mo.hstack([dvymin_slider, dvymax_slider]),
        mo.hstack([dvzmin_slider, dvzmax_slider]),
        mo.hstack([dacc_slider, dangle_slider]),
        run_button,
    ])
    return (
        dacc_slider,
        dangle_slider,
        dvxmax_slider,
        dvxmin_slider,
        dvymax_slider,
        dvymin_slider,
        dvzmax_slider,
        dvzmin_slider,
        run_button,
    )


@app.cell
def _(cals, cpar, dataset_dir, load_run, os, spar, tpar, vpar):
    # A separate, fresh, unstepped-past run dedicated to probing -- reusing
    # `run` (which step_and_capture already advanced) would violate
    # probe_particle's precondition. Stepped up to (not through)
    # first_number.value, matching what the real-capture cell assumes.
    _cwd = os.getcwd()
    os.chdir(dataset_dir)
    try:
        probe_run = load_run(cpar, spar, vpar, tpar, cals)
    finally:
        os.chdir(_cwd)
    return (probe_run,)


@app.cell
def _(
    dacc_slider,
    dangle_slider,
    dvxmax_slider,
    dvxmin_slider,
    dvymax_slider,
    dvymin_slider,
    dvzmax_slider,
    dvzmin_slider,
    first_number,
    mo,
    particle_number,
    probe_particle,
    probe_run,
    result,
    run_button,
    step_and_capture,
):
    is_script_mode = mo.app_meta().mode == "script"
    if is_script_mode or run_button.value:
        step_and_capture(probe_run, probe_run.seq_par.first, int(first_number.value))
        active_result = probe_particle(
            probe_run,
            int(first_number.value),
            int(particle_number.value),
            dvxmin=dvxmin_slider.value,
            dvxmax=dvxmax_slider.value,
            dvymin=dvymin_slider.value,
            dvymax=dvymax_slider.value,
            dvzmin=dvzmin_slider.value,
            dvzmax=dvzmax_slider.value,
            dacc=dacc_slider.value,
            dangle=dangle_slider.value,
        )
    else:
        active_result = result
    return (active_result,)


@app.cell
def _(active_result, mo):
    mo.md(
        "**showing:** "
        + (
            "isolated probe with the tuned parameters above (press Run again after changing them)"
            if active_result.is_isolated
            else "the real trackcorr link (unmodified parameters) -- press Run above to probe with tuned parameters instead"
        )
    )
    return


@app.cell
def _(active_result, cals, cpar, dataset_dir, imread, np, os, spar):
    # Camera images for the frame the CANDIDATES live in
    # (active_result.step + 1), so the overlay sits on the same image the
    # detections came from.
    _next_frame = active_result.step + 1
    _cwd = os.getcwd()
    os.chdir(dataset_dir)
    try:
        images = []
        for _c in range(cpar.num_cams):
            _base = spar.get_img_base_name(_c)
            _path = _base % _next_frame
            images.append(np.asarray(imread(_path)))
    finally:
        os.chdir(_cwd)
    return (images,)


@app.cell
def _(active_result, images, plt, snapshot):
    # All-detections context dots come from the real step's snapshot
    # (frame-2 detections don't change with the probed parameters, only
    # which of them are considered candidates does).
    _n = len(images)
    fig, axes = plt.subplots(1, _n, figsize=(5 * _n, 5))
    if _n == 1:
        axes = [axes]

    _cmap = plt.get_cmap("plasma")
    _max_rank = max((c.rank for c in active_result.candidates), default=0)

    for _cam, _ax in enumerate(axes):
        _ax.imshow(images[_cam], cmap="gray")
        _ax.set_title(f"camera {_cam}")

        # All detections in this frame, for context.
        _tx = snapshot["targ_x_2"][_cam]
        _ty = snapshot["targ_y_2"][_cam]
        _ax.scatter(_tx, _ty, s=4, c="lightgray", alpha=0.6, label="detections")

        for _cand in active_result.candidates:
            if _cam not in _cand.cameras:
                continue
            _tnr, _x, _y = _cand.cameras[_cam]
            _color = _cmap(_cand.rank / max(_max_rank, 1))
            _is_winner = _cand.row == active_result.winner_row
            _ax.scatter(
                [_x],
                [_y],
                s=120 if _is_winner else 70,
                facecolors=_color,
                edgecolors="red" if _is_winner else "black",
                linewidths=2.5 if _is_winner else 0.8,
                zorder=5,
            )
            _ax.annotate(
                f"#{_cand.rank} tnr={_tnr}",
                (_x, _y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
                color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6),
            )

        _ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    return (fig,)


@app.cell
def _(fig, mo):
    mo.mpl.interactive(fig)
    return


@app.cell
def _(candidates_for_particle):
    def test_winner_resolves_when_linked():
        """Self-check: candidates_for_particle().winner is well-formed
        whenever a real link exists -- mirrors
        tests/unit/test_trackcorr_debug.py's correctness contract, run
        inline so this notebook has its own runnable check."""
        snap = {
            "step": 0,
            "num_cams": 1,
            "num_parts_1": 1,
            "path_x_1": [[0.0, 0.0, 0.0]],
            "path_next_1": [0],
            "path_inlist_1": [1],
            "path_decis_1": [[0.5]],
            "path_linkdecis_1": [[0]],
            "num_parts_2": 1,
            "path_x_2": [[0.1, 0.1, 0.1]],
            "corres_p_2": [[0]],
            "targ_x_2": [[1.0]],
            "targ_y_2": [[2.0]],
            "targ_tnr_2": [[7]],
        }
        import numpy as np

        for key in ("path_x_1", "path_next_1", "path_inlist_1", "path_decis_1",
                    "path_linkdecis_1", "path_x_2", "corres_p_2"):
            snap[key] = np.asarray(snap[key])

        r = candidates_for_particle(snap, 0)
        assert r.winner_row == 0
        assert r.winner is not None
        assert r.winner.cameras[0] == (7, 1.0, 2.0)

    test_winner_resolves_when_linked()
    return


if __name__ == "__main__":
    app.run()
