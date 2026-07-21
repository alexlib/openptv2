"""Interactive 3D calibration viewer/comparator (marimo notebook).

Renders one or more calibration models (camera poses + calibration body) in
an orbit-able 3D plot, plus per-camera sanity diagnostics: distance to the
calibration-body centroid, sight-line angle (is the camera actually pointed
at the target?), and reprojection RMS when a `calib_matches/cam*_matches.txt`
file sits next to the model. A camera aimed >15deg off centroid, or a
cross-camera centroid-distance spread that's large relative to the rig, is
flagged -- those are the checks that catch a self-consistent-but-wrong bundle
adjustment that plain reprojection RMS misses.

The loading/diagnostic logic lives in `openptv2.calibration_diagnostics`,
shared with the headless CLI (`scripts/calibration_diagnostics.py`) -- this
notebook is presentation only.

Run as an app (read-only, sliders/checkboxes live, no visible code):
    uv run --extra viz marimo run src/openptv2/gui/visualize_calibration_nb.py -- \\
        --models "current=path/to/parameters_Run1.yaml" --calblock path/to/calblock.txt

Compare two (or more) models -- e.g. before/after full_calibration, or two
independently-calibrated experiments:
    ... -- --models "before=cal/backup,after=cal" --calblock cal/target.txt

Edit interactively (shows code, lets you tweak the loader):
    uv run --extra viz marimo edit src/openptv2/gui/visualize_calibration_nb.py

Each `--models` entry is `label=path`, where path is either a
`parameters_*.yaml` (its `cal_ori` section supplies img_ori + fixp_name) or a
directory containing `cam*.tif.ori`/`.addpar` files directly. `--calblock`
is optional when at least one model is a YAML with `fixp_name` set.
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
    # Calibration viewer

    Camera poses + calibration body in 3D, with sanity checks a passing
    reprojection RMS alone won't catch (sight-line angle to target,
    cross-camera pose symmetry). Orbit with the sliders below (or drag/scroll
    directly on the plot -- marimo's `mo.mpl.interactive` adds that for free).
    """)
    return


@app.cell
def _(mo):
    args = mo.cli_args()
    models_arg = args.get("models") or ""
    calblock_arg = args.get("calblock") or ""
    return calblock_arg, models_arg


@app.cell
def _(calblock_arg, mo, models_arg):
    # Editable fallback for `marimo edit` with no --models passed.
    models_text = mo.ui.text(
        value=models_arg or "current=../parameters_Run1.yaml",
        label="models (label=path, comma-separated)",
        full_width=True,
    )
    calblock_text = mo.ui.text(
        value=calblock_arg,
        label="calibration block .txt (optional if a model YAML has fixp_name)",
        full_width=True,
    )
    if not models_arg:
        mo.md(f"""
        No `--models` CLI arg was passed -- fill these in and re-run the cell below.
        {models_text}
        {calblock_text}
        """)
    return calblock_text, models_text


@app.cell
def _(calblock_arg, calblock_text, models_arg, models_text):
    models_spec = models_arg or models_text.value
    calblock_override = calblock_arg or calblock_text.value
    return calblock_override, models_spec


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from openptv2.calibration_diagnostics import (
        compute_diagnostics,
        load_model,
        parse_models_arg,
        resolve_centroid,
        viewing_dir,
    )

    return (
        Path,
        compute_diagnostics,
        load_model,
        np,
        parse_models_arg,
        plt,
        resolve_centroid,
        viewing_dir,
    )


@app.cell
def _(Path, calblock_override, load_model, models_spec, parse_models_arg):
    models = {}
    calblock_path = (
        Path(calblock_override).expanduser().resolve() if calblock_override else None
    )
    load_errors = []
    for _label, _path in parse_models_arg(models_spec):
        try:
            _cams, _calblock_guess = load_model(_path)
            models[_label] = _cams
            if calblock_path is None and _calblock_guess is not None:
                calblock_path = _calblock_guess
        except Exception as _e:  # surfaced in the UI, not raised -- one bad
            load_errors.append(
                f"{_label} ({_path}): {_e}"
            )  # model shouldn't kill the notebook
    return calblock_path, load_errors, models


@app.cell
def _(load_errors, mo):
    if load_errors:
        mo.md(
            "**Some models failed to load:**\n\n"
            + "\n".join(f"- {e}" for e in load_errors)
        )
    return


@app.cell
def _(calblock_path, mo, models, resolve_centroid):
    body, centroid = resolve_centroid(models, calblock_path)
    if body is None and models:
        mo.md(
            "_No calibration body found -- sight-line checks use the "
            "camera-cluster centroid instead._"
        )
    return body, centroid


@app.cell
def _(centroid, compute_diagnostics, models):
    diagnostics = compute_diagnostics(models, centroid)
    return (diagnostics,)


@app.cell
def _(mo, models):
    azim_slider = mo.ui.slider(-180, 180, value=-35, step=1, label="azimuth")
    elev_slider = mo.ui.slider(-90, 90, value=20, step=1, label="elevation")
    sight_toggle = mo.ui.checkbox(value=False, label="sightlines to centroid")
    model_toggles = {label: mo.ui.checkbox(value=True, label=label) for label in models}
    mo.hstack([azim_slider, elev_slider, sight_toggle, *model_toggles.values()])
    return azim_slider, elev_slider, model_toggles, sight_toggle


@app.cell
def _(
    azim_slider,
    body,
    centroid,
    elev_slider,
    mo,
    model_toggles,
    models,
    np,
    plt,
    sight_toggle,
    viewing_dir,
):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if body is not None and len(body):
        ax.scatter(
            body[:, 0],
            body[:, 1],
            body[:, 2],
            s=6,
            c="gray",
            alpha=0.4,
            label="calibration body",
        )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_pts = [body] if body is not None and len(body) else []
    all_pts += [pos for cams in models.values() for _, pos, _, _ in cams]
    span = float(np.ptp(np.vstack(all_pts), axis=0).max()) if all_pts else 100.0
    axis_len = span * 0.1

    for _i, (_label, _cams) in enumerate(models.items()):
        if _label in model_toggles and not model_toggles[_label].value:
            continue
        _color = color_cycle[_i % len(color_cycle)]
        for _name, _pos, _rot, _ori_path in _cams:
            ax.scatter(*_pos, s=70, c=_color, marker="^")
            ax.text(*_pos, f"  {_label}:{_name}", fontsize=7, color=_color)
            _v = viewing_dir(_rot) * axis_len
            ax.quiver(*_pos, *_v, color=_color, linewidth=1.5, arrow_length_ratio=0.2)
            if sight_toggle.value:
                ax.plot(
                    *zip(_pos, centroid),
                    c=_color,
                    alpha=0.3,
                    linestyle="--",
                    linewidth=1,
                )

    ax.scatter(*centroid, s=40, c="black", marker="x", label="centroid")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(
        " vs ".join(models.keys()) if len(models) > 1 else next(iter(models), "")
    )
    ax.legend(loc="upper left")
    ax.view_init(elev=elev_slider.value, azim=azim_slider.value)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(diagnostics, mo):
    def fmt_rows(cams):
        lines = [
            "| cam | dist mm | axis off deg | RMS px | matched |",
            "|---|---|---|---|---|",
        ]
        for c in cams:
            mark = " ⚠️" if c.flag else ""
            rms = f"{c.rms:.2f}" if c.rms is not None else "n/a"
            matched = str(c.matched) if c.matched is not None else "n/a"
            lines.append(
                f"| {c.name} | {c.dist:.1f} | {c.angle:.1f}{mark} | {rms} | {matched} |"
            )
        return "\n".join(lines)

    sections = []
    for _label, _d in diagnostics.items():
        spread_mark = " ⚠️ large spread relative to rig size" if _d.flag else ""
        sections.append(
            f"### {_label}\n{fmt_rows(_d.cameras)}\n\ncentroid-distance spread: "
            f"**{_d.spread:.1f} mm**{spread_mark}"
        )
    mo.md("\n\n".join(sections) if sections else "_no models loaded_")
    return


@app.cell
def _(models, np):
    def test_calibration_sanity():
        """Smoke test: every loaded rotation matrix is a valid rotation."""
        for label, cams in models.items():
            for name, _pos, rot, _ori_path in cams:
                det = np.linalg.det(rot)
                assert abs(abs(det) - 1.0) < 1e-3, (
                    f"{label}:{name}: not a rotation matrix (det={det})"
                )
                assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-3), (
                    f"{label}:{name}: not orthonormal"
                )

    test_calibration_sanity()
    return


if __name__ == "__main__":
    app.run()
