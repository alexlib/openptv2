# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "matplotlib>=3.7.0",
#     "pyyaml>=6.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pathlib import Path
    import yaml

    return mo, np, plt, patches, Path, yaml


@app.cell
def _(Path):
    base_path = Path("/home/user/Documents/GitHub/openptv2/test_data/test_cavity")
    res_dir = base_path / "res"
    res_dir.mkdir(exist_ok=True)
    img_dir = base_path / "img"
    return base_path, res_dir, img_dir


@app.cell
def _(mo):
    mo.md(
        """
        # Tracking Step Visualizer

        This notebook runs the **Python tracking engine** with an observer that
        records every per-particle decision: predicted position, search volume,
        candidates found, angle/acceleration scores, and the final link.

        Use the sliders below to explore individual particles at each frame.
        """
    )
    return


@app.cell
def _(base_path, res_dir, yaml):
    """Run tracking with the Python engine + observer."""
    import sys, os
    sys.path.insert(0, str(base_path.parent.parent))
    os.chdir(str(base_path))

    from gui.pyptv import pyptv_batch
    from openptv2.engine import set_engine

    # Use optv for the correspondence step (batch), then Python for tracking
    set_engine("optv")

    yaml_path = base_path / "parameters_Run1.yaml"
    with open(yaml_path) as f:
        params = yaml.safe_load(f)

    params["sequence"]["output"] = str(res_dir)
    params["sequence"]["first"] = 10001
    params["sequence"]["last"] = 10004

    temp_yaml = base_path / "temp_run.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(params, f)

    print("Running batch (correspondence): frames 10001-10004")
    pyptv_batch.main(temp_yaml, 10001, 10004)
    print("Batch complete!")
    return params, pyptv_batch, set_engine, temp_yaml, yaml_path


@app.cell
def _(base_path, params, set_engine, yaml):
    """Run tracking with the Python engine and observer attached."""
    from gui.pyptv.ptv import py_start_proc_c
    from gui.pyptv.parameter_manager import ParameterManager
    from algorithms.track import Tracker, TrackingObserver, default_naming
    from algorithms.parameters import (
        TrackPar, SequencePar, VolumePar, convert_track_par_to_tuple,
    )
    from algorithms.calibration import Calibration as PythonCalibration
    from algorithms.parameters_adapter import ControlParams

    set_engine("python")

    # Load parameters via the same path the batch uses
    pm = ParameterManager(base_path / "parameters_Run1.yaml")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)

    # Convert optv objects to Python-engine objects
    num_cams = cpar.get_num_cams()

    # ControlPar
    from algorithms.parameters import ControlPar, MultimediaPar
    mm = MultimediaPar(
        nlay=1,
        n1=params["ptv"]["mmp_n1"],
        n2=[params["ptv"]["mmp_n2"]],
        d=[params["ptv"]["mmp_d"]],
        n3=params["ptv"]["mmp_n3"],
    )
    img_base = [f"img/cam{i+1}." for i in range(num_cams)]
    cpar_py = ControlPar(
        num_cams=num_cams,
        img_base_name=img_base,
        cal_img_base_name=[params["ptv"]["img_cal"][i] for i in range(num_cams)],
        imx=params["ptv"]["imx"],
        imy=params["ptv"]["imy"],
        pix_x=params["ptv"]["pix_x"],
        pix_y=params["ptv"]["pix_y"],
        mm=mm,
    )

    # VolumePar
    crit = params["criteria"]
    vpar_py = VolumePar(
        x_lay=crit["X_lay"],
        z_min_lay=crit["Zmin_lay"],
        z_max_lay=crit["Zmax_lay"],
        cn=crit["cn"], cnx=crit["cnx"], cny=crit["cny"],
        csumg=crit["csumg"], eps0=crit["eps0"], corrmin=crit["corrmin"],
    )

    # TrackPar
    tp = params["track"]
    tpar_py = TrackPar(
        dvxmin=tp["dvxmin"], dvxmax=tp["dvxmax"],
        dvymin=tp["dvymin"], dvymax=tp["dvymax"],
        dvzmin=tp["dvzmin"], dvzmax=tp["dvzmax"],
        dacc=tp["dacc"], dangle=tp["angle"],
        add=tp.get("flagNewParticles", True),
    )

    # SequencePar
    spar_py = SequencePar(
        img_base_name=[spar.get_img_base_name(i) for i in range(num_cams)],
        first=spar.get_first(),
        last=spar.get_last(),
    )

    # Calibrations
    cals_py = []
    for cal_obj in cals:
        ext_par = cal_obj.get_ext_par() if hasattr(cal_obj, "get_ext_par") else cal_obj.ext_par
        int_par = cal_obj.get_int_par() if hasattr(cal_obj, "get_int_par") else cal_obj.int_par
        glass_par = cal_obj.get_glass_par() if hasattr(cal_obj, "get_glass_par") else None
        add_par = cal_obj.get_add_par() if hasattr(cal_obj, "get_add_par") else None
        cals_py.append(PythonCalibration(
            ext_par=ext_par, int_par=int_par,
            glass_par=glass_par, added_par=add_par,
        ))

    # Create observer + tracker
    observer = TrackingObserver()
    tracker = Tracker(
        cpar_py, vpar_py, tpar_py, spar_py, cals_py, default_naming,
    )
    tracker.full_forward(observer=observer)

    print(f"Tracking complete. Collected {len(observer.events)} particle events.")
    return observer, tracker, cpar_py, cals_py, num_cams


@app.cell
def _(observer, mo, np):
    """Build summary statistics and frame list."""
    all_steps = sorted(set(e["step"] for e in observer.events))
    step_counts = {s: len(observer.events_for_frame(s)) for s in all_steps}

    mo.md(
        f"**{len(observer.events)} events** across **{len(all_steps)} frames**: "
        + ", ".join(f"step {s}: {c} particles" for s, c in step_counts.items())
    )
    return all_steps, step_counts


@app.cell
def _(mo, all_steps):
    frame_slider = mo.ui.slider(
        start=min(all_steps), stop=max(all_steps), step=1,
        value=min(all_steps), label="Frame",
    )
    frame_slider
    return (frame_slider,)


@app.cell
def _(mo, observer, frame_slider):
    frame_events = observer.events_for_frame(frame_slider.value)
    n_particles = len(frame_events)
    particle_slider = mo.ui.slider(
        start=0, stop=max(0, n_particles - 1), step=1,
        value=0, label=f"Particle (0–{max(0, n_particles-1)})",
    )
    particle_slider
    return frame_events, n_particles, particle_slider


@app.cell
def _(frame_events, particle_slider, mo):
    """Display the selected event's data in a summary table."""
    evt = frame_events[particle_slider.value] if frame_events else None
    if evt is None:
        summary = mo.md("No events for this frame.")
    else:
        cands_info = ""
        for i, c in enumerate(evt.get("candidates", [])):
            cands_info += (
                f"| {i} | {c['ftnr']} | ({c['cand_3d'][0]:.2f}, {c['cand_3d'][1]:.2f}, {c['cand_3d'][2]:.2f}) "
                f"| {c['freq']} | {'yes' if c['registered'] else 'no'} |\n"
            )

        linked_str = "none"
        if evt.get("linked_3d") is not None:
            lp = evt["linked_3d"]
            linked_str = f"({lp[0]:.2f}, {lp[1]:.2f}, {lp[2]:.2f})"

        summary = mo.md(f"""
### Particle {evt['particle_id']} at frame {evt['step']}

| Property | Value |
|----------|-------|
| **3D position** | ({evt['pos_3d'][0]:.2f}, {evt['pos_3d'][1]:.2f}, {evt['pos_3d'][2]:.2f}) |
| **Predicted 3D** | ({evt['predicted_3d'][0]:.2f}, {evt['predicted_3d'][1]:.2f}, {evt['predicted_3d'][2]:.2f}) |
| **Has previous** | {evt.get('has_prev', False)} |
| **Candidates found** | {len(evt.get('candidates', []))} |
| **In-list (registered)** | {evt.get('inlist', 0)} |
| **Final decision** | {evt.get('finaldecis', 'n/a')} |
| **Linked to** | next={evt.get('next_frame', 'n/a')}, pos={linked_str} |

#### Candidates

| # | ftnr | 3D position | freq | registered |
|---|------|-------------|------|------------|
{cands_info if cands_info else "| — | — | — | — | — |"}
""")
    summary
    return (evt,)


@app.cell
def _(img_dir, np, plt, frame_slider, num_cams):
    """Load camera images for the selected frame."""
    frame_num = frame_slider.value
    cam_images = []
    for c in range(num_cams):
        img_path = img_dir / f"cam{c+1}.{frame_num}"
        if img_path.exists():
            cam_images.append(plt.imread(str(img_path)))
        else:
            cam_images.append(np.zeros((1024, 1280), dtype=np.uint8))
    return (cam_images,)


@app.cell
def _(mo, evt, cam_images, num_cams, np, plt, patches, cals_py, cpar_py):
    """Draw 4-camera view with search rectangles and candidates."""
    from algorithms.track import point_to_pixel

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for cam_idx in range(num_cams):
        ax = axes_flat[cam_idx]
        ax.imshow(cam_images[cam_idx], cmap="gray")
        ax.set_title(f"Camera {cam_idx + 1}")

        if evt is None:
            continue

        # --- Current particle position (projected) ---
        pos_px = point_to_pixel(evt["pos_3d"], cals_py[cam_idx], cpar_py)
        ax.plot(pos_px[0], pos_px[1], "o", color="cyan", markersize=10, label="current")
        ax.annotate(
            f"ID {evt['particle_id']}",
            (pos_px[0], pos_px[1]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=8, color="cyan",
        )

        # --- Predicted position ---
        pred_px = point_to_pixel(evt["predicted_3d"], cals_py[cam_idx], cpar_py)
        ax.plot(pred_px[0], pred_px[1], "x", color="lime", markersize=10, mew=2, label="predicted")

        # --- Previous position ---
        if evt.get("prev_3d") is not None:
            prev_px = point_to_pixel(evt["prev_3d"], cals_py[cam_idx], cpar_py)
            ax.plot(prev_px[0], prev_px[1], "s", color="yellow", markersize=7, label="previous")
            # Velocity arrow
            ax.annotate(
                "", xy=(pos_px[0], pos_px[1]),
                xytext=(prev_px[0], prev_px[1]),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
            )

        # --- Search rectangle ---
        sr = evt.get("search_rect")
        if sr is not None:
            center_px = evt["search_center_px"][cam_idx]
            xl, xr = sr["xl"][cam_idx], sr["xr"][cam_idx]
            yu, yd = sr["yu"][cam_idx], sr["yd"][cam_idx]
            rect = patches.Rectangle(
                (center_px[0] - xl, center_px[1] - yu),
                xl + xr, yu + yd,
                linewidth=1.5, edgecolor="magenta", facecolor="none",
                linestyle="--", label="search region",
            )
            ax.add_patch(rect)

        # --- Candidates ---
        for i, c in enumerate(evt.get("candidates", [])):
            c_px = point_to_pixel(c["cand_3d"], cals_py[cam_idx], cpar_py)
            color = "red" if c["registered"] else "orange"
            ax.plot(c_px[0], c_px[1], "^", color=color, markersize=8)
            ax.annotate(
                f"c{i}(f{c['freq']})",
                (c_px[0], c_px[1]),
                textcoords="offset points", xytext=(5, -10),
                fontsize=7, color=color,
            )

        # --- Linked position ---
        if evt.get("linked_3d") is not None:
            link_px = point_to_pixel(evt["linked_3d"], cals_py[cam_idx], cpar_py)
            ax.plot(link_px[0], link_px[1], "*", color="white", markersize=14, label="linked")
            ax.annotate(
                "", xy=(link_px[0], link_px[1]),
                xytext=(pos_px[0], pos_px[1]),
                arrowprops=dict(arrowstyle="->", color="white", lw=2),
            )

    # Legend on first subplot only
    if evt is not None:
        axes_flat[0].legend(loc="upper right", fontsize=7, framealpha=0.7)

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo, evt, plt, np):
    """3D scatter: current particle, prediction, candidates, and linked."""
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection="3d")

    if evt is not None:
        p = evt["pos_3d"]
        ax3d.scatter(*p, c="cyan", s=120, marker="o", label=f"ID {evt['particle_id']}")

        pr = evt["predicted_3d"]
        ax3d.scatter(*pr, c="lime", s=100, marker="x", label="predicted")
        ax3d.plot([p[0], pr[0]], [p[1], pr[1]], [p[2], pr[2]], "g--", alpha=0.5)

        if evt.get("prev_3d") is not None:
            pv = evt["prev_3d"]
            ax3d.scatter(*pv, c="yellow", s=80, marker="s", label="previous")
            ax3d.plot([pv[0], p[0]], [pv[1], p[1]], [pv[2], p[2]], "y-", alpha=0.5)

        for i, c in enumerate(evt.get("candidates", [])):
            cp = c["cand_3d"]
            color = "red" if c["registered"] else "orange"
            ax3d.scatter(*cp, c=color, s=60, marker="^", label=f"cand {i}" if i < 5 else "")

        if evt.get("linked_3d") is not None:
            lp = evt["linked_3d"]
            ax3d.scatter(*lp, c="white", s=150, marker="*", label="linked", edgecolors="black")
            ax3d.plot([p[0], lp[0]], [p[1], lp[1]], [p[2], lp[2]], "w-", lw=2)

        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title(f"3D View — Particle {evt['particle_id']}, Frame {evt['step']}")
        ax3d.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    mo.mpl.interactive(fig3d)
    return


@app.cell
def _(mo):
    mo.md(
        """
        **Legend:**
        - **Cyan circle**: current particle position
        - **Lime ×**: predicted next position (search center)
        - **Yellow square**: previous position
        - **Magenta dashed rect**: search region on each camera
        - **Red/orange triangles**: candidates (red = registered, orange = not)
        - **White star + arrow**: final linked particle
        """
    )
    return


if __name__ == "__main__":
    app.run()
