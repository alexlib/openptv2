"""Illmenau calibration plates, reconstructed in 3D inside the barrel.

For both rigs (cameras 1-4 and 5-8) every labelled plate view is triangulated
with the DELIVERED .ori/.addpar, so what is drawn is what the calibration
actually reconstructs -- not the poses it was fitted from.  Optionally the
ideal rigid 6x7 plate is drawn at the pose the joint bundle solved for that
frame (cal/bundle_plate_poses.npz), so the two can be compared.

Frames: each rig is its own world whose origin is the datum dot of plate frame
00000000, on the barrel axis.  A rig moves into the barrel frame by a pure y
shift (plate.yaml datum barrel_frame: y = -1175 mm, origin at mid-height); no
rotation is assumed.  +Y is up.

Calibration version: the dropdown at the top picks which .ori files to draw --
the delivered ones or a kept earlier version (.pre_selfcal, .prebundle,
.pre_reanchor), wherever a rig has it.  The table below the plot then shows
how well each version closes on the plates.

Display: plotly's 3D axes are right-handed, so drawing world (X, Z, Y) as
plotly (x, y, z) would mirror the scene.  The Z axis is drawn reversed instead,
which keeps the view true and the hover values in world coordinates.

Open with:  uv run marimo edit notebooks/illmenau_plates_barrel_marimo.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import yaml

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.parameters import ControlPar, MmNp
    from openptv2.plate_bundle import rodrigues
    from openptv2.plate_multiplane import (
        PlateGrid,
        load_plate_views,
        plate_frame_metrics,
        triangulate_plate,
    )

    RAW = Path(os.environ.get("ILLMENAU_RAW", "/Users/alex/Downloads/Ilmenau"))
    RIGS = {
        "1-4": (RAW / "openptv_illmenau_4cam", [1, 2, 3, 4], "#ff7f0e"),
        "5-8": (RAW / "openptv_illmenau_5678", [5, 6, 7, 8], "#1f77b4"),
    }
    IMX, IMY, PIX = 2560, 2048, 0.005
    _plate = yaml.safe_load((RIGS["1-4"][0] / "plate.yaml").read_text())["plate"]
    R_WALL = float(_plate["test_section"]["radius"])
    H_BARREL = float(_plate["test_section"]["height"])
    Y_SHIFT = float(_plate["datum"]["barrel_frame"][1])  # datum dot, barrel y
    return (
        Calibration,
        ControlPar,
        H_BARREL,
        IMX,
        IMY,
        MmNp,
        PIX,
        PlateGrid,
        RIGS,
        R_WALL,
        Y_SHIFT,
        go,
        load_plate_views,
        mo,
        np,
        plate_frame_metrics,
        rodrigues,
        triangulate_plate,
    )


@app.cell
def _(mo):
    ori_version = mo.ui.dropdown(
        ["delivered", ".pre_selfcal", ".prebundle", ".pre_reanchor"],
        value="delivered",
        label="calibration (.ori) version; a rig without it uses its delivered one",
    )
    ori_version
    return (ori_version,)


@app.cell
def _(
    Calibration,
    ControlPar,
    IMX,
    IMY,
    MmNp,
    PIX,
    PlateGrid,
    RIGS,
    Y_SHIFT,
    load_plate_views,
    np,
    ori_version,
    plate_frame_metrics,
    rodrigues,
    triangulate_plate,
):
    def to_barrel(P):
        P = np.atleast_2d(np.asarray(P, float))
        return P + [0.0, Y_SHIFT, 0.0]

    def load_rig(folder, cams):
        """Triangulate every frame; returns cameras, plates and bundle poses."""
        cal_dir = folder / "cal"
        grid = PlateGrid.from_yaml(folder / "plate.yaml")
        suffix = "" if ori_version.value == "delivered" else ori_version.value
        if not all((cal_dir / f"cam{n}.tif.ori{suffix}").exists() for n in cams):
            suffix = ""
        cals = [
            Calibration.from_file(
                str(cal_dir / f"cam{n}.tif.ori{suffix}"),
                str(cal_dir / f"cam{n}.tif.addpar"),
            )
            for n in cams
        ]
        cpar = ControlPar(
            num_cams=len(cams),
            imx=IMX,
            imy=IMY,
            pix_x=PIX,
            pix_y=PIX,
            mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
            chfield=0,
            tiff_flag=1,
            hp_flag=1,
            allCam_flag=0,
            img_base_name=[""] * len(cams),
            cal_img_base_name=[""] * len(cams),
        )
        views = load_plate_views(cal_dir / "labelled_all_frames.npz")
        poses_file = cal_dir / "bundle_plate_poses.npz"
        used, bundle = None, {}
        if poses_file.exists():
            bp = np.load(poses_file)
            if "used_views" in bp.files:
                used = set(bp["used_views"].tolist())
            ref = str(grid_ref(folder))
            bundle[ref] = (np.eye(3), np.zeros(3))
            for fr, rv, tv in zip(bp["frames"], bp["plate_rvec"], bp["plate_tvec"]):
                bundle[str(fr)] = (rodrigues(rv), np.asarray(tv, float))

        plates = []
        for fr in sorted({f for _, f in views}):
            per_all = {
                ci: views[(ci, fr)] for ci in range(len(cams)) if (ci, fr) in views
            }
            per_used = {
                ci: v
                for ci, v in per_all.items()
                if used is None or f"{ci}_{fr}" in used
            }
            for kind, per in (("all", per_all), ("used", per_used)):
                tri = (
                    triangulate_plate(per, cals, cpar, min_dots=8)
                    if len(per) >= 2
                    else None
                )
                if tri is None:
                    continue
                m = plate_frame_metrics(tri.pos, tri.ids, grid)
                plates.append(
                    dict(
                        frame=fr,
                        kind=kind,
                        ids=tri.ids,
                        P=to_barrel(tri.pos),
                        rcm=tri.rcm,
                        grid_dev=m.grid_dev,
                        planarity=m.planarity_rms,
                        pitch=(
                            float(np.median(m.pitch_x)) if len(m.pitch_x) else np.nan,
                            float(np.median(m.pitch_y)) if len(m.pitch_y) else np.nan,
                        ),
                    )
                )
        cam_C = to_barrel([[c.ext_par.x0, c.ext_par.y0, c.ext_par.z0] for c in cals])
        corners = grid.object_points(
            [1, grid.nx, grid.n_points, grid.n_points - grid.nx + 1, 1]
        )
        outlines = {fr: to_barrel(corners @ R.T + t) for fr, (R, t) in bundle.items()}
        return dict(
            cams=cams,
            cam_C=cam_C,
            plates=plates,
            outlines=outlines,
            ori=f"cam{cams[0]}.tif.ori{suffix}",
            gated=used is not None,
        )

    def grid_ref(folder):
        import yaml

        p = yaml.safe_load((folder / "plate.yaml").read_text())["plate"]
        return p.get("origin_frame", "00000000")

    rigs = {name: load_rig(folder, cams) for name, (folder, cams, _) in RIGS.items()}
    return (rigs,)


@app.cell
def _(mo, rigs):
    frames_all = sorted({p["frame"] for r in rigs.values() for p in r["plates"]})
    rig_sel = mo.ui.multiselect(list(rigs), value=list(rigs), label="rigs")
    frame_sel = mo.ui.multiselect(frames_all, value=frames_all, label="frames")
    only_used = mo.ui.checkbox(
        True, label="only views the bundle accepted (rejected views are mislabelled)"
    )
    color_by = mo.ui.dropdown(
        ["rig", "frame", "ray-convergence miss [mm]", "grid deviation [mm]"],
        value="rig",
        label="colour dots by",
    )
    show_outline = mo.ui.checkbox(True, label="bundle-fitted plate outlines")
    show_cams = mo.ui.checkbox(True, label="cameras")
    show_barrel = mo.ui.checkbox(True, label="barrel wireframe")
    psize = mo.ui.slider(1, 6, value=2.5, step=0.5, label="dot size")
    mo.vstack(
        [
            mo.hstack([rig_sel, color_by, psize]),
            frame_sel,
            mo.hstack([only_used, show_outline, show_cams, show_barrel]),
        ]
    )
    return (
        color_by,
        frame_sel,
        only_used,
        psize,
        rig_sel,
        show_barrel,
        show_cams,
        show_outline,
    )


@app.cell
def _(
    H_BARREL,
    RIGS,
    R_WALL,
    color_by,
    frame_sel,
    go,
    np,
    only_used,
    psize,
    rig_sel,
    rigs,
    show_barrel,
    show_cams,
    show_outline,
):
    def xyz(P):  # world (X, Y, Z) -> plotly (x, y, z) with Y up
        return dict(x=P[:, 0], y=P[:, 2], z=P[:, 1])

    kind = "used" if only_used.value else "all"
    frames = set(frame_sel.value)
    fig = go.Figure()

    if show_barrel.value:
        th = np.linspace(0, 2 * np.pi, 200)
        for y in (-H_BARREL / 2, H_BARREL / 2):
            ring = np.stack(
                [R_WALL * np.cos(th), np.full_like(th, y), R_WALL * np.sin(th)], 1
            )
            fig.add_trace(
                go.Scatter3d(
                    **xyz(ring),
                    mode="lines",
                    line=dict(color="black", width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            seg = np.array(
                [
                    [R_WALL * np.cos(a), -H_BARREL / 2, R_WALL * np.sin(a)],
                    [R_WALL * np.cos(a), H_BARREL / 2, R_WALL * np.sin(a)],
                ]
            )
            fig.add_trace(
                go.Scatter3d(
                    **xyz(seg),
                    mode="lines",
                    line=dict(color="gray", width=1),
                    opacity=0.3,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    all_frames = sorted({p["frame"] for r in rigs.values() for p in r["plates"]})
    fidx = {f: k for k, f in enumerate(all_frames)}
    for name in rig_sel.value:
        rig, col = rigs[name], RIGS[name][2]
        sel = [p for p in rig["plates"] if p["kind"] == kind and p["frame"] in frames]
        if sel:
            P = np.vstack([p["P"] for p in sel])
            rcm = np.concatenate([p["rcm"] for p in sel])
            dev = np.concatenate([p["grid_dev"] for p in sel])
            fr = np.concatenate([[p["frame"]] * len(p["ids"]) for p in sel])
            ids = np.concatenate([p["ids"] for p in sel])
            if color_by.value == "rig":
                marker = dict(size=psize.value, color=col)
            elif color_by.value == "frame":
                marker = dict(
                    size=psize.value,
                    color=[fidx[f] for f in fr],
                    colorscale="Turbo",
                    showscale=False,
                )
            else:
                v = rcm if color_by.value.startswith("ray") else dev
                marker = dict(
                    size=psize.value,
                    color=v,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=float(np.percentile(v, 98)),
                    colorbar=dict(title=color_by.value, len=0.5),
                )
            hover = [
                f"rig {name}  frame {f}  dot {i}<br>"
                f"X {p[0]:.0f}  Y {p[1]:.0f}  Z {p[2]:.0f} mm<br>"
                f"ray miss {r:.2f} mm, grid dev {d:.2f} mm"
                for f, i, p, r, d in zip(fr, ids, P, rcm, dev)
            ]
            fig.add_trace(
                go.Scatter3d(
                    **xyz(P),
                    mode="markers",
                    marker=marker,
                    text=hover,
                    hoverinfo="text",
                    name=f"plates rig {name} (triangulated)",
                )
            )
        if show_outline.value:
            first = True
            for f, Q in sorted(rig["outlines"].items()):
                if f not in frames:
                    continue
                fig.add_trace(
                    go.Scatter3d(
                        **xyz(Q),
                        mode="lines",
                        line=dict(color=col, width=2),
                        opacity=0.7,
                        hoverinfo="text",
                        text=f"rig {name} frame {f} (bundle)",
                        name=f"plates rig {name} (bundle pose)",
                        legendgroup=f"o{name}",
                        showlegend=first,
                    )
                )
                first = False
        if show_cams.value:
            C = rig["cam_C"]
            fig.add_trace(
                go.Scatter3d(
                    **xyz(C),
                    mode="markers+text",
                    marker=dict(
                        size=7,
                        color=col,
                        symbol="diamond",
                        line=dict(color="black", width=1),
                    ),
                    text=[f"c{n}" for n in rig["cams"]],
                    textposition="top center",
                    name=f"cameras {name}",
                )
            )

    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(title="X [mm]", range=[-4000, 4000]),
            # reversed so (X, Z, Y) displays right-handed, i.e. not mirrored
            yaxis=dict(title="Z [mm]", range=[4000, -4000]),
            zaxis=dict(title="Y up [mm]", range=[-H_BARREL / 2, H_BARREL / 2]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=H_BARREL / 8000),
            camera=dict(eye=dict(x=1.2, y=1.4, z=0.8)),
        ),
        legend=dict(x=0.01, y=0.99),
    )
    fig
    return


@app.cell
def _(R_WALL, mo, np, only_used, rigs):
    _kind = "used" if only_used.value else "all"
    rows = []
    for _name, _rig in rigs.items():
        for _p in _rig["plates"]:
            if _p["kind"] != _kind:
                continue
            _P = _p["P"]
            rows.append(
                {
                    "rig": _name,
                    "frame": _p["frame"],
                    "dots": len(_p["ids"]),
                    "centre X": round(float(_P[:, 0].mean())),
                    "centre Y": round(float(_P[:, 1].mean())),
                    "centre Z": round(float(_P[:, 2].mean())),
                    "max radius": round(float(np.hypot(_P[:, 0], _P[:, 2]).max())),
                    "planarity RMS mm": round(_p["planarity"], 2),
                    "pitch X mm": round(_p["pitch"][0], 2),
                    "pitch Y mm": round(_p["pitch"][1], 2),
                    "ray miss med mm": round(float(np.median(_p["rcm"])), 2),
                    "grid dev max mm": round(float(_p["grid_dev"].max()), 1),
                }
            )
    _outside = sum(r["max radius"] > R_WALL for r in rows)
    _summary = " | ".join(
        f"rig {n}: `{r['ori']}`, median ray miss "
        f"{np.median([np.median(p['rcm']) for p in r['plates'] if p['kind'] == _kind]):.2f} mm"
        + (
            ""
            if r["gated"]
            else " (its bundle file lists no accepted views: unfiltered)"
        )
        for n, r in rigs.items()
    )
    mo.vstack(
        [
            mo.md(_summary),
            mo.md(
                f"**{len(rows)} plate reconstructions.** Plates reaching past the "
                f"{R_WALL:.0f} mm wall: **{_outside}**. A large *grid dev* means the "
                "dots were mislabelled; a large *ray miss* with a small grid dev means "
                "the calibration itself does not close there."
            ),
            mo.ui.table(rows, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
