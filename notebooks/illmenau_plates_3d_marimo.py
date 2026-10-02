import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "scripts" / "illmenau")
    )
    import reconstruct_plates_barrel as rp

    return go, mo, np, os, rp


@app.cell
def _(mo):
    mo.md(r"""
    # Ilmenau calibration plates, interactive 3D

    Every labelled plate view is **triangulated with the current `.ori`/`.addpar`**
    (not the bundle poses it was fitted from), then both rigs are drawn in one barrel
    frame: origin on the axis at mid-height, +Y up. Each rig's origin dot is on the axis,
    so a rig is moved into the barrel by a vertical shift only.

    **Orange = cams 1-4, blue = cams 5-8.** Rotate with the mouse; pick a frame to
    highlight it with point ids.
    """)
    return


@app.cell
def _(np, rp):
    def _load():
        out = {}
        for name, (folder, cams, _col) in rp.RIGS.items():
            cals, views = rp.load_rig(folder, cams)
            cpar = rp.cpar_for(len(cals))
            xyz = np.array([[c.ext_par.x0, c.ext_par.y0, c.ext_par.z0] for c in cals])
            plates = []
            for fr in sorted({f for (_, f) in views}):
                r = rp.triangulate(fr, cals, views, cpar)
                if r is None:
                    continue
                P, ids = r
                c, n = rp.plane_fit(P)
                rms = float(np.sqrt((((P - c) @ n) ** 2).mean()))
                B = np.stack([P[:, 0], P[:, 1] + rp.Y_DATUM_BARREL, P[:, 2]], 1)
                plates.append(dict(frame=fr, P=B, ids=ids, rms=rms))
            cb = np.stack([xyz[:, 0], xyz[:, 1] + rp.Y_DATUM_BARREL, xyz[:, 2]], 1)
            out[name] = dict(cams=cams, cam_xyz=cb, plates=plates)
        return out

    res = _load()
    return (res,)


@app.cell
def _(mo, res, rp):
    rig_sel = mo.ui.multiselect(
        options=list(rp.RIGS), value=list(rp.RIGS), label="rigs"
    )
    max_rms = mo.ui.slider(
        1, 100, value=100, step=1, label="hide plates with plane RMS above [mm]"
    )
    frame_opts = {"(none)": ""} | {
        f"{n}  {p['frame']}  rms {p['rms']:.1f}": f"{n}|{p['frame']}"
        for n in res
        for p in res[n]["plates"]
    }
    frame_sel = mo.ui.dropdown(frame_opts, value="(none)", label="highlight frame")
    show_cyl = mo.ui.checkbox(True, label="barrel")
    show_cams = mo.ui.checkbox(True, label="cameras + sight lines")
    psize = mo.ui.slider(1, 6, value=2, step=0.5, label="dot size")
    mo.hstack([rig_sel, max_rms, frame_sel], justify="start", gap=2)
    return frame_sel, max_rms, psize, rig_sel, show_cams, show_cyl


@app.cell
def _(
    frame_sel,
    go,
    max_rms,
    mo,
    np,
    psize,
    res,
    rig_sel,
    rp,
    show_cams,
    show_cyl,
):
    COL = {"1-4": "#ff7f0e", "5-8": "#1f77b4"}  # matplotlib tab:orange / tab:blue
    # plotly axes: (x, z, y) so the vertical axis on screen is +Y
    fig = go.Figure()
    R, H = rp.R_WALL, rp.H_BARREL
    if show_cyl.value:
        th = np.linspace(0, 2 * np.pi, 200)
        for y in (-H / 2, H / 2):
            fig.add_trace(
                go.Scatter3d(
                    x=R * np.cos(th),
                    y=R * np.sin(th),
                    z=np.full_like(th, y),
                    mode="lines",
                    line=dict(color="black", width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            fig.add_trace(
                go.Scatter3d(
                    x=[R * np.cos(a)] * 2,
                    y=[R * np.sin(a)] * 2,
                    z=[-H / 2, H / 2],
                    mode="lines",
                    line=dict(color="gray", width=1),
                    opacity=0.3,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    hl = frame_sel.value.split("|") if frame_sel.value else None
    for n in rig_sel.value:
        d = res[n]
        keep = [p for p in d["plates"] if p["rms"] <= max_rms.value]
        rest = [p for p in keep if not (hl and hl == [n, p["frame"]])]
        if rest:
            P = np.vstack([p["P"] for p in rest])
            fr = np.concatenate([[p["frame"]] * len(p["P"]) for p in rest])
            ids = np.concatenate([p["ids"] for p in rest])
            fig.add_trace(
                go.Scatter3d(
                    x=P[:, 0],
                    y=P[:, 2],
                    z=P[:, 1],
                    mode="markers",
                    marker=dict(size=psize.value, color=COL[n], opacity=0.55),
                    text=[f"rig {n} frame {f} id {i}" for f, i in zip(fr, ids)],
                    hoverinfo="text",
                    name=f"plates rig {n}",
                )
            )
        if show_cams.value:
            C = d["cam_xyz"]
            fig.add_trace(
                go.Scatter3d(
                    x=C[:, 0],
                    y=C[:, 2],
                    z=C[:, 1],
                    mode="markers+text",
                    marker=dict(
                        size=7,
                        color=COL[n],
                        symbol="diamond",
                        line=dict(color="black", width=1),
                    ),
                    text=[f"c{c}" for c in d["cams"]],
                    textposition="top center",
                    name=f"cameras {n}",
                )
            )
            o = np.array([0.0, 0.0, rp.Y_DATUM_BARREL])  # origin dot, barrel frame
            for c in C:
                fig.add_trace(
                    go.Scatter3d(
                        x=[c[0], o[0]],
                        y=[c[2], o[1]],
                        z=[c[1], o[2]],
                        mode="lines",
                        line=dict(color=COL[n], width=1, dash="dot"),
                        opacity=0.5,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        if hl and hl[0] == n:
            p = next((p for p in d["plates"] if p["frame"] == hl[1]), None)
            if p is not None:
                P = p["P"]
                fig.add_trace(
                    go.Scatter3d(
                        x=P[:, 0],
                        y=P[:, 2],
                        z=P[:, 1],
                        mode="markers+text",
                        marker=dict(size=6, color="crimson"),
                        text=[str(i) for i in p["ids"]],
                        textposition="top center",
                        name=f"frame {p['frame']} (rms {p['rms']:.1f} mm)",
                    )
                )
    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(title="X [mm]", range=[-R, R]),
            yaxis=dict(title="Z [mm]", range=[-R, R]),
            zaxis=dict(title="Y up [mm]", range=[-H / 2, H / 2]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=H / (2 * R)),
            camera=dict(eye=dict(x=1.1, y=-1.5, z=0.8)),
        ),
        legend=dict(x=0.01, y=0.99),
    )
    mo.vstack([mo.hstack([show_cyl, show_cams, psize], justify="start", gap=2), fig])
    return


@app.cell
def _(mo, np, res):
    rows = [
        {
            "rig": n,
            "frame": p["frame"],
            "dots": len(p["P"]),
            "centre_x": round(float(p["P"][:, 0].mean())),
            "centre_z": round(float(p["P"][:, 2].mean())),
            "centre_y": round(float(p["P"][:, 1].mean())),
            "plane_rms_mm": round(p["rms"], 2),
        }
        for n in res
        for p in res[n]["plates"]
    ]
    mo.vstack(
        [
            mo.md("## Per-frame table (sort by `plane_rms_mm` to find bad labels)"),
            mo.ui.table(rows, page_size=15),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
