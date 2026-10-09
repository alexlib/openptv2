"""Dropbox postptv visualization: longest tracks of the merged dual-rig set.

Self-contained plotly HTML (plotly.js embedded, no server needed).
Source: /Users/alex/Downloads/Ilmenau/merged_fulldiam.zarr
(meters, barrel frame; trajid 1000..1000+N14-1 = rig 1-4, rest = rig 5-8).

Usage:
    uv run python scripts/illmenau/build_dropbox_viz.py [--n58 6000] [--out PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import zarr

RAW = Path("/Users/alex/Downloads/Ilmenau")
N14 = 3402  # len(ta) at merge time: rig 1-4 ids are 1000..1000+N14-1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n58", type=int, default=6000)
    ap.add_argument("--src", default=str(RAW / "merged_fulldiam.zarr"))
    ap.add_argument("--out", default=str(
        Path("/Users/alex/Library/CloudStorage/Dropbox/3DPTV_Illmenau/"
             "August2026/illmenau_dual_rig_postptv_v5.html")))
    a = ap.parse_args()

    g = zarr.open(a.src, mode="r")["trajectories"]
    time = np.asarray(g["time"])
    pos = np.asarray(g["pos"])
    tid = np.asarray(g["trajid"])
    vel = np.asarray(g["vel"])
    # vertical velocity component: +Y is up (against gravity), so +w = red.
    wvel = vel[:, 1]
    wmax = float(np.percentile(np.abs(wvel), 99))

    order = np.argsort(tid, kind="stable")
    tids = np.unique(tid)
    lens = {t: int((tid == t).sum()) for t in tids}
    rig_of = {t: ("1-4" if t < 1000 + N14 else "5-8") for t in tids}
    t14 = sorted([t for t in tids if rig_of[t] == "1-4"], key=lambda t: -lens[t])
    t58 = sorted([t for t in tids if rig_of[t] == "5-8"], key=lambda t: -lens[t])
    sel = t14 + t58[:a.n58]
    print(f"tracks: 1-4 all {len(t14)}, 5-8 longest {min(a.n58, len(t58))} "
          f"of {len(t58)}", flush=True)

    fig = go.Figure()
    # matplotlib coolwarm: blue (down) -> neutral -> red (up), midpoint at 0.
    COOLWARM = [[0.0, "rgb(59,76,192)"], [0.25, "rgb(119,170,203)"],
                [0.5, "rgb(221,221,221)"], [0.75, "rgb(245,156,125)"],
                [1.0, "rgb(180,4,38)"]]
    for tag, tidsel, col in (("1-4", t14, "#ff7f0e"), ("5-8", t58[:a.n58], "#1f77b4")):
        xs, ys, zs, cs = [], [], [], []
        for t in tidsel:
            m = tid == t
            p = pos[m]
            # time order within track
            o = np.argsort(time[m], kind="stable")
            p, w = p[o], wvel[m][o]
            xs += p[:, 0].tolist() + [None]  # plotly x = barrel X
            ys += p[:, 2].tolist() + [None]  # plotly y = barrel Z
            zs += p[:, 1].tolist() + [None]  # plotly z = barrel Y (up)
            cs += w.tolist() + [None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=[c if c is not None else 0 for c in cs],
                      colorscale=COOLWARM, cmin=-wmax, cmax=+wmax,
                      width=2, showscale=(tag == "5-8"),
                      colorbar=dict(title="vertical w [m/s], +up", x=1.02)),
            opacity=0.75, hoverinfo="skip",
            name=f"rig {tag}: {len(tidsel)} longest"))
        # sparse track-id labels: longest 60 per rig
        lx, ly, lz, lt = [], [], [], []
        for t in tidsel[:60]:
            m = tid == t
            p = pos[m]
            mid = p[len(p) // 2]
            lx.append(mid[0])
            ly.append(mid[1])
            lz.append(mid[2])
            lt.append(f"T{t}")
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="text", text=lt,
            textfont=dict(size=8), hoverinfo="skip", showlegend=False))

    # cameras (both rigs, barrel meters) + barrel rings
    folders = {"1-4": RAW / "openptv_illmenau_4cam",
               "5-8": RAW / "openptv_illmenau_5678"}
    for rig, cams, col in (("1-4", [1, 2, 3, 4], "#ff7f0e"),
                           ("5-8", [5, 6, 7, 8], "#1f77b4")):
        folder = folders[rig]
        C = []
        for n in cams:
            v = [float(x) for x in
                 (folder / f"cal/cam{n}.tif.ori").read_text().splitlines()[0].split()]
            C.append([v[0] / 1000, v[1] / 1000 - 1.175, v[2] / 1000])
        C = np.array(C)
        fig.add_trace(go.Scatter3d(
            x=C[:, 0], y=C[:, 2], z=C[:, 1], mode="markers+text",
            marker=dict(size=5, color=col, symbol="diamond",
                        line=dict(color="black", width=1)),
            text=[f"c{n}" for n in cams], textposition="top center",
            name=f"cameras {rig}"))
    R, H = 3.575, 3.58
    th = np.linspace(0, 2 * np.pi, 200)
    for yy in (-H / 2, H / 2):
        fig.add_trace(go.Scatter3d(
            x=R * np.cos(th), y=R * np.sin(th), z=np.full_like(th, yy),
            mode="lines", line=dict(color="black", width=3),
            hoverinfo="skip", showlegend=False))
    # translucent barrel wall + floor/ceiling: inside/outside is decidable
    # from any angle (rings alone fool the eye in perspective).
    uu = np.linspace(0, 2 * np.pi, 120)
    vv = np.linspace(-H / 2, H / 2, 2)
    U, V = np.meshgrid(uu, vv)
    fig.add_trace(go.Surface(
        x=R * np.cos(U), y=R * np.sin(U), z=V,
        opacity=0.08, showscale=False, hoverinfo="skip",
        showlegend=False, name="wall"))
    for yy in (-H / 2, H / 2):
        rr = np.linspace(0, R, 40)
        TT, RR = np.meshgrid(th[:120], rr)
        fig.add_trace(go.Surface(
            x=RR * np.cos(TT), y=RR * np.sin(TT),
            z=np.full_like(TT, yy),
            opacity=0.06, showscale=False, hoverinfo="skip",
            showlegend=False, name="cap"))

    med = float(np.median(list(lens.values())))
    fig.update_layout(
        title=(f"Illmenau dual-rig postptv — merged barrel set "
               f"({len(tids)} tracks, median len {med:.0f}; "
               f"self-cal .ori, full-diameter boxes, v_max=40, no confirm)"),
        height=860, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(title="X [m]"), yaxis=dict(title="Z [m]"),
            zaxis=dict(title="Y up [m]"),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=H / (2 * R)),
            camera=dict(eye=dict(x=1.1, y=-1.5, z=0.8))),
        legend=dict(x=0.01, y=0.99))
    fig.write_html(a.out, include_plotlyjs=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
