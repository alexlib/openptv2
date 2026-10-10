"""Illmenau barrel: trajectories of both rigs in ONE barrel-frame dataset.

Reads one of the merged datasets built by scripts/illmenau/merge_dual_rig.py:
  * merged_fulldiam_marked.zarr -- EVERY trajectory of both rigs; rig 5-8 tracks
    that duplicate a rig 1-4 track are kept and marked (track_info/duplicate),
  * merged_fulldiam.zarr -- the same with those duplicates removed.
Both are both rigs' cleaned trajectories in metres,
barrel frame (origin on the axis at mid-height, +Y up; floor at Y = -1.79 m).
Rig 5-8 was mapped into rig 1-4's frame with the rig transform the merge
MEASURED from bubbles both rigs see (stored in the store's attributes), and
track_info gives each trajectory its rig, its partner in the other rig (the
track it shares most same-frame points with, closer than dup_mm), and whether it
is a duplicate.

Speeds are shown in mm per frame -- independent of the frame rate, which is
still to be settled (readme: 100 ms between images; post-processing used 50 fps).

Display: plotly's 3D axes are right-handed, so world (X, Z, Y) is drawn with the
Z axis reversed; the view is true and hover values are world coordinates.

Open with:  uv run marimo edit notebooks/illmenau_barrel_trajectories_marimo.py
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
    import zarr

    RAW = Path(os.environ.get("ILLMENAU_RAW", "/Users/alex/Downloads/Ilmenau"))
    R_WALL, H_BARREL = 3.575, 3.58  # m
    return H_BARREL, Path, RAW, R_WALL, go, mo, np, zarr


@app.cell
def _(mo):
    dataset = mo.ui.dropdown(
        {"all trajectories, duplicates marked (merged_fulldiam_marked.zarr)":
             "merged_fulldiam_marked.zarr",
         "duplicates removed (merged_fulldiam.zarr)": "merged_fulldiam.zarr"},
        value="all trajectories, duplicates marked (merged_fulldiam_marked.zarr)",
        label="dataset")
    dataset
    return (dataset,)


@app.cell
def _(RAW, dataset, np, zarr):
    MERGED = RAW / dataset.value
    _g = zarr.open_group(str(MERGED), mode="r")
    attrs = dict(_g.attrs)
    _t = _g["trajectories"]
    trajid = np.asarray(_t["trajid"][:]).astype(np.int64)
    ttime = np.asarray(_t["time"][:]).astype(np.int64)
    tpos = np.asarray(_t["pos"][:], float)  # m, barrel frame
    _ti = _g["track_info"]
    _ids = np.asarray(_ti["trajid"][:])
    _o = np.argsort(_ids)
    _k = _o[np.searchsorted(_ids, trajid, sorter=_o)]  # info row of every point
    rig_of = np.asarray(_ti["rig"][:])[_k]  # 0 = rig 1-4, 1 = rig 5-8
    partner_of = np.asarray(_ti["partner"][:])[_k]
    dup_of = np.asarray(_ti["duplicate"][:])[_k]
    # 0 rig 1-4 alone, 1 rig 1-4 with a rig 5-8 partner,
    # 2 rig 5-8 alone, 3 rig 5-8 with a partner (kept), 4 rig 5-8 duplicate
    cat_of = np.where(rig_of == 0, np.where(partner_of >= 0, 1, 0),
                      np.where(dup_of, 4, np.where(partner_of >= 0, 3, 2)))
    return MERGED, attrs, cat_of, partner_of, rig_of, tpos, trajid, ttime


@app.cell
def _(RAW, attrs, np):
    def _centres(folder, cams):
        out = []
        for n in cams:
            line = (RAW / folder / "cal" / f"cam{n}.tif.ori").read_text().split("\n")[0]
            out.append([float(v) for v in line.split()])
        return np.array(out)  # mm, rig frame

    _shift = np.array(attrs["barrel_from_rig14_mm"])
    _M = np.array(attrs["rig58_to_rig14"]["matrix"])
    _off = np.array(attrs["rig58_to_rig14"]["offset_mm"])
    cams14 = (_centres("openptv_illmenau_4cam", (1, 2, 3, 4)) + _shift) / 1000.0
    cams58 = (_centres("openptv_illmenau_5678", (5, 6, 7, 8)) @ _M.T - _off + _shift) / 1000.0
    return cams14, cams58


@app.cell
def _(MERGED, attrs, cat_of, mo, np, rig_of, trajid, ttime):
    _tr = attrs["rig58_to_rig14"]
    _u, _first = np.unique(trajid, return_index=True)
    _c = np.bincount(cat_of[_first], minlength=5)
    mo.md(
        f"""
        # Illmenau barrel — trajectories of both rigs

        `{MERGED.name}`: **{len(_u):,}** trajectories,
        {len(trajid):,} points, frames {ttime.min()}–{ttime.max()};
        rig 1-4 {int((rig_of == 0).sum()):,} points, rig 5-8 {int((rig_of == 1).sum()):,}.
        Rig 5-8 → rig 1-4 measured from {_tr['n_pairs']:,} common bubbles:
        {'180°' if _tr['matrix'][0][0] < 0 else '0°'} about the vertical, offset
        ({', '.join(f'{v:+.1f}' for v in _tr['offset_mm'])}) mm, agreement
        {' / '.join(f'{v:.1f}' for v in _tr['spread_mm'])} mm (x/y/z, MAD).

        Overlap (same-frame points closer than {attrs.get('dup_mm', 20):g} mm):
        rig 1-4 tracks with a rig 5-8 partner **{_c[1]:,}** (alone {_c[0]:,});
        rig 5-8 tracks with a partner **{_c[3] + _c[4]:,}**, of which
        **{_c[4]:,} duplicates** (partner at least as long){' — kept and marked' if attrs.get('duplicates_kept') else ' — removed from this dataset'};
        rig 5-8 alone {_c[2]:,}.
        """
    )
    return


@app.cell
def _(mo, ttime):
    frame_range = mo.ui.range_slider(
        start=int(ttime.min()), stop=int(ttime.max()), step=1,
        value=(int(ttime.min()), min(int(ttime.min()) + 100, int(ttime.max()))),
        label="frame range", show_value=True, full_width=True,
    )
    n_traj = mo.ui.slider(10, 3000, step=10, value=600, label="longest trajectories",
                          show_value=True)
    min_len = mo.ui.slider(2, 100, value=10, label="min frames in window", show_value=True)
    rigs = mo.ui.multiselect(["rig 1-4", "rig 5-8"], value=["rig 1-4", "rig 5-8"],
                             label="rigs")
    color_by = mo.ui.dropdown(["rig", "rig + duplicates", "speed [mm/frame]", "height Y",
                               "frame"], value="rig + duplicates", label="colour by")
    show_set = mo.ui.dropdown(["all", "paired only (both rigs)", "duplicates + their partners",
                               "unpaired only"], value="all", label="show")
    width = mo.ui.slider(1, 8, value=3, label="line width", show_value=True)
    show_cams = mo.ui.checkbox(True, label="cameras")
    show_barrel = mo.ui.checkbox(True, label="barrel")
    mo.vstack([frame_range,
               mo.hstack([n_traj, min_len, width]),
               mo.hstack([rigs, show_set, color_by, show_cams, show_barrel])])
    return (color_by, frame_range, min_len, n_traj, rigs, show_barrel, show_cams,
            show_set, width)


@app.cell
def _(
    cat_of,
    frame_range,
    min_len,
    n_traj,
    np,
    partner_of,
    rig_of,
    rigs,
    show_set,
    tpos,
    trajid,
    ttime,
):
    _f0, _f1 = frame_range.value
    _want = [i for i, r in enumerate(("rig 1-4", "rig 5-8")) if r in rigs.value]
    _m = (ttime >= _f0) & (ttime <= _f1) & np.isin(rig_of, _want)
    if show_set.value == "paired only (both rigs)":
        _m &= partner_of >= 0
    elif show_set.value == "unpaired only":
        _m &= partner_of < 0
    elif show_set.value == "duplicates + their partners":
        _dups = np.unique(trajid[cat_of == 4])
        _partners = np.unique(partner_of[cat_of == 4])
        _m &= np.isin(trajid, _dups) | np.isin(trajid, _partners)
    _tid, _t, _p, _r = trajid[_m], ttime[_m], tpos[_m], rig_of[_m]
    _cat = cat_of[_m]
    _ids, _cnt = np.unique(_tid, return_counts=True)
    _keep = _ids[_cnt >= min_len.value]
    _keep = _keep[np.argsort(-_cnt[_cnt >= min_len.value], kind="stable")][: n_traj.value]
    _sel = np.isin(_tid, _keep)
    _o = np.lexsort((_t[_sel], _tid[_sel]))
    sel_tid, sel_t, sel_p, sel_rig = _tid[_sel][_o], _t[_sel][_o], _p[_sel][_o], _r[_sel][_o]
    sel_cat = _cat[_sel][_o]
    # per-point speed in mm per frame, within each trajectory
    _same = np.r_[False, sel_tid[1:] == sel_tid[:-1]]
    _dt = np.maximum(np.r_[1, np.diff(sel_t)], 1)
    _v = np.r_[0.0, np.linalg.norm(np.diff(sel_p, axis=0), axis=1)] * 1000.0 / _dt
    _v[~_same] = np.nan
    _first = np.flatnonzero(~_same)
    _v[_first] = np.where(_first + 1 < len(_v), _v[np.minimum(_first + 1, len(_v) - 1)], 0.0)
    sel_speed = _v
    return sel_cat, sel_p, sel_rig, sel_speed, sel_t, sel_tid


@app.cell
def _(
    H_BARREL,
    R_WALL,
    cams14,
    cams58,
    color_by,
    go,
    np,
    sel_cat,
    sel_p,
    sel_rig,
    sel_speed,
    sel_t,
    sel_tid,
    show_barrel,
    show_cams,
    width,
):
    RIG_COLOURS = ("#ff7f0e", "#1f77b4")

    def xyz(P):  # world (X, Y, Z) -> plotly (x, y, z) with Y up
        P = np.asarray(P, float)
        return dict(x=P[:, 0], y=P[:, 2], z=P[:, 1])

    def with_breaks(mask):
        """Points of the selected trajectories, NaN between trajectories."""
        p, v, t, tid = sel_p[mask], sel_speed[mask], sel_t[mask], sel_tid[mask]
        brk = np.flatnonzero(np.diff(tid) != 0) + 1
        nan3 = np.full((1, 3), np.nan)
        P = np.insert(p, brk, nan3, axis=0)
        V = np.insert(v, brk, np.nan)
        T = np.insert(t.astype(float), brk, np.nan)
        return P, V, T

    fig = go.Figure()
    if show_barrel.value:
        _th = np.linspace(0, 2 * np.pi, 120)
        for _y in (-H_BARREL / 2, H_BARREL / 2):
            _ring = np.column_stack([R_WALL * np.cos(_th), np.full_like(_th, _y),
                                     R_WALL * np.sin(_th)])
            fig.add_trace(go.Scatter3d(**xyz(_ring), mode="lines",
                                       line=dict(color="black", width=3),
                                       hoverinfo="skip", showlegend=False))
        for _a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            _s = np.array([[R_WALL * np.cos(_a), -H_BARREL / 2, R_WALL * np.sin(_a)],
                           [R_WALL * np.cos(_a), H_BARREL / 2, R_WALL * np.sin(_a)]])
            fig.add_trace(go.Scatter3d(**xyz(_s), mode="lines",
                                       line=dict(color="gray", width=1), opacity=0.3,
                                       hoverinfo="skip", showlegend=False))
    CATS = ((0, "rig 1-4, no rig 5-8 partner", "#ff7f0e"),
            (1, "rig 1-4 with a rig 5-8 partner", "#d62728"),
            (2, "rig 5-8, no rig 1-4 partner", "#1f77b4"),
            (3, "rig 5-8 with a partner (kept: longer)", "#17becf"),
            (4, "rig 5-8 DUPLICATE of a rig 1-4 track", "#9467bd"))
    _groups = (
        [(sel_cat == c, n, col) for c, n, col in CATS]
        if color_by.value == "rig + duplicates"
        else [(sel_rig == 0, "rig 1-4", RIG_COLOURS[0]), (sel_rig == 1, "rig 5-8", RIG_COLOURS[1])]
    )
    for _m, _name, _col in _groups:
        if not _m.any():
            continue
        _P, _V, _T = with_breaks(_m)
        if color_by.value in ("rig", "rig + duplicates"):
            _line = dict(color=_col, width=width.value)
        else:
            _c = {"speed [mm/frame]": _V, "height Y": _P[:, 1], "frame": _T}[color_by.value]
            _ok = np.isfinite(_c)
            _line = dict(color=np.where(_ok, _c, np.nanmedian(_c)), colorscale="Viridis",
                         width=width.value, showscale=_name == _groups[0][1],
                         colorbar=dict(title=color_by.value, len=0.5),
                         cmin=float(np.nanpercentile(_c, 2)),
                         cmax=float(np.nanpercentile(_c, 98)))
        fig.add_trace(go.Scatter3d(**xyz(_P), mode="lines", line=_line,
                                   name=f"{_name} ({len(np.unique(sel_tid[_m]))} tracks)",
                                   hoverinfo="skip"))
    if show_cams.value:
        for _C, _names, _col in ((cams14, (1, 2, 3, 4), RIG_COLOURS[0]),
                                 (cams58, (5, 6, 7, 8), RIG_COLOURS[1])):
            fig.add_trace(go.Scatter3d(
                **xyz(_C), mode="markers+text", text=[f"c{n}" for n in _names],
                textposition="top center",
                marker=dict(size=6, color=_col, symbol="diamond",
                            line=dict(color="black", width=1)),
                name=f"cameras {_names[0]}-{_names[-1]}"))
    fig.update_layout(
        height=780, margin=dict(l=0, r=0, t=0, b=0), legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(title="X [m]", range=[-4, 4]),
            yaxis=dict(title="Z [m]", range=[4, -4]),  # reversed: right-handed view
            zaxis=dict(title="Y up [m]", range=[-H_BARREL / 2, H_BARREL / 2]),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=H_BARREL / 8),
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.7))))
    fig
    return


if __name__ == "__main__":
    app.run()
