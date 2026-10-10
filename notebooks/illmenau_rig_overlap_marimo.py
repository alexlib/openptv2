"""Illmenau: do the two rigs see the SAME bubbles and trajectories?

Cameras 1-4 and 5-8 look into the barrel from opposite sides; cams 5-8 fire
3 ms after cams 1-4 (readme), so a bubble moves < 3 mm between the two rigs'
images of the same frame.  Where their views overlap, the same bubble should
appear in both rigs' results -- as two dots a few mm apart and as two
trajectories running side by side.  This notebook shows both rigs in different
colours (rig 1-4 orange, rig 5-8 blue), pairs up trajectories that stay within
a tolerance of each other frame after frame, and says how close they are.

Each rig's numbers come from its own files, untouched:
  * trajectories: <rig>/res/run_fulldiam_clean.zarr (metres, the rig's own frame)
  * bubbles:      <rig>/res/run_fulldiam.zarr (4-camera points, mm)

"rig 5-8 coordinates" switch:
  * as calibrated -- rig 5-8 drawn in its own coordinates, no conversion;
  * measured conversion -- rig 5-8 converted into rig 1-4's coordinates with
    the rotation + offset that merge_dual_rig.py measured from common bubbles
    (stored in merged_fulldiam.zarr).  Each rig's calibration names its own
    axes from the face of the plate it saw, so the two can differ by a turn.
Both rigs are then shifted into the barrel frame (datum dot on the axis,
615 mm above the heating plate -> y - 1175 mm).

Open with:  uv run marimo edit notebooks/illmenau_rig_overlap_marimo.py
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
    from scipy.spatial import cKDTree

    from openptv2.storage import RunStore

    RAW = Path(os.environ.get("ILLMENAU_RAW", "/Users/alex/Downloads/Ilmenau"))
    RIG_DIRS = ("openptv_illmenau_4cam", "openptv_illmenau_5678")
    RIG_NAMES = ("rig 1-4", "rig 5-8")
    COLOURS = ("#ff7f0e", "#1f77b4")
    Y_SHIFT_MM = -1175.0
    return (
        COLOURS,
        RAW,
        RIG_DIRS,
        RIG_NAMES,
        RunStore,
        Y_SHIFT_MM,
        cKDTree,
        go,
        mo,
        np,
        zarr,
    )


@app.cell
def _(RAW, RIG_DIRS, np, zarr):
    def _load(rig_dir):
        t = zarr.open_group(str(RAW / rig_dir / "res" / "run_fulldiam_clean.zarr"),
                            mode="r")["trajectories"]
        return (np.asarray(t["trajid"][:]).astype(np.int64),
                np.asarray(t["time"][:]).astype(np.int64),
                np.asarray(t["pos"][:], float) * 1000.0)  # mm, rig frame

    tracks = [_load(d) for d in RIG_DIRS]
    _attrs = dict(zarr.open_group(str(RAW / "merged_fulldiam.zarr"), mode="r").attrs)
    measured = _attrs.get("rig58_to_rig14")
    return measured, tracks


@app.cell
def _(measured, mo, tracks):
    _f = tracks[0][1]
    conv = mo.ui.dropdown(
        ["as calibrated (no conversion)", "measured conversion"],
        value="measured conversion" if measured else "as calibrated (no conversion)",
        label="rig 5-8 coordinates")
    start = mo.ui.slider(int(_f.min()), int(_f.max()) - 5, value=10200, step=1,
                         label="first frame", show_value=True, full_width=True)
    length = mo.ui.slider(3, 100, value=20, label="frames in window", show_value=True)
    tol = mo.ui.slider(2.0, 30.0, value=8.0, step=0.5, label="pairing tolerance [mm]",
                       show_value=True)
    min_shared = mo.ui.slider(2, 50, value=3, label="min frames together", show_value=True)
    show_unpaired = mo.ui.checkbox(True, label="unpaired trajectories near the pairs (faint)")
    show_bubbles = mo.ui.checkbox(True, label="bubbles of the first frame")
    mo.vstack([
        mo.hstack([conv, length, tol, min_shared]),
        start,
        mo.hstack([show_unpaired, show_bubbles]),
    ])
    return conv, length, min_shared, show_bubbles, show_unpaired, start, tol


@app.cell
def _(Y_SHIFT_MM, conv, measured, np):
    if conv.value == "measured conversion" and measured:
        M58 = np.array(measured["matrix"], float)
        OFF58 = np.array(measured["offset_mm"], float)
    else:
        M58, OFF58 = np.eye(3), np.zeros(3)

    def to_barrel(p_mm, rig):
        """mm, rig frame -> mm, barrel frame."""
        p = np.asarray(p_mm, float)
        if rig == 1:
            p = p @ M58.T - OFF58
        return p + [0.0, Y_SHIFT_MM, 0.0]

    return M58, OFF58, to_barrel


@app.cell
def _(length, np, start, to_barrel, tracks):
    f0, f1 = start.value, start.value + length.value - 1
    win = []  # per rig: (trajid, frame, pos barrel mm)
    for _rig, (_tid, _t, _p) in enumerate(tracks):
        _m = (_t >= f0) & (_t <= f1)
        win.append((_tid[_m], _t[_m], to_barrel(_p[_m], _rig)))
    return f0, f1, win


@app.cell
def _(cKDTree, f0, f1, min_shared, np, tol, win):
    # same-frame proximity between rig 1-4 and rig 5-8 trajectory points
    (_ta, _fa, _pa), (_tb, _fb, _pb) = win
    _rows = []
    for _f in range(f0, f1 + 1):
        _ia, _ib = np.flatnonzero(_fa == _f), np.flatnonzero(_fb == _f)
        if not len(_ia) or not len(_ib):
            continue
        _d, _j = cKDTree(_pb[_ib]).query(_pa[_ia], distance_upper_bound=tol.value)
        _ok = np.isfinite(_d)
        for _a, _b in zip(_ia[_ok], _ib[_j[_ok]]):
            _rows.append((_ta[_a], _tb[_b], _f, *(_pb[_b] - _pa[_a])))
    close = np.array(_rows, float).reshape(-1, 6)  # tid_a, tid_b, frame, dx, dy, dz
    pairs = []  # each rig-1-4 track with the rig-5-8 track it stays closest to
    if len(close):
        _key = close[:, 0] * 1e7 + close[:, 1]
        _uk, _inv, _n = np.unique(_key, return_inverse=True, return_counts=True)
        _best = {}
        for _k, _cnt in enumerate(_n):
            _sel = _inv == _k
            _a, _b = int(close[_sel][0, 0]), int(close[_sel][0, 1])
            _sep = np.median(np.linalg.norm(close[_sel][:, 3:], axis=1))
            if _cnt >= min_shared.value and (_a not in _best or _cnt > _best[_a][1]):
                _best[_a] = (_b, int(_cnt), float(_sep), np.median(close[_sel][:, 3:], 0))
        _used_b = {}
        for _a, (_b, _cnt, _sep, _dm) in sorted(_best.items(), key=lambda kv: -kv[1][1]):
            if _b in _used_b:
                continue  # one partner per rig-5-8 track too
            _used_b[_b] = _a
            _la = int((_ta == _a).sum())
            _lb = int((_tb == _b).sum())
            pairs.append(dict(a=_a, b=_b, shared=_cnt, len_a=_la, len_b=_lb, sep=_sep, d=_dm))
    return close, pairs


@app.cell
def _(conv, f0, f1, mo, np, pairs, tol, win):
    (_ta, _fa, _pa), (_tb, _fb, _pb) = win
    _na, _nb = len(np.unique(_ta)), len(np.unique(_tb))
    if pairs:
        _sh = np.array([p["shared"] for p in pairs])
        _cov = np.array([p["shared"] / max(1, min(p["len_a"], p["len_b"])) for p in pairs])
        _sep = np.array([p["sep"] for p in pairs])
        _d = np.array([p["d"] for p in pairs])
        _txt = (
            f"**{len(pairs)} trajectory pairs** run together (≥ the min frames within "
            f"{tol.value:g} mm), out of {_na:,} rig 1-4 and {_nb:,} rig 5-8 trajectories "
            f"in frames {f0}–{f1}.  Together for a median of **{np.median(_sh):.0f} frames**, "
            f"i.e. **{np.median(_cov) * 100:.0f} %** of the shorter partner's frames in the "
            f"window; median separation **{np.median(_sep):.1f} mm**, median offset "
            f"rig 5-8 − rig 1-4 = ({', '.join(f'{v:+.1f}' for v in np.median(_d, 0))}) mm."
        )
    else:
        _txt = (f"**No trajectory pairs** within {tol.value:g} mm in frames {f0}–{f1} "
                f"({_na:,} rig 1-4 and {_nb:,} rig 5-8 trajectories).")
    mo.md(f"### rig 5-8 coordinates: *{conv.value}*\n\n{_txt}")
    return


@app.cell
def _(
    COLOURS,
    RAW,
    RIG_DIRS,
    RunStore,
    f0,
    go,
    np,
    pairs,
    show_bubbles,
    show_unpaired,
    to_barrel,
    win,
):
    def xyz(P):  # world (X, Y, Z) mm -> plotly (x, y, z) with Y up
        P = np.asarray(P, float)
        return dict(x=P[:, 0], y=P[:, 2], z=P[:, 1])

    def lines(tid, frame, pos, keep):
        m = np.isin(tid, list(keep))
        o = np.lexsort((frame[m], tid[m]))
        t, p = tid[m][o], pos[m][o]
        brk = np.flatnonzero(np.diff(t) != 0) + 1
        return np.insert(p, brk, np.full((1, 3), np.nan), axis=0)

    fig = go.Figure()
    _pa_ids = {p["a"] for p in pairs}
    _pb_ids = {p["b"] for p in pairs}
    _paired = [_pa_ids, _pb_ids]
    if pairs:
        _allp = np.vstack([win[0][2][np.isin(win[0][0], list(_pa_ids))],
                           win[1][2][np.isin(win[1][0], list(_pb_ids))]])
        _lo, _hi = _allp.min(0) - 150, _allp.max(0) + 150
    for _rig in (0, 1):
        _tid, _fr, _p = win[_rig]
        if show_unpaired.value and pairs:
            _near = np.all((_p >= _lo) & (_p <= _hi), axis=1)
            _others = set(np.unique(_tid[_near]).tolist()) - _paired[_rig]
            if _others:
                fig.add_trace(go.Scatter3d(**xyz(lines(_tid, _fr, _p, _others)), mode="lines",
                                           line=dict(color=COLOURS[_rig], width=1.5),
                                           opacity=0.25, hoverinfo="skip",
                                           name=f"{('rig 1-4', 'rig 5-8')[_rig]} unpaired"))
        if _paired[_rig]:
            fig.add_trace(go.Scatter3d(**xyz(lines(_tid, _fr, _p, _paired[_rig])),
                                       mode="lines+markers",
                                       line=dict(color=COLOURS[_rig], width=4),
                                       marker=dict(size=2, color=COLOURS[_rig]),
                                       name=f"{('rig 1-4', 'rig 5-8')[_rig]} paired "
                                            f"({len(_paired[_rig])})"))
    if show_bubbles.value:
        for _rig in (0, 1):
            _st = RunStore(str(RAW / RIG_DIRS[_rig] / "res" / "run_fulldiam.zarr"), mode="r")
            _pos, _ids = _st.read_correspondences(f0)
            _q = to_barrel(_pos[(_ids >= 0).sum(1) == 4], _rig)
            if pairs:
                _q = _q[np.all((_q >= _lo) & (_q <= _hi), axis=1)]
            fig.add_trace(go.Scatter3d(**xyz(_q), mode="markers",
                                       marker=dict(size=3, color=COLOURS[_rig],
                                                   symbol="circle-open"),
                                       name=f"{('rig 1-4', 'rig 5-8')[_rig]} bubbles, frame {f0}"))
    fig.update_layout(
        height=760, margin=dict(l=0, r=0, t=0, b=0), legend=dict(x=0.01, y=0.99),
        scene=dict(xaxis_title="X [mm]",
                   yaxis=dict(title="Z [mm]", autorange="reversed"),  # right-handed view
                   zaxis_title="Y up [mm]", aspectmode="data"))
    fig
    return (xyz,)


@app.cell
def _(close, go, mo, np):
    _fig = go.Figure()
    if len(close):
        for _k, _ax in enumerate("xyz"):
            _fig.add_trace(go.Histogram(x=close[:, 3 + _k], name=f"d{_ax}",
                                        xbins=dict(size=0.5), opacity=0.6))
    _fig.update_layout(barmode="overlay", height=300, margin=dict(l=40, r=10, t=30, b=40),
                       title="rig 5-8 − rig 1-4, same frame, every close point pair [mm]",
                       xaxis_title="mm")
    mo.vstack([_fig, mo.md(f"{len(close):,} same-frame point pairs within the tolerance "
                           f"(MAD {', '.join(f'{v:.2f}' for v in np.median(np.abs(close[:, 3:] - np.median(close[:, 3:], 0)), 0)) if len(close) else '-'} mm)")])
    return


@app.cell
def _(mo, pairs):
    _opts = {f"#{k + 1}: 1-4 track {p['a']} / 5-8 track {p['b']} — {p['shared']} frames, "
             f"{p['sep']:.1f} mm": k for k, p in enumerate(pairs[:200])}
    pick = mo.ui.dropdown(_opts, value=next(iter(_opts)) if _opts else None,
                          label="inspect a pair")
    table = mo.ui.table(
        [dict(rank=k + 1, rig14_track=p["a"], rig58_track=p["b"], frames_together=p["shared"],
              rig14_frames=p["len_a"], rig58_frames=p["len_b"], median_sep_mm=round(p["sep"], 2),
              dx=round(float(p["d"][0]), 2), dy=round(float(p["d"][1]), 2),
              dz=round(float(p["d"][2]), 2)) for k, p in enumerate(pairs)],
        page_size=10, selection=None)
    mo.vstack([pick, table])
    return (pick,)


@app.cell
def _(COLOURS, go, mo, np, pairs, pick, win, xyz):
    if pick.value is None or not pairs:
        _out = mo.md("No pair to inspect.")
    else:
        _p = pairs[pick.value]
        _A = win[0][2][win[0][0] == _p["a"]], win[0][1][win[0][0] == _p["a"]]
        _B = win[1][2][win[1][0] == _p["b"]], win[1][1][win[1][0] == _p["b"]]
        _oa, _ob = np.argsort(_A[1]), np.argsort(_B[1])
        _f3 = go.Figure()
        for _rig, (_P, _F), _o in ((0, _A, _oa), (1, _B, _ob)):
            _f3.add_trace(go.Scatter3d(**xyz(_P[_o]), mode="lines+markers",
                                       line=dict(color=COLOURS[_rig], width=5),
                                       marker=dict(size=3, color=COLOURS[_rig]),
                                       text=[f"frame {f}" for f in _F[_o]],
                                       name=("rig 1-4", "rig 5-8")[_rig]))
        _common = np.intersect1d(_A[1], _B[1])
        _sep = []
        for _f in _common:
            _a = _A[0][_A[1] == _f][0]
            _b = _B[0][_B[1] == _f][0]
            _sep.append(np.linalg.norm(_b - _a))
            _f3.add_trace(go.Scatter3d(**xyz(np.array([_a, _b])), mode="lines",
                                       line=dict(color="black", width=2),
                                       showlegend=False, hoverinfo="skip"))
        _f3.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                          title="the pair, with same-frame connectors",
                          scene=dict(xaxis_title="X [mm]",
                                     yaxis=dict(title="Z [mm]", autorange="reversed"),
                                     zaxis_title="Y up [mm]", aspectmode="data"))
        _f4 = go.Figure(go.Scatter(x=_common, y=_sep, mode="lines+markers"))
        _f4.update_layout(height=260, margin=dict(l=40, r=10, t=30, b=40),
                          title="separation per frame [mm]", xaxis_title="frame",
                          yaxis_title="mm")
        _out = mo.hstack([_f3, _f4], widths=[2, 1])
    _out
    return


if __name__ == "__main__":
    app.run()
