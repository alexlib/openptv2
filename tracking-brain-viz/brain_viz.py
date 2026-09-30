# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "matplotlib>=3.7.0",
#     "imageio>=2.19.0",
#     "pandas>=2.0.0",
#     "zarr>=2.18.0",
# ]
# ///

"""Tracking brain viz: click-a-tracer image-space replay from zarr + raw images.

Run:  uv run marimo run tracking-brain-viz/brain_viz.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Tracking brain viz

        Full-frame camera view with target IDs → pick a 3D particle row →
        zoom strip over `t…t+3` following it through linkage `next` pointers,
        with gate, candidates, and the tracker's chosen link.
        Rarely-used tool: for understanding *one* link decision, not batches.
        """
    )
    return


@app.cell
def _(mo):
    zarr_path = mo.ui.text(
        value="test_data/test_cavity/run.zarr", label="zarr store"
    )
    img_dir = mo.ui.text(
        value="test_data/test_cavity/img", label="image dir"
    )
    frame = mo.ui.number(value=10000, label="frame t")
    cam = mo.ui.number(value=1, label="camera (1-based)")
    prow = mo.ui.number(value=0, label="particle row at t (correspondences)")
    hw = mo.ui.slider(20, 400, step=10, value=120, label="zoom half-width px")
    nframes = mo.ui.slider(2, 4, step=1, value=4, label="strip length")
    gate_px = mo.ui.slider(5, 200, step=5, value=40, label="pixel gate radius")
    show_image = mo.ui.checkbox(value=True, label="image")
    show_ids = mo.ui.checkbox(value=True, label="target ids")
    show_links = mo.ui.checkbox(value=True, label="linkage arrows")
    show_gate = mo.ui.checkbox(value=True, label="gate + candidates")
    mo.vstack(
        [
            mo.hstack([zarr_path, img_dir]),
            mo.hstack([frame, cam, prow, hw, nframes, gate_px]),
            mo.hstack([show_image, show_ids, show_links, show_gate]),
        ]
    )
    return (
        zarr_path,
        img_dir,
        frame,
        cam,
        prow,
        hw,
        nframes,
        gate_px,
        show_image,
        show_ids,
        show_links,
        show_gate,
    )


@app.cell
def _(np, zarr_path, img_dir):
    import imageio.v2 as iio
    from pathlib import Path

    from openptv2.storage import RunStore

    _store = RunStore(str(Path(zarr_path.value)), mode="r")
    _imgdir = Path(img_dir.value)

    def load_image(_cam, _f):
        for _name in (f"cam{_cam}.{_f}", f"Cam{_cam}.{_f}", f"cam_{_cam}.{_f}"):
            _p = _imgdir / _name
            if _p.exists():
                try:
                    return np.asarray(iio.imread(str(_p)))
                except Exception:
                    return None
        return None

    def load_targets(_cam0, _f):
        try:
            return np.asarray(_store.root[f"targets/cam_{_cam0}/frame_{_f:06d}"])
        except KeyError:
            return np.zeros((0, 8))

    def load_corr(_f):
        try:
            _d = np.asarray(_store.root[f"correspondences/frame_{_f:06d}"])
            return _d[:, :3], _d[:, 3:].astype(int)
        except KeyError:
            return np.zeros((0, 3)), np.zeros((0, 0), dtype=int)

    def load_link(_f, _name="ptv_is"):
        try:
            _g = _store.root[f"linkage/{_name}/frame_{_f:06d}"]
            return (
                np.asarray(_g["prev"]).astype(int),
                np.asarray(_g["next"]).astype(int),
            )
        except KeyError:
            return None, None

    _cams = _store.target_cameras()
    _frames = _store.frames()
    tstore, tcams, tframes = _store, _cams, _frames
    return (
        tstore,
        tcams,
        tframes,
        load_image,
        load_targets,
        load_corr,
        load_link,
    )


@app.cell
def _(mo, tcams, tframes):
    mo.md(
        f"Store: cams `{tcams}`, frames `{tframes[0]}…{tframes[-1]}` "
        f"({len(tframes)} frames). "
        "Particle row = row in `correspondences` at frame `t`; "
        "followed via linkage `next` (row ids are per-frame array rows)."
    )
    return


@app.cell
def _(
    np,
    frame,
    cam,
    prow,
    nframes,
    load_link,
    load_corr,
    load_targets,
):
    _t0, _c0, _r0, _n = int(frame.value), int(cam.value) - 1, int(prow.value), int(nframes.value)
    _trail = []  # per strip frame: dict(row, tid, x, y, next_row)
    _r = _r0
    for _k in range(_n):
        _f = _t0 + _k
        _pos, _cids = load_corr(_f)
        _tg = load_targets(_c0, _f)
        _txy = {int(_row[0]): (float(_row[1]), float(_row[2])) for _row in _tg}
        _tid, _x, _y = -1, float("nan"), float("nan")
        if 0 <= _r < len(_pos) and _c0 < _cids.shape[1]:
            _tid = int(_cids[_r, _c0])
            if _tid in _txy:
                _x, _y = _txy[_tid]
        _nxt, _nn = load_link(_f)
        _nr = int(_nn[_r]) if _nn is not None and 0 <= _r < len(_nn) else -1
        _trail.append(
            {"frame": _f, "row": _r, "tid": _tid, "x": _x, "y": _y, "next": _nr}
        )
        _r = _nr if _nr is not None and _nr >= 0 else -1
    (_trail, _t0, _c0, _r0)
    trail, t0, c0, r0 = _trail, _t0, _c0, _r0
    return (trail, t0, c0, r0)


@app.cell
def _(
    plt,
    np,
    trail,
    t0,
    c0,
    load_image,
    load_targets,
    load_link,
    show_image,
    show_ids,
    show_links,
    cam,
):
    _img = load_image(int(cam.value), t0)
    _tg0 = load_targets(c0, t0)
    _pv0, _nx0 = load_link(t0)
    _sel = trail[0]
    _fig, _ax = plt.subplots(figsize=(8, 6))
    if show_image.value and _img is not None:
        _ax.imshow(_img, cmap="gray")
    else:
        _ax.set_xlim(0, 1000)
        _ax.set_ylim(1000, 0)
        _ax.set_title("image not found — overlay only")
    if len(_tg0):
        _ax.scatter(_tg0[:, 1], _tg0[:, 2], s=18, marker="x")
        if show_ids.value:
            for _row in _tg0[:: max(1, len(_tg0) // 60)]:
                _ax.text(float(_row[1]) + 3, float(_row[2]) - 3, f"{int(_row[0])}", fontsize=7)
    if show_links.value and _nx0 is not None and len(_tg0):
        _id2xy = {int(_row[0]): (float(_row[1]), float(_row[2])) for _row in _tg0}
        _tg1 = load_targets(c0, t0 + 1)
        _id2xy1 = {int(_row[0]): (float(_row[1]), float(_row[2])) for _row in _tg1}
        for _i in range(min(len(_nx0), 400)):
            _j = int(_nx0[_i])
            if _j < 0 or _i >= len(_tg0) or _j >= len(_tg1):
                continue
            _a, _b = _id2xy.get(int(_tg0[_i, 0])), _id2xy1.get(int(_tg1[_j, 0]))
            if _a and _b:
                _ax.annotate("", xy=_b, xytext=_a,
                             arrowprops=dict(arrowstyle="->", lw=0.7))
    if np.isfinite(_sel["x"]):
        _ax.scatter([_sel["x"]], [_sel["y"]], s=200, facecolors="none", edgecolors="lime", lw=2)
        _ax.text(_sel["x"] + 5, _sel["y"] + 5,
                 f"row{_sel['row']}/tid{_sel['tid']}", color="lime", fontsize=10)
    _ax.set_title(f"cam {cam.value} frame {t0}: crosses=targets, arrows=linkage next")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    plt,
    np,
    pd,
    mo,
    trail,
    c0,
    hw,
    gate_px,
    load_image,
    load_targets,
    show_image,
    show_gate,
    cam,
):
    _H = int(hw.value)
    _figs = []
    _tables = []
    for _s in trail:
        _img = load_image(int(cam.value), _s["frame"])
        _tg = load_targets(c0, _s["frame"])
        _cx, _cy = _s["x"], _s["y"]
        _fig, _ax = plt.subplots(figsize=(4, 4))
        if show_image.value and _img is not None and np.isfinite(_cx):
            _x0, _x1 = max(0, int(_cx) - _H), int(_cx) + _H
            _y0, _y1 = max(0, int(_cy) - _H), int(_cy) + _H
            _ax.imshow(_img[_y0:_y1, _x0:_x1], cmap="gray",
                       extent=[_x0, _x1, _y1, _y0])
            _ax.set_xlim(_x0, _x1)
            _ax.set_ylim(_y1, _y0)
        elif len(_tg):
            _ax.scatter(_tg[:, 1], _tg[:, 2], s=10, marker=".")
            _ax.set_aspect("equal")
        _cand = []
        if np.isfinite(_cx) and len(_tg):
            _d = np.hypot(_tg[:, 1] - _cx, _tg[:, 2] - _cy)
            _inside = np.flatnonzero(_d <= float(gate_px.value))
            for _i in _inside:
                _is_win = int(_tg[_i, 0]) == _s["tid"]
                _cand.append(
                    {"tid": int(_tg[_i, 0]),
                     "dx": round(float(_tg[_i, 1] - _cx), 1),
                     "dy": round(float(_tg[_i, 2] - _cy), 1),
                     "dist_px": round(float(_d[_i]), 1),
                     "chosen": _is_win})
                _ax.scatter([float(_tg[_i, 1])], [float(_tg[_i, 2])],
                            s=90, facecolors="none",
                            edgecolors="lime" if _is_win else "red", lw=2)
                _ax.text(float(_tg[_i, 1]) + 2, float(_tg[_i, 2]) - 2,
                         f"{int(_tg[_i, 0])}", fontsize=8,
                         color="lime" if _is_win else "red")
            if show_gate.value:
                _circ = plt.Circle((_cx, _cy), float(gate_px.value),
                                   fill=False, ls="--")
                _ax.add_patch(_circ)
            _ax.scatter([_cx], [_cy], s=60, marker="+", color="cyan")
        _ax.set_title(f"f={_s['frame']} row={_s['row']} tid={_s['tid']} "
                      f"next={_s['next']} cands={len(_cand)}")
        _fig.tight_layout()
        _figs.append(_fig)
        _df = pd.DataFrame(_cand).sort_values("dist_px") if _cand else pd.DataFrame(
            [{"note": "tracer not visible in this cam/frame (tid -1 or lost)"}])
        _tables.append((_s["frame"], _df))
    mo.vstack(
        [mo.hstack(_figs)] +
        [mo.hstack([mo.md(f"**f={_f}**"), mo.ui.table(_df, selection=None)])
         for _f, _df in _tables]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        **Reading a strip:** cyan `+` = tracer anchor (linkage position);
        dashed circle = pixel gate (approximation of the 3D gate);
        **green** = candidate the linkage kept, **red** = rejected neighbours.
        Green on the wrong tid, or an empty gate with the truth nearby in the
        full-frame view, tells you whether to suspect Phase 1 (gate/prediction)
        or Phase 2 (scoring/Hungarian) — then reproduce exactly in
        `notebooks/two_phase_link_lab.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
