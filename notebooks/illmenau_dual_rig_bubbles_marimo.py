"""Illmenau dual-rig bubbles: 5 consecutive frames, quadruplets, tracker chains.

Data flow (all files already generated, nothing is recomputed on load):
  * dots: /Users/alex/Downloads/Ilmenau/dual_rig_bubbles_fulldiam.npz
    (fresh re-triangulation with self-cal .ori + full-diameter criteria boxes;
    falls back to res/run.zarr correspondences when absent)
  * lines: /Users/alex/Downloads/Ilmenau/eye_test_chains.pkl
    (re-tracked fresh quadruplets: v_max=40 mm/frame, no confirm gate,
    bidirectional; backup of the old transferred links: eye_test_chains.ptvis.pkl)

Open with:  uv run marimo edit notebooks/illmenau_dual_rig_bubbles_marimo.py
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
    from scipy.spatial import cKDTree

    RAW = Path(os.environ.get("ILLMENAU_RAW", "/Users/alex/Downloads/Ilmenau"))
    RIG1 = RAW / "openptv_illmenau_4cam"
    RIG2 = RAW / "openptv_illmenau_5678"
    Y_SHIFT = -1175.0  # barrel_from_plate: P + [0,-1175,0], plate.yaml datum
    # 5 CONSECUTIVE mid-run frames for the eye-vs-tracker test.
    FRAMES = [10200, 10201, 10202, 10203, 10204]
    R_WALL, H_BARREL = 3575.0, 3580.0

    return FRAMES, H_BARREL, RIG1, RIG2, R_WALL, RAW, Y_SHIFT, cKDTree, go, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Illmenau dual-rig bubbles: 5 consecutive frames, quadruplets + tracks

    Triangulated bubbles from **both rigs** in one barrel frame
    (`barrel = rig + [0,−1175,0]`, origin on the axis at mid-height, `+Y` up).

    * **Orange = rig 1-4** (cams 1-4 at `+Z`), **blue = rig 5-8** (cams 5-8 at `−Z`).
    * Dots are **quadruplets inside the barrel walls only** (`r ≤ 3575`, `|Y| ≤ 1790`).
    * Labels read `rig:offset:id3d`, e.g. `1-4:202:1843` (offset = frame − 10000).
      Hover adds the track id (`track T123`) or `(untracked)`.
    * Lines are the tracker's answer (re-tracked fresh quadruplets, `v_max=40`,
      no confirm gate). Eye-test: follow a dot across frames by eye, then check
      the line — and the tracks table below — did the same.
    """)
    return


@app.cell
def _(FRAMES, RIG1, RIG2, Y_SHIFT, mo, np):
    from openptv2.storage import RunStore as RigStore

    fresh_path = RIG1.parent / "dual_rig_bubbles_fulldiam.npz"
    fresh = None
    if fresh_path.exists():
        with np.load(str(fresh_path), allow_pickle=True) as z:
            fresh = {k: z[k] for k in z.files}

    def load_rig(rig_path):
        rig_key = rig_path.name  # openptv_illmenau_4cam | openptv_illmenau_5678
        store = None
        out = {}
        for f in FRAMES:
            if fresh is not None and f"{rig_key}/{f}/pos" in fresh:
                pos = np.asarray(fresh[f"{rig_key}/{f}/pos"], float)
                ncam = np.asarray(fresh[f"{rig_key}/{f}/ncam"], int)
                rest = None
            else:
                if store is None:
                    store = RigStore(str(rig_path / "res" / "run.zarr"), mode="r")
                try:
                    pos, rest = store.read_correspondences(int(f))
                except Exception as exc:
                    print(f"{rig_key} frame {f}: {exc}")
                    continue
                pos = np.asarray(pos, float)
                rest = np.asarray(rest, np.int32)
                ncam = ((rest >= 0).sum(axis=1) if rest.size
                        else np.zeros(len(pos), int))
            b = pos.copy()
            b[:, 1] += Y_SHIFT
            out[str(f)] = dict(pos=b, ncam=ncam, rest=rest)
        return out

    rigA = load_rig(RIG1)
    rigB = load_rig(RIG2)

    def cam_xyz(rig_path, names):
        xyz = []
        for n in names:
            line = (rig_path / "cal" / f"cam{n}.tif.ori").read_text().splitlines()[0]
            xyz.append([float(v) for v in line.split()])
        xyz = np.array(xyz)
        xyz[:, 1] += Y_SHIFT
        return xyz

    camA = cam_xyz(RIG1, [1, 2, 3, 4])
    camB = cam_xyz(RIG2, [5, 6, 7, 8])
    src = ("fresh re-triangulation (self-cal `.ori` + full-diameter boxes)"
           if fresh is not None else "stored `res/run.zarr`")
    mo.md(f"**Source:** {src}. " + ", ".join(
        f"frame {f}: 1-4 N={len(rigA[str(f)]['pos'])} / "
        f"5-8 N={len(rigB[str(f)]['pos'])}" for f in FRAMES))
    return camA, camB, rigA, rigB


@app.cell
def _(FRAMES, mo):
    frame_sel = mo.ui.multiselect(
        options=[str(f) for f in FRAMES],
        value=[str(f) for f in FRAMES],
        label="frames (offsets: "
              + ", ".join(f"{f}→{f - 10000}" for f in FRAMES) + ")",
    )
    show_ids = mo.ui.checkbox(
        True, label="id labels rig:offset:id3d")
    id_stride = mo.ui.slider(
        1, 20, value=4, step=1, label="label every Nth id")
    show_tracks = mo.ui.checkbox(
        True, label="tracker chains (lines + T-ids)")
    overlap_only = mo.ui.checkbox(
        False, label="centre slab |Z|<150mm only")
    mask_barrel = mo.ui.checkbox(
        True, label="hide points outside barrel walls")
    psize = mo.ui.slider(1, 6, value=2, step=0.5, label="dot size")
    show_cams = mo.ui.checkbox(True, label="cameras")
    show_cyl = mo.ui.checkbox(True, label="barrel wireframe")
    mo.vstack([
        mo.hstack([frame_sel, psize], justify="start", gap=2),
        mo.hstack([show_ids, id_stride, show_tracks, overlap_only,
                   mask_barrel, show_cams, show_cyl],
                  justify="start", gap=2),
    ])
    return (frame_sel, id_stride, mask_barrel, overlap_only, psize,
            show_cams, show_cyl, show_ids, show_tracks)


@app.cell
def _(RIG1, mo, np):
    import pickle

    chain_path = RIG1.parent / "eye_test_chains.pkl"
    chains_eye = (pickle.load(open(str(chain_path), "rb"))
                  if chain_path.exists() else {})
    dot2chain = {}
    for _rk, _chs in chains_eye.items():
        for _ci, _ch in enumerate(_chs):
            for _fr, _rw in _ch:
                dot2chain[(str(_rk), str(_fr), int(_rw))] = int(_ci)
    counts = {r: len(c) for r, c in chains_eye.items()}
    mo.md(f"**Tracker chains** (re-tracked fresh quadruplets, `v_max=40`, "
          f"no confirm gate, bidirectional): {counts}. "
          f"Lines pass exactly through the dots they link.")
    return chains_eye, dot2chain


@app.cell
def _(
    H_BARREL,
    R_WALL,
    camA,
    camB,
    chains_eye,
    dot2chain,
    frame_sel,
    go,
    id_stride,
    mask_barrel,
    mo,
    np,
    overlap_only,
    psize,
    rigA,
    rigB,
    show_cams,
    show_cyl,
    show_ids,
    show_tracks,
):
    RIGKEY = {"1-4": "openptv_illmenau_4cam", "5-8": "openptv_illmenau_5678"}

    def filt(d):
        m = d["ncam"] == 4  # quadruplets only
        p, n = d["pos"][m], d["ncam"][m]
        ids = np.where(m)[0]
        if mask_barrel.value:  # pos already barrel frame
            inside = ((np.hypot(p[:, 0], p[:, 2]) <= R_WALL)
                      & (np.abs(p[:, 1]) <= H_BARREL / 2))
            p, n, ids = p[inside], n[inside], ids[inside]
        if overlap_only.value:
            z = np.abs(p[:, 2]) < 150.0
            p, n, ids = p[z], n[z], ids[z]
        return p, n, ids

    fig = go.Figure()
    if show_cyl.value:
        th = np.linspace(0, 2 * np.pi, 200)
        for y in (-H_BARREL / 2, H_BARREL / 2):
            fig.add_trace(go.Scatter3d(
                x=R_WALL * np.cos(th), y=R_WALL * np.sin(th),
                z=np.full_like(th, y), mode="lines",
                line=dict(color="black", width=3), hoverinfo="skip",
                showlegend=False))
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            fig.add_trace(go.Scatter3d(
                x=[R_WALL * np.cos(a)] * 2, y=[R_WALL * np.sin(a)] * 2,
                z=[-H_BARREL / 2, H_BARREL / 2], mode="lines",
                line=dict(color="gray", width=1), opacity=0.3,
                hoverinfo="skip", showlegend=False))

    for tag, rig, col in (("1-4", rigA, "#ff7f0e"),
                          ("5-8", rigB, "#1f77b4")):
        Pall, Nall, Iall, Fall, Call = [], [], [], [], []
        for fs in frame_sel.value:
            if fs not in rig:
                continue
            p, n, ids = filt(rig[fs])
            if len(p) == 0:
                continue
            Pall.append(p)
            Nall.append(n)
            Iall.append(ids)
            Fall.append(np.full(len(p), fs))
            Call.append(np.array([
                dot2chain.get((RIGKEY[tag], str(fs), int(i)), -1)
                for i in ids]))
        if not Pall:
            continue
        P = np.vstack(Pall)
        I = np.concatenate(Iall)
        F = np.concatenate(Fall)
        TT = np.full(len(P), tag)
        CC = np.concatenate(Call)
        hover = [f"rig {t} frame {f} (-10000: {int(f) - 10000}) "
                 f"id3d {i}"
                 + (f" track T{c}" if c >= 0 else " (untracked)") + "<br>"
                 f"X {p[0]:.1f} Y {p[1]:.1f} Z {p[2]:.1f} mm"
                 for t, f, i, c, p in zip(TT, F, I, CC, P)]
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 2], z=P[:, 1], mode="markers",
            marker=dict(size=psize.value, color=col, opacity=0.6),
            text=hover, hoverinfo="text", name=f"bubbles rig {tag}"))

    if show_ids.value:
        st = max(1, int(id_stride.value))
        for tag, rig, col in (("1-4", rigA, "crimson"),
                              ("5-8", rigB, "darkblue")):
            for fs in frame_sel.value:
                if fs not in rig:
                    continue
                p, _n, ids = filt(rig[fs])
                p, ids = p[::st], ids[::st]
                if len(p) == 0 or len(p) > 800:
                    continue
                fig.add_trace(go.Scatter3d(
                    x=p[:, 0], y=p[:, 2], z=p[:, 1], mode="markers+text",
                    marker=dict(size=5, color=col),
                    text=[f"{tag}:{int(fs) - 10000}:{i}" for i in ids],
                    textposition="top center",
                    name=f"ids {tag} {fs}"))

    if show_tracks.value:
        for _tag, _key, lcol in (
                ("1-4", "openptv_illmenau_4cam", "black"),
                ("5-8", "openptv_illmenau_5678", "navy")):
            rig = rigA if _tag == "1-4" else rigB
            sel = set(frame_sel.value)
            xs, ys, zs, tx, ty, tz, tt = [], [], [], [], [], [], []
            for _ci, _ch in enumerate(chains_eye.get(_key, [])):
                pts = [(fr, rw) for fr, rw in _ch if str(fr) in sel]
                if len(pts) < 2:
                    continue
                Pch = np.array([rig[str(fr)]["pos"][rw] for fr, rw in pts])
                xs += Pch[:, 0].tolist() + [None]
                ys += Pch[:, 2].tolist() + [None]
                zs += Pch[:, 1].tolist() + [None]
                if len(pts) >= 3:  # label sustained tracks
                    mid = Pch[len(Pch) // 2]
                    tx.append(mid[0])
                    ty.append(mid[2])
                    tz.append(mid[1])
                    tt.append(f"T{_ci}")
            if xs:
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color=lcol, width=3), opacity=0.85,
                    hoverinfo="skip", name=f"tracks rig {_tag}"))
                if tt:
                    fig.add_trace(go.Scatter3d(
                        x=tx, y=ty, z=tz, mode="text",
                        text=tt, textfont=dict(size=9),
                        hoverinfo="skip", showlegend=False))

    if show_cams.value:
        for C, names, col in ((camA, [1, 2, 3, 4], "#ff7f0e"),
                              (camB, [5, 6, 7, 8], "#1f77b4")):
            fig.add_trace(go.Scatter3d(
                x=C[:, 0], y=C[:, 2], z=C[:, 1], mode="markers+text",
                marker=dict(size=7, color=col, symbol="diamond",
                            line=dict(color="black", width=1)),
                text=[f"c{n}" for n in names], textposition="top center",
                name=f"cameras {names[0]}-{names[-1]}"))

    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(title="X [mm]", range=[-4000, 4000]),
            yaxis=dict(title="Z [mm]", range=[-4000, 4000]),
            zaxis=dict(title="Y up [mm]", range=[-H_BARREL / 2, H_BARREL / 2]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=H_BARREL / 7000),
            camera=dict(eye=dict(x=1.1, y=-1.5, z=0.8))),
        legend=dict(x=0.01, y=0.99))
    fig
    return


@app.cell
def _(chains_eye, mo):
    trows = []
    for _rk2 in ("openptv_illmenau_4cam", "openptv_illmenau_5678"):
        _tg = "1-4" if _rk2.endswith("4cam") else "5-8"
        for _ci2, _ch2 in enumerate(chains_eye.get(_rk2, [])):
            trows.append({
                "track": f"T{_ci2}", "rig": _tg, "n_frames": len(_ch2),
                "links": " → ".join(
                    f"{int(fr) - 10000}:{rw}" for fr, rw in _ch2),
            })
    mo.vstack([
        mo.md(f"## Tracks linking quadruplets ({len(trows)} chains, "
              f"`offset:id3d` per frame)"),
        mo.ui.table(trows, page_size=15),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Overlap check: same physical bubble seen from both sides?

    Nearest neighbour in the other rig (barrel frame, quadruplets inside
    walls only). Pairs under the threshold are hand-check candidates.
    """)
    return


@app.cell
def _(mo):
    ov_frame = mo.ui.dropdown(
        options=["10200", "10201", "10202", "10203", "10204"],
        value="10202",
        label="frame for overlap search (offset = frame − 10000)",
    )
    ov_thr = mo.ui.slider(5, 100, value=25, step=5,
                           label="pair threshold [mm]", show_value=True)
    ov_max = mo.ui.slider(10, 500, value=200, step=10,
                           label="max pairs listed", show_value=True)
    mo.hstack([ov_frame, ov_thr, ov_max], justify="start", gap=2)
    return ov_frame, ov_max, ov_thr


@app.cell
def _(H_BARREL, R_WALL, cKDTree, mo, np, ov_frame, ov_max, ov_thr, rigA, rigB):
    of = ov_frame.value

    def quad_inside(d):
        m = ((d["ncam"] == 4)
             & (np.hypot(d["pos"][:, 0], d["pos"][:, 2]) <= R_WALL)
             & (np.abs(d["pos"][:, 1]) <= H_BARREL / 2))
        return d["pos"][m], np.where(m)[0]

    Aq, ix = quad_inside(rigA[of])
    Bq, jx = quad_inside(rigB[of])
    tree = cKDTree(Bq)
    dist, j = tree.query(Aq, k=1)
    order = np.argsort(dist)
    prows, pair_ids = [], []
    for kk in order:
        if dist[kk] > ov_thr.value:
            break
        if len(prows) >= ov_max.value:
            break
        i2 = int(j[kk])
        prows.append({
            "id_1-4": int(ix[kk]), "id_5-8": int(jx[i2]),
            "dist_mm": round(float(dist[kk]), 2),
            "X_1-4": round(float(Aq[kk, 0]), 1),
            "Y_1-4": round(float(Aq[kk, 1]), 1),
            "Z_1-4": round(float(Aq[kk, 2]), 1),
            "X_5-8": round(float(Bq[i2, 0]), 1),
            "Y_5-8": round(float(Bq[i2, 1]), 1),
            "Z_5-8": round(float(Bq[i2, 2]), 1),
        })
        pair_ids.append((int(ix[kk]), int(jx[i2]), float(dist[kk])))
    mo.vstack([
        mo.md(f"**Frame {of} (-10000: {int(of) - 10000}), quadruplets inside "
              f"walls:** {len(prows)} pairs < {ov_thr.value} mm "
              f"(nearest {dist[order[0]]:.1f} mm, "
              f"median {np.median(dist):.1f} mm)."),
        mo.ui.table(prows, page_size=15),
    ])
    return (pair_ids,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Self-calibration export (on demand)

    Collects every `ncam==4` bubble of the 5 frames with its 2D pixel
    observations for tracer self-calibration / shaking. Press the button —
    it reads the target stores, so it takes ~1 minute.
    """)
    return


@app.cell
def _(FRAMES, RIG1, RIG2, mo, np):
    from openptv2.storage import RunStore as ExportStore

    export_btn = mo.ui.button(label="Export ncam==4 + pixel obs → "
                                    "dual_rig_selfcal_candidates.npz")
    export_btn
    return ExportStore, export_btn


@app.cell
def _(ExportStore, FRAMES, RIG1, RIG2, export_btn, mo, np):
    msg = mo.md("Press the button above to build the export file.")
    if (export_btn.value or 0) > 0:
        def export_rig(rig_path):
            s = ExportStore(str(rig_path / "res" / "run.zarr"), mode="r")
            fr = RIG1.parent / "dual_rig_bubbles_fulldiam.npz"
            fz = (dict(np.load(str(fr), allow_pickle=True))
                  if fr.exists() else None)
            pts, obs, meta = [], [], []
            for ff in FRAMES:
                if fz is not None and f"{rig_path.name}/{ff}/pos" in fz:
                    pos = np.asarray(fz[f"{rig_path.name}/{ff}/pos"], float)
                    rest = np.asarray(fz[f"{rig_path.name}/{ff}/ids"],
                                      np.int32)
                else:
                    pos, rest = s.read_correspondences(int(ff))
                    pos, rest = np.asarray(pos, float), np.asarray(rest,
                                                                   np.int32)
                good = np.where((rest >= 0).sum(axis=1) == 4)[0]
                tg = {}
                for cam in range(4):
                    try:
                        t = np.asarray(
                            s.root["targets"][f"cam_{cam}"]
                            [f"frame_{int(ff):06d}"])
                        tg[cam] = dict(zip(t[:, 0].astype(int),
                                           t[:, 1:3].tolist()))
                    except KeyError:
                        tg[cam] = None
                for i in good:
                    px, ok = [], True
                    for cam in range(4):
                        t = tg[cam]
                        if t is None or int(rest[i, cam]) not in t:
                            ok = False
                            break
                        px.append(t[int(rest[i, cam])])
                    if not ok:
                        continue
                    pts.append(pos[i].tolist())
                    obs.append(px)
                    meta.append([str(ff), int(i)])
            return np.array(pts), np.array(obs), meta

        pA, oA, mA = export_rig(RIG1)
        pB, oB, mB = export_rig(RIG2)
        out = RIG1.parent / "dual_rig_selfcal_candidates.npz"
        np.savez_compressed(
            out, pts_14=pA, obs_14=oA, meta_14=np.array(mA, dtype=str),
            pts_58=pB, obs_58=oB, meta_58=np.array(mB, dtype=str),
            frames=np.array(FRAMES, dtype=str))
        msg = mo.md(f"**Wrote `{out.name}`**: 1-4: `{len(pA)}`, "
                    f"5-8: `{len(pB)}` ncam==4 points with pixel observations.")
    msg
    return


if __name__ == "__main__":
    app.run()
