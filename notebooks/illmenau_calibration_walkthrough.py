import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # Plate calibration, end to end

        Every step from folders of coded-plate images to the `.ori` / `.addpar`
        files, run live, with the reasoning beside the code. Companion to
        `docs/plate-calibration-howto.md` (the procedure) and
        `docs/illmenau-4cam-calibration.md` (what happened on this dataset).

        Switch between the two Illmenau camera groups with the selector below —
        they are **separate calibrations of separate worlds**, each anchored to
        its own reference frame's coded dot.

        ```bash
        uv run --with opencv-python-headless marimo edit             notebooks/illmenau_calibration_walkthrough.py
        ```

        OpenCV is not an openptv2 dependency, so it is pulled in at launch. The
        notebook still runs without it — everything except the live pose fit in
        §4 reads the `.ori` on disk.

        /// admonition | This notebook does not write anything
            type: warning

        It recomputes the calibration and shows you the `.ori` it arrives at, so
        you can compare against the files on disk. Writing is done by
        `scripts/illmenau/refit_plate_pinhole.py` and `bundle_plate_poses.py`,
        which is deliberate: a notebook cell that overwrites a verified
        calibration is too easy to run by accident.
        ///
        """
    )
    return (mo,)


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import numpy as np

    OPTV = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
    sys.path.insert(0, str(OPTV / "scripts" / "illmenau"))

    RAW = Path(os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau"))

    RIGS = {
        "cameras 1-4  (near wall, +Z, front face of the plate)": {
            "dir": RAW / "openptv_illmenau_4cam", "cams": [1, 2, 3, 4],
            "cc": 8.5858, "colour": "#C8951F",
        },
        "cameras 5-8  (far wall, -Z, back face of the plate)": {
            "dir": RAW / "openptv_illmenau_5678", "cams": [5, 6, 7, 8],
            "cc": 8.6313, "colour": "#B3323C",
        },
    }
    return OPTV, Path, RAW, RIGS, np, os


@app.cell(hide_code=True)
def _(RIGS, mo):
    rig_pick = mo.ui.dropdown(
        options=list(RIGS), value=list(RIGS)[0], label="**camera group**"
    )
    rig_pick
    return (rig_pick,)


@app.cell
def _(Path, RIGS, np, rig_pick):
    import yaml

    RIG = RIGS[rig_pick.value]
    DIR, CAMS, CC = Path(RIG["dir"]), RIG["cams"], RIG["cc"]
    PLATE = yaml.safe_load((DIR / "plate.yaml").read_text(encoding="utf-8"))["plate"]
    PITCH = float(PLATE["pitch_x"])
    NX, NY = int(PLATE["nx"]), int(PLATE["ny"])
    REF = str(PLATE.get("origin_frame", "00000000"))
    DATUM = (int(PLATE["datum"]["ix"]), int(PLATE["datum"]["iy"]))
    IMX, IMY, PIX = 2560, 2048, 0.005

    NPZ = np.load(DIR / "cal" / "labelled_all_frames.npz")
    VIEWS = {}
    for _k in NPZ.files:
        if _k.endswith("_ids"):
            _c, _f, _ = _k.split("_")
            VIEWS[(int(_c[1:]), _f)] = (NPZ[_k], NPZ[f"{_c}_{_f}_px"])
    FRAMES = sorted({f for _, f in VIEWS})
    return (
        CAMS, CC, DATUM, DIR, FRAMES, IMX, IMY, NX, NY, PITCH, PIX, PLATE, REF,
        VIEWS,
    )


@app.cell(hide_code=True)
def _(CAMS, CC, DATUM, DIR, FRAMES, NX, NY, PITCH, REF, VIEWS, mo):
    mo.md(
        f"""
        ## 0 · What this rig supplies

        | | |
        |---|---|
        | working folder | `{DIR.name}` |
        | cameras | {", ".join(f"cam{c}" for c in CAMS)} |
        | plate | {NX}×{NY} lattice, {PITCH:.0f} mm pitch |
        | datum — the coded L corner | grid `{DATUM}` → point id **{DATUM[1]*NX+DATUM[0]+1}** |
        | reference frame (defines the world) | `{REF}` |
        | frames captured | {len(FRAMES)} |
        | labelled views in the cache | {len(VIEWS)} of {len(CAMS)*len(FRAMES)} |
        | fitted focal length | **{CC} mm** |

        The nominal focal length is *not* an input — it is only a bracket for the
        sweep in §3. Everything here reads the detection cache written once by
        `detect_plate_frames.py`; **one labelling, used by every step.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · Detect and label

        `detect_plate.detect_plate_targets` finds the dots and flags the three
        **coded** ones (bright centre, dark ring) by comparing a 5×5 centre mean
        against an annulus mean. `plate_labeler.label_coded_6x7` then turns them
        into identities: of the three coded dots the **corner** is the one whose
        partners lie at ≈1·pitch and ≈2·pitch at a right angle — the 1·pitch
        partner defines `+Y`, the 2·pitch partner `+X`. Every other dot follows
        from `ix = round((p − corner)·e_x / pitch)`.

        Because the code gives every dot its identity, **no manual orientation
        clicking and no `sortgrid`** are needed.
        """
    )
    return


@app.cell(hide_code=True)
def _(FRAMES, mo):
    frame_pick = mo.ui.dropdown(options=FRAMES, value=FRAMES[0], label="frame")
    cam_pick = mo.ui.number(1, 4, 1, value=1, label="camera (index within the group)")
    mo.hstack([frame_pick, cam_pick], justify="start")
    return cam_pick, frame_pick


@app.cell
def _(CAMS, NX, VIEWS, cam_pick, frame_pick, mo, np):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ci = int(cam_pick.value) - 1
    _key = (_ci, frame_pick.value)
    if _key not in VIEWS:
        label_fig = mo.md(f"*cam{CAMS[_ci]} has no labelled dots on frame {frame_pick.value}*")
    else:
        _ids, _px = VIEWS[_key]
        _f, _ax = plt.subplots(figsize=(7.2, 5.8))
        _ax.scatter(_px[:, 0], _px[:, 1], s=26, facecolor="none", edgecolor="#C8951F")
        for _i, _p in zip(_ids, _px):
            _ax.annotate(str(int(_i)), _p, textcoords="offset points",
                         xytext=(4, 3), fontsize=7)
        # mark the datum dot: the world origin
        _d = np.where(_ids == (2 * 0 + 21))[0]
        _ax.set(xlim=(0, 2560), ylim=(2048, 0), aspect="equal",
                xlabel="image x [px]", ylabel="image y [px]",
                title=f"cam{CAMS[_ci]}  frame {frame_pick.value} — "
                      f"{len(_ids)} dots labelled  (id = iy·{NX} + ix + 1)")
        _ax.grid(alpha=.25)
        label_fig = _f
    label_fig
    return (plt,)


@app.cell(hide_code=True)
def _(DATUM, NX, mo):
    mo.md(
        f"""
        ## 2 · Pin the world to a physical dot

        **Not the plate centre** — the coded L-corner dot on the reference frame,
        a specific piece of plastic someone can point at. Its grid index
        `{DATUM}` is *verified from the data* by `find_datum.py`, never assumed;
        guessing it offsets the entire world by a multiple of the pitch.

        Shifting the lattice so that dot is the origin gives the object points,
        which are the same in every frame — nothing about where the plate is
        being held enters here:

        ```python
        ix, iy = (ids - 1) % nx, (ids - 1) // nx
        obj = [(ix - {DATUM[0]}) * pitch, (iy - {DATUM[1]}) * pitch, 0.0]
        ```

        Point ids run `id = iy·{NX} + ix + 1`, so the datum is
        **id {DATUM[1]*NX+DATUM[0]+1}** and `cal/calibration_block.txt`
        carries it as `{DATUM[1]*NX+DATUM[0]+1} 0.0 0.0 0.0`.
        """
    )
    return


@app.cell
def _(DATUM, NX, PITCH, np):
    def obj_of(ids):
        """Plate coordinates of point ids, datum dot at the origin, plate in z=0."""
        ids = np.asarray(ids)
        ix, iy = (ids - 1) % NX, (ids - 1) // NX
        return np.stack([(ix - DATUM[0]) * PITCH, (iy - DATUM[1]) * PITCH,
                         np.zeros(len(ids))], 1).astype(float)
    return (obj_of,)


@app.cell(hide_code=True)
def _(CC, mo):
    mo.md(
        f"""
        ## 3 · The one shared focal length

        `cc` is **exactly degenerate on a single plane**: re-fit the pose at a
        different focal length and the reconstruction is self-similar — identical
        residuals, identical recovered pitch. Any fit of `cc` from one plane is
        fitting noise.

        It *is* observable across many planes. Fit each camera's pose on the
        reference frame, then ask each camera **separately** where the plate is
        in some other frame: the right `cc` makes the answers coincide, the wrong
        one spreads them apart, worse the further from the reference plane.
        `fit_plate_cc.py` minimises that spread and lands on **{CC} mm** for this
        group.

        The sweep takes a couple of minutes, so it is behind a button.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    run_cc = mo.ui.run_button(label="run the cc sweep  (~2 min)")
    run_cc
    return (run_cc,)


@app.cell
def _(CC, IMX, IMY, PIX, REF, VIEWS, mo, np, obj_of, plt, run_cc):
    if not run_cc.value:
        cc_out = mo.md("*not run — press the button above*")
    else:
        import cv2

        def _pose(K, ids, px):
            o = obj_of(ids)
            if len(o) < 6:
                return None
            ok, rv, tv = cv2.solvePnP(o, px.astype(float), K, np.zeros(5))
            if not ok:
                return None
            rv, tv = cv2.solvePnPRefineLM(o, px.astype(float), K, np.zeros(5), rv, tv)
            rep, _ = cv2.projectPoints(o, rv, tv, K, np.zeros(5))
            return rv, tv, float(np.sqrt(np.mean(np.sum((rep.reshape(-1, 2) - px) ** 2, 1))))

        _frames = [f for f in sorted({f for _, f in VIEWS})
                   if all((c, f) in VIEWS and len(VIEWS[(c, f)][0]) >= 12 for c in range(4))]

        def _spread(cc):
            K = np.array([[cc / PIX, 0, IMX / 2], [0, cc / PIX, IMY / 2], [0, 0, 1.0]])
            ref = [_pose(K, *VIEWS[(c, REF)]) for c in range(4)]
            if any(p is None for p in ref):
                return np.inf
            out = []
            for f in _frames:
                if f == REF:
                    continue
                ts = []
                for c in range(4):
                    p = _pose(K, *VIEWS[(c, f)])
                    if p is None or p[2] > 1.5:
                        ts = None
                        break
                    R0, _ = cv2.Rodrigues(ref[c][0])
                    ts.append((R0.T @ (p[1] - ref[c][1])).ravel())
                if ts and len(ts) == 4:
                    ts = np.array(ts)
                    out.append(float(np.max(np.linalg.norm(ts - ts.mean(0), axis=1))))
            return float(np.median(out)) if out else np.inf

        _ccs = np.arange(7.8, 10.61, 0.2)
        _sp = [_spread(float(c)) for c in _ccs]
        _f, _ax = plt.subplots(figsize=(7.2, 4.2))
        _ax.plot(_ccs, _sp, "o-", color="#245D6C")
        _ax.axvline(CC, color="#B3323C", ls="--", label=f"fitted {CC} mm")
        _ax.set(xlabel="trial cc [mm]",
                ylabel="median cross-camera spread of the plate [mm]",
                title="a clean single minimum means the frames span enough depth")
        _ax.legend()
        _ax.grid(alpha=.3)
        cc_out = _f
    cc_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · Pose → `.ori`

        One `cv2.solvePnP` per camera on the reference frame, with **zero
        distortion** and the principal point at the sensor centre. Then a pure
        change of basis, in `calibration_import.calibration_from_opencv`:

        | OpenCV | openptv2 `.ori` |
        |---|---|
        | `R`, `t` | `dm = Rᵀ·S`, `S = diag(1, −1, −1)` applied **on the right** |
        | `R`, `t` | `C = −Rᵀ·t` — the projection centre |
        | `dm` | `ω, φ, κ` via `angles_from_dm` |
        | `fx` | `cc = fx · pix_x` |

        `S` is there because OpenCV's camera frame is x-right, y-**down**,
        z-**forward** while openptv2's is x-right, y-**up**, z-**backward** — the
        camera views along `−dm[:,2]`.

        Distortion stays zero: fitted from a single plane, `k1,k2,k3` trade
        against pose and produce a polynomial that fits those 42 points and
        diverges everywhere else. The signature is unmistakable — *the projection
        of a straight 3D ray doubles back inside the sensor*, which a pinhole
        cannot do.
        """
    )
    return


@app.cell
def _(CAMS, CC, DIR, IMX, IMY, PIX, REF, VIEWS, np, obj_of):
    from openptv2.algorithms.calibration import Calibration
    from openptv2.calibration_import import calibration_from_opencv

    try:
        import cv2 as _cv2
        HAVE_CV2 = True
    except ModuleNotFoundError:
        HAVE_CV2 = False

    _K = np.array([[CC / PIX, 0, IMX / 2], [0, CC / PIX, IMY / 2], [0, 0, 1.0]])
    fitted, on_disk = {}, {}
    for _i, _cam in enumerate(CAMS):
        if HAVE_CV2:
            _ids, _px = VIEWS[(_i, REF)]
            _obj = obj_of(_ids)
            _ok, _rv, _tv = _cv2.solvePnP(_obj, _px.astype(float), _K, np.zeros(5))
            _rv, _tv = _cv2.solvePnPRefineLM(_obj, _px.astype(float), _K,
                                             np.zeros(5), _rv, _tv)
            _cal, _ = calibration_from_opencv(_K, np.zeros(5), _rv, _tv, imx=IMX,
                                              imy=IMY, pix_x=PIX, pixel_origin="corner")
            _rep, _ = _cv2.projectPoints(_obj, _rv, _tv, _K, np.zeros(5))
            fitted[_cam] = (_cal, float(np.sqrt(np.mean(np.sum(
                (_rep.reshape(-1, 2) - _px) ** 2, 1)))), len(_obj))
        _d = Calibration()
        _d.from_file(str(DIR / f"cal/cam{_cam}.tif.ori"),
                     str(DIR / f"cal/cam{_cam}.tif.addpar"))
        on_disk[_cam] = _d
    return Calibration, HAVE_CV2, fitted, on_disk


@app.cell(hide_code=True)
def _(CAMS, HAVE_CV2, fitted, mo, np, on_disk):
    _rows = []
    for _cam in (CAMS if HAVE_CV2 else []):
        _c, _rms, _n = fitted[_cam]
        _e, _o = _c.ext_par, on_disk[_cam].ext_par
        _moved = np.linalg.norm(np.array([_e.x0, _e.y0, _e.z0])
                                - np.array([_o.x0, _o.y0, _o.z0]))
        _rows.append(
            f"| cam{_cam} | {_n} | ({_e.x0:.0f}, {_e.y0:.0f}, {_e.z0:.0f}) | "
            f"{_rms:.3f} px | ({_o.x0:.0f}, {_o.y0:.0f}, {_o.z0:.0f}) | {_moved:.0f} mm |")
    mo.md(
        "### What this reproduces, against the files on disk\n\n"
        "| camera | dots | pose fitted here | reproj RMS | `.ori` on disk | differ by |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(_rows) + "\n\n"
        "The two agree when the `.ori` on disk is the single-plane fit. They differ "
        "by a few tens of mm once `bundle_plate_poses.py` has run, because the "
        "bundle re-solves every pose jointly — see §6."
    )
    return


@app.cell(hide_code=True)
def _(CAMS, HAVE_CV2, mo):
    ori_pick = mo.ui.dropdown(options=[str(c) for c in CAMS],
                              value=str(CAMS[0]), label="show the .ori text for cam")
    ori_pick if HAVE_CV2 else mo.md("")
    return (ori_pick,)


@app.cell
def _(HAVE_CV2, fitted, mo, ori_pick):
    import tempfile
    from pathlib import Path as _P

    mo.stop(not HAVE_CV2, mo.md(""))
    _cal = fitted[int(ori_pick.value)][0]
    with tempfile.TemporaryDirectory() as _td:
        _o, _a = _P(_td) / "c.ori", _P(_td) / "c.addpar"
        _cal.to_file(str(_o), str(_a))
        _ori_txt, _add_txt = _o.read_text(), _a.read_text()
    mo.md(
        f"""
        ```text
        # cam{ori_pick.value}.tif.ori — x0 y0 z0 / omega phi kappa / dm (3x3) / xh yh / cc
        {_ori_txt.strip()}
        ```
        ```text
        # cam{ori_pick.value}.tif.addpar — k1 k2 k3 p1 p2 scx she, all zero by design
        {_add_txt.strip()}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5 · Gate the result — never on reprojection RMS

        Per-camera reprojection RMS is **blind** here. It sat at 0.5 px through
        unphysical distortion, a focal length wrong by 10 %, poses anchored to one
        plane, and half the frames mislabelled — and it is *anti-correlated* with
        ray convergence.

        Two numbers that do work, computed below from the `.ori` on disk:

        * **absolute positions** of the triangulated plate against the known block
          coordinates, with **no alignment applied**;
        * **ray-convergence miss (RCM)** — the closest approach of the sight lines
          of one dot. It uses *no plate model at all*, so the grid cannot fool it:
          a bad RCM with a good grid is a calibration error, not a labelling one.
        """
    )
    return


@app.cell
def _(DIR, REF, VIEWS, np, obj_of, on_disk):
    from openptv2.algorithms.orientation import COORD_UNUSED
    from openptv2.algorithms.parameters import ControlPar, MmNp
    from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
    from openptv2.orientation import multi_cam_point_positions

    cpar = ControlPar(num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                      mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0,
                      tiff_flag=1, hp_flag=1, allCam_flag=0,
                      img_base_name=[""] * 4, cal_img_base_name=[""] * 4)
    cals = list(on_disk.values())

    def triangulate(frame):
        per = {c: dict(zip(VIEWS[(c, frame)][0].tolist(), VIEWS[(c, frame)][1].tolist()))
               for c in range(4) if (c, frame) in VIEWS}
        ids = [i for i in sorted({i for m in per.values() for i in m})
               if sum(i in m for m in per.values()) >= 2]
        if len(ids) < 6:
            return None
        t = np.full((len(ids), 4, 2), COORD_UNUSED)
        for k, pid in enumerate(ids):
            for c, m in per.items():
                if pid in m:
                    mx, my = pixel_to_metric(m[pid][0], m[pid][1], cpar)
                    a = cals[c].added_par
                    t[k, c] = dist_to_flat(mx, my, cals[c].int_par.xh, cals[c].int_par.yh,
                                           a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she)
        pos, rcm = multi_cam_point_positions(t, cpar, cals)
        rcm = np.asarray(rcm, float)
        ok = np.isfinite(pos).all(1) & (np.abs(pos) < 1e5).all(1) & np.isfinite(rcm)
        return pos[ok], np.array(ids)[ok], rcm[ok]

    _p, _i, _r = triangulate(REF)
    _nom = obj_of(_i)
    _ctr = _p.mean(0)
    _nrm = np.linalg.svd(_p - _ctr)[2][2]
    _err = np.linalg.norm(_p - _nom, axis=1)
    gate_a = dict(
        n=len(_i), normal=_nrm,
        planarity=float(np.sqrt(np.mean(((_p - _ctr) @ _nrm) ** 2))),
        abs_med=float(np.median(_err)), abs_max=float(_err.max()),
        rcm=float(np.median(_r)),
    )
    return cals, cpar, gate_a, triangulate


@app.cell(hide_code=True)
def _(DIR, REF, gate_a, mo):
    _g = gate_a
    mo.md(
        f"""
        ### Gate A — reference frame `{REF}`, from `{DIR.name}/cal/*.ori`

        | | |
        |---|---|
        | dots triangulated | {_g['n']} |
        | plane normal | ({_g['normal'][0]:+.4f}, {_g['normal'][1]:+.4f}, {_g['normal'][2]:+.4f}) |
        | planarity residual RMS | **{_g['planarity']:.3f} mm** |
        | absolute error vs the block, no alignment | **{_g['abs_med']:.2f} mm** median, {_g['abs_max']:.2f} max |
        | ray-convergence miss | **{_g['rcm']:.2f} mm** median |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Accuracy across the whole volume

        The reference plane is exact by construction, so the interesting question
        is what happens away from it. Anchoring every pose to one plane costs
        **~0.58 % of the distance** from it; the joint bundle in
        `bundle_plate_poses.py` — every camera pose plus one plate pose per frame,
        with `cc` fixed and the reference plate pose held as the gauge — brings
        that to **~0.13 %**.
        """
    )
    return


@app.cell
def _(FRAMES, np, obj_of, plt, triangulate):
    _rows = []
    for _f in FRAMES:
        _t = triangulate(_f)
        if _t is None:
            continue
        _p, _i, _r = _t
        _c = _p.mean(0)
        _n = np.linalg.svd(_p - _c)[2][2]
        # rigid ideal grid fitted onto the dots -> is the pattern still the pattern?
        _G = obj_of(_i)
        _ca, _cb = _G.mean(0), _p.mean(0)
        _U, _, _Vt = np.linalg.svd((_G - _ca).T @ (_p - _cb))
        _R = _U @ np.diag([1, 1, np.sign(np.linalg.det(_U @ _Vt))]) @ _Vt
        _dev = np.linalg.norm((_G - _ca) @ _R + _cb - _p, axis=1)
        _rows.append((np.linalg.norm(_c), float(np.sqrt(np.mean(((_p - _c) @ _n) ** 2))),
                      float(np.median(_r)), float(_dev.max())))
    _a = np.array(_rows)
    _good = _a[:, 3] < 30.0

    _f2, _ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    _ax[0].semilogy(_a[_good, 0], _a[_good, 2], "o", color="#245D6C")
    _ax[0].set(xlabel="plate distance from the world origin [mm]",
               ylabel="ray-convergence miss [mm]", title="RCM vs depth")
    _ax[1].semilogy(_a[_good, 0], _a[_good, 1], "o", color="#C8951F")
    _ax[1].set(xlabel="plate distance from the world origin [mm]",
               ylabel="planarity RMS [mm]", title="planarity vs depth")
    for _x in _ax:
        _x.grid(alpha=.3, which="both")
    _f2.suptitle(f"{int(_good.sum())} of {len(_a)} frames pass the grid-deviation gate "
                 "(> 30 mm means the labeller, not the calibration)")
    _f2.tight_layout()
    _f2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6 · The rig, as calibrated

        Fitted camera positions against the test section. Nothing constrained
        these to land on the wall — comparing them with the physical mounts, and
        with pairwise camera distances (which are frame-invariant), is a real
        check on the whole chain.
        """
    )
    return


@app.cell
def _(CAMS, PLATE, REF, np, on_disk, triangulate):
    import plotly.graph_objects as go

    _R = float(PLATE["test_section"]["radius"])
    _H = float(PLATE["test_section"]["height"])
    _Y0 = _H / 2 + float(PLATE["datum"]["barrel_frame"][1])
    _th = np.linspace(0, 2 * np.pi, 160)

    rig_fig = go.Figure()
    for _y in (-_Y0, _H - _Y0):
        rig_fig.add_trace(go.Scatter3d(
            x=_R * np.cos(_th), y=np.full_like(_th, _y), z=_R * np.sin(_th),
            mode="lines", line=dict(color="#8894A2", width=2),
            name="test section", showlegend=(_y < 0), hoverinfo="skip"))

    _p, _i, _ = triangulate(REF)
    rig_fig.add_trace(go.Scatter3d(
        x=_p[:, 0], y=_p[:, 1], z=_p[:, 2], mode="markers",
        marker=dict(size=3, color="#2F6B45"),
        name=f"plate, frame {REF}",
        hovertext=[f"id {int(v)}" for v in _i], hoverinfo="text"))

    for _cam in CAMS:
        _e = on_disk[_cam].ext_par
        _C = np.array([_e.x0, _e.y0, _e.z0])
        _look = -np.asarray(_e.dm)[:, 2] * 900.0
        rig_fig.add_trace(go.Scatter3d(
            x=[_C[0]], y=[_C[1]], z=[_C[2]], mode="markers+text",
            marker=dict(size=6, color="#B3323C", symbol="diamond"),
            text=[f"cam{_cam}"], textposition="top center",
            name=f"cam{_cam}", showlegend=False,
            hovertext=[f"cam{_cam}<br>({_C[0]:.0f}, {_C[1]:.0f}, {_C[2]:.0f}) mm"],
            hoverinfo="text"))
        rig_fig.add_trace(go.Scatter3d(
            x=[_C[0], _C[0] + _look[0]], y=[_C[1], _C[1] + _look[1]],
            z=[_C[2], _C[2] + _look[2]], mode="lines",
            line=dict(color="#B3323C", width=3), opacity=.55,
            showlegend=False, hoverinfo="skip"))

    rig_fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text",
        marker=dict(size=6, color="#C8951F", symbol="x"),
        text=["datum (0,0,0)"], textposition="bottom center",
        name="world origin", hoverinfo="skip"))
    for _v, _c, _l in ((np.eye(3)[0], "#B3323C", "X"), (np.eye(3)[1], "#2F6B45", "Y"),
                       (np.eye(3)[2], "#245D6C", "Z")):
        _q = _v * 800.0
        rig_fig.add_trace(go.Scatter3d(
            x=[0, _q[0]], y=[0, _q[1]], z=[0, _q[2]], mode="lines+text",
            text=["", f"+{_l}"], line=dict(color=_c, width=5),
            showlegend=False, hoverinfo="skip"))

    rig_fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=40, b=0),
        title="fitted cameras, the plate at the reference frame, and the test section",
        scene=dict(aspectmode="data", xaxis_title="X [mm]",
                   yaxis_title="Y [mm]  (up)", zaxis_title="Z [mm]  (object→camera)"),
        legend=dict(orientation="h", yanchor="bottom", y=0))
    rig_fig
    return


@app.cell(hide_code=True)
def _(CAMS, mo, np, on_disk):
    _rows = []
    for _a in range(len(CAMS)):
        for _b in range(_a + 1, len(CAMS)):
            _ea, _eb = on_disk[CAMS[_a]].ext_par, on_disk[CAMS[_b]].ext_par
            _d = np.linalg.norm(np.array([_ea.x0, _ea.y0, _ea.z0])
                                - np.array([_eb.x0, _eb.y0, _eb.z0]))
            _rows.append(f"| cam{CAMS[_a]} – cam{CAMS[_b]} | {_d:.0f} mm |")
    mo.md("### Pairwise camera distances\n\nFrame-invariant, so they can be checked "
          "against the physical mounts directly.\n\n| pair | distance |\n|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What this notebook leaves to the scripts

        Writing files, and the two long-running steps:

        ```bash
        export ILLMENAU_DIR=.../openptv_illmenau_5678
        export ILLMENAU_CAMS=5,6,7,8

        python scripts/illmenau/detect_plate_frames.py --cams 5,6,7,8   # ~10 min
        python scripts/illmenau/fit_plate_cc.py                          # ~2 min
        python scripts/illmenau/refit_plate_pinhole.py 8.6313            # writes .ori
        python scripts/illmenau/bundle_plate_poses.py 8.6313 --write     # ~8 min
        python scripts/illmenau/check_epipolar.py                        # gate B
        ```

        `_config.py` cross-checks `ILLMENAU_CAMS` against the camera names in the
        folder's `parameters_Run1.yaml` and refuses to run on a mismatch — the two
        groups are separate worlds, and writing one into the other's folder
        destroys it.

        Full reasoning: `docs/plate-calibration-howto.md`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
