import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import yaml
    from pathlib import Path
    import matplotlib.pyplot as plt
    from PIL import Image

    return Image, Path, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Illmenau 4-Cam Calibration — `Kalibrierung_1..4` → OpenPTV2

    **Rig:** `1,2` SE bottom/top right, `3,4` SW bottom/top left — `45°` around `Y`, `Ø7150` (`r=3575`), `Z≈+2528` wall, `Y=700/2900` (2200 sep), looking at `0,615,0` (datum: 4th row from below, 3rd col from left at `615 mm` above heating plate `Y=0`). Opposite rig `5-8` is `180°` rotated (`X,Z → -X,-Z`) when added.

    **Plate:** `6×7`, `pitch 120`, dot `60`, thickness `6 mm` (front/back planes `6 mm` apart). `000000_*.tiff` defines origin; frame number is **pre-underscore** and synchronised across `cams 1-4`.

    This notebook runs the hand-held `6×7` pipeline for **cams 1-4 together**: per-frame grouped detection → labeling (`pitch 120`) → `Z` grouping → `DLT` + `presorted` refine → `.ori` + `parameters_Run1.yaml` ready for GUI/batch.
    """)
    return


@app.cell
def _(Path, mo):
    base = Path(r"C:\Users\alex\Downloads\Illmenau")
    out = Path(r"C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam")
    src1 = mo.ui.text(value=str(base / "Kalibrierung_1"), label="Cam1")
    src2 = mo.ui.text(value=str(base / "Kalibrierung_2"), label="Cam2")
    src3 = mo.ui.text(value=str(base / "Kalibrierung_3"), label="Cam3")
    src4 = mo.ui.text(value=str(base / "Kalibrierung_4"), label="Cam4")
    pitch = mo.ui.number(value=120, start=10, stop=200, label="Pitch [mm]")
    dotsize = mo.ui.number(value=60, start=10, stop=200, label="Dot size [mm]")
    gv = mo.ui.slider(start=5, stop=80, step=1, value=20, label="gvthres")
    sumg = mo.ui.slider(start=500, stop=8000, step=500, value=5000, label="sumg_min")
    mo.vstack([mo.hstack([src1, src2]), mo.hstack([src3, src4]), mo.hstack([pitch, dotsize, gv, sumg]), mo.md(f"Output: `{out}` — see `rig.yaml`/`plate.yaml`/`top_view.png` there")])
    return gv, pitch, src1, src2, src3, src4, sumg


@app.cell
def _(Path, mo, src1, src2, src3, src4):
    folders = [Path(p.value) for p in [src1, src2, src3, src4]]
    counts = []
    for _f in folders:
        _n = len(list(_f.glob("*.tiff"))) + len(list(_f.glob("*.tif")))
        counts.append(_n)
    from collections import defaultdict
    groups = defaultdict(dict)
    for _ci, _fld in enumerate(folders):
        _tifs = list(_fld.glob("*.tiff")) + list(_fld.glob("*.tif"))
        for _p in sorted([_f for _f in _tifs if _f.name[:8].isdigit()]):
            _frame = _p.name.split("_")[0]
            groups[_frame][_ci] = _p
    sync_frames = [k for k,v in groups.items() if len(v)==4]
    mo.vstack([
        mo.md(f"Folders: `{ [str(f) for f in folders] }` — counts `{counts}` (each 48 expected)"),
        mo.md(f"**Synchronised frames** (present in all 4 cams, by pre-underscore): **{len(sync_frames)}** — e.g. `{sorted(sync_frames)[:5]}`"),
        mo.md("`000000` defines origin; datum marker `ix=2,iy=3` should be at `615 mm` above heating plate") if "00000000" in sync_frames else mo.md("`00000000` not found in all 4 — check naming").callout(kind="warn"),
    ])
    return (sync_frames,)


@app.cell(hide_code=True)
def _(np):

    from scipy.ndimage import gaussian_filter as _gf, label as _lab
    from scipy.spatial import cKDTree as _KD

    def find_plate_roi(work8, sigma=25, pad=0.07):
        imy, imx = work8.shape
        blurred = _gf(work8.astype(float), sigma=sigma)
        hist, _ = np.histogram(blurred, bins=256, range=(0,255))
        total = blurred.size
        sum_tot = (hist * np.arange(256)).sum()
        sumB = 0; wB = 0; max_var = 0; thresh = 0
        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB; mF = (sum_tot - sumB) / wF
            var = wB * wF * (mB - mF) ** 2
            if var > max_var:
                max_var = var; thresh = t
        bw = (blurred > thresh).astype(np.uint8) * 255
        labeled, n = _lab(bw)
        if n == 0:
            return 1, imx-1, 1, imy-1, thresh, bw
        areas = []
        for i in range(1, n+1):
            ys, xs = np.where(labeled == i)
            if len(xs) == 0:
                continue
            area = len(xs)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            areas.append((area, (x0, y0, x1-x0+1, y1-y0+1)))
        areas.sort(reverse=True)
        _, (x, y, w, h) = areas[0]
        x0 = int(max(1, x - w*pad))
        y0 = int(max(1, y - h*pad))
        x1 = int(min(imx-1, x + w + w*pad))
        y1 = int(min(imy-1, y + h + h*pad))
        if x1 - x0 < 80 or y1 - y0 < 80:
            return 1, imx-1, 1, imy-1, thresh, bw
        return x0, x1, y0, y1, thresh, bw

    def _outer_mean(work8, x, y, r_outer=25, r_inner=6):
        xi=int(round(x)); yi=int(round(y))
        imy, imx = work8.shape
        x0=max(0,xi-r_outer); x1=min(imx,xi+r_outer+1); y0=max(0,yi-r_outer); y1=min(imy,yi+r_outer+1)
        win=work8[y0:y1, x0:x1]
        if win.size==0:
            return 0
        cx=xi - x0; cy=yi - y0
        cx0=max(0,cx-r_inner); cx1=min(win.shape[1], cx+r_inner+1); cy0=max(0,cy-r_inner); cy1=min(win.shape[0], cy+r_inner+1)
        outer_sum=int(win.sum()) - int(win[cy0:cy1, cx0:cx1].sum())
        outer_area=win.size - (cy1-cy0)*(cx1-cx0)
        return outer_sum/outer_area if outer_area>0 else 0

    def _estimate_grid_axes(cent):
        cov=np.cov(cent.T)
        vals, vecs=np.linalg.eigh(cov)
        ex=vecs[:,np.argmax(vals)]
        ey=np.array([-ex[1], ex[0]])
        ex/=np.linalg.norm(ex); ey/=np.linalg.norm(ey)
        tree=_KD(cent)
        dists,_=tree.query(cent,k=2)
        pitch=float(np.median(dists[:,1]))
        return ex, ey, pitch

    def reject_outside_grid(cent, work8=None, target=42, outer_thresh=100):
        cent=np.asarray(cent,float)
        if len(cent)==0:
            return cent
        if work8 is not None:
            outer=np.array([_outer_mean(work8, x, y) for x,y in cent])
            keep_int=outer > outer_thresh
            if keep_int.sum() >= target and keep_int.sum() < len(cent):
                cent=cent[keep_int]
                if len(cent)==target:
                    return cent
        if len(cent) <= target:
            return cent
        ex, ey, pitch = _estimate_grid_axes(cent)
        median=np.median(cent, axis=0)
        vx=ey*pitch; vy=ex*pitch
        expected=[]
        for iy in [-3,-2,-1,0,1,2,3]:
            for ix in [-2.5,-1.5,-0.5,0.5,1.5,2.5]:
                expected.append(median + ix*vx + iy*vy)
        expected=np.array(expected)
        tree_exp=_KD(expected)
        dists,_=tree_exp.query(cent,k=1)
        order=np.argsort(dists)
        keep_idx=order[:target]
        return cent[keep_idx]

    def reject_by_neighbor_cost(cent, target=42, k=5):
        cent=np.asarray(cent,float)
        if len(cent) <= target:
            return cent
        tree=_KD(cent)
        dists,_=tree.query(cent,k=k)
        cost=np.sum(dists[:,1:],axis=1)
        keep=np.argsort(cost)[:target]
        return cent[keep]

    def reject_outside_grid_v2(cent, work8=None, target=42, outer_thresh=100):
        filt=reject_by_neighbor_cost(cent, target=target, k=5)
        if work8 is not None and len(filt)==target:
            tree=_KD(cent)
            dists,_=tree.query(cent,k=5)
            cost=np.sum(dists[:,1:],axis=1)
            order=np.argsort(cost)
            outer_filt=np.array([_outer_mean(work8,x,y) for x,y in filt])
            low_mask=outer_filt < outer_thresh
            if low_mask.any():
                for idx in order[target:]:
                    if _outer_mean(work8, cent[idx,0], cent[idx,1]) > outer_thresh:
                        worst=int(np.argmin(outer_filt))
                        filt[worst]=cent[idx]
                        outer_filt[worst]= _outer_mean(work8, cent[idx,0], cent[idx,1])
                        if (outer_filt < outer_thresh).sum()==0:
                            break
        return filt


    return find_plate_roi, reject_outside_grid_v2


@app.cell
def _(
    Image,
    Path,
    find_plate_roi,
    gv,
    mo,
    np,
    plt,
    reject_outside_grid_v2,
    src1,
    sumg,
):
    from openptv2.algorithms.parameters import ControlPar as _CP, MmNp as _MM, TargetPar as _TP
    from openptv2.image_processing import preprocess_image as _prep
    from openptv2.segmentation import target_recognition as _tr

    cpar_prev = _CP(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005, mm=_MM(n1=1,n2=[1],d=[0],n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=[""], cal_img_base_name=[""])
    p_preview = sorted(Path(src1.value).glob("*.tiff"))[0] if list(Path(src1.value).glob("*.tiff")) else None

    raw_prev = np.array(Image.open(p_preview))
    if raw_prev.ndim == 3:
        raw_prev = np.mean(raw_prev, axis=2).astype(raw_prev.dtype)
    lo_prev, hi_prev = float(np.percentile(raw_prev,1)), float(np.percentile(raw_prev,99.5))
    work8_prev = np.clip((raw_prev.astype(float)-lo_prev)/(hi_prev-lo_prev)*255,0,255).astype(np.uint8)

    xmin_p, xmax_p, ymin_p, ymax_p, thresh_p, bw_p = find_plate_roi(work8_prev, sigma=25, pad=0.07)
    _work8_neg_prev = (255 - work8_prev).astype(np.uint8)
    hp_prev = _prep(_work8_neg_prev,1,cpar_prev,25)
    tpar_prev = _TP(gvthres=[int(gv.value)]*4, discont=80, nnmin=10, nnmax=5000, nxmin=8, nxmax=80, nymin=8, nymax=80, sumg_min=int(sumg.value), cr_sz=3)
    tg_prev = _tr(hp_prev,tpar_prev,0,cpar_prev, subrange_x=(xmin_p,xmax_p), subrange_y=(ymin_p,ymax_p))
    tg_prev = [t for t in tg_prev if not (t.n==1 and t.x==1 and t.y==1)]
    cent_prev = np.array([[t.x,t.y] for t in tg_prev], float) if tg_prev else np.zeros((0,2))
    n_raw_prev = len(cent_prev)
    cent_filt_prev = reject_outside_grid_v2(cent_prev, work8=work8_prev, target=42, outer_thresh=100)
    tg_filt_prev=[]
    for pt in cent_filt_prev:
        dists=[np.hypot(t.x-pt[0], t.y-pt[1]) for t in tg_prev]
        _idx_pt=int(np.argmin(dists))
        tg_filt_prev.append(tg_prev[_idx_pt])
    try:
        import cv2
        _roi_gray = work8_prev[ymin_p:ymax_p, xmin_p:xmax_p]
        _found, _corners = cv2.findCirclesGrid(_roi_gray, (6,7), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        _cv2_msg = f"OpenCV found={_found} {len(_corners) if _found else 0}"
    except Exception as _e:
        _cv2_msg = f"OpenCV N/A: {_e}"
    tg_prev = tg_filt_prev
    fig_prev, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
    ax0.imshow(np.clip((raw_prev.astype(float)-lo_prev)/(hi_prev-lo_prev),0,1), cmap="gray")
    ax0.add_patch(plt.Rectangle((xmin_p,ymin_p), xmax_p-xmin_p, ymax_p-ymin_p, edgecolor="cyan", facecolor="none", lw=1.5, ls="--"))
    ax0.set_title(f"ROI [{xmin_p}:{xmax_p}, {ymin_p}:{ymax_p}] Otsu={thresh_p} NEG raw {n_raw_prev}")
    ax0.axis("off")
    ax1.imshow(np.clip((raw_prev.astype(float)-lo_prev)/(hi_prev-lo_prev),0,1), cmap="gray")
    ax1.set_xlim(xmin_p, xmax_p); ax1.set_ylim(ymax_p, ymin_p)
    if tg_prev:
        ax1.scatter([t.x for t in tg_prev],[t.y for t in tg_prev], s=40, facecolors="none", edgecolors="lime", linewidths=1.2)
    ax1.set_title(f"Filtered 42: {len(tg_prev)} gv={gv.value} sumg={sumg.value} | {_cv2_msg}")
    ax1.axis("off")
    fig_prev.tight_layout()
    mo.vstack([mo.md(f"`{p_preview.name}` ROI `{xmin_p}:{xmax_p} x {ymin_p}:{ymax_p}` NEG raw **{n_raw_prev}** -> **{len(tg_prev)}** (neighbor-cost+plate)"), fig_prev])

    return


@app.cell
def _(
    Path,
    find_plate_roi,
    gv,
    mo,
    np,
    pitch,
    reject_outside_grid_v2,
    src1,
    src2,
    src3,
    src4,
    sumg,
    sync_frames,
):
    from openptv2.algorithms.parameters import ControlPar as _CP2, MmNp as _MM2, TargetPar as _TP2
    from openptv2.image_processing import preprocess_image as _prep2
    from openptv2.segmentation import target_recognition as _tr2
    from openptv2.plate_labeler import label_plate as _label2
    from PIL import Image as _PIL2
    pitch_val2 = float(pitch.value); gv_val2 = int(gv.value); sumg_val2 = int(sumg.value)
    cpar_lab2 = _CP2(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005, mm=_MM2(n1=1,n2=[1],d=[0],n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=[""], cal_img_base_name=[""])
    all_lab2 = []
    for frame in sorted(sync_frames)[:6]:
        frame_results2 = []
        for ci, fld in enumerate([Path(src1.value), Path(src2.value), Path(src3.value), Path(src4.value)]):
            cands = list(fld.glob(f"{frame}_*.tiff"))
            path2 = cands[0] if cands else None
            if path2 is None:
                continue
            raw2 = np.array(_PIL2.open(path2))
            if raw2.ndim == 3:
                raw2 = np.mean(raw2, axis=2).astype(raw2.dtype)
            lo2, hi2 = float(np.percentile(raw2,1)), float(np.percentile(raw2,99.5))
            work8_2 = np.clip((raw2.astype(float)-lo2)/(hi2-lo2)*255,0,255).astype(np.uint8)
            xmin2,xmax2,ymin2,ymax2,_,_ = find_plate_roi(work8_2, sigma=25, pad=0.07)
            _work8_neg2 = (255 - work8_2).astype(np.uint8)
            hp2 = _prep2(_work8_neg2,1,cpar_lab2,25)
            tpar2 = _TP2(gvthres=[gv_val2]*4, discont=80, nnmin=10, nnmax=5000, nxmin=8, nxmax=80, nymin=8, nymax=80, sumg_min=sumg_val2, cr_sz=3)
            tg2 = _tr2(hp2,tpar2,0,cpar_lab2, subrange_x=(xmin2,xmax2), subrange_y=(ymin2,ymax2))
            tg2 = [t for t in tg2 if not (t.n==1 and t.x==1 and t.y==1)]
            cent2 = np.array([[t.x,t.y] for t in tg2], float) if tg2 else np.zeros((0,2))
            n_raw2=len(cent2)
            if n_raw2>42:
                cent2 = reject_outside_grid_v2(cent2, work8=work8_2, target=42, outer_thresh=100)
            try:
                img_pts2, ref_pts2, _idx2 = _label2(cent2, None, pitch_x=pitch_val2, pitch_y=pitch_val2, nx=6, ny=7, y_sign=1)
            except Exception:
                img_pts2, ref_pts2 = cent2, np.zeros((0,3))
            frame_results2.append((f"cam{ci+1}", n_raw2, len(cent2), len(img_pts2), f"{xmin2}:{xmax2}"))
        all_lab2.append((frame, frame_results2))
    rows2 = []
    for frame, fr in all_lab2:
        rows2.append("| " + frame + " | " + " | ".join([f"{c}: {nraw}->{nfilt}->{nlab} roi {roi}" for c,nraw,nfilt,nlab,roi in fr]) + " |")
    mo.md("\n".join(["| frame | cam1 | cam2 | cam3 | cam4 |", "|---|---|---|---|---|"] + rows2))

    return


@app.cell(hide_code=True)
def _(
    Path,
    find_plate_roi,
    gv,
    mo,
    np,
    pitch,
    reject_outside_grid_v2,
    src1,
    src2,
    src3,
    src4,
    sumg,
    sync_frames,
):

    from openptv2.algorithms.parameters import ControlPar as _CPc, MmNp as _MMc, TargetPar as _TPc
    from openptv2.image_processing import preprocess_image as _prepc
    from openptv2.segmentation import target_recognition as _trc
    from openptv2.plate_labeler import label_plate as _labelc
    from PIL import Image as _PILc2

    def _detect_plate_points(_image_path: Path, _pitch_val: float, _gv_val: int, _sumg_val: int):
        _raw = np.array(_PILc2.open(_image_path))
        if _raw.ndim == 3:
            _raw = np.mean(_raw, axis=2).astype(_raw.dtype)
        _lo, _hi = float(np.percentile(_raw, 1)), float(np.percentile(_raw, 99.5))
        _work8 = np.clip((_raw.astype(float)-_lo)/(_hi-_lo)*255,0,255).astype(np.uint8)
        _xmin,_xmax,_ymin,_ymax,_,_ = find_plate_roi(_work8, sigma=25, pad=0.07)
        _work8_neg = (255 - _work8).astype(np.uint8)
        _cpar = _CPc(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005, mm=_MMc(n1=1,n2=[1],d=[0],n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=[""], cal_img_base_name=[""])
        _hp = _prepc(_work8_neg, 1, _cpar, 25)
        _tpar = _TPc(gvthres=[_gv_val]*4, discont=80, nnmin=10, nnmax=5000, nxmin=8, nxmax=80, nymin=8, nymax=80, sumg_min=_sumg_val, cr_sz=3)
        _tg = _trc(_hp, _tpar, 0, _cpar, subrange_x=(_xmin,_xmax), subrange_y=(_ymin,_ymax))
        _tg = [t for t in _tg if not (t.n==1 and t.x==1 and t.y==1)]
        _cent = np.array([[t.x,t.y] for t in _tg], float) if _tg else np.zeros((0,2))
        _n_raw = len(_cent)
        if _n_raw > 42:
            _cent = reject_outside_grid_v2(_cent, work8=_work8, target=42, outer_thresh=100)
        _cv2_alt = None
        try:
            import cv2
            _roi = _work8[_ymin:_ymax, _xmin:_xmax]
            _found, _corners = cv2.findCirclesGrid(_roi, (6,7), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            if _found:
                _corners = _corners.reshape(-1,2)
                _corners[:,0] += _xmin
                _corners[:,1] += _ymin
                _cv2_alt = _corners
        except Exception:
            _cv2_alt = None
        return {"raw_path": _image_path, "work8": _work8, "roi": (_xmin,_xmax,_ymin,_ymax), "cent_raw": np.array([[t.x,t.y] for t in _tg], float) if _tg else np.zeros((0,2)), "cent_filt": _cent, "n_raw": _n_raw, "n_filt": len(_cent), "cv2_corners": _cv2_alt}

    _pitch_val = float(pitch.value); _gv_val = int(gv.value); _sumg_val = int(sumg.value)
    _folders = [Path(p.value) for p in [src1, src2, src3, src4]]
    collections = {}
    _all_stats = []
    for _frame in sorted(sync_frames)[:6]:  # notebook preview: first 12; full 48 via flat_collections export
        _frame_entry = {}
        for _ci, _fld in enumerate(_folders):
            _cands = list(_fld.glob(f"{_frame}_*.tiff"))
            _path = _cands[0] if _cands else None
            if _path is None:
                continue
            _det = _detect_plate_points(_path, _pitch_val, _gv_val, _sumg_val)
            _cent = _det["cent_filt"]
            try:
                _img_pts, _ref_pts, __idx = _labelc(_cent, None, pitch_x=_pitch_val, pitch_y=_pitch_val, nx=6, ny=7, y_sign=1)
            except Exception:
                _img_pts, _ref_pts = _cent, np.zeros((0,3))
            _frame_entry[_ci] = {"img_pts": _img_pts, "ref_pts": _ref_pts, "det": _det, "path": _path}
        collections[_frame] = _frame_entry
        _row = [_frame]
        for _ci in range(4):
            if _ci in _frame_entry:
                _e=_frame_entry[_ci]
                _row.append(f"{_e['det']['n_raw']}->{len(_e['det']['cent_filt'])}->{len(_e['img_pts'])}")
            else:
                _row.append("missing")
        _all_stats.append(_row)

    _header = "| frame | cam1 raw->filt->labeled | cam2 | cam3 | cam4 |"
    _sep = "|---|---|---|---|---|"
    _rows = ["| " + " | ".join(r) + " |" for r in _all_stats]
    _md_all = "\n".join([_header, _sep] + _rows)

    flat_collections = {ci: [] for ci in range(4)}
    for _frame, _entry in collections.items():
        for _ci, _dat in _entry.items():
            if len(_dat["img_pts"]) >= 20:
                flat_collections[_ci].append((_dat["ref_pts"], _dat["img_pts"], _frame))

    _flat_counts = {ci: (len(v), sum(len(x[0]) for x in v)) for ci,v in flat_collections.items()}

    mo.vstack([
        mo.md(f"### All-frames XYZ->xy collections  (pitch {_pitch_val}, gv {_gv_val}, sumg {_sumg_val})"),
        mo.md(f"**sync_frames:** {len(sync_frames)}  |  **kept per cam (frames, total points):** {_flat_counts}"),
        mo.md(_md_all),
        mo.md(f"`collections[frame][cam]` -> {{img_pts (n,2) xy, ref_pts (n,3) XYZ}}  |  `flat_collections[cam]` -> list of (XYZ, xy, frame) for solver | OpenCV alt on ROI also computed per frame (see _detect_plate_points)"),
    ])
    return


@app.cell
def _(Path, mo, np, plt):
    rig_path = Path(r"C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam\rig_1-4.yaml")
    import yaml

    doc = yaml.safe_load(rig_path.read_text())
    cams = doc["cameras"]
    fig3 = plt.figure(figsize=(8,6))
    ax3 = fig3.add_subplot(111, projection="3d")
    for i,c in enumerate(cams):
        p = np.array(c["position"])
        ax3.scatter([p[0]],[p[1]],[p[2]], s=80, marker="^", label=f"cam{i+1}")
        t = np.array(c["target"])
        ax3.plot([p[0],t[0]],[p[1],t[1]],[p[2],t[2]], ls="--", alpha=0.5)
    ax3.scatter([0],[615],[0], s=60, c="gold", marker="*", label="datum 615")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y (up)"); ax3.set_zlabel("Z")
    ax3.set_title("Rig 1-4 (Y up, Z depth) — wall r=3575, Y 700/2900")
    ax3.legend()
    mo.mpl.interactive(fig3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Next** — tune `gvthres/sumg_min` until each `cam` shows `~42` labeled, then run the headless solver:

    ```bash
    uv run python scripts/import_calibration.py --model-dir ... --points-dir ... --imx 2560 --imy 2048 --pix 0.005 --out openptv_illmenau_4cam/cal
    uv run python scripts/verify_plate.py --cals openptv_illmenau_4cam/cal --points-dir openptv_illmenau_4cam/cal
    ```

    The `6 mm` thickness is handled as two `Z` planes (`front` at `0`, `back` at `6` along plate normal) in `plate_calibration.py` — enable when you supply `5-8`.
    """)
    return


if __name__ == "__main__":
    app.run()
