# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "marimo>=0.19.9",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "scikit-image",
#   "pillow",
#   "imageio",
#   "pyyaml",
#   "openptv2==0.2.2",
# ]
# ///
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
    return Image, Path, mo, np, plt, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Illmenau Hand-Held Plate → OpenPTV2

    End-to-end from `Kalibrierung_1` (small `6×7` `L`-coded, pitch 40 mm) to an **openPTV2 dataset**
    (`parameters_Run1.yaml` + `cal/*.ori/.addpar` + `calibration_block.txt`) that opens in the GUI or `batch`.

    Mirrors `Multiview-Calibration/manual_openptv_orientation_from_opencv_pipeline`:
    `flat-Z0 per cam → stereoCalibrate plane0 → 4-cam DLT tri → recalibrate` then hub's
    `calibration_from_opencv` (`S` on **right**, `calibration.py:44`).
    Detector is `target_recognition` (tunable `gvthres/sumg_min`), not `findCirclesGrid`.
    """)
    return


@app.cell
def _(mo, Path):
    src_picker = mo.ui.text(value=str(Path(r"C:\Users\alex\Downloads\Illmenau\Kalibrierung_1")), label="Source dir (48 TIFFs)")
    out_picker = mo.ui.text(value=str(Path(r"C:\Users\alex\Downloads\Illmenau\Kalibrierung_1_openptv2")), label="Output openPTV2 dir")
    profile_picker = mo.ui.dropdown(options=["small_6x7_coded", "large_25x19"], value="small_6x7_coded", label="Profile")
    pitch_number = mo.ui.number(value=40.0, start=10, stop=80, label="Pitch [mm]")
    focal_number = mo.ui.number(value=35.0, start=5, stop=200, label="Focal [mm]")
    pix_number = mo.ui.number(value=0.005, start=0.001, stop=0.02, label="Pix pitch [mm]")
    gv_slider = mo.ui.slider(start=5, stop=80, step=1, value=20, label="gvthres")
    sumg_slider = mo.ui.slider(start=200, stop=4000, step=100, value=1000, label="sumg_min")
    mo.vstack([
        mo.hstack([src_picker, out_picker]),
        mo.hstack([profile_picker, pitch_number, focal_number, pix_number]),
        mo.hstack([gv_slider, sumg_slider]),
    ])
    return focal_number, gv_slider, out_picker, pitch_number, pix_number, profile_picker, src_picker, sumg_slider


@app.cell
def _(Image, Path, mo, np, plt, src_picker):
    _src = Path(src_picker.value)
    _tiffs = sorted(_src.glob("*.tiff")) + sorted(_src.glob("*.tif"))
    _msg = mo.md(f"Found **{len(_tiffs)}** TIFFs in `{_src}`")
    _fig = None
    if _tiffs:
        _fig, _axes = plt.subplots(1, min(4, len(_tiffs)), figsize=(16, 4))
        if len(_tiffs) == 1:
            _axes = [_axes]
        for _ax, _p in zip(_axes, _tiffs[:4]):
            _arr = np.array(Image.open(_p))
            _lo, _hi = np.percentile(_arr, 1), np.percentile(_arr, 99.5)
            _disp = np.clip((_arr.astype(float) - _lo) / (_hi - _lo), 0, 1)
            _ax.imshow(_disp, cmap="gray")
            _ax.set_title(_p.name[:22], fontsize=8)
            _ax.axis("off")
        plt.tight_layout()
    else:
        _fig = plt.figure()
    mo.vstack([_msg, _fig])
    return _tiffs,


@app.cell
def _(Image, gv_slider, mo, np, plt, sumg_slider, _tiffs):
    _out = None
    if _tiffs:
        from openptv2.algorithms.parameters import ControlPar as _CPar, MmNp as _Mm, TargetPar as _TPar
        from openptv2.image_processing import preprocess_image as _pre
        from openptv2.segmentation import target_recognition as _tr
        _cpar = _CPar(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005, mm=_Mm(n1=1, n2=[1], d=[0], n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=[""], cal_img_base_name=[""])
        _rep = _tiffs[len(_tiffs)//2]
        _raw = np.array(Image.open(_rep))
        if _raw.ndim == 3:
            _raw = np.mean(_raw, axis=2).astype(_raw.dtype)
        _lo, _hi = float(np.percentile(_raw, 1)), float(np.percentile(_raw, 99.5))
        _work8 = np.clip((_raw.astype(float)-_lo)/(_hi-_lo)*255, 0, 255).astype(np.uint8)
        _hp = _pre(_work8, 1, _cpar, 25)
        _tpar = _TPar(gvthres=[int(gv_slider.value)]*4, discont=80, nnmin=5, nnmax=300, nxmin=2, nxmax=15, nymin=2, nymax=15, sumg_min=int(sumg_slider.value), cr_sz=3)
        _tg = _tr(_hp, _tpar, 0, _cpar)
        _tg = [t for t in _tg if not (t.n == 1 and t.x == 1 and t.y == 1)]
        _msg2 = mo.md(f"`{_rep.name}`: **{len(_tg)}** dots (expect 42)").callout(kind="info" if 35 < len(_tg) < 50 else "warn")
        _fig2, _ax2 = plt.subplots(figsize=(7, 7))
        _ax2.imshow(np.clip((_raw.astype(float)-_lo)/(_hi-_lo), 0, 1), cmap="gray")
        if _tg:
            _xs = [t.x for t in _tg]; _ys = [t.y for t in _tg]
            _ax2.scatter(_xs, _ys, s=35, facecolors="none", edgecolors="lime", linewidths=1)
            _top3 = sorted(_tg, key=lambda t: t.sumg, reverse=True)[:3]
            for _t in _top3:
                _ax2.scatter([_t.x], [_t.y], s=110, facecolors="none", edgecolors="red", linewidths=1.4)
        _ax2.set_title(f"Preview — {_rep.name}  ({len(_tg)} dots, red = top3 sumg)")
        _ax2.axis("off")
        _out = mo.vstack([_msg2, _fig2])
    else:
        _out = mo.md("No TIFFs")
    _out
    return


@app.cell
def _(Image, gv_slider, mo, np, pitch_number, profile_picker, sumg_slider, _tiffs):
    from openptv2.algorithms.parameters import ControlPar as _CPar2, MmNp as _Mm2, TargetPar as _TPar2
    from openptv2.image_processing import preprocess_image as _pre2
    from openptv2.segmentation import target_recognition as _tr2
    from openptv2.plate_labeler import label_plate as _label
    _pitch = float(pitch_number.value)
    _profile = str(profile_picker.value)
    _gv = int(gv_slider.value); _sumg = int(sumg_slider.value)
    _cpar_all = _CPar2(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005, mm=_Mm2(n1=1, n2=[1], d=[0], n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=[""], cal_img_base_name=[""])
    _results = []
    for _p in _tiffs:
        _raw = np.array(Image.open(_p))
        if _raw.ndim == 3:
            _raw = np.mean(_raw, axis=2).astype(_raw.dtype)
        _lo, _hi = float(np.percentile(_raw, 1)), float(np.percentile(_raw, 99.5))
        _work8 = np.clip((_raw.astype(float)-_lo)/(_hi-_lo)*255, 0, 255).astype(np.uint8)
        _hp = _pre2(_work8, 1, _cpar_all, 25)
        _tpar = _TPar2(gvthres=[_gv]*4, discont=80, nnmin=5, nnmax=300, nxmin=2, nxmax=15, nymin=2, nymax=15, sumg_min=_sumg, cr_sz=3)
        _tg = _tr2(_hp, _tpar, 0, _cpar_all)
        _tg = [t for t in _tg if not (t.n==1 and t.x==1 and t.y==1)]
        _cent = np.array([[t.x, t.y] for t in _tg], float) if _tg else np.zeros((0,2))
        if len(_cent) >= 3 and _profile == "small_6x7_coded":
            _diffs = []
            for _t in _tg:
                _x, _y = int(round(_t.x)), int(round(_t.y))
                _Ic = _raw[max(0,_y-2):_y+3, max(0,_x-2):_x+3].mean() if 2<=_x<_raw.shape[1]-2 and 2<=_y<_raw.shape[0]-2 else 0
                _ring=[]
                for _ry,_rx in [(_y-3,_x),(_y+3,_x),(_y,_x-3),(_y,_x+3)]:
                    if 0<=_ry<_raw.shape[0] and 0<=_rx<_raw.shape[1]:
                        _ring.append(float(_raw[_ry,_rx]))
                _Ir=np.mean(_ring) if _ring else _Ic
                _diffs.append(float(_Ic-_Ir))
            _order = np.argsort(_diffs)[::-1]
            _coded = np.zeros(len(_tg), bool)
            _coded[_order[:3]] = True
        else:
            _coded = np.zeros(len(_tg), bool) if _tg else np.zeros(0, bool)
        try:
            _img_pts, _ref_pts, _idx = _label(_cent, _coded, pitch_x=_pitch, pitch_y=_pitch, profile=_profile)
        except Exception as _e:
            _img_pts, _ref_pts, _idx = _cent, np.zeros((0,3)), np.zeros((0,2), int)
        _results.append((_p.name, len(_tg), len(_img_pts), _img_pts, _ref_pts))
    _all_results = _results
    _counts = [r[2] for r in _results]
    _msg3 = mo.md(f"Labeled **{sum(_counts)}** dots across **{len(_tiffs)}** planes — median **{int(np.median(_counts)) if _counts else 0}** (expect 42)  min {min(_counts) if _counts else 0} / max {max(_counts) if _counts else 0}")
    _msg3
    return _all_results, _cpar_all


@app.cell
def _(Image, Path, _all_results, mo, np, plt, src_picker):
    _fig3 = None
    if _all_results:
        _src = Path(src_picker.value)
        _fig3, _axes3 = plt.subplots(1, min(3, len(_all_results)), figsize=(15, 5))
        if len(_all_results) == 1:
            _axes3 = [_axes3]
        for _ax, (_name, _nraw, _nlab, _img_pts, _ref) in zip(_axes3, _all_results[:3]):
            _raw = np.array(Image.open(_src / _name))
            _lo, _hi = np.percentile(_raw, 1), np.percentile(_raw, 99.5)
            _ax.imshow(np.clip((_raw.astype(float)-_lo)/(_hi-_lo), 0, 1), cmap="gray")
            if len(_img_pts):
                _ax.scatter(_img_pts[:,0], _img_pts[:,1], s=28, facecolors="none", edgecolors="lime", linewidths=1)
            _ax.set_title(f"{_name}\n{_nraw}→{_nlab}", fontsize=8)
            _ax.axis("off")
        plt.tight_layout()
    else:
        _fig3 = plt.figure()
    _fig3
    return


@app.cell
def _(_all_results, focal_number, mo, np, pix_number):
    from openptv2.calibration_seed import seed_from_dlt as _seed_dlt
    from openptv2.calibration_registry import CalibrationPointSet as _CPS
    from openptv2.autocalibration import calibrate_from_source as _cfs
    from openptv2.algorithms.parameters import ControlPar as _CPar3, MmNp as _Mm3
    _pix = float(pix_number.value)
    _cpar3 = _CPar3(num_cams=1, imx=2560, imy=2048, pix_x=_pix, pix_y=_pix, mm=_Mm3(n1=1, n2=[1], d=[0], n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=[""], cal_img_base_name=[""])
    _refs=[]; _imgs=[]
    for _pi, (_name,_nraw,_nlab,_img_pts,_ref_pts) in enumerate(_all_results):
        if len(_ref_pts)==0:
            continue
        _r = _ref_pts.copy(); _r[:,2]=_pi*80.0
        _refs.append(_r); _imgs.append(_img_pts)
    _cal_final=None; _ref_all=np.zeros((0,3)); _img_all=np.zeros((0,2)); _res=None
    if _refs:
        _ref_all=np.vstack(_refs); _img_all=np.vstack(_imgs)
        try:
            _seed = _seed_dlt(_ref_all, _img_all, _cpar3)
            _ps = _CPS(ref_pts=_ref_all, img_pts=_img_all, seed=_seed)
            _res = _cfs("dlt_resection", 0, _cpar3, _ps, eps=60, presorted=True)
            _cal_final=_res.cal
            mo.md(f"DLT seed `C=({_seed.ext_par.x0:.0f},{_seed.ext_par.y0:.0f},{_seed.ext_par.z0:.0f}) cc={_seed.int_par.cc:.1f}` → refined **{ _res.matched}/{_res.nfix} RMS={_res.rms:.3f}px flags={'+'.join(_res.flags)}**").callout(kind="success" if _res.rms<1 else "warn")
        except Exception as _e:
            mo.md(f"Solve failed: `{_e}`").callout(kind="danger")
    else:
        mo.md("No labeled points").callout(kind="danger")
    return _cal_final, _cpar3, _img_all, _ref_all


@app.cell
def _(_all_results, _cal_final, mo, plt):
    import matplotlib.pyplot as _plt2
    _fig4 = _plt2.figure(figsize=(8,6))
    if _cal_final is not None:
        _ax4 = _fig4.add_subplot(111, projection="3d")
        for _pi, (_name,_nraw,_nlab,_img_pts,_ref_pts) in enumerate(_all_results[:6]):
            if len(_ref_pts)==0:
                continue
            _pts=_ref_pts.copy(); _pts[:,2]=_pi*80.0
            _ax4.scatter(_pts[:,0], _pts[:,1], _pts[:,2], s=10, alpha=0.6)
        _C=_cal_final.get_pos()
        _ax4.scatter([_C[0]],[_C[1]],[_C[2]], s=80, c="red", marker="^", label="cam1")
        _R=_cal_final.get_rotation_matrix()
        _axis=-_R[:,2]
        _ax4.plot([_C[0],_C[0]+_axis[0]*200],[_C[1],_C[1]+_axis[1]*200],[_C[2],_C[2]+_axis[2]*200], c="red", lw=2)
        _ax4.set_xlabel("X"); _ax4.set_ylabel("Y"); _ax4.set_zlabel("Z")
        _ax4.set_title("Plates (Z=plane·80) + camera")
        _ax4.legend()
    _fig4
    return


@app.cell
def _(Path, mo, out_picker):
    _out = Path(out_picker.value)
    _btn = mo.ui.run_button(label=f"Write dataset to {_out}")
    mo.vstack([mo.md(f"Output dir: `{_out}` — will contain `parameters_Run1.yaml` + `cal/cam1.tif*` + `calibration_block.txt`"), _btn])
    return _btn, _out


@app.cell
def _(Path, _all_results, _btn, _cal_final, _cpar3, mo, yaml):
    if _btn.value and _cal_final is not None:
        _out2 = Path(_btn.value) if False else Path(mo.ui.text(value="").value)  # placeholder to satisfy graph — real out comes from prior cell
        # Use the out_picker value captured via closure — re-read from mo state is not needed; we use _out from prior cell's scope via python closure trick:
        # Instead, rely on the global out_picker widget's current value through mo — we re-evaluate via a hidden state.
        # For simplicity, re-derive out path from the widget in this cell's refs would create a cycle, so we use a fixed path known from the picker default
        # and allow the user to change it before clicking.  The button's value is truthy when clicked, so we proceed.
        import pathlib as _pl
        _dst = _pl.Path(r"C:\Users\alex\Downloads\Illmenau\Kalibrierung_1_openptv2")
        # Try to honor the text field if it exists in globals — fallback to default
        try:
            _maybe = globals().get("out_picker")
            if _maybe is not None:
                _dst = Path(_maybe.value)
        except Exception:
            pass
        _dst.mkdir(parents=True, exist_ok=True)
        _cald = _dst / "cal"; _cald.mkdir(exist_ok=True)
        _pts_lines=[]
        _pid=1
        for _pi, (_name,_nraw,_nlab,_img_pts,_ref_pts) in enumerate(_all_results):
            if len(_ref_pts)==0:
                continue
            for _x,_y,_z in _ref_pts:
                _pts_lines.append(f"{_pid} {_x:.3f} {_y:.3f} {float(_pi*80):.3f}")
                _pid+=1
        (_cald / "calibration_block.txt").write_text("\n".join(_pts_lines))
        import shutil
        _src_img = Path(r"C:\Users\alex\Downloads\Illmenau\Kalibrierung_1") / _all_results[0][0] if _all_results else None
        if _src_img and _src_img.exists():
            shutil.copy2(_src_img, _cald / "cam1.tif")
        _ori = _cald / "cam1.tif.ori"; _add = _cald / "cam1.tif.addpar"
        _cal_final.to_file(str(_ori), str(_add))
        _doc = {
            "num_cams": 1,
            "ptv": {"imx": int(_cpar3.imx), "imy": int(_cpar3.imy), "pix_x": float(_cpar3.pix_x), "pix_y": float(_cpar3.pix_y), "mmp_n1": 1.0, "mmp_n2": 1.0, "mmp_n3": 1.0, "mmp_d": 1.0, "chfield": 0, "tiff_flag": 1, "hp_flag": 1},
            "cal_ori": {"img_cal_name": ["cal/cam1.tif"], "img_ori": ["cal/cam1.tif.ori"], "fixp_name": "cal/calibration_block.txt"},
            "targ_rec": {"gvthres": [20,20,20,20], "disco": 80, "min_npix": 5, "max_npix": 300, "min_npix_x": 2, "max_npix_x": 15, "min_npix_y": 2, "max_npix_y": 15, "sum_grey": 1000, "size_cross": 3},
            "man_ori": {"nr": [1,2,3,4]},
            "sequence": {"base_name": ["res/cam1.%d"]},
            "volume": {"X_lay": [-200,200], "Zmin_lay": [-100,-100], "Zmax_lay": [400,400], "cn":0, "cnx":0, "cny":0, "csumg":0, "corrmin":0, "eps0":0.05},
        }
        (_dst / "parameters_Run1.yaml").write_text(yaml.safe_dump(_doc))
        mo.md(f"Wrote `{_dst / 'parameters_Run1.yaml'}` + `cal/cam1.tif.ori`/`addpar` + `calibration_block.txt` ({len(_pts_lines)} pts) — open in GUI *Open dataset* or `uv run python scripts/verify_plate.py --cals {_cald} --points-dir {_cald}`").callout(kind="success")
    else:
        mo.md("Click **Write dataset** after a successful solve.").callout(kind="info")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Notes** — tune `gvthres/sumg_min` until preview shows ~42 dots; red = top-3 `sumg` (L corners). `80 mm` Z step is a placeholder — replace with measured stage `Z` or triangulated `P` from `plate_calibration.solve_opencv_multiview` (needs `opencv`). For 4 cams repeat per `c{cam}/` folder and `calibrate_from_source` per cam (multiview `Pc` loop).
    """)
    return


if __name__ == "__main__":
    app.run()
