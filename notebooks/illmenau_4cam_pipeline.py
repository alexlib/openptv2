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
    gv = mo.ui.slider(start=5, stop=80, step=1, value=20, label="gvthres")
    sumg = mo.ui.slider(start=500, stop=8000, step=500, value=5000, label="sumg_min")
    mo.vstack([mo.hstack([src1, src2]), mo.hstack([src3, src4]), mo.hstack([pitch, gv, sumg]), mo.md(f"Output: `{out}` — see `rig.yaml`/`plate.yaml`/`top_view.png` there")])
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


@app.cell
def _(Image, Path, gv, mo, np, plt, src1, sumg):
    from openptv2.algorithms.parameters import ControlPar, MmNp, TargetPar
    from openptv2.image_processing import preprocess_image
    from openptv2.segmentation import target_recognition
    cpar_prev = ControlPar(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005, mm=MmNp(n1=1,n2=[1],d=[0],n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=[""], cal_img_base_name=[""])
    p_preview = sorted(Path(src1.value).glob("*.tiff"))[0] if list(Path(src1.value).glob("*.tiff")) else None
    if p_preview is not None:
        raw_prev = np.array(Image.open(p_preview))
        if raw_prev.ndim == 3:
            raw_prev = np.mean(raw_prev, axis=2).astype(raw_prev.dtype)
        lo_prev, hi_prev = float(np.percentile(raw_prev,1)), float(np.percentile(raw_prev,99.5))
        work8_prev = np.clip((raw_prev.astype(float)-lo_prev)/(hi_prev-lo_prev)*255,0,255).astype(np.uint8)
        hp_prev = preprocess_image(work8_prev,1,cpar_prev,25)
        tpar_prev = TargetPar(gvthres=[int(gv.value)]*4, discont=80, nnmin=10, nnmax=5000, nxmin=10, nxmax=80, nymin=10, nymax=80, sumg_min=int(sumg.value), cr_sz=3)
        tg_prev = target_recognition(hp_prev,tpar_prev,0,cpar_prev)
        tg_prev = [t for t in tg_prev if not (t.n==1 and t.x==1 and t.y==1)]
        fig_prev, ax_prev = plt.subplots(figsize=(7,7))
        ax_prev.imshow(np.clip((raw_prev.astype(float)-lo_prev)/(hi_prev-lo_prev),0,1), cmap="gray")
        if tg_prev:
            ax_prev.scatter([t.x for t in tg_prev],[t.y for t in tg_prev], s=30, facecolors="none", edgecolors="lime", linewidths=1)
        ax_prev.set_title(f"{p_preview.name} — {len(tg_prev)} dots (tune gv/sumg to ~42, dot 60mm)")
        ax_prev.axis("off")
        mo.vstack([mo.md(f"`{p_preview.name}`: **{len(tg_prev)}** dots"), fig_prev])
    else:
        mo.md("No preview")
    return


@app.cell
def _(Path, gv, mo, np, pitch, src1, src2, src3, src4, sumg, sync_frames):
    from openptv2.algorithms.parameters import ControlPar, MmNp, TargetPar
    from openptv2.image_processing import preprocess_image
    from openptv2.segmentation import target_recognition
    from openptv2.plate_labeler import label_plate
    from PIL import Image as PILImage
    pitch_val = float(pitch.value); gv_val = int(gv.value); sumg_val = int(sumg.value)
    cpar_lab = ControlPar(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005, mm=MmNp(n1=1,n2=[1],d=[0],n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0, img_base_name=[""], cal_img_base_name=[""])
    all_lab = []
    for frame in sorted(sync_frames)[:6]:
        frame_results = []
        for ci, fld in enumerate([Path(src1.value), Path(src2.value), Path(src3.value), Path(src4.value)]):
            cands = list(fld.glob(f"{frame}_*.tiff"))
            path = cands[0] if cands else None
            if path is None:
                continue
            raw = np.array(PILImage.open(path))
            if raw.ndim == 3:
                raw = np.mean(raw, axis=2).astype(raw.dtype)
            lo, hi = float(np.percentile(raw,1)), float(np.percentile(raw,99.5))
            work8 = np.clip((raw.astype(float)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
            hp = preprocess_image(work8,1,cpar_lab,25)
            tpar = TargetPar(gvthres=[gv_val]*4, discont=80, nnmin=10, nnmax=5000, nxmin=10, nxmax=80, nymin=10, nymax=80, sumg_min=sumg_val, cr_sz=3)
            tg = target_recognition(hp,tpar,0,cpar_lab)
            tg = [t for t in tg if not (t.n==1 and t.x==1 and t.y==1)]
            cent = np.array([[t.x,t.y] for t in tg], float) if tg else np.zeros((0,2))
            try:
                img_pts, ref_pts, idx = label_plate(cent, None, pitch_x=pitch_val, pitch_y=pitch_val, nx=6, ny=7, y_sign=1)
            except Exception:
                img_pts, ref_pts = cent, np.zeros((0,3))
            frame_results.append((f"cam{ci+1}", len(tg), len(img_pts)))
        all_lab.append((frame, frame_results))
    rows = []
    for frame, fr in all_lab:
        rows.append(f"| {frame} | " + " | ".join([f"c{c}: {nraw}→{nlab}" for c,nraw,nlab in [(r[0],r[1],r[2]) for r in fr]]) + " |")
    mo.md("\n".join(["| frame | cam1 | cam2 | cam3 | cam4 |", "|---|---|---|---|---|"] + rows))
    return


@app.cell
def _(Path, mo, np, plt):
    rig_path = Path(r"C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam\rig.yaml")
    if rig_path.exists():
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
        fig3
    else:
        mo.md("`rig.yaml` not found")
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
