# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-image",
#     "trackpy",
#     "dask",
#     "opencv-python-headless",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial import cKDTree

    return cKDTree, mo, np, plt, time


@app.cell
def _(mo):
    mo.md("""
    # Detection prototypes on crops

    Same crop of `test_cavity/img/cam1.10000` under every algorithm, one
    section per method. Tune each method's sliders and watch the overlay:
    green circles = `targ_rec` baseline on the full frame (frozen at
    `targ_rec.par` defaults, shifted into crop coords), red crosses =
    prototype output **on the crop** (already crop-local, no shifting needed).

    Crop cases: **overlap** (pair 1.3 px apart + neighbour), **faint**
    (n=6/sumg=169 next to a big one), **single** (two faint targets),
    **bright** (sumg=1733 + medium neighbour).

    Note: particles cut by the crop border can differ from the full-frame
    baseline — that is a cropping artifact, not a detector error.
    """)
    return


@app.cell
def _(np):
    from skimage.io import imread

    from openptv2.algorithms import detection_prototypes as dp
    from openptv2.algorithms.parameters import ControlPar, TargetPar
    from openptv2.image_processing import preprocess_image

    raw_full = np.ascontiguousarray(
        imread("test_data/test_cavity/img/cam1.10000"), dtype=np.uint8
    )
    cpar = ControlPar(1)
    cpar.set_image_size((raw_full.shape[1], raw_full.shape[0]))
    hp_full = np.ascontiguousarray(preprocess_image(raw_full, 0, cpar, 25), dtype=np.uint8)

    tpar = TargetPar.from_file("test_data/test_cavity/parameters/targ_rec.par")
    base_params = {
        "gvthres": int(tpar.get_grey_thresholds()[0]),
        "discont": int(tpar.get_max_discontinuity()),
        "nnmin": int(tpar.get_pixel_count_bounds()[0]),
        "nnmax": int(tpar.get_pixel_count_bounds()[1]),
        "nxmin": int(tpar.get_xsize_bounds()[0]),
        "nxmax": int(tpar.get_xsize_bounds()[1]),
        "nymin": int(tpar.get_ysize_bounds()[0]),
        "nymax": int(tpar.get_ysize_bounds()[1]),
        "sumg_min": int(tpar.get_min_sum_grey()),
    }
    base_all = dp.baseline_targets(hp_full, base_params)

    CROPS = {
        "overlap (pair 1.3px)": (443, 330),
        "faint (n=6,sumg=169)": (589, 62),
        "single (2x faint)": (321, 37),
        "bright (sumg=1733)": (792, 112),
    }
    return CROPS, base_all, base_params, dp, hp_full, raw_full


@app.cell
def _(cKDTree, np, plt):
    def crop_of(img, cx, cy, size):
        x0 = int(np.clip(cx - size // 2, 0, img.shape[1] - size))
        y0 = int(np.clip(cy - size // 2, 0, img.shape[0] - size))
        return np.ascontiguousarray(img[y0 : y0 + size, x0 : x0 + size]), x0, y0

    def in_crop(rows, x0, y0, size):
        m = (
            (rows[:, 1] >= x0)
            & (rows[:, 1] < x0 + size)
            & (rows[:, 2] >= y0)
            & (rows[:, 2] < y0 + size)
        )
        return rows[m]

    def to_local(rows, x0, y0):
        out = rows.copy()
        out[:, 1] -= x0
        out[:, 2] -= y0
        return out

    def match_stats(base, cand, tol=1.5):
        if len(base) == 0 or len(cand) == 0:
            return float("nan"), 0.0
        d1, _ = cKDTree(base[:, 1:3]).query(cand[:, 1:3], k=1)
        d2, _ = cKDTree(cand[:, 1:3]).query(base[:, 1:3], k=1)
        return float(np.median(d1)), float(np.mean(d2 <= tol))

    def overlay_fig(crop, base_local, cand_local, title):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(crop, cmap="gray", interpolation="nearest")
        if len(base_local):
            ax.scatter(
                base_local[:, 1], base_local[:, 2], s=120, facecolors="none",
                edgecolors="lime", linewidths=1.5, label=f"baseline x{len(base_local)}",
            )
        if len(cand_local):
            ax.scatter(
                cand_local[:, 1], cand_local[:, 2], s=60, marker="x", c="red",
                linewidths=1.5, label=f"proto x{len(cand_local)}",
            )
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig

    return crop_of, in_crop, match_stats, overlay_fig, to_local


@app.cell
def _(CROPS, mo):
    crop_pick = mo.ui.dropdown(CROPS, value="overlap (pair 1.3px)", label="crop case")
    size_pick = mo.ui.dropdown({16: 16, 24: 24, 32: 32, 48: 48}, value=32, label="crop size")
    input_pick = mo.ui.dropdown(
        {"highpassed (pipeline)": "hp", "raw": "raw"},
        value="highpassed (pipeline)",
        label="crop input (proPTV section can override)",
    )
    mo.hstack([crop_pick, size_pick, input_pick])
    return crop_pick, input_pick, size_pick


@app.cell
def _(
    CROPS,
    base_all,
    crop_of,
    crop_pick,
    hp_full,
    in_crop,
    input_pick,
    mo,
    raw_full,
    size_pick,
    to_local,
):
    src = hp_full if input_pick.value == "hp" else raw_full
    cx, cy = crop_pick.value
    crop_label = next(k for k, v in CROPS.items() if v == crop_pick.value)
    size = int(size_pick.value)
    crop, x0, y0 = crop_of(src, cx, cy, size)
    base_local = to_local(in_crop(base_all, x0, y0, size), x0, y0)
    mo.md(
        f"**{crop_label}** — center ({cx},{cy}), origin ({x0},{y0}), "
        f"{size}×{size}, input={input_pick.value}. "
        f"Baseline targets in view: **{len(base_local)}**."
    )
    return base_local, crop, size, x0, y0


@app.cell
def _(base_local, crop, overlay_fig, plt):
    plt.close("all")
    overlay_fig(crop, base_local, base_local[:0], "crop + baseline (green)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Baseline `targ_rec` — reference, rerun on the crop (expect close to baseline, modulo border cuts)
    """)
    return


@app.cell
def _(base_params, mo):
    tr_gv = mo.ui.slider(1, 60, 1, value=base_params["gvthres"], label="gvthres")
    tr_disco = mo.ui.slider(0, 200, 5, value=base_params["discont"], label="discont")
    tr_nnmin = mo.ui.slider(1, 20, 1, value=base_params["nnmin"], label="nnmin")
    tr_nnmax = mo.ui.slider(50, 1000, 10, value=base_params["nnmax"], label="nnmax")
    tr_sumg = mo.ui.slider(0, 1000, 10, value=base_params["sumg_min"], label="sumg_min")
    mo.hstack([tr_gv, tr_disco, tr_nnmin, tr_nnmax, tr_sumg])
    return tr_disco, tr_gv, tr_nnmax, tr_nnmin, tr_sumg


@app.cell
def _(
    base_local,
    base_params,
    crop,
    dp,
    match_stats,
    overlay_fig,
    plt,
    time,
    tr_disco,
    tr_gv,
    tr_nnmax,
    tr_nnmin,
    tr_sumg,
):
    tr_params = dict(
        base_params, gvthres=tr_gv.value, discont=tr_disco.value,
        nnmin=tr_nnmin.value, nnmax=tr_nnmax.value, sumg_min=tr_sumg.value,
    )
    tr_t0 = time.perf_counter()
    tr_rows = dp.baseline_targets(crop, tr_params)
    tr_ms = 1000 * (time.perf_counter() - tr_t0)
    tr_med, tr_rec = match_stats(base_local, tr_rows)
    plt.close("all")
    overlay_fig(crop, base_local, tr_rows, f"targ_rec: {len(tr_rows)} | medNN {tr_med:.2f}px rec {tr_rec:.2f} | {tr_ms:.1f} ms")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. dask (scipy kernel, dask graph) — `gvthres` + size/sum filters
    """)
    return


@app.cell
def _(base_params, mo):
    da_gv = mo.ui.slider(1, 60, 1, value=base_params["gvthres"], label="gvthres")
    da_nnmin = mo.ui.slider(1, 20, 1, value=base_params["nnmin"], label="nnmin")
    da_nnmax = mo.ui.slider(50, 1000, 10, value=base_params["nnmax"], label="nnmax")
    da_sumg = mo.ui.slider(0, 1000, 10, value=base_params["sumg_min"], label="sumg_min")
    mo.hstack([da_gv, da_nnmin, da_nnmax, da_sumg])
    return da_gv, da_nnmax, da_nnmin, da_sumg


@app.cell
def _(
    base_local,
    base_params,
    crop,
    da_gv,
    da_nnmax,
    da_nnmin,
    da_sumg,
    dp,
    match_stats,
    overlay_fig,
    plt,
    time,
):
    da_params = dict(
        base_params, gvthres=da_gv.value, nnmin=da_nnmin.value,
        nnmax=da_nnmax.value, sumg_min=da_sumg.value,
    )
    da_t0 = time.perf_counter()
    da_rows = dp.detect_dask_label(crop, da_params)
    da_ms = 1000 * (time.perf_counter() - da_t0)
    da_med, da_rec = match_stats(base_local, da_rows)
    plt.close("all")
    overlay_fig(crop, base_local, da_rows, f"dask: {len(da_rows)} | medNN {da_med:.2f}px rec {da_rec:.2f} | {da_ms:.1f} ms")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. trackpy.locate — `diameter` / `minmass` / `separation` / `threshold`
    """)
    return


@app.cell
def _(mo):
    tp_diam = mo.ui.slider(3, 15, 2, value=7, label="diameter")
    tp_mass = mo.ui.slider(0, 2000, 25, value=150, label="minmass")
    tp_sep = mo.ui.slider(1, 10, 1, value=3, label="separation")
    tp_thr = mo.ui.slider(0, 60, 1, value=9, label="threshold")
    mo.hstack([tp_diam, tp_mass, tp_sep, tp_thr])
    return tp_diam, tp_mass, tp_sep, tp_thr


@app.cell
def _(
    base_local,
    crop,
    dp,
    match_stats,
    overlay_fig,
    plt,
    time,
    tp_diam,
    tp_mass,
    tp_sep,
    tp_thr,
):
    tp_t0 = time.perf_counter()
    tp_rows = dp.detect_trackpy(
        crop,
        {"diameter": tp_diam.value, "sumg_min": tp_mass.value,
         "separation": tp_sep.value, "gvthres": tp_thr.value},
    )
    tp_ms = 1000 * (time.perf_counter() - tp_t0)
    tp_med, tp_rec = match_stats(base_local, tp_rows)
    plt.close("all")
    overlay_fig(crop, base_local, tp_rows, f"trackpy: {len(tp_rows)} | medNN {tp_med:.2f}px rec {tp_rec:.2f} | {tp_ms:.1f} ms")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. skimage peak + regionprops — threshold / min_distance / size filters
    """)
    return


@app.cell
def _(base_params, mo):
    sk_gv = mo.ui.slider(1, 60, 1, value=base_params["gvthres"], label="gvthres")
    sk_md = mo.ui.slider(1, 5, 1, value=2, label="min_distance")
    sk_nnmin = mo.ui.slider(1, 20, 1, value=base_params["nnmin"], label="nnmin")
    sk_sumg = mo.ui.slider(0, 1000, 10, value=base_params["sumg_min"], label="sumg_min")
    mo.hstack([sk_gv, sk_md, sk_nnmin, sk_sumg])
    return sk_gv, sk_md, sk_nnmin, sk_sumg


@app.cell
def _(
    base_local,
    base_params,
    crop,
    dp,
    match_stats,
    overlay_fig,
    plt,
    sk_gv,
    sk_md,
    sk_nnmin,
    sk_sumg,
    time,
):
    sk_params = dict(
        base_params, gvthres=sk_gv.value, min_distance=sk_md.value,
        nnmin=sk_nnmin.value, sumg_min=sk_sumg.value,
    )
    sk_t0 = time.perf_counter()
    sk_rows = dp.detect_skimage(crop, sk_params)
    sk_ms = 1000 * (time.perf_counter() - sk_t0)
    sk_med, sk_rec = match_stats(base_local, sk_rows)
    plt.close("all")
    overlay_fig(crop, base_local, sk_rows, f"skimage: {len(sk_rows)} | medNN {sk_med:.2f}px rec {sk_rec:.2f} | {sk_ms:.1f} ms")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. proPTV port (Barta) — expects **raw** input; keep the box checked to use the raw crop regardless of the global toggle
    """)
    return


@app.cell
def _(mo):
    pp_raw = mo.ui.checkbox(value=True, label="use raw crop")
    pp_thr = mo.ui.slider(20, 300, 5, value=100, label="threshold")
    pp_ps = mo.ui.slider(1, 5, 1, value=2, label="particleSize")
    pp_runs = mo.ui.slider(1, 5, 1, value=3, label="runs_search")
    pp_std = mo.ui.slider(0.3, 1.5, 0.1, value=0.7, label="std")
    mo.hstack([pp_raw, pp_thr, pp_ps, pp_runs, pp_std])
    return pp_ps, pp_raw, pp_runs, pp_std, pp_thr


@app.cell
def _(
    base_local,
    crop,
    crop_of,
    dp,
    match_stats,
    overlay_fig,
    plt,
    pp_ps,
    pp_raw,
    pp_runs,
    pp_std,
    pp_thr,
    raw_full,
    size,
    time,
    x0,
    y0,
):
    pp_src = crop_of(raw_full, x0 + size // 2, y0 + size // 2, size)[0] if pp_raw.value else crop
    pp_t0 = time.perf_counter()
    pp_rows = dp.detect_proptv(
        pp_src,
        {"threshold": pp_thr.value, "particleSize": pp_ps.value,
         "runs_search": pp_runs.value, "std": pp_std.value},
    )
    pp_ms = 1000 * (time.perf_counter() - pp_t0)
    pp_med, pp_rec = match_stats(base_local, pp_rows)
    plt.close("all")
    overlay_fig(pp_src, base_local, pp_rows, f"proPTV: {len(pp_rows)} | medNN {pp_med:.2f}px rec {pp_rec:.2f} | {pp_ms:.1f} ms")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Scoreboard (tol 1.5 px, baseline in view: "
        f"{len(base_local)})

    "
        "| method | found | medNN px | recall | ms |
    "
        "|---|---|---|---|---|
    "
        f"| targ_rec | {len(tr_rows)} | {tr_med:.2f} | {tr_rec:.2f} | {tr_ms:.1f} |
    "
        f"| dask | {len(da_rows)} | {da_med:.2f} | {da_rec:.2f} | {da_ms:.1f} |
    "
        f"| trackpy | {len(tp_rows)} | {tp_med:.2f} | {tp_rec:.2f} | {tp_ms:.1f} |
    "
        f"| skimage | {len(sk_rows)} | {sk_med:.2f} | {sk_rec:.2f} | {sk_ms:.1f} |
    "
        f"| proPTV | {len(pp_rows)} | {pp_med:.2f} | {pp_rec:.2f} | {pp_ms:.1f} |
    """)
    return


if __name__ == "__main__":
    app.run()
