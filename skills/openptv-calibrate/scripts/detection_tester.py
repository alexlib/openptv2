# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "imageio",
# ]
# ///
"""Live target-detection tester on a REAL image — a marimo replacement for the
GUI's Detection panel.

Loads an actual TIFF, optionally splits a 4-view splitter frame into quadrants,
and runs openptv2's REAL `targ_rec_fast` (the exact detection the pipeline and
the calibration GUI use) with a slider per parameter and a live overlay of the
detected targets. Change a slider, see the detection update instantly.

Companion to `detection_params_demo.py` (which *explains* what each parameter
does on a synthetic scene, sandbox-runnable). This one *tests* real parameters
on your real data, so it needs the compiled openptv2 — run it from the checkout
WITHOUT `--sandbox`:

    cd openptv2
    uv run marimo edit skills/openptv-calibrate/scripts/detection_tester.py

Goal: if this covers the workflow, it can replace `detection_gui` in
openptv2-gui (same algorithm, same parameters, live feedback, no Chaco/Qt).
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Detection tester (real image, real `targ_rec_fast`)

        Point it at a calibration or sequence TIFF, set the parameters, and see
        exactly what openptv2 detects. Same detection the pipeline runs — this
        just gives it a live UI.
        """
    )
    return


@app.cell
def _(HAS_OPTV, mo):
    warning = (
        None
        if HAS_OPTV
        else mo.callout(
            "**openptv2 could not be imported.** Run this from the openptv2 "
            "checkout **without** `--sandbox` so the compiled detection module "
            "is available:\n\n"
            "`uv run marimo edit skills/openptv-calibrate/scripts/detection_tester.py`",
            kind="danger",
        )
    )
    warning
    return


@app.cell
def _(mo):
    image_path = mo.ui.text(
        value="",
        label="image path (.tif)",
        full_width=True,
    )
    is_splitter = mo.ui.checkbox(value=True, label="splitter frame → split into 4")
    quadrant = mo.ui.dropdown(
        options={"cam1 (TL)": 0, "cam2 (TR)": 1, "cam3 (BR)": 2, "cam4 (BL)": 3},
        value="cam1 (TL)",
        label="quadrant",
    )
    mo.vstack([image_path, mo.hstack([is_splitter, quadrant])])
    return image_path, is_splitter, quadrant


@app.cell
def _(mo):
    # Parameter sliders — the full targ_rec / detect_plate set.
    gvthres = mo.ui.slider(2, 150, value=10, label="grey threshold (gvthres / gvth)")
    disco = mo.ui.slider(1, 250, value=100, label="discontinuity (disco / tol_dis)")
    nnmin = mo.ui.slider(1, 80, value=15, label="min pixels (nnmin / min_npix)")
    nnmax = mo.ui.slider(
        50, 2000, value=900, step=10, label="max pixels (nnmax / max_npix)"
    )
    nxmin = mo.ui.slider(1, 20, value=5, label="min width (nxmin)")
    nxmax = mo.ui.slider(3, 60, value=30, label="max width (nxmax)")
    nymin = mo.ui.slider(1, 20, value=5, label="min height (nymin)")
    nymax = mo.ui.slider(3, 60, value=30, label="max height (nymax)")
    sumg_min = mo.ui.slider(
        0, 4000, value=100, step=50, label="min sum-grey (sumg_min / sum_grey)"
    )
    mo.vstack([gvthres, disco, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min])
    return disco, gvthres, nnmax, nnmin, nxmax, nxmin, nymax, nymin, sumg_min


@app.cell
def _(image_path, is_splitter, np, quadrant):
    def load_image():
        if not image_path.value.strip():
            return None
        import imageio.v3 as iio

        raw = iio.imread(image_path.value.strip())
        if raw.ndim > 2:
            raw = raw[..., :3].mean(axis=2)
        raw = raw.astype(np.float64)
        span = raw.max() - raw.min()
        raw = (255 * (raw - raw.min()) / (span if span else 1)).astype(np.uint8)
        if not is_splitter.value:
            return raw
        h, w = raw.shape
        h2, w2 = h // 2, w // 2
        # raw quadrant indices [TL,TR,BL,BR]=[0,1,2,3]; splitter_order [0,1,3,2]
        # -> cam1=TL cam2=TR cam3=BR cam4=BL.
        tiles = {0: raw[:h2, :w2], 1: raw[:h2, w2:], 3: raw[h2:, :w2], 2: raw[h2:, w2:]}
        return np.ascontiguousarray(tiles[quadrant.value])

    real_img = load_image()
    return (real_img,)


@app.cell
def _(
    HAS_OPTV,
    disco,
    gvthres,
    mo,
    nnmax,
    nnmin,
    nxmax,
    nxmin,
    nymax,
    nymin,
    np,
    plt,
    real_img,
    sumg_min,
    targ_rec_fast,
):
    def view():
        if not HAS_OPTV:
            return mo.md("*openptv2 unavailable — see the note above.*")
        if real_img is None:
            return mo.md("*Enter an image path above to run detection.*")
        img = np.ascontiguousarray(real_img, dtype=np.uint8)
        imy, imx = img.shape
        res = targ_rec_fast(
            img,
            img.copy(),
            int(gvthres.value),
            int(disco.value),
            int(nnmin.value),
            int(nnmax.value),
            int(nxmin.value),
            int(nxmax.value),
            int(nymin.value),
            int(nymax.value),
            int(sumg_min.value),
            1,
            1,
            imx - 1,
            imy - 1,
            20000,
        )
        n, xs, ys = int(res[0]), res[1], res[2]
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("image")
        ax[0].axis("off")
        ax[1].imshow(img, cmap="gray")
        ax[1].axis("off")
        ax[1].set_title(f"targ_rec_fast → {n} targets")
        if n:
            ax[1].scatter(
                np.asarray(xs)[:n],
                np.asarray(ys)[:n],
                s=60,
                facecolors="none",
                edgecolors="lime",
                linewidths=1.3,
            )
        fig.tight_layout()
        return fig

    view()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        Not sure what a parameter does? Open the explainer notebook
        `detection_params_demo.py` (sandbox-ok) — it shows, on a synthetic
        scene, exactly which rule rejects each blob and how `disco` splits
        overlapping particles.
        """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    _src = Path(__file__).resolve().parents[3] / "src"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    try:
        from openptv2.algorithms.track_kernels_batch import targ_rec_fast

        HAS_OPTV = True
    except Exception:
        targ_rec_fast = None
        HAS_OPTV = False

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return HAS_OPTV, mo, np, plt, targ_rec_fast


if __name__ == "__main__":
    app.run()
