# ruff: noqa: E501
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "scikit-image",
#     "matplotlib",
# ]
# ///
"""Interactive explainer for OpenPTV target-detection parameters.

Self-contained (pure numpy/scipy/skimage -- no compiled openptv2 needed) so it
runs in a marimo sandbox. It models the SAME detection semantics OpenPTV's
target_recognition uses -- grey threshold, blob size/shape bounds, min sum-grey,
and (the confusing one) the discontinuity `disco` that splits overlapping
particles -- on a synthetic calibration-like image, with a slider per parameter.

    uv run marimo edit --sandbox
    skills/openptv-calibrate/scripts/detection_params_demo.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # OpenPTV detection parameters — visual explainer

        Detection turns a grey image into a list of **targets** (particle/dot
        centroids). Every parameter is a **filter** on candidate blobs. The
        pipeline:

        1. **Threshold** the image at the grey value `gvthres` → foreground pixels.
        2. **Group** connected foreground into blobs.
        3. **Split** overlapping blobs using the discontinuity `disco` (below).
        4. **Filter** each blob by size/shape/brightness bounds.
        5. Keep survivors; their grey-weighted centroid is the target position.

        `detect_plate` (calibration) and `targ_rec` (particles) are the **same
        algorithm with different values** — plate dots are big/bright/isolated,
        particles are small/dim/crowded, so the two are tuned oppositely.
        Move the sliders and watch which blobs survive and why.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## The synthetic scene (stands in for a split calibration quadrant)""")
    return


@app.cell
def _(np):
    def make_scene(seed=0):
        """A calibration-like grey image: a grid of bright dots, a couple of
        deliberately OVERLAPPING dot pairs (to demo `disco`), a few tiny noise
        specks and one oversized smear (to demo the size bounds), on a gently
        ramped background."""
        rng = np.random.default_rng(seed)
        H = W = 220
        yy, xx = np.mgrid[0:H, 0:W]
        img = 8 + 6 * (xx / W)  # gentle background ramp

        def blob(cx, cy, amp, sx, sy=None):
            sy = sy or sx
            return amp * np.exp(
                -(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2))
            )

        # regular grid of normal dots
        for gy in range(35, W - 20, 45):
            for gx in range(35, W - 20, 45):
                img = img + blob(gx, gy, 180, 3.2)
        # two OVERLAPPING pairs (centres ~5 px apart -> one merged blob)
        img = img + blob(110, 70, 190, 3.4) + blob(118, 70, 185, 3.4)
        img = img + blob(70, 150, 185, 3.4) + blob(77, 156, 190, 3.4)
        # tiny noise specks (small n -> nnmin should reject)
        for _ in range(12):
            img = img + blob(rng.uniform(10, W - 10), rng.uniform(10, H - 10), 70, 1.1)
        # one big smear (large n / wide -> nnmax / nxmax should reject)
        img = img + blob(170, 120, 150, 11, 4)
        img = img + rng.normal(0, 3, img.shape)  # sensor noise
        return np.clip(img, 0, 255).astype(np.uint8)

    scene = make_scene()
    return (scene,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The parameters

        | slider | OpenPTV key(s) | what it filters |
        |---|---|---|
        | **grey threshold** | `gvthres` / `gvth_N` | pixels dimmer than this are background — nothing is detected there |
        | **min / max pixels** | `nnmin`,`nnmax` / `min_npix`,`max_npix` | blob area: rejects tiny noise specks / oversized smears |
        | **max width / height** | `nxmax`,`nymax` | bounding-box size: rejects streaks and merged clumps |
        | **min sum-grey** | `sumg_min` / `sum_grey` | total brightness above threshold: rejects dim blobs |
        | **discontinuity** | `disco` / `tol_dis` | **peak prominence needed to split two overlapping dots** (see the dedicated demo below) |
        """
    )
    return


@app.cell
def _(mo):
    gvthres = mo.ui.slider(2, 120, value=40, label="grey threshold (gvthres)")
    nnmin = mo.ui.slider(1, 60, value=8, label="min pixels (nnmin)")
    nnmax = mo.ui.slider(50, 1200, value=400, label="max pixels (nnmax)")
    nxmax = mo.ui.slider(3, 40, value=18, label="max width (nxmax)")
    nymax = mo.ui.slider(3, 40, value=18, label="max height (nymax)")
    sumg_min = mo.ui.slider(
        0, 3000, value=200, step=50, label="min sum-grey (sumg_min)"
    )
    disco = mo.ui.slider(1, 200, value=25, label="discontinuity (disco)")
    mo.vstack([gvthres, nnmin, nnmax, nxmax, nymax, sumg_min, disco])
    return disco, gvthres, nnmax, nnmin, nxmax, nymax, sumg_min


@app.cell
def _(np, peak_local_max, watershed):
    def segment(img, gvthres, disco):
        """Split the thresholded foreground into targets, applying `disco`
        exactly as OpenPTV means it: two overlapping peaks are separated only
        if the dip (saddle) between them is at least `disco` grey levels below
        the lower peak. LOW disco splits crowded/overlapping particles; HIGH
        disco merges them into one.

        Implementation: seed a watershed at every local maximum, then merge any
        two adjacent regions whose shared saddle is shallower than `disco`
        (union-find over the region-adjacency graph). Returns the merged label
        image and each label's peak grey value.
        """
        fg = img > int(gvthres)
        peaks = peak_local_max(
            img, min_distance=2, labels=fg.astype(int), threshold_abs=int(gvthres) + 1
        )
        if len(peaks) == 0:
            return np.zeros(img.shape, int), {}
        markers = np.zeros(img.shape, int)
        for i, p in enumerate(peaks):
            markers[p[0], p[1]] = i + 1
        lab = watershed(-img.astype(float), markers=markers, mask=fg)
        peakval = {i + 1: float(img[p[0], p[1]]) for i, p in enumerate(peaks)}

        # saddle grey between each pair of adjacent regions = max boundary value
        saddle = {}
        ys, xs = np.where(fg)
        for r, c in zip(ys, xs):
            a = lab[r, c]
            for dr, dc in ((1, 0), (0, 1)):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < img.shape[0] and 0 <= c2 < img.shape[1]:
                    b = lab[r2, c2]
                    if a and b and a != b:
                        key = (min(a, b), max(a, b))
                        saddle[key] = max(
                            saddle.get(key, 0), min(int(img[r, c]), int(img[r2, c2]))
                        )

        parent = list(range(len(peaks) + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b), sv in saddle.items():
            if min(peakval[a], peakval[b]) - sv < float(
                disco
            ):  # dip shallower than disco -> merge
                parent[find(a)] = find(b)
        merged = np.zeros_like(lab)
        for i in range(1, len(peaks) + 1):
            merged[lab == i] = find(i)
        pv = {}
        for i in range(1, len(peaks) + 1):
            pv[find(i)] = max(pv.get(find(i), 0), peakval[i])
        return merged, pv

    def detect(img, gvthres, disco, nnmin, nnmax, nxmax, nymax, sumg_min):
        """Full OpenPTV-style target_recognition: segment, then filter each
        blob by size/shape/brightness bounds."""
        above = img.astype(np.int16) - int(gvthres)
        lab, _ = segment(img, gvthres, disco)

        kept, rejected = [], []
        for i in np.unique(lab):
            if i == 0:
                continue
            ys, xs = np.where(lab == i)
            if len(xs) == 0:
                continue
            n = len(xs)
            nx = xs.max() - xs.min() + 1
            ny = ys.max() - ys.min() + 1
            g = above[ys, xs]
            g = g[g > 0]
            sumg = int(g.sum())
            w = above[ys, xs].clip(min=0).astype(float)
            cx = (xs * w).sum() / w.sum()
            cy = (ys * w).sum() / w.sum()
            # apply each filter, remember the FIRST rule that rejects it
            why = None
            if n < nnmin:
                why = "n<nnmin"
            elif n > nnmax:
                why = "n>nnmax"
            elif nx > nxmax:
                why = "nx>nxmax"
            elif ny > nymax:
                why = "ny>nymax"
            elif sumg < sumg_min:
                why = "sumg<min"
            rec = dict(cx=cx, cy=cy, n=n, nx=nx, ny=ny, sumg=sumg, why=why)
            (rejected if why else kept).append(rec)
        return kept, rejected

    return (detect,)


@app.cell
def _(
    detect,
    disco,
    gvthres,
    nnmax,
    nnmin,
    nxmax,
    nymax,
    plt,
    scene,
    sumg_min,
):
    kept, rejected = detect(
        scene,
        gvthres.value,
        disco.value,
        nnmin.value,
        nnmax.value,
        nxmax.value,
        nymax.value,
        sumg_min.value,
    )
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    for a in ax:
        a.imshow(scene, cmap="gray")
        a.axis("off")
    ax[0].set_title(f"threshold @ {gvthres.value}: foreground")
    ax[0].imshow((scene > gvthres.value), cmap="Reds", alpha=0.35)
    ax[1].set_title(f"KEPT (green)={len(kept)}   rejected (red)={len(rejected)}")
    for r in kept:
        ax[1].scatter(
            [r["cx"]],
            [r["cy"]],
            s=90,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.6,
        )
    rej_colors = {
        "n<nnmin": "orange",
        "n>nnmax": "red",
        "nx>nxmax": "magenta",
        "ny>nymax": "cyan",
        "sumg<min": "yellow",
    }
    for r in rejected:
        ax[1].scatter(
            [r["cx"]],
            [r["cy"]],
            s=90,
            marker="x",
            c=rej_colors.get(r["why"], "red"),
            linewidths=1.6,
        )
    ax[1].scatter([], [], marker="x", c="orange", label="n<nnmin (noise speck)")
    ax[1].scatter([], [], marker="x", c="red", label="n>nnmax (too big)")
    ax[1].scatter([], [], marker="x", c="magenta", label="nx>nxmax (too wide)")
    ax[1].scatter([], [], marker="x", c="yellow", label="sumg<min (too dim)")
    ax[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The confusing one: `disco` (discontinuity) — splitting overlapping particles

        When two particles overlap, their blobs merge into one connected region
        with **two bright peaks and a dip (saddle) between them**. `disco` is the
        **minimum depth of that dip**, in grey levels, required to treat the two
        peaks as **separate** targets:

        - **`disco` low** → even a shallow dip splits them → overlapping particles
          are resolved as two (but noise ripples can over-split one particle).
        - **`disco` high** → only a deep dip splits → overlapping particles
          **merge into one** target (and one centroid sitting between them).

        Below: two Gaussian dots a fixed distance apart. Slide `disco` and the
        overlap depth, and watch one blob split into two — or not.
        """
    )
    return


@app.cell
def _(mo):
    sep = mo.ui.slider(4, 20, value=8, label="separation between the two dots (px)")
    disco2 = mo.ui.slider(1, 200, value=25, label="discontinuity (disco)")
    mo.vstack([sep, disco2])
    return disco2, sep


@app.cell
def _(disco2, np, plt, segment, sep):
    H = 60
    yy2, xx2 = np.mgrid[0:H, 0:H]
    c = H / 2
    two = 190 * np.exp(
        -(((xx2 - (c - sep.value / 2)) ** 2) + (yy2 - c) ** 2) / (2 * 3.4**2)
    )
    two = two + 190 * np.exp(
        -(((xx2 - (c + sep.value / 2)) ** 2) + (yy2 - c) ** 2) / (2 * 3.4**2)
    )
    two = two.astype(np.uint8)

    lab2, _pv2 = segment(two, 20, disco2.value)
    ntarg = len(np.unique(lab2)) - (1 if 0 in np.unique(lab2) else 0)

    saddle = int(two[int(c), int(c)])
    peak = int(two.max())
    dip_depth = peak - saddle

    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
    ax2[0].imshow(two, cmap="gray")
    ax2[0].set_title(
        f"image: peaks sep={sep.value}px\ndip depth = {dip_depth} grey levels"
    )
    ax2[0].axis("off")
    ax2[1].plot(two[int(c), :], "-o", ms=3)
    ax2[1].axhline(
        peak - disco2.value,
        color="red",
        ls="--",
        label=f"peak - disco ({peak - disco2.value})",
    )
    ax2[1].axhline(saddle, color="green", ls=":", label=f"saddle ({saddle})")
    ax2[1].set_title("horizontal profile through both peaks")
    ax2[1].legend(fontsize=8)
    ax2[1].set_xlabel("x")
    ax2[1].set_ylabel("grey")
    ax2[2].imshow(lab2, cmap="tab10")
    ax2[2].axis("off")
    verdict = "SPLIT → 2 targets" if ntarg >= 2 else "MERGED → 1 target"
    ax2[2].set_title(f"disco={disco2.value}: {verdict}")
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Read the middle panel:** if the green saddle line sits **below** the red
        `peak − disco` line, the dip is deeper than `disco` → the peaks **split**.
        If the saddle is **above** it, the dip is too shallow → they **merge**.

        That's why the two detectors differ on the *same* image:

        - **`detect_plate`** (calibration plate): dots are big, bright, well
          separated. High discontinuity, generous size window — grab whole dots,
          don't bother splitting. Result on your plate: ~95 dots/camera.
        - **`targ_rec`** (particles): particles are small, dim, often touching.
          Low discontinuity to split touching particles, higher `nnmin` to reject
          noise, lower grey threshold to catch faint ones. Run it on the *plate*
          and those particle-tuned settings fragment/reject the big plate dots →
          ~40 dots/camera → correspondences starve. Run it on real *particle*
          frames and it's correct.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    return mo, np, peak_local_max, plt, watershed


if __name__ == "__main__":
    app.run()
