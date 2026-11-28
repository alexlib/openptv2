import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Seeing through glass: the independent ray tracer against the radial-shift model

    A camera outside a water cell sees a particle through air, a glass window and water. Light bends at both
    surfaces (Snell's law), so the particle appears at a different place than the straight line from the camera
    suggests. Every 3D-PTV code needs a model of that bend.

    * **openptv2's model** (`openptv2.algorithms.multimed`, inherited from OpenPTV): pretend the whole path is one
      2D problem in the radial distance `r` from the glass axis and find a *radial shift factor* iteratively. It is
      fast (a look-up table) and exact **if** the layers are flat, parallel and perpendicular to the glass vector.
    * **The truth rig** (`openptv2.synthetic_rig.TruthRig`, written independently): trace the real 3D ray, with
      vector Snell's law at each surface, for any orientation of the surfaces. It is slower, but it can *break the
      assumptions one at a time*, which a model that is built on them cannot.

    This notebook shows both, side by side, with the cavity-like rig of `docs/self-calibration-cavity-synthetic.md`
    (2 cameras each side of a water cell, glass 6 mm, n = 1.0 / 1.46 / 1.33). It ends with what the tracer suggests
    for the multimedia model, honestly separated into what we measured and what is only an idea.
    """)
    return


@app.cell
def _():
    import copy
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import least_squares

    from openptv2.loco import project_pixels
    from openptv2.synthetic_rig import TruthRig, cavity_rig
    from openptv2.synthetic_rig import _tilted as tilted

    rig0 = cavity_rig(truth_lens=False)
    CAM = 0
    return (
        CAM,
        TruthRig,
        cavity_rig,
        copy,
        least_squares,
        np,
        plt,
        project_pixels,
        rig0,
        tilted,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One ray, step by step

    Move the particle and change the glass, then look at the picture. **Red** is the traced path; **dotted grey** is
    the straight line from the camera to the same particle; **dashed orange** is where the camera *thinks* the
    particle is (the air ray, continued to the same distance). The distance between the dotted line and the red
    path at the particle is what the multimedia model must supply.

    The tilt sliders rotate the window surfaces (degrees). With *front* = the water-side surface and *back* = the
    air-side surface, a different tilt of the two makes a wedge. At zero tilt both models agree.
    """)
    return


@app.cell
def _(mo):
    lateral = mo.ui.slider(-45, 45, value=30, step=1, label="particle lateral offset (mm)", show_value=True)
    depth = mo.ui.slider(-40, 40, value=-20, step=1, label="particle depth z (mm)", show_value=True)
    thick = mo.ui.slider(2, 14, value=6, step=0.5, label="glass thickness (mm)", show_value=True)
    nwater = mo.ui.slider(1.30, 1.40, value=1.33, step=0.005, label="water index", show_value=True)
    tfront = mo.ui.slider(0, 3, value=0, step=0.1, label="front (water-side) tilt (deg)", show_value=True)
    tback = mo.ui.slider(0, 3, value=0, step=0.1, label="back (air-side) tilt (deg)", show_value=True)
    mo.vstack([mo.hstack([lateral, depth]), mo.hstack([thick, nwater]), mo.hstack([tfront, tback])])
    return depth, lateral, nwater, thick, tback, tfront


@app.cell
def _(CAM, depth, lateral, np, nwater, rig0, tback, tfront, thick):
    deg = np.pi / 180
    rig1 = rig0.with_(
        glass_thickness=float(thick.value),
        n_water=float(nwater.value),
        tilt_front=[(tfront.value * deg, 0.0)] * 4,
        tilt_back=[(tback.value * deg, 0.0)] * 4,
    )
    cam = np.asarray(rig1.cals[CAM].get_pos(), float)
    g = np.asarray(rig1.cals[CAM].get_glass_vec(), float)
    ghat = g / np.linalg.norm(g)
    # particle: lateral offset along x from the camera axis, depth z in the lab frame
    P = np.array([[cam[0] + lateral.value, cam[1], float(depth.value)]])
    path = rig1.ray_path(CAM, P)[0]
    return P, cam, deg, g, ghat, path, rig1


@app.cell
def _(CAM, P, cam, g, ghat, np, path, plt, rig1):
    s_front = float(np.linalg.norm(g))
    d_ = rig1.glass_thickness
    e = (P[0] - cam) - ghat * ((P[0] - cam) @ ghat)
    e = e / np.linalg.norm(e)

    def sx(p):  # (lateral, axial s) of a point
        v = np.asarray(p) - cam
        return float(v @ e), float(np.asarray(p) @ ghat)

    pts_ = [sx(p) for p in path]
    fig1, axs = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1, 1.3]))
    for _a, _zoom in zip(axs, (False, True)):
        _a.axhspan(s_front, s_front + d_, color="#9ec5e8", alpha=0.85, lw=0)
        _a.axhspan(-60, s_front, color="#d7ecf7", alpha=0.7, lw=0)
        _a.axhspan(s_front + d_, 700, color="#f6f3e8", alpha=0.5, lw=0)
        _a.plot([q[0] for q in pts_], [q[1] for q in pts_], "-", color="#b23a48", lw=1.5, label="traced (Snell, 3D)")
        _a.plot(*pts_[-1], "o", color="#b23a48")
        pc, pp = sx(path[0]), sx(P[0])
        _a.plot([pc[0], pp[0]], [pc[1], pp[1]], ":", color="#555", lw=1.1, label="straight line (no refraction)")
        # apparent position: the air ray continued to the same distance
        ua = (path[1] - path[0]) / np.linalg.norm(path[1] - path[0])
        eff = cam + ua * np.linalg.norm(P[0] - cam)
        ee = sx(eff)
        _a.plot([pc[0], ee[0]], [pc[1], ee[1]], "--", color="#e08a1e", lw=1.2, label="where the camera thinks it is")
        _a.plot(*ee, "x", color="#e08a1e", ms=8)
        _a.set_xlabel("lateral (mm)")
        if _zoom:
            _a.set_xlim(min(pts_[1][0], pts_[2][0]) - 25, max(pts_[1][0], pts_[2][0]) + 25)
            _a.set_ylim(s_front - 35, s_front + d_ + 30)
            _a.set_title("zoom on the window")
            _a.text(_a.get_xlim()[1] - 1, s_front + d_ / 2, f"glass n={rig1.n_glass}", ha="right", va="center", fontsize=9)
            _a.text(_a.get_xlim()[1] - 1, s_front - 15, f"water n={rig1.n_water}", ha="right", fontsize=9)
            _a.text(_a.get_xlim()[1] - 1, s_front + d_ + 15, "air n=1", ha="right", fontsize=9)
            _a.legend(loc="upper left", fontsize=8)
        else:
            _a.set_xlim(-60, 80)
            _a.set_ylim(-60, float(cam @ ghat) + 30)
            _a.set_ylabel("axial position s along the glass vector (mm)")
            _a.set_title("camera to particle (side view)")
            _a.plot(0, float(cam @ ghat), "ks")
            _a.text(2, float(cam @ ghat) - 25, "camera", fontsize=9)
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(CAM, P, ghat, np, path, rig1, tilted):
    # angles to the surface normals, and the invariant n*sin(theta) (equal in all layers only if the surfaces are parallel)
    nb_, nf_ = tilted(ghat, rig1.tilt_back[CAM]), tilted(ghat, rig1.tilt_front[CAM])

    def ang(v, n):
        v = v / np.linalg.norm(v)
        return np.degrees(np.arccos(abs(v @ n)))

    ua_, ug_, uw_ = path[1] - path[0], path[2] - path[1], path[3] - path[2]
    rows = [
        ("air", 1.0, ang(ua_, nb_), ang(ua_, nf_)),
        ("glass", rig1.n_glass, ang(ug_, nb_), ang(ug_, nf_)),
        ("water", rig1.n_water, ang(uw_, nb_), ang(uw_, nf_)),
    ]
    import marimo as _mo

    table = "| layer | n | angle to air-side normal | angle to water-side normal | n·sin(angle to air-side) | n·sin(angle to water-side) |\n|---|---|---|---|---|---|\n"
    for nm, n_, a1, a2 in rows:
        table += f"| {nm} | {n_:.3f} | {a1:.3f}° | {a2:.3f}° | {n_ * np.sin(np.radians(a1)):.5f} | {n_ * np.sin(np.radians(a2)):.5f} |\n"
    _mo.md(
        "**Snell's invariant.** Across the air-side surface, `n·sin θ` (to its own normal) must be the same in air and in "
        "glass; across the water-side surface, in glass and in water. With parallel surfaces the two normals are one "
        "vector, so the product is constant through all three layers (the first and third column pairs below are "
        "identical) and the whole path is a single curve in one plane. That is the fact the radial-shift model is built "
        "on. Tilt a surface and the columns separate: there is no single invariant any more.\n\n" + table
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. How the tracer finds the ray

    The question the tracer answers is the *forward* one: **for a particle at `P`, in which direction does the
    camera have to look so that the ray, after both surfaces, ends at `P`?**

    1. Pick a trial direction `u` in the air (two unknowns: two angles).
    2. Intersect with the air-side plane, refract with vector Snell, `v = η u + (η cosθᵢ − cosθₜ) n`, `η = nₐ/n_b`.
    3. Intersect with the water-side plane, refract again.
    4. Measure how far the final ray passes from `P` (perpendicular miss, 3 numbers, 2 unknowns).
    5. Gauss-Newton on the two angles; 4-6 iterations to 1e-9 mm. (`air_direction`, vectorised over all points.)

    The image is then the pinhole-plus-lens-distortion projection of `C + u·|P−C|`, the *apparent* point. Nothing
    in steps 1-5 asks for parallel surfaces: the two planes are just a point and a normal each.

    What the radial-shift model does instead: it assumes the path lies in one plane through the glass axis, writes
    `r = (z_c − d)·tan β₁ − z_p·tan β₃ + d·tan β₂` for the radial distance at the particle, and iterates the single
    unknown `r_q` (the apparent radius) until `r_q` matches. The result is one factor, `r_q / r`, applied to the
    radial coordinates in a look-up table. One scalar per point, hence the speed.

    **They solve the same equation when the surfaces are parallel.** Check:
    """)
    return


@app.cell
def _(CAM, np, plt, project_pixels, rig0, time):
    rng_ = np.random.default_rng(1)
    vol_ = rng_.uniform(-30, 30, (3000, 3))
    t0 = time.perf_counter()
    px_t_ = rig0.project(vol_)[:, CAM]
    t_tracer = time.perf_counter() - t0
    t0 = time.perf_counter()
    px_l_ = project_pixels(vol_, rig0.cals[CAM], rig0.model_cpar())
    t_lib = time.perf_counter() - t0
    diff_ = np.hypot(*(px_t_ - px_l_).T)
    fig2, ax2 = plt.subplots(figsize=(6.5, 3.2))
    ax2.hist(diff_[np.isfinite(diff_)], bins=50, color="#4f7cac")
    ax2.set_xlabel("|tracer − library| (pixels)")
    ax2.set_ylabel("points (of 3000)")
    ax2.set_title(f"ideal flat window: max {np.nanmax(diff_):.4f} px, median {np.nanmedian(diff_):.5f} px")
    fig2.tight_layout()
    print(f"time for 3000 points, one camera: tracer {t_tracer * 1e3:.0f} ms, library {t_lib * 1e3:.1f} ms")
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two models agree to well under a thousandth of a pixel, so for the ideal case each one checks the other. The
    library is about 100 times faster per point (a vectorised Gauss-Newton in NumPy against a compiled look-up),
    which is why the tracer is a *truth* and a test tool, not a replacement for the tracking loop.

    ## 3. What happens when the glass is not ideal

    Choose one deviation of the truth from the nominal model. Left: how many pixels the image of each volume point
    moves (camera 1). Right: what is **left** after the calibration has done what it can, i.e. after re-fitting the
    camera position, angles, `cc` and the principal point to the tracer's pixels over the volume, using the
    library's model. The first number is what a wrong assumption costs if you do not calibrate; the second is what
    still shows up after calibration.
    """)
    return


@app.cell
def _(mo):
    effect = mo.ui.dropdown(
        options={
            "fluid index +0.04": "n",
            "glass 4 mm thicker": "d",
            "front (water-side) tilt 1.2°": "tf",
            "back (air-side) tilt 1.2°  (wedge)": "tb",
            "back tilt 0.3° (small wedge)": "tb_small",
        },
        value="back (air-side) tilt 1.2°  (wedge)",
        label="truth deviates by",
    )
    effect
    return (effect,)


@app.cell
def _(CAM, copy, effect, least_squares, np, plt, project_pixels, rig0):
    kw = {
        "n": dict(n_water=1.37),
        "d": dict(glass_thickness=10.0),
        "tf": dict(tilt_front=[(0.021, 0)] * 4),
        "tb": dict(tilt_back=[(0.021, 0)] * 4),
        "tb_small": dict(tilt_back=[(0.005, 0)] * 4),
    }[effect.value]
    truth = rig0.with_(**kw)
    rng3 = np.random.default_rng(2)
    vol3 = rng3.uniform(-30, 30, (3000, 3))
    px_truth = truth.project(vol3)[:, CAM]
    px_nom = rig0.project(vol3)[:, CAM]
    ok3 = np.isfinite(px_truth[:, 0]) & np.isfinite(px_nom[:, 0])
    cpar3 = rig0.model_cpar()
    cal3 = copy.deepcopy(rig0.cals[CAM])
    p0 = np.r_[cal3.get_pos(), cal3.get_angles(), cal3.int_par.cc, cal3.int_par.xh, cal3.int_par.yh]

    def resid3(p):
        cal3.set_pos(p[:3])
        cal3.set_angles(p[3:6])
        cal3.int_par.cc, cal3.int_par.xh, cal3.int_par.yh = p[6], p[7], p[8]
        cal3.invalidate_mmlut()
        return (project_pixels(vol3[ok3], cal3, cpar3) - px_truth[ok3]).ravel()

    sol = least_squares(resid3, p0, x_scale=np.r_[[1.0] * 3, [1e-3] * 3, [1.0], [0.1, 0.1]])
    r_after = sol.fun.reshape(-1, 2)
    mag_before = np.hypot(*(px_truth[ok3] - px_nom[ok3]).T)
    mag_after = np.hypot(*r_after.T)
    fig3, ax3 = plt.subplots(1, 2, figsize=(11, 4.2))
    for _a, _mag, _title in (
        (ax3[0], mag_before, f"image shift without calibrating (median {np.median(mag_before):.2f}, max {mag_before.max():.2f} px)"),
        (ax3[1], mag_after, f"left after re-fitting the camera (rms {np.sqrt(np.mean(mag_after**2)):.3f}, max {mag_after.max():.2f} px)"),
    ):
        sc = _a.scatter(px_truth[ok3, 0], px_truth[ok3, 1], c=_mag, s=5, cmap="viridis")
        _a.invert_yaxis()
        _a.set_aspect("equal")
        _a.set_title(_title, fontsize=9)
        fig3.colorbar(sc, ax=_a, label="px")
    fig3.tight_layout()
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading it.** Every one of these deviations moves the image by 1-7 px, which is huge, and a camera re-fit absorbs
    almost all of it (0.01-0.07 px left for an index or thickness error; about 0.2 px for a 1.2°
    wedge; all as the rms of the pixel distance). That is the calibration doing its job, *and* the reason it is easy to be wrong without noticing: the
    residual pixels look fine while the camera position, `cc` and the principal point have been moved to hide a
    physical effect. They are not at their true values any more. (This is why the self-calibration studies compare
    against the truth in image space and not by reprojection error.)

    Notice what the leftover looks like (right panel): it is **not** random noise but a smooth pattern across the
    sensor, and its size relative to the visible shift differs by cause: about 0.4 % of the shift for a thickness
    error, 1.5 % for an index error, 2 % for a tilt. A tilt or a wedge is therefore the least well absorbed of the
    four. A model without a tilt parameter cannot remove it; one with it might (idea b below).

    ## 4. What this suggests for openptv2's multimedia model

    I want to keep **measured** and **idea** apart.

    **Measured (this repository, this notebook):**

    1. For flat parallel windows the radial-shift model is correct to under 0.001 px. Nothing to gain there; the
       library's model is the faster of the two by about 100×.
    2. For tilted or wedge windows the radial-shift model *cannot* be right, and after camera re-fitting 0.05-0.22 px
       (rms of the pixel distance) remain on this rig for a 0.3-1.2° tilt. At the usual detection noise
       (0.1-0.3 px) this is below the noise of one point but systematic across all points of a camera.
    3. Index and thickness errors, the other things the model assumes known, are absorbed to 0.01-0.05 px but are
       **not recovered**: in the synthetic cavity study a +1 mm glass distance came back as +1 mm after the fit
       (`docs/self-calibration-cavity-synthetic.md`, section 3).

    **Ideas, not tested:**

    * **(a) Exact backward rays.** Going from a pixel to the ray in the water needs no iteration: undistort the pixel to
      a direction in air, then apply two vector refractions (`_refract`, `_hit` in `synthetic_rig.py`). The look-up table could be replaced
      by this for any window orientation, including per-camera tilt, with no interpolation error. The cost is the
      two refractions per ray, which is cheap compared to the radial iteration.
    * **(b) Tilt as calibration parameters.** The model has glass thickness and glass vector, but no tilt. The tracer
      shows that a wedge leaves a structured 0.05-0.2 px; a free tilt (2 parameters per camera) in the self-calibration
      could capture it. Whether the data can identify it against the camera angles is the open question. The
      synthetic cavity is the right place to test it, since the truth is known: recover a planted 0.3° tilt, and
      check that a 0° truth stays at 0°.
    * **(c) Test the look-up table.** The tracker uses an interpolated table (`init_mmlut`), not the iterative
      solution. The tracer is an independent reference to measure the table's error over the whole volume,
      especially at the corners and for the opposite-sides rigs, where `z` of the camera is negative.
    * **(d) A better *truth* for the benchmarks.** Synthetic benchmarks that render with the library's own model cannot
      reveal a flaw in it. Rendering with the tracer, with realistic wedge and thickness errors, tests the whole chain
      honestly. That is already how `TruthRig` is used here.

    **What the tracer does not replace.** It still needs the same inputs (indices, thickness, glass vector), so it is
    not "better physics" in itself: it is the same physics without the flat-parallel shortcut. For the usual rig the
    shortcut is exact. The gain, if any, is in the cases that depart from it, and (b) and (c) above are the cheapest
    ways to find out whether this rig class has any.
    """)
    return


if __name__ == "__main__":
    app.run()
