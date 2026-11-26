"""Is the calibration consistent between the cameras? A truth-free check on a finished run, and what to do about it.

When all four cameras see a particle, four rays meet at one point (up to blob noise). When a camera drops out, three rays
meet at a slightly different point if the calibration is not perfectly consistent: the position *jumps* when a track
changes from "4 cameras" to "3 cameras, camera m missing". Averaged over many particles, the jump is a systematic
offset that a velocity does not cancel, and it is the cheapest direct measure of calibration quality a recording offers.

``consistency_report`` measures it per missing camera, together with the smooth residual field of every camera (how far
its blob centres lie from the projection of the 4-camera position, as a function of position) and whether that field is
real (the same on two independent halves of the frames) or noise. ``verdict`` turns it into an action:

* ``inconclusive``  the offsets are not larger than the noise of this sample (too few points / frames): measure on more frames
* ``consistent``    nothing to do (a perfect calibration shows 1-2 um here, a good real one 10-20 um)
* ``marginal``      a small systematic offset: ``align_partial_points`` (explicit, marked, optional) or leave as is
* ``inconsistent``  re-calibrate: ``self_calibrate_from_store`` fits the cameras on the tracers of the run itself

The check is cheap (seconds) and works on any run with >= 4 cameras; with fewer cameras it reports ``not applicable``.
Validation on rendered images with a known calibration error and on the calibration ladder: ``bench/calibration_lab``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openptv2.loco import pixels_to_flat, project_pixels

#: thresholds on the largest systematic 3-vs-4-camera jump, in units of the image footprint (mm per pixel at the point)
CONSISTENT_PX = 0.05
INCONSISTENT_PX = 0.30


def collect_clean_points(store: Any, frames: list[int], n_cams: int, step: int = 1) -> dict:
    """Points seen by every camera on a blob of its own (a blob used by two points is skipped): 3D position ``pos``,
    blob centres ``centre`` (n, n_cams, 2) from the stored targets, and the ``frame`` of each."""
    pos_l, ctr_l, fr_l = [], [], []
    for f in list(frames)[::step]:
        try:
            pos, ids = store.read_correspondences(int(f))
        except Exception:  # noqa: BLE001 - a frame without correspondences simply has no points
            continue
        pos, ids = np.asarray(pos, float), np.asarray(ids, np.int64)
        if len(pos) == 0 or ids.shape[1] < n_cams:
            continue
        ids = ids[:, :n_cams]
        targs = []
        for c in range(n_cams):
            key = f"targets/cam_{c}/frame_{int(f):06d}"
            if key not in store.root:
                break
            targs.append(np.asarray(store.root[key], float))
        else:
            n = len(pos)
            centre = np.full((n, n_cams, 2), np.nan)
            shared = np.zeros((n, n_cams), bool)
            for c in range(n_cams):
                sel = np.flatnonzero((ids[:, c] >= 0) & (ids[:, c] < len(targs[c])))
                if len(sel):
                    centre[sel, c] = targs[c][ids[sel, c], 1:3]
                    _, inv, cnt = np.unique(ids[sel, c], return_inverse=True, return_counts=True)
                    shared[sel, c] = cnt[inv] >= 2
            ok = ((ids >= 0) & ~shared & np.isfinite(centre[:, :, 0])).all(axis=1) & np.isfinite(pos[:, 0])
            if ok.any():
                pos_l.append(pos[ok])
                ctr_l.append(centre[ok])
                fr_l.append(np.full(int(ok.sum()), int(f)))
    if not pos_l:
        return dict(pos=np.zeros((0, 3)), centre=np.zeros((0, n_cams, 2)), frame=np.zeros(0, int))
    return dict(pos=np.concatenate(pos_l), centre=np.concatenate(ctr_l), frame=np.concatenate(fr_l))


def _triangulate(centre: np.ndarray, cals: list, cpar: Any, vpar: Any, use: list[int]) -> np.ndarray:
    from openptv2.orientation import point_positions

    n_cams = len(cals)
    flat = np.full((len(centre), n_cams, 2), -1e10)
    for c in use:
        flat[:, c] = pixels_to_flat(centre[:, c], cals[c], cpar)
    return np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)


def _mm_per_px(pos: np.ndarray, cals: list, cpar: Any) -> float:
    """Footprint of one pixel (mm) at the points: shift them 1 mm sideways, see how far the image moves (median of cameras)."""
    p = pos[:2000]
    out = []
    for cal in cals:
        a = project_pixels(p, cal, cpar)
        shifts = [np.hypot(*(project_pixels(p + np.eye(3)[k], cal, cpar) - a).T) for k in (0, 1, 2)]
        out.append(np.nanmedian(np.max(shifts, axis=0)))
    return float(1.0 / max(np.nanmedian(out), 1e-9))


def inlier_mask(centre: np.ndarray, pos: np.ndarray, cals: list, cpar: Any, floor_px: float = 1.5, k: float = 3.0) -> tuple:
    """Points whose cameras agree: the rms reprojection residual over the cameras is at most ``max(floor_px, k * median)``.
    Matched quadruplets always contain some wrong matches (ghosts, merged blobs); they are not calibration information and
    would dominate a mean. Returns (mask, threshold in px)."""
    n = len(cals)
    r2 = np.mean([np.sum((project_pixels(pos, cals[c], cpar) - centre[:, c]) ** 2, axis=1) for c in range(n)], axis=0)
    r = np.sqrt(r2)
    fin = np.isfinite(r)
    thr = max(floor_px, k * float(np.median(r[fin]))) if fin.any() else floor_px
    return fin & (r <= thr), thr


@dataclass
class ConsistencyReport:
    applicable: bool = False
    n_points: int = 0
    n_frames: int = 0
    mm_per_px: float = float("nan")
    #: per camera: median / p90 of |blob centre - projection of the 4-camera position| (px), the smooth field's size (px),
    #: and the share of the residual variance a field fitted on half of the frames explains on the other half
    cameras: list = field(default_factory=list)
    #: per missing camera: systematic jump |mean(x_3cam - x_4cam)| and rms, in micrometres
    jump_um: list = field(default_factory=list)
    rms_um: list = field(default_factory=list)
    se_um: list = field(default_factory=list)        # standard error of each systematic jump (rms / sqrt(n)): the noise floor
    worst_jump_px: float = float("nan")      # the largest systematic jump in pixel footprints
    verdict: str = "not applicable"
    message: str = ""
    seconds: float = 0.0
    outlier_fraction: float = 0.0            # points left out because their cameras disagree (wrong matches)

    def text(self) -> str:
        if not self.applicable:
            return f"calibration consistency: {self.message}"
        lines = [f"calibration consistency: {self.verdict.upper()}  ({self.n_points} four-camera points in {self.n_frames} frames, "
                 f"1 px = {self.mm_per_px * 1000:.0f} um; {100 * self.outlier_fraction:.0f} % left out as wrong matches)"]
        lines.append("  camera   residual median/p90 (px)   smooth field (px)   field reproducible   jump when missing (um): systematic +- s.e. / rms")
        for c, (cam, j, r, se) in enumerate(zip(self.cameras, self.jump_um, self.rms_um, self.se_um)):
            lines.append(f"  {c + 1:>6}   {cam['res_med']:>8.3f} / {cam['res_p90']:<8.3f}   {cam['field_px']:>12.3f}   "
                         f"{('n/a' if not np.isfinite(cam['field_explained']) else format(100 * cam['field_explained'], '.0f') + ' %'):>16}   {j:>8.1f} +- {se:.1f} / {r:.1f}")
        lines.append(f"  worst systematic jump {self.worst_jump_px:.2f} px footprint -> {self.message}")
        return "\n".join(lines)


def consistency_report(
    store: Any,
    frames: list[int],
    cals: list,
    cpar: Any,
    vpar: Any,
    *,
    min_points: int = 300,
    max_points: int = 60000,
) -> ConsistencyReport:
    """Measure the camera consistency of a run from its stored correspondences and targets (see the module docstring)."""
    t0 = time.perf_counter()
    n_cams = len(cals)
    rep = ConsistencyReport()
    if n_cams < 4:
        rep.message = f"{n_cams} cameras: a camera-subset jump needs at least four"
        return rep
    d = collect_clean_points(store, frames, n_cams)
    if len(d["pos"]) < min_points:
        rep.message = f"only {len(d['pos'])} four-camera points on their own blobs (need {min_points}): too few to judge"
        return rep
    sel = np.random.default_rng(0).permutation(len(d["pos"]))[:max_points]
    pos, ctr, fr = d["pos"][sel], d["centre"][sel], d["frame"][sel]
    rep.applicable, rep.n_points, rep.n_frames = True, len(pos), len(np.unique(fr))
    rep.mm_per_px = _mm_per_px(pos, cals, cpar)

    # position from the pixels (the stored one can come from another step); then residuals and the field per camera
    x4 = _triangulate(ctr, cals, cpar, vpar, list(range(n_cams)))
    ok = np.isfinite(x4[:, 0])
    if ok.any():
        keep, _thr = inlier_mask(ctr[ok], x4[ok], cals, cpar)
        idx = np.flatnonzero(ok)
        rep.outlier_fraction = float(1.0 - keep.mean())
        ok = np.zeros(len(ok), bool)
        ok[idx[keep]] = True
    from openptv2.merge_refine import eval_4cam_residual_field, fit_4cam_residual_field

    odd = (fr % 2 == 1) & ok
    even = (fr % 2 == 0) & ok
    field_all = fit_4cam_residual_field(x4[ok], ctr[ok], cals, cpar)
    field_odd = fit_4cam_residual_field(x4[odd], ctr[odd], cals, cpar) if odd.sum() > 100 else field_all
    pred_all = eval_4cam_residual_field(x4[ok], field_all)
    pred_odd = eval_4cam_residual_field(x4[even], field_odd) if even.sum() else pred_all
    for c in range(n_cams):
        res = ctr[ok, c] - project_pixels(x4[ok], cals[c], cpar)
        rn = np.hypot(res[:, 0], res[:, 1])
        fn = np.hypot(pred_all[:, c, 0], pred_all[:, c, 1])
        # reproducibility: the field fitted on odd frames explains how much of the residual variance on even frames
        res_e = ctr[even, c] - project_pixels(x4[even], cals[c], cpar)
        left = res_e - pred_odd[:, c] if even.sum() else res_e
        explained = 1.0 - float(np.mean(np.sum(left**2, 1)) / max(np.mean(np.sum(res_e**2, 1)), 1e-12)) if even.sum() and odd.sum() > 100 else float('nan')
        rep.cameras.append(dict(res_med=float(np.median(rn)), res_p90=float(np.percentile(rn, 90)),
                                field_px=float(np.median(fn)), field_explained=max(explained, 0.0) if np.isfinite(explained) else float('nan')))
    # jump when camera m is missing
    for m in range(n_cams):
        use = [c for c in range(n_cams) if c != m]
        x3 = _triangulate(ctr, cals, cpar, vpar, use)
        diff = (x3 - x4)[ok & np.isfinite(x3[:, 0])]
        rep.jump_um.append(float(np.linalg.norm(np.mean(diff, axis=0)) * 1e3))
        rep.rms_um.append(float(np.sqrt(np.mean(np.sum(diff**2, axis=1))) * 1e3))
        rep.se_um.append(rep.rms_um[-1] / np.sqrt(max(len(diff), 1)))
    rep.worst_jump_px = max(rep.jump_um) / 1e3 / max(rep.mm_per_px, 1e-12)
    significant = max((j - 3.0 * se for j, se in zip(rep.jump_um, rep.se_um)), default=0.0) > 0
    if rep.worst_jump_px < CONSISTENT_PX:
        rep.verdict, rep.message = "consistent", "nothing to do"
    elif not significant:
        rep.verdict = "inconclusive"
        rep.message = ("the offsets are within 3 standard errors of the noise of this sample: use more points / frames "
                       f"(now {rep.n_points} points in {rep.n_frames} frames)")
    elif rep.worst_jump_px < INCONSISTENT_PX:
        rep.verdict, rep.message = "marginal", "small systematic offsets between camera subsets; align_partial_points() can remove them, or re-calibrate"
    else:
        rep.verdict, rep.message = "inconsistent", "re-calibrate: self_calibrate_from_store() fits the cameras on this run's tracers"
    rep.seconds = time.perf_counter() - t0
    return rep




# --------------------------------------------------------------------------------------------------------------------
# layer 2: re-calibrate from the tracers of the run itself


@dataclass
class RecalibrationResult:
    cals: list
    before: ConsistencyReport                # on the held-out frames, old cameras
    after: ConsistencyReport                 # on the same frames, new cameras
    rms_before_px: float = float("nan")      # reprojection rms of the fit points (px)
    rms_after_px: float = float("nan")
    scale: float = 1.0                       # linear scale of the volume, new cameras vs old (tracers cannot define it)
    scale_axes: tuple = (1.0, 1.0, 1.0)      # the three principal stretches of the old -> new map (isotropic when equal)
    stretch_dirs: Any = None                 # (3, 3): the directions (in lab coordinates) of those stretches, one per row
    anchored: bool = False                   # a KnownBody / Dumbbell fixed the scale
    shift_mm: float = 0.0                    # mean shift of the points, new vs old (mm)
    accepted: bool = False
    message: str = ""
    n_fit: int = 0
    cpar: Any = None                         # control parameters with the fitted multimedia (use them with ``cals``)
    yield_ratio: float = float("nan")        # 4-camera matches with the new cameras / with the old ones, on held-out frames
    multimedia: dict | None = None           # fitted glass thickness / indices when they were free

    def text(self) -> str:
        gauge = (f"\n  SCALE NOT ANCHORED: tracers cannot define the scale. Volume scale vs the old cameras: {self.scale:.5f} "
                 f"(principal stretches {', '.join(f'{a:.5f}' for a in self.scale_axes)}). A uniform scale error s scales every velocity "
                 f"and length by s and leaves strain rates and vorticity unchanged; unequal stretches do change them. See docs/self-calibration-scale-bias.md."
                 if not self.anchored else "")
        return (f"self-calibration {'ACCEPTED' if self.accepted else 'NOT accepted'}: {self.message}{gauge}\n"
                f"  fit: {self.n_fit} points, reprojection rms {self.rms_before_px:.3f} -> {self.rms_after_px:.3f} px\n"
                f"  matching yield with the new cameras (held-out frames): "
                f"{'not measured' if not np.isfinite(self.yield_ratio) else format(100 * self.yield_ratio, '.1f') + ' % of the old'}\n"
                f"  held-out frames: worst jump {self.before.worst_jump_px:.2f} -> {self.after.worst_jump_px:.2f} px footprint "
                f"({max(self.before.jump_um):.1f} -> {max(self.after.jump_um):.1f} um)\n"
                f"  volume vs old cameras: scale {self.scale:.5f}, mean shift {self.shift_mm:.3f} mm")


def matching_yield(store: Any, frames: list[int], cals: list, cpar: Any, vpar: Any) -> tuple:
    """(4-camera matches, all matches) of the stored blobs of ``frames`` matched with ``cals`` (nothing is written). The matcher
    is the one of the pipeline; ``None`` when the store has no path or the matching fails."""
    from openptv2.correspondences import match_correspondences_batch_parallel

    path = getattr(store, "store_path", None)
    if path is None or not frames:
        return None
    try:
        res = match_correspondences_batch_parallel(frames=list(frames), cpar=cpar, cals=list(cals), vpar=vpar,
                                                   zarr_store_path=str(path), n_workers=1, write_to_store=False)
    except Exception:  # noqa: BLE001 - the yield check is an extra safeguard, not a reason to fail
        return None
    quads = total = 0
    for _pos, ids in res.values():
        ids = np.asarray(ids)[:, : len(cals)]
        quads += int((ids >= 0).all(axis=1).sum())
        total += len(ids)
    return quads, total


def self_calibrate_from_store(
    store: Any,
    frames: list[int],
    cals: list,
    cpar: Any,
    vpar: Any,
    *,
    max_points: int = 12000,
    min_improvement: float = 0.5,
    min_yield: float = 1.0,            # 4-camera matches with the new cameras / the old ones, on held-out frames; 0 = check off
    body: Any = None,
    dumbbell: Any = None,
    **selfcal_kwargs: Any,
) -> RecalibrationResult:
    """Fit the cameras (exterior, principal point, k1 k2, p1 p2) on the clean 4-camera points of the run, using the odd
    frames, and judge the result on the even frames: the new cameras are ``accepted`` only when the worst camera-subset
    jump on held-out frames falls by at least ``min_improvement`` (a fraction) AND matching again with the new cameras
    does not lose 4-camera matches (``min_yield``, ratio on held-out frames; 0 switches it off). The second test exists
    because the jump is measured on points the OLD cameras matched, so it can improve while the calibration gets worse: on
    lv_multi it did, and tracking lost 1-10 % of the good trajectories. Nothing is written or overwritten; the input cameras
    are left as they are.

    Tracers fix the consistency between the cameras, not the absolute scale: ``scale`` and ``shift_mm`` show how much
    the volume moved; pass a ``KnownBody`` / ``Dumbbell`` (see ``openptv2.selfcal``) to anchor it. After new cameras
    the correspondences have to be matched again (the stored points belong to the old ones)."""
    from openptv2.selfcal import joint_self_calibrate

    n_cams = len(cals)
    frames = list(frames)
    odd = [f for f in frames if f % 2 == 1]
    even = [f for f in frames if f % 2 == 0]
    before = consistency_report(store, even, cals, cpar, vpar)
    if not before.applicable:
        return RecalibrationResult(cals, before, before, message=before.message)
    d = collect_clean_points(store, odd, n_cams)
    if len(d["pos"]) < 300:
        return RecalibrationResult(cals, before, before, message=f"only {len(d['pos'])} usable points on the odd frames")
    sel = np.random.default_rng(1).permutation(len(d["pos"]))[:max_points]
    ctr = d["centre"][sel]
    x0 = _triangulate(ctr, cals, cpar, vpar, list(range(n_cams)))
    ok = np.isfinite(x0[:, 0])
    if ok.any():
        keep, _thr = inlier_mask(ctr[ok], x0[ok], cals, cpar)
        idx = np.flatnonzero(ok)
        ok = np.zeros(len(ok), bool)
        ok[idx[keep]] = True
    kw = dict(max_nfev=200, free_radial=("k1", "k2"), ext_sigma_scale=3.0)
    kw.update(selfcal_kwargs)
    res = joint_self_calibrate(ctr[ok], x0[ok], cals, cpar, body=body, dumbbell=dumbbell, vpar=vpar, **kw)
    cpar2 = res.cpar if res.cpar is not None else cpar
    after = consistency_report(store, even, res.cals, cpar2, vpar)
    # what the new cameras do to the volume: best affine map between the old and the new triangulation
    xn = _triangulate(ctr[ok], res.cals, cpar2, vpar, list(range(n_cams)))
    good = np.isfinite(xn[:, 0])
    a_ = np.c_[x0[ok][good], np.ones(int(good.sum()))]
    coef = np.linalg.lstsq(a_, xn[good], rcond=None)[0]
    _u, stretches, _vt = np.linalg.svd(coef[:3])
    order = np.argsort(stretches)
    stretch_dirs = _u[:, order].T
    scale = float(np.prod(stretches) ** (1 / 3))
    shift = float(np.linalg.norm(np.mean(xn[good] - x0[ok][good], axis=0)))
    worst_b, worst_a = max(before.jump_um), max(after.jump_um)
    already_fine = worst_b / 1e3 / max(before.mm_per_px, 1e-12) < CONSISTENT_PX
    accepted = (not already_fine) and worst_a <= (1.0 - min_improvement) * worst_b
    if accepted:
        msg = "the camera-subset jump on held-out frames fell"
        if body is None and dumbbell is None and abs(scale - 1.0) > 1e-3:
            msg += f"; the scale moved by {100 * (scale - 1):+.2f} % and no anchor was given: check it against a known length"
    elif already_fine:
        msg = "the cameras were already consistent: nothing to gain (a fit would only move the scale)"
    else:
        msg = "no clear improvement on held-out frames: the inconsistency is not a camera-parameter error (keep the old cameras)"
    yield_ratio = float("nan")
    if accepted and min_yield > 0:
        sample = even[:: max(1, len(even) // 4)][:4]
        y_old = matching_yield(store, sample, cals, cpar, vpar)
        y_new = matching_yield(store, sample, res.cals, cpar2, vpar)
        if y_old and y_new and y_old[0] > 0:
            yield_ratio = y_new[0] / y_old[0]
            if yield_ratio < min_yield:
                accepted = False
                msg = (f"matching again with the new cameras finds {100 * (1 - yield_ratio):.1f} % fewer 4-camera matches on held-out "
                       f"frames ({y_new[0]} against {y_old[0]}): a smaller jump among the points the old cameras matched is not "
                       f"a better calibration (lower min_yield to accept a loss of matches for more accuracy). Keep the old cameras")
    return RecalibrationResult(res.cals, before, after, res.rms_before_px, res.rms_after_px, scale,
                               tuple(float(v) for v in stretches[order]), stretch_dirs, body is not None or dumbbell is not None, shift,
                               accepted, msg, int(ok.sum()), cpar2, yield_ratio, res.multimedia)


def write_calibration(cals: list, folder: Any, *, prefix: str = "cam_", suffix: str = ".tif") -> list:
    """Write ``cals`` as ``<folder>/<prefix><n><suffix>.ori`` and ``.addpar`` (a new folder: never over the originals)."""
    from pathlib import Path

    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, cal in enumerate(cals):
        ori = out / f"{prefix}{i + 1}{suffix}.ori"
        cal.write(str(ori), str(ori).replace(".ori", ".addpar"))
        paths.append(ori)
    return paths


def _partial_positions(store: Any, frames: list[int], n_cams: int) -> dict:
    """{frame: positions (n, 3)} of the stored points, with the mask of the partial-camera ones (2 <= seen < n_cams)."""
    out = {}
    for f in frames:
        try:
            pos, ids = store.read_correspondences(int(f))
        except Exception:  # noqa: BLE001
            continue
        pos, ids = np.asarray(pos, float), np.asarray(ids)[:, :n_cams]
        seen = (ids >= 0).sum(1)
        out[int(f)] = (pos.copy(), (seen >= 2) & (seen < n_cams) & np.isfinite(pos[:, 0]))
    return out


# --------------------------------------------------------------------------------------------------------------------
# layer 3: the explicit stopgap: correct the partial-camera points when the cameras cannot be re-calibrated


@dataclass
class AlignResult:
    applied: bool
    n_updated: int = 0
    message: str = ""
    before: ConsistencyReport | None = None
    shift_median_um: float = float("nan")    # how far the partial-camera points moved (median, um)
    shift_p99_um: float = float("nan")
    four_cam_changed: int = 0                # 4-camera points whose position changed (must be 0)

    def text(self) -> str:
        head = f"align_partial_points: {'applied to ' + str(self.n_updated) + ' partial-camera points' if self.applied else 'not applied'}: {self.message}"
        if self.applied:
            head += (f"\n  partial-camera points moved by {self.shift_median_um:.1f} um (median), {self.shift_p99_um:.1f} um (99 %); "
                     f"4-camera points changed: {self.four_cam_changed}")
        return head


ALIGN_ATTR = "align_partial_points"
#: the stored 3D points must come from the cameras given: their median distance to the pixel triangulation, in pixel footprints
CALIBRATION_MATCH_PX = 0.5


def align_partial_points(
    store: Any,
    frames: list[int],
    cals: list,
    cpar: Any,
    vpar: Any,
    *,
    check_calibration: bool = True,
    force: bool = False,
    min_verdict: str = "marginal",
) -> AlignResult:
    """Subtract the smooth 4-camera residual field from the pixel centres of the 2- and 3-camera points and triangulate
    them again (``merge_refine.align_store_correspondences_4cam``), as an explicit, recorded step.

    Refuses when (1) the cameras are already consistent (``min_verdict``), (2) the stored points do not come from
    ``cals`` (a re-calibration without a new matching: the correction would move partial points by the calibration
    difference only; ``check_calibration``), or (3) it was applied before to this store (``force`` overrides). On success
    the run store root gets the attribute ``align_partial_points`` (frames and the size of the field) so the state of the
    points is not a secret.

    Use it when the validated scale and frame of the calibration must stay (4-camera points are not changed at all) or the
    cameras cannot be changed; avoid it when the cameras are ``inconsistent`` (re-calibrate instead), with the legacy
    ``trackcorr``, with few partial-camera points, or when most of them lie far from the 4-camera region. Full guidance:
    docs/calibration-consistency.md, section 3."""
    from openptv2.merge_refine import align_store_correspondences_4cam

    attrs = store.root.attrs
    if ALIGN_ATTR in attrs and not force:
        return AlignResult(False, 0, f"already applied to this store ({dict(attrs[ALIGN_ATTR])}); force=True to repeat")
    before = consistency_report(store, frames, cals, cpar, vpar)
    if not before.applicable:
        return AlignResult(False, 0, before.message, before)
    if check_calibration:
        d = collect_clean_points(store, frames, len(cals))
        sel = np.random.default_rng(2).permutation(len(d["pos"]))[:5000]
        x = _triangulate(d["centre"][sel], cals, cpar, vpar, list(range(len(cals))))
        gap = float(np.nanmedian(np.linalg.norm(x - d["pos"][sel], axis=1))) / max(before.mm_per_px, 1e-12)
        if gap > CALIBRATION_MATCH_PX:
            return AlignResult(False, 0, f"the stored points are {gap:.2f} px footprints away from the triangulation with these "
                               f"cameras: they were matched with other calibration files. Match again first.", before)
    order = ("consistent", "marginal", "inconsistent")
    if order.index(before.verdict) < order.index(min_verdict) and not force:
        return AlignResult(False, 0, f"the cameras are {before.verdict}: nothing to correct", before)
    old = _partial_positions(store, list(frames), len(cals))
    n = align_store_correspondences_4cam(store, list(frames), list(cals), cpar, vpar)
    new = _partial_positions(store, list(frames), len(cals))
    moved, changed4 = [], 0
    for f, (p0, m0) in old.items():
        p1, _m1 = new[f]
        moved.append(np.linalg.norm(p1[m0] - p0[m0], axis=1))
        changed4 += int((np.linalg.norm(p1[~m0] - p0[~m0], axis=1) > 1e-9).sum())
    mv = np.concatenate(moved) * 1e3 if moved else np.zeros(0)
    med = float(np.median(mv)) if len(mv) else float("nan")
    p99 = float(np.percentile(mv, 99)) if len(mv) else float("nan")
    try:
        attrs[ALIGN_ATTR] = dict(frames=[int(min(frames)), int(max(frames))], n_updated=int(n),
                                 worst_jump_um_before=round(max(before.jump_um), 2))
    except Exception:  # noqa: BLE001 - a read-only store cannot carry the marker; the result still says what happened
        pass
    return AlignResult(True, int(n), "recorded in the store attributes", before, med, p99, changed4)


__all__ = [
    "AlignResult",
    "ConsistencyReport",
    "RecalibrationResult",
    "align_partial_points",
    "collect_clean_points",
    "consistency_report",
    "self_calibrate_from_store",
    "write_calibration",
]


# --------------------------------------------------------------------------------------------------------------------
# command line:  python -m openptv2.calibration_consistency {check,selfcal,align} <run folder> [--first N --last M]


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    import yaml

    from openptv2.recorrespond import load_parameters
    from openptv2.storage import RunStore

    ap = argparse.ArgumentParser(prog="python -m openptv2.calibration_consistency", description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("check", "selfcal", "align"),
                    help="check: report only | selfcal: fit new cameras on the run's tracers (writes a NEW folder) | "
                         "align: explicit correction of the partial-camera points (rewrites the stored correspondences, marked); keeps the "
                         "calibration's scale and frame, not for 'inconsistent' cameras or the legacy trackcorr: see docs/calibration-consistency.md")
    ap.add_argument("run", help="run folder (parameters_Run1.yaml, cal/, res/run.zarr)")
    ap.add_argument("--first", type=int)
    ap.add_argument("--last", type=int)
    ap.add_argument("--out", help="selfcal: folder for the new .ori/.addpar files (default <run>/cal_selfcal)")
    ap.add_argument("--min-yield", type=float, default=1.0,
                    help="selfcal: refuse new cameras that match fewer 4-camera points than the old ones on held-out frames (ratio, default 1.0; 0 = off)")
    ap.add_argument("--force", action="store_true", help="align: apply even if the cameras are consistent / it was applied before")
    a = ap.parse_args(argv)
    run = Path(a.run)
    yml = run / "parameters_Run1.yaml"
    cpar, vpar, cals, _ = load_parameters(yml)
    cals = list(cals)
    seq = (yaml.safe_load(yml.read_text()) or {}).get("sequence", {})
    f0 = a.first if a.first is not None else int(seq.get("first", 1))
    f1 = a.last if a.last is not None else int(seq.get("last", f0))
    frames = list(range(f0, f1 + 1))
    store = RunStore.open(run, mode="r" if a.action != "align" else "a")
    if a.action == "check":
        rep = consistency_report(store, frames, cals, cpar, vpar)
        print(rep.text())
        return 0 if rep.verdict in ("consistent", "not applicable") else 1
    if a.action == "selfcal":
        res = self_calibrate_from_store(store, frames, cals, cpar, vpar, min_yield=a.min_yield)
        print(res.text())
        if res.accepted:
            out = Path(a.out) if a.out else run / "cal_selfcal"
            paths = write_calibration(res.cals, out)
            print(f"new cameras written to {out} ({len(paths)} files); the originals are untouched. Point the yaml's cal_ori at them "
                  f"and match the frames again before tracking (the stored points belong to the old cameras).")
        return 0 if res.accepted else 1
    res = align_partial_points(store, frames, cals, cpar, vpar, force=a.force, min_verdict="consistent" if a.force else "marginal")
    print(res.text())
    return 0 if res.applied else 1


if __name__ == "__main__":
    raise SystemExit(main())
