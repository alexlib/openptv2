"""Reconstruct trackcorr's real per-particle candidate search for interactive
debugging (see docs/plans/2026-08-21-trackcorr-interactive-debug.md).

trackcorr only (track_mode=0). The compiled kernel
(``track_kernels_corr.trackcorr_loop_fast``) writes each particle's
*accepted* candidates -- ones that passed the angle/acceleration gate --
into ``Frame.path_decis``/``path_linkdecis``/``path_inlist`` before
``trackcorr_c_loop`` rotates the 4-frame ring buffer away
(``fb.fb_next()``). This module calls the exact same compiled kernel
`trackcorr_c_loop` does, with the exact same packed arguments, but snapshots
those arrays *before* the rotation instead of letting them be overwritten by
the next step -- so what this module reports is the real trackcorr
decision, not a reimplementation of the search.

Candidates that were found in the search box but *rejected* by the
angle/acc gate are not recorded by the compiled kernel and so are not
visible here; only the accepted, ranked candidate set (including the
winner) is.
"""

from __future__ import annotations

import numpy as np

from openptv2.algorithms.constants import CORRES_NONE, NEXT_NONE, POSI
from openptv2.algorithms.parameters import convert_track_par_to_tuple
from openptv2.algorithms.track import (
    _pack_cams_fast,
    _pack_cams_fast_tuples,
    _sync_soa_to_aos,
    track_forward_start,
)
from openptv2.algorithms.track_kernels import (
    trackcorr_loop_fast as _trackcorr_loop_fast,
)
from openptv2.algorithms.tracking_run import TrackingRun


def _trackcorr_c_loop_capture(run_info, step, num_threads=1):
    """Verbatim copy of ``track.trackcorr_c_loop``'s real search call, with a
    capture point inserted between the compiled kernel call and
    ``fb.fb_next()`` (the ring-buffer rotation that would otherwise
    overwrite path_decis/path_linkdecis/path_inlist for this step before we
    can read them). Everything else -- packing, the kernel call itself,
    bookkeeping, rotation -- is identical to the real loop, so tracking
    state after this call is exactly what a real ``trackcorr_c_loop(step)``
    would leave it at.

    Returns a dict snapshot for this step (see ``candidates_for_particle``
    for the shape) plus (count1, num_added) as the real loop reports.
    """
    fb = run_info.fb
    cal = run_info.cal
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar

    c_imx = cpar.imx
    c_imy = cpar.imy
    imx_half = c_imx * 0.5
    imy_half = c_imy * 0.5
    inv_pix_x = 1.0 / cpar.pix_x
    inv_pix_y = 1.0 / cpar.pix_y
    c_chfield = cpar.chfield
    c_mm = cpar.mm

    fast_cals, fast_mmluts = _pack_cams_fast(cal, c_mm)
    cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t = _pack_cams_fast_tuples(fast_cals, fast_mmluts)
    cal_arr = np.asarray(list(cal_t), dtype=np.float64)
    md_arr = list(md_t)
    mo_arr = np.asarray(list(mo_t), dtype=np.float64)
    mnr_arr = np.array(list(mnr_t), dtype=np.int32)
    mnz_arr = np.array(list(mnz_t), dtype=np.int32)
    mrw_arr = np.array(list(mrw_t), dtype=np.float64)

    nc = fb.num_cams
    orig_parts = fb.buf[1].num_parts

    fb.buf[0]._sync_path_to_soa()
    fb.buf[1]._sync_path_to_soa()
    fb.buf[2]._sync_path_to_soa()
    fb.buf[3]._sync_path_to_soa()

    np2 = np.array([fb.buf[2].num_parts], dtype=np.int32)
    np3 = np.array([fb.buf[3].num_parts], dtype=np.int32)
    nt2 = np.array(fb.buf[2].num_targets[:nc], dtype=np.int32)
    nt3 = np.array(fb.buf[3].num_targets[:nc], dtype=np.int32)

    count1, num_added = _trackcorr_loop_fast(
        orig_parts,
        fb.buf[0].path_x,
        fb.buf[1].path_x,
        fb.buf[1].path_prev,
        fb.buf[1].path_next,
        fb.buf[1].path_inlist,
        fb.buf[1].path_finaldecis,
        fb.buf[1].path_decis,
        fb.buf[1].path_linkdecis,
        fb.buf[1].corres_p,
        fb.buf[1].targ_x,
        fb.buf[1].targ_y,
        fb.buf[1].targ_tnr,
        fb.buf[2].path_x,
        fb.buf[2].path_prev,
        fb.buf[2].path_next,
        fb.buf[2].path_inlist,
        fb.buf[2].path_prio,
        fb.buf[2].path_finaldecis,
        fb.buf[2].path_decis,
        fb.buf[2].path_linkdecis,
        fb.buf[2].corres_p,
        fb.buf[2].corres_nr,
        fb.buf[2].targ_x,
        fb.buf[2].targ_y,
        fb.buf[2].targ_tnr,
        nt2,
        np2,
        fb.buf[3].path_x,
        fb.buf[3].path_prev,
        fb.buf[3].path_next,
        fb.buf[3].path_inlist,
        fb.buf[3].path_prio,
        fb.buf[3].path_finaldecis,
        fb.buf[3].path_decis,
        fb.buf[3].path_linkdecis,
        fb.buf[3].corres_p,
        fb.buf[3].corres_nr,
        fb.buf[3].targ_x,
        fb.buf[3].targ_y,
        fb.buf[3].targ_tnr,
        nt3,
        np3,
        cal_arr,
        md_arr,
        mo_arr,
        mnr_arr,
        mnz_arr,
        mrw_arr,
        tpar.dvxmin,
        tpar.dvxmax,
        tpar.dvymin,
        tpar.dvymax,
        tpar.dvzmin,
        tpar.dvzmax,
        tpar.dacc,
        tpar.dangle,
        int(tpar.add),
        run_info.lmax,
        vpar.X_lay[0],
        vpar.X_lay[1],
        run_info.ymin,
        run_info.ymax,
        vpar.Zmin_lay[0],
        vpar.Zmax_lay[1],
        nc,
        imx_half,
        imy_half,
        inv_pix_x,
        inv_pix_y,
        c_chfield,
        float(c_imx),
        float(c_imy),
        cpar.pix_x,
        cpar.pix_y,
        run_info.flatten_tol,
        num_threads,
    )

    # --- Capture point: fb.buf[1]/[2] still hold `step`'s data, pre-rotation. ---
    snapshot = {
        "step": step,
        "num_cams": nc,
        "num_parts_1": orig_parts,
        "path_x_1": np.array(fb.buf[1].path_x[:orig_parts]),
        "path_next_1": np.array(fb.buf[1].path_next[:orig_parts]),
        "path_inlist_1": np.array(fb.buf[1].path_inlist[:orig_parts]),
        "path_decis_1": np.array(fb.buf[1].path_decis[:orig_parts]),
        "path_linkdecis_1": np.array(fb.buf[1].path_linkdecis[:orig_parts]),
        "num_parts_2": int(np2[0]),
        "path_x_2": np.array(fb.buf[2].path_x[: int(np2[0])]),
        "corres_p_2": np.array(fb.buf[2].corres_p[: int(np2[0])]),
        "targ_x_2": [np.array(fb.buf[2].targ_x[c][: nt2[c]]) for c in range(nc)],
        "targ_y_2": [np.array(fb.buf[2].targ_y[c][: nt2[c]]) for c in range(nc)],
        "targ_tnr_2": [np.array(fb.buf[2].targ_tnr[c][: nt2[c]]) for c in range(nc)],
    }

    fb.buf[2].num_parts = int(np2[0])
    fb.buf[3].num_parts = int(np3[0])

    _sync_soa_to_aos(fb.buf[1])
    _sync_soa_to_aos(fb.buf[2])
    _sync_soa_to_aos(fb.buf[3])

    print(
        f"step: {step}, curr: {fb.buf[1].num_parts}, "
        f"next: {fb.buf[2].num_parts}, links: {count1}, "
        f"lost: {fb.buf[1].num_parts - count1}, add: {num_added}"
    )

    run_info.npart = run_info.npart + fb.buf[1].num_parts
    run_info.nlinks = run_info.nlinks + count1

    fb.fb_next()
    fb.write_frame_from_start(step)
    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, read_links=False)
    else:
        fb.buf[fb.buf_len - 1].num_parts = 0

    return snapshot, count1, num_added


def load_run(cpar, spar, vpar, tpar, cals, flatten_tol=0.0001, max_targets=10000):
    """Build a fresh TrackingRun ready for forward stepping (mirrors
    ``openptv2.tracker.Tracker.restart()``, minus store/naming plumbing this
    debug tool doesn't need). Reads the run's own res/rt_is.*, res/ptv_is.*,
    res/added.* -- the standard file bases -- from the current working
    directory (same convention as ``openptv2.tracker.default_naming``); this
    does not write any output back."""
    tpar_tuple = convert_track_par_to_tuple(tpar)
    run = TrackingRun(
        seq_par=spar,
        tpar=tpar_tuple,
        vpar=vpar,
        cpar=cpar,
        buf_len=4,
        max_targets=max_targets,
        corres_file_base="res/rt_is",
        linkage_file_base="res/ptv_is",
        prio_file_base="res/added",
        cal=list(cals),
        flatten_tol=flatten_tol,
    )
    track_forward_start(run)
    return run


def step_and_capture(run, first, last):
    """Step trackcorr forward from `first` to `last` (inclusive of the last
    *transition*, i.e. last-1 -> last), capturing each step's real candidate
    data. Returns {frame_number: snapshot}, keyed by the frame the particles
    in `path_x_1` belong to (== the step number).

    After this call, ``run.fb.buf[1]`` is positioned at frame `last`,
    unprocessed -- ready for ``probe_particle(run, last, ...)`` without any
    further stepping. Right after ``load_run()`` (no stepping at all),
    ``run.fb.buf[1]`` is already positioned at ``run.seq_par.first`` for the
    same reason (``track_forward_start`` leaves it there)."""
    snapshots = {}
    for step in range(first, last):
        snap, _count1, _num_added = _trackcorr_c_loop_capture(run, step)
        snapshots[step] = snap
    return snapshots


def probe_particle(
    run,
    step,
    particle_index,
    *,
    dvxmin=None,
    dvxmax=None,
    dvymin=None,
    dvymax=None,
    dvzmin=None,
    dvzmax=None,
    dacc=None,
    dangle=None,
    num_threads=1,
):
    """Fast, real-kernel-backed candidate recompute for ONE particle, for
    "tune parameters then press Run" workflows -- avoids the ~7.5s (for a
    ~700-particle frame) cost of ``_trackcorr_c_loop_capture``'s full-frame
    real step by slicing the frame-1 side of the exact same compiled kernel
    call down to just this one particle. Each `dv*`/`dacc`/`dangle` kwarg
    overrides the run's configured value for this probe only; omitted ones
    use ``run.tpar``'s real value.

    Precondition: ``run.fb.buf[1]`` must currently be positioned at `step`,
    unprocessed -- true right after ``load_run()`` (for
    ``step == run.seq_par.first``) or after
    ``step_and_capture(run, first, step)`` for any later `step`. This
    function does not advance or mutate `run`'s real tracking state (frame
    2/3 arrays are copied before the kernel call, never written back), so
    any number of probe calls -- and different parameter values -- can be
    made for the same `step` before a real step is taken through it.

    Returns a ParticleSearchResult, same shape as
    ``candidates_for_particle()`` returns from a real captured snapshot.
    """
    fb = run.fb
    tpar = run.tpar
    vpar = run.vpar
    cpar = run.cpar
    cal = run.cal

    if particle_index < 0 or particle_index >= fb.buf[1].num_parts:
        raise IndexError(
            f"particle {particle_index} out of range "
            f"(frame {step} has {fb.buf[1].num_parts} particles)"
        )

    c_imx = cpar.imx
    c_imy = cpar.imy
    imx_half = c_imx * 0.5
    imy_half = c_imy * 0.5
    inv_pix_x = 1.0 / cpar.pix_x
    inv_pix_y = 1.0 / cpar.pix_y
    c_chfield = cpar.chfield
    c_mm = cpar.mm

    fast_cals, fast_mmluts = _pack_cams_fast(cal, c_mm)
    cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t = _pack_cams_fast_tuples(fast_cals, fast_mmluts)
    cal_arr = np.asarray(list(cal_t), dtype=np.float64)
    md_arr = list(md_t)
    mo_arr = np.asarray(list(mo_t), dtype=np.float64)
    mnr_arr = np.array(list(mnr_t), dtype=np.int32)
    mnz_arr = np.array(list(mnz_t), dtype=np.int32)
    mrw_arr = np.array(list(mrw_t), dtype=np.float64)

    nc = fb.num_cams

    # Frame 1: a single throwaway row for the particle being probed -- built
    # fresh, never touches the live fb.buf[1] arrays. path_prev indexes into
    # buf[0], which we pass through unmodified (read-only, real data).
    path_x_1 = np.array([fb.buf[1].path_x[particle_index]], dtype=np.float64)
    path_prev_1 = np.array([fb.buf[1].path_prev[particle_index]], dtype=np.int32)
    path_next_1 = np.full(1, NEXT_NONE, dtype=np.int32)
    path_inlist_1 = np.zeros(1, dtype=np.int32)
    path_finaldecis_1 = np.full(1, 1e6, dtype=np.float64)
    path_decis_1 = np.zeros((1, POSI), dtype=np.float64)
    path_linkdecis_1 = np.zeros((1, POSI), dtype=np.int32)
    corres_p_1 = np.array([fb.buf[1].corres_p[particle_index]], dtype=np.int32)

    # Frames 2/3: fresh copies. The kernel may write link/add-particle
    # results into these -- copies mean the live run is never touched.
    def _frame_copy(frm):
        return {
            "path_x": np.array(frm.path_x),
            "path_prev": np.array(frm.path_prev),
            "path_next": np.array(frm.path_next),
            "path_inlist": np.array(frm.path_inlist),
            "path_prio": np.array(frm.path_prio),
            "path_finaldecis": np.array(frm.path_finaldecis),
            "path_decis": np.array(frm.path_decis),
            "path_linkdecis": np.array(frm.path_linkdecis),
            "corres_p": np.array(frm.corres_p),
            "corres_nr": np.array(frm.corres_nr),
            "targ_x": np.array(frm.targ_x),
            "targ_y": np.array(frm.targ_y),
            "targ_tnr": np.array(frm.targ_tnr),
            "num_targets": np.array(frm.num_targets[:nc], dtype=np.int32),
            "num_parts": frm.num_parts,
        }

    f2 = _frame_copy(fb.buf[2])
    f3 = _frame_copy(fb.buf[3])
    np2 = np.array([f2["num_parts"]], dtype=np.int32)
    np3 = np.array([f3["num_parts"]], dtype=np.int32)

    def _v(override, real):
        return real if override is None else override

    count1, _num_added = _trackcorr_loop_fast(
        1,  # orig_parts -- just this one particle
        np.array(fb.buf[0].path_x),
        path_x_1,
        path_prev_1,
        path_next_1,
        path_inlist_1,
        path_finaldecis_1,
        path_decis_1,
        path_linkdecis_1,
        corres_p_1,
        fb.buf[1].targ_x,
        fb.buf[1].targ_y,
        fb.buf[1].targ_tnr,
        f2["path_x"],
        f2["path_prev"],
        f2["path_next"],
        f2["path_inlist"],
        f2["path_prio"],
        f2["path_finaldecis"],
        f2["path_decis"],
        f2["path_linkdecis"],
        f2["corres_p"],
        f2["corres_nr"],
        f2["targ_x"],
        f2["targ_y"],
        f2["targ_tnr"],
        f2["num_targets"],
        np2,
        f3["path_x"],
        f3["path_prev"],
        f3["path_next"],
        f3["path_inlist"],
        f3["path_prio"],
        f3["path_finaldecis"],
        f3["path_decis"],
        f3["path_linkdecis"],
        f3["corres_p"],
        f3["corres_nr"],
        f3["targ_x"],
        f3["targ_y"],
        f3["targ_tnr"],
        f3["num_targets"],
        np3,
        cal_arr,
        md_arr,
        mo_arr,
        mnr_arr,
        mnz_arr,
        mrw_arr,
        _v(dvxmin, tpar.dvxmin),
        _v(dvxmax, tpar.dvxmax),
        _v(dvymin, tpar.dvymin),
        _v(dvymax, tpar.dvymax),
        _v(dvzmin, tpar.dvzmin),
        _v(dvzmax, tpar.dvzmax),
        _v(dacc, tpar.dacc),
        _v(dangle, tpar.dangle),
        int(tpar.add),
        run.lmax,
        vpar.X_lay[0],
        vpar.X_lay[1],
        run.ymin,
        run.ymax,
        vpar.Zmin_lay[0],
        vpar.Zmax_lay[1],
        nc,
        imx_half,
        imy_half,
        inv_pix_x,
        inv_pix_y,
        c_chfield,
        float(c_imx),
        float(c_imy),
        cpar.pix_x,
        cpar.pix_y,
        run.flatten_tol,
        num_threads,
    )

    snapshot = {
        "step": step,
        "num_cams": nc,
        "num_parts_1": 1,
        "path_x_1": path_x_1,
        "path_next_1": path_next_1,
        "path_inlist_1": path_inlist_1,
        "path_decis_1": path_decis_1,
        "path_linkdecis_1": path_linkdecis_1,
        "num_parts_2": int(np2[0]),
        "path_x_2": np.array(f2["path_x"][: int(np2[0])]),
        "corres_p_2": np.array(f2["corres_p"][: int(np2[0])]),
        "targ_x_2": [f2["targ_x"][c][: f2["num_targets"][c]] for c in range(nc)],
        "targ_y_2": [f2["targ_y"][c][: f2["num_targets"][c]] for c in range(nc)],
        "targ_tnr_2": [f2["targ_tnr"][c][: f2["num_targets"][c]] for c in range(nc)],
    }
    result = candidates_for_particle(snapshot, 0, is_isolated=True)
    result.particle_index = particle_index  # report the real particle index, not 0
    return result


class ParticleCandidate:
    """One accepted candidate for a particle's link into the next frame."""

    __slots__ = ("row", "cost", "rank", "pos_3d", "cameras")

    def __init__(self, row, cost, rank, pos_3d, cameras):
        self.row = row  # row index into next frame's path_x/corres_p
        self.cost = cost  # trackcorr's rr cost (lower = better)
        self.rank = rank  # 0 = best (lowest cost)
        self.pos_3d = pos_3d
        self.cameras = cameras  # {cam_idx: (tnr, x, y)} for cams that saw it

    def __repr__(self):
        cams = ",".join(str(c) for c in sorted(self.cameras))
        return (
            f"ParticleCandidate(row={self.row}, rank={self.rank}, "
            f"cost={self.cost:.4g}, cams=[{cams}])"
        )


class ParticleSearchResult:
    """The trackcorr search result for one particle at one step.

    ``winner_row``/``winner`` mean different things depending on how this
    was produced:

    - From ``candidates_for_particle()`` on a real ``step_and_capture()``
      snapshot: the REAL link trackcorr made, resolved against every other
      particle competing in that step.
    - From ``probe_particle()``: only the lowest-cost candidate *for this
      one particle in isolation* -- ``is_isolated`` is True. trackcorr's
      real winner can legitimately differ, because two particles can both
      want the same next-frame target; the real run resolves that
      cross-particle competition globally, which a single-particle probe
      cannot see (confirmed empirically: probing gives an identical
      candidate set/costs to the real run, but the isolated best-by-cost
      candidate does not always match the real winner -- see
      tests/unit/test_trackcorr_debug.py). Read this as "what this
      particle would prefer if nothing else was competing for the same
      target", not as a prediction of the real link.
    """

    def __init__(self, step, particle_index, pos_3d, candidates, winner_row, is_isolated=False):
        self.step = step
        self.particle_index = particle_index
        self.pos_3d = pos_3d
        self.candidates = candidates  # list[ParticleCandidate], rank-ordered
        self.winner_row = winner_row  # row in next frame, or NEXT_NONE
        self.is_isolated = is_isolated

    @property
    def winner(self):
        for c in self.candidates:
            if c.row == self.winner_row:
                return c
        return None

    def __repr__(self):
        return (
            f"ParticleSearchResult(step={self.step}, particle={self.particle_index}, "
            f"pos_3d={tuple(round(v, 2) for v in self.pos_3d)}, "
            f"n_candidates={len(self.candidates)}, winner_row={self.winner_row})"
        )


def candidates_for_particle(snapshot, particle_index, is_isolated=False) -> ParticleSearchResult:
    """Read the real, already-computed candidate list for one particle from
    a step's snapshot -- no re-search, no approximation."""
    if particle_index < 0 or particle_index >= snapshot["num_parts_1"]:
        raise IndexError(
            f"particle {particle_index} out of range "
            f"(step has {snapshot['num_parts_1']} particles)"
        )

    n_in = int(snapshot["path_inlist_1"][particle_index])
    n_in = min(n_in, POSI)
    rows = snapshot["path_linkdecis_1"][particle_index, :n_in]
    costs = snapshot["path_decis_1"][particle_index, :n_in]

    order = np.argsort(costs)
    candidates = []
    for rank, idx in enumerate(order):
        row = int(rows[idx])
        cost = float(costs[idx])
        pos_3d = tuple(float(v) for v in snapshot["path_x_2"][row])
        cameras = {}
        for cam in range(snapshot["num_cams"]):
            ti = int(snapshot["corres_p_2"][row, cam])
            if ti == CORRES_NONE:
                continue
            tnr = int(snapshot["targ_tnr_2"][cam][ti])
            x = float(snapshot["targ_x_2"][cam][ti])
            y = float(snapshot["targ_y_2"][cam][ti])
            cameras[cam] = (tnr, x, y)
        candidates.append(ParticleCandidate(row, cost, rank, pos_3d, cameras))

    if is_isolated:
        # No cross-particle competition was resolved for this call (see
        # ParticleSearchResult's docstring) -- report the best candidate by
        # cost, not the kernel's raw path_next_1 (which, for an
        # orig_parts=1 call, is *also* just "best by cost in isolation",
        # but naming it "winner" here would overstate what it means).
        winner_row = candidates[0].row if candidates else NEXT_NONE
    else:
        winner_row = int(snapshot["path_next_1"][particle_index])
    pos_3d = tuple(float(v) for v in snapshot["path_x_1"][particle_index])
    return ParticleSearchResult(
        snapshot["step"], particle_index, pos_3d, candidates, winner_row, is_isolated
    )
