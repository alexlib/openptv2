import cython

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt

from .track_kernels import track3d_loop_fast as _track3d_loop_fast

MAX_CANDS = 32


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def find_candidates_in_3d(frm, pos, dx, dy, dz, max_cands=MAX_CANDS):
    """Find up to max_cands closest candidates within a 3D box.

    Maintains a running top-N by distance (like candsearch_in_pix does in 2D),
    so the returned candidates are the closest, not just the first found.
    """
    indices = [-1] * max_cands
    dists = [1e20] * max_cands
    i: cython.Py_ssize_t
    slot: cython.Py_ssize_t
    s: cython.Py_ssize_t
    ddx: cython.double
    ddy: cython.double
    ddz: cython.double
    d: cython.double
    for i in range(frm.num_parts):
        x = frm.path_info[i].x
        ddx = x[0] - pos[0]
        ddy = x[1] - pos[1]
        ddz = x[2] - pos[2]
        if abs(ddx) < dx and abs(ddy) < dy and abs(ddz) < dz:
            d = c_sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            for slot in range(max_cands):
                if d < dists[slot]:
                    for s in range(max_cands - 1, slot, -1):
                        indices[s] = indices[s - 1]
                        dists[s] = dists[s - 1]
                    indices[slot] = i
                    dists[slot] = d
                    break
    return [idx for idx in indices if idx >= 0]


@cython.ccall
def sort(n, a, b):
    """
    Sorts float array a and int array b in ascending order of a. Returns sorted arrays.
    """
    combined = list(zip(a, b))
    combined.sort()
    sorted_a = [af for af, _ in combined]
    sorted_b = [bf for _, bf in combined]
    return sorted_a, sorted_b


def _sync_soa_to_aos(frm):
    """Fast SoA->AoS sync — only copies fields needed for file I/O."""
    for i in range(frm.num_parts):
        p = frm.path_info[i]
        p.x[:] = frm.path_x[i]
        p.prev = int(frm.path_prev[i])
        p.next_idx = int(frm.path_next[i])
        p.prio = int(frm.path_prio[i])

        c = frm.correspond[i]
        c.nr = int(frm.corres_nr[i])
        c.p[:] = frm.corres_p[i]

    for cam in range(frm.num_cams):
        tnr_arr = frm.targ_tnr[cam]
        for j in range(frm.num_targets[cam]):
            frm.targets[cam][j].tnr = int(tnr_arr[j])


def estimate_level1_dist_weight(pos_a, pos_b, w_min=0.1, w_max=2.0, r0=0.3):
    """Data-driven Level 1 distance-tiebreak weight (see LEVEL1_DIST_WEIGHT
    in track_kernels_track3d.py) from two frames of raw 3D positions.

    The right balance between "trust proximity" and "trust the velocity
    prediction" depends on one ratio: how far particles actually move
    between frames (R = median displacement) relative to how close together
    they sit (S = median nearest-neighbor spacing). When true motion is tiny
    compared to spacing (R/S << 1, e.g. a slow flow densely seeded -- this
    is what test_cavity turned out to be), the velocity prediction carries
    almost no disambiguating power: its own noise floor (from finite
    z-reconstruction precision) is comparable to the true signal, so a
    distant candidate can align with it by chance as easily as the true one
    -- proximity is the more reliable cue, and the distance term should
    dominate. When true motion is a meaningful fraction of the spacing
    (R/S ~ 1, a fast or sparsely-seeded flow -- the regime
    test_track3d_level1_ranks_by_forward_acceleration_not_decoy_behind
    exercises), only the velocity-informed prediction can tell a real
    continuation from a nearer-but-wrong neighbor, so the distance term
    should stay small and let acceleration dominate as it does today.

    Estimating displacement uses the 10th percentile (not the median) of
    each frame-a point's distance to its nearest frame-b neighbor. This
    matters for real correspondence data (unlike a clean synthetic test):
    raw rt_is includes 2-camera "pair" correspondences that are often
    spurious epipolar accidents, not real particles (on test_cavity,
    measured up to 64% ghost rate for pairs vs 16% for 4-camera
    correspondences -- see docs/plans/two-subrig-calibration.md). A ghost
    has no real match next frame, so it contributes a large, noisy
    nearest-neighbor distance; genuine matches cluster tightly near the
    true (small) displacement. The median is dominated by ghost noise (on
    test_cavity: median match distance 2.9mm vs the PIV-verified true
    motion of ~0.2-0.3mm); the 10th percentile isolates the tight cluster
    of genuine matches and recovers ~0.4mm -- close to ground truth -- while
    still separating cleanly from a genuinely fast flow in synthetic tests
    with no ghost contamination at all.

    Returns a weight in [w_min, w_max], smoothly decreasing in R/S:
    w_max at R/S -> 0, the R/S == r0 midpoint at R/S == r0, w_min as
    R/S -> infinity. Falls back to 1.0 (today's fixed default) when there
    isn't enough data to estimate R or S reliably.
    """
    import numpy as np

    from openptv2.tracking_feasibility import measure_motion_scale

    scale = measure_motion_scale(pos_a, pos_b)
    if scale is None:
        return 1.0
    displacement, spacing = scale
    r = displacement / spacing
    weight = w_min + (w_max - w_min) / (1.0 + r / r0)
    return float(np.clip(weight, w_min, w_max))


@cython.ccall
def track3d_loop(run_info, step):
    """
    Python translation of C track3d_loop.
    run_info: tracking_run object with .fb, .tpar, .npart, .nlinks, .seq_par
    step: int
    """
    fb = run_info.fb
    tpar = run_info.tpar
    fb.buf[0]
    curr = fb.buf[1]
    fb.buf[2]
    orig_parts = curr.num_parts
    dx = tpar.dvxmax
    dy = tpar.dvymax
    dz = tpar.dvzmax
    dacc = float(getattr(tpar, "dacc", 0.0))

    fb.buf[0]._sync_path_to_soa()
    fb.buf[1]._sync_path_to_soa()
    fb.buf[2]._sync_path_to_soa()

    dist_weight = getattr(run_info, "_level1_dist_weight", None)
    if dist_weight is None:
        # buf[0] ("previous frame") is empty on the very first step of a
        # forward run -- there is no frame before the first one. buf[1] and
        # buf[2] (the actual first real frame pair) are always populated by
        # the time this runs, and are the pair Level 1 will condition its
        # very next prediction on anyway.
        try:
            dist_weight = estimate_level1_dist_weight(
                fb.buf[1].path_x[: fb.buf[1].num_parts],
                fb.buf[2].path_x[: fb.buf[2].num_parts],
            )
        except Exception:
            dist_weight = 1.0
        run_info._level1_dist_weight = dist_weight

    count1 = _track3d_loop_fast(
        orig_parts,
        fb.buf[0].path_x,
        fb.buf[0].path_prev,
        fb.buf[0].num_parts,
        fb.buf[1].path_x,
        fb.buf[1].path_prev,
        fb.buf[1].path_next,
        fb.buf[1].num_parts,
        fb.buf[2].path_x,
        fb.buf[2].path_prev,
        fb.buf[2].path_next,
        fb.buf[2].num_parts,
        dx,
        dy,
        dz,
        MAX_CANDS,
        dacc,
        dist_weight,
    )

    _sync_soa_to_aos(fb.buf[1])
    _sync_soa_to_aos(fb.buf[2])

    print(
        f"track3d step: {step}, curr: {fb.buf[1].num_parts}, "
        f"next: {fb.buf[2].num_parts}, links: {count1}"
    )

    run_info.npart += fb.buf[1].num_parts
    run_info.nlinks += count1

    fb.fb_next()
    fb.write_frame_from_start(step)
    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, read_links=False)


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
