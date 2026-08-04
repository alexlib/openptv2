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
