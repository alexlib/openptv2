import cython
import numpy as np

from .track_kernels import track3d_loop_jit as _track3d_loop_jit, HAS_NUMBA

MAX_CANDS = 4


def find_candidates_in_3d(frm, pos, dx, dy, dz, max_cands=MAX_CANDS):
    """Find up to max_cands closest candidates within a 3D box.

    Maintains a running top-N by distance (like candsearch_in_pix does in 2D),
    so the returned candidates are the closest, not just the first found.
    """
    import math
    indices = [-1] * max_cands
    dists = [1e20] * max_cands
    for i in range(frm.num_parts):
        x = frm.path_info[i].x
        ddx = x[0] - pos[0]
        ddy = x[1] - pos[1]
        ddz = x[2] - pos[2]
        if abs(ddx) < dx and abs(ddy) < dy and abs(ddz) < dz:
            d = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            for slot in range(max_cands):
                if d < dists[slot]:
                    for s in range(max_cands - 1, slot, -1):
                        indices[s] = indices[s - 1]
                        dists[s] = dists[s - 1]
                    indices[slot] = i
                    dists[slot] = d
                    break
    return [idx for idx in indices if idx >= 0]

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
        p.next = int(frm.path_next[i])
        p.prio = int(frm.path_prio[i])

        c = frm.correspond[i]
        c.nr = int(frm.corres_nr[i])
        c.p[:] = frm.corres_p[i]

    for cam in range(frm.num_cams):
        tnr_arr = frm.targ_tnr[cam]
        for j in range(frm.num_targets[cam]):
            frm.targets[cam][j].tnr = int(tnr_arr[j])


def track3d_loop(run_info, step):
    """
    Python translation of C track3d_loop.
    run_info: tracking_run object with .fb, .tpar, .npart, .nlinks, .seq_par
    step: int
    """
    fb = run_info.fb
    tpar = run_info.tpar
    prev = fb.buf[0]
    curr = fb.buf[1]
    nextf = fb.buf[2]
    orig_parts = curr.num_parts
    dx = tpar.dvxmax
    dy = tpar.dvymax
    dz = tpar.dvzmax

    if HAS_NUMBA:
        fb.buf[0]._sync_path_to_soa()
        fb.buf[1]._sync_path_to_soa()
        fb.buf[2]._sync_path_to_soa()

        count1 = _track3d_loop_jit(
            orig_parts,
            fb.buf[0].path_x, fb.buf[0].path_prev, fb.buf[0].num_parts,
            fb.buf[1].path_x, fb.buf[1].path_prev, fb.buf[1].path_next,
            fb.buf[1].num_parts,
            fb.buf[2].path_x, fb.buf[2].path_prev, fb.buf[2].path_next,
            fb.buf[2].num_parts,
            dx, dy, dz, MAX_CANDS,
        )

        _sync_soa_to_aos(fb.buf[1])
        _sync_soa_to_aos(fb.buf[2])

        print(f"track3d step: {step}, curr: {fb.buf[1].num_parts}, "
              f"next: {fb.buf[2].num_parts}, links: {count1}")

        run_info.npart += fb.buf[1].num_parts
        run_info.nlinks += count1

        fb.fb_next()
        fb.write_frame_from_start(step)
        if step < run_info.seq_par.last - 2:
            fb.read_frame_at_end(step + 3, read_links=False)
        return

    # ===== Python fallback (no Numba) =====
    count1 = 0
    # Level 1: Particles with previous links
    for i in range(orig_parts):
        curr_path_inf = curr.path_info[i]
        if curr_path_inf.prev < 0:
            continue
        prev_idx = curr_path_inf.prev
        if prev_idx < 0 or prev_idx >= prev.num_parts:
            continue
        prev_path_inf = prev.path_info[prev_idx]
        predicted = 2 * curr_path_inf.x - prev_path_inf.x
        cand_indices = find_candidates_in_3d(nextf, predicted, dx, dy, dz)
        decis = []
        linkdecis = []
        for k in cand_indices:
            diff = curr_path_inf.x - 2 * nextf.path_info[k].x + prev_path_inf.x
            acc = np.linalg.norm(diff)
            decis.append(acc)
            linkdecis.append(k)
        if len(decis) > 1:
            decis, linkdecis = sort(len(decis), decis, linkdecis)
        if len(linkdecis) > 0 and nextf.path_info[linkdecis[0]].prev < 0:
            curr_path_inf.next = linkdecis[0]
            nextf.path_info[linkdecis[0]].prev = i
            count1 += 1
        else:
            curr_path_inf.next = -1
    # Level 2: No previous link, but neighbors have previous links
    for i in range(orig_parts):
        curr_path_inf = curr.path_info[i]
        if curr_path_inf.prev >= 0 or curr_path_inf.next >= 0:
            continue
        nvel = 0
        vel = np.zeros(3)
        for j in range(orig_parts):
            if j == i:
                continue
            nbr = curr.path_info[j]
            if (abs(nbr.x[0] - curr_path_inf.x[0]) < dx and
                abs(nbr.x[1] - curr_path_inf.x[1]) < dy and
                abs(nbr.x[2] - curr_path_inf.x[2]) < dz and
                nbr.prev >= 0):
                vel += nbr.x - prev.path_info[nbr.prev].x
                nvel += 1
        if nvel == 0:
            continue
        vel /= nvel
        predicted = curr_path_inf.x + vel
        cand_indices = find_candidates_in_3d(nextf, predicted, dx, dy, dz)
        decis = []
        linkdecis = []
        for k in cand_indices:
            diff = curr_path_inf.x - 2 * nextf.path_info[k].x + predicted
            acc = np.linalg.norm(diff)
            decis.append(acc)
            linkdecis.append(k)
        if len(decis) > 1:
            decis, linkdecis = sort(len(decis), decis, linkdecis)
        if len(linkdecis) > 0 and nextf.path_info[linkdecis[0]].prev < 0:
            curr_path_inf.next = linkdecis[0]
            nextf.path_info[linkdecis[0]].prev = i
            count1 += 1
        else:
            curr_path_inf.next = -1
    # Level 3: No previous link, no neighbors with previous links
    for i in range(orig_parts):
        curr_path_inf = curr.path_info[i]
        if curr_path_inf.prev >= 0 or curr_path_inf.next >= 0:
            continue
        predicted = curr_path_inf.x
        cand_indices = find_candidates_in_3d(nextf, predicted, dx, dy, dz)
        decis = []
        linkdecis = []
        for k in cand_indices:
            diff = curr_path_inf.x - 2 * nextf.path_info[k].x + predicted
            acc = np.linalg.norm(diff)
            decis.append(acc)
            linkdecis.append(k)
        if len(decis) > 1:
            decis, linkdecis = sort(len(decis), decis, linkdecis)
        if len(linkdecis) > 0 and nextf.path_info[linkdecis[0]].prev < 0:
            curr_path_inf.next = linkdecis[0]
            nextf.path_info[linkdecis[0]].prev = i
            count1 += 1
        else:
            curr_path_inf.next = -1
    print(f"track3d step: {step}, curr: {fb.buf[1].num_parts}, next: {fb.buf[2].num_parts}, links: {count1}")
    run_info.npart += fb.buf[1].num_parts
    run_info.nlinks += count1
    fb.fb_next()
    fb.write_frame_from_start(step)
    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, read_links=False)


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
