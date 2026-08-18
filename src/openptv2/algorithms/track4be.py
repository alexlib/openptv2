"""4BE — four-frame best-estimate tracking loop.

Pure stereo-3D, exactly like :mod:`track3d`: it consumes only the
correspondence cloud (``path_x``) and never touches 2D targets or camera
models. The difference from track3d is the candidate cost:

* track3d (3MA) scores a candidate in frame n+1 by the acceleration it
  implies against frames n-1 and n.
* 4BE scores it by how well it *predicts a real particle in frame n+2* —
  the candidate's own constant-velocity extrapolation is compared against
  the nearest actual particle two frames ahead.

Source: Ouellette, Xu & Bodenschatz, "A quantitative study of
three-dimensional Lagrangian particle tracking algorithms", Exp. Fluids
40:301-313 (2006), eqs. 10, 12 and 14; the kernel docstring in
``track_kernels_track3d.track4be_loop_fast`` derives why eq. 12 reduces to
``2*q - x1``.
"""

import cython

from .track3d import MAX_CANDS, _sync_soa_to_aos
from .track_kernels import track4be_loop_fast as _track4be_loop_fast

#: Kernel variants, exposed here so a benchmark can A/B them without
#: threading a parameter through the plugin/preset/YAML stack. Both default
#: to the paper's behaviour; see ``track4be_loop_fast``'s docstring.
STRICT_SUPPORT = 0
GREEDY_CONFLICTS = 0


@cython.ccall
def track4be_loop(run_info, step):
    """One 4BE tracking step: link frame n to frame n+1.

    Mirrors :func:`track3d.track3d_loop`'s buffer contract, but reads a
    fourth frame (``fb.buf[3]``, i.e. n+2) which the 4-slot frame buffer
    already carries for trackcorr's benefit — no buffer changes needed.
    """
    fb = run_info.fb
    tpar = run_info.tpar
    orig_parts = fb.buf[1].num_parts

    dx = tpar.dvxmax
    dy = tpar.dvymax
    dz = tpar.dvzmax

    for b in range(4):
        fb.buf[b]._sync_path_to_soa()

    count1 = _track4be_loop_fast(
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
        fb.buf[3].path_x,
        fb.buf[3].num_parts,
        dx,
        dy,
        dz,
        MAX_CANDS,
        STRICT_SUPPORT,
        GREEDY_CONFLICTS,
    )

    _sync_soa_to_aos(fb.buf[1])
    _sync_soa_to_aos(fb.buf[2])

    print(
        f"4be step: {step}, curr: {fb.buf[1].num_parts}, "
        f"next: {fb.buf[2].num_parts}, links: {count1}"
    )

    run_info.npart += fb.buf[1].num_parts
    run_info.nlinks += count1

    fb.fb_next()
    fb.write_frame_from_start(step)
    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, read_links=False)
    else:
        # Frame n+2 is out of range near the sequence tail -- clear buf[3]
        # rather than leaving it holding stale positions from an earlier
        # read. 4BE's candidate cost scores against "the nearest real
        # particle in frame n+2" (see module docstring); a stale buf[3]
        # scores real candidates against garbage several frames old and
        # silently prefers wrong, distant matches instead. Mirrors
        # trackcorr_c_loop's identical guard in track.py.
        fb.buf[fb.buf_len - 1].num_parts = 0
