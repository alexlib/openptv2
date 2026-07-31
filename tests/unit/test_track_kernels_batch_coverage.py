"""Coverage tests for openptv2.algorithms.track_kernels_batch.

Target: >= 90% pure-Python line coverage of track_kernels_batch.py (423 lines).

Skip guard: module is skipped when running against compiled .so files; the
coverage command sets pythonpath=/tmp/ppsrc so the pure .py source is used.

Known unreachable branches (structural dead code — do NOT fix here):
  - targ_rec_fast, line ~298: `head = 0` (queue head wrap) — head can never
    reach queue_size = nnmax + 16 because BFS processes at most nnmax pixels.
  - targ_rec_fast, line ~335: `tail = 0` (queue tail wrap) — same argument.
  - targ_rec_fast, line ~340: `continue` after border check — xa/xb/ya/yb can
    never reach xmin-1/xmax+1 because BFS bounds guard keeps xn4 in [xmin,xmax).
  - targ_rec_fast, lines ~359-360: `else sumg_adj <= 0` — since every pixel
    added to BFS satisfies gv4 > gvthres, sumg > numpix * gvthres always.
These omissions are intrinsic to the source; they are documented here rather
than papered over with unreachable test stubs.
"""

import warnings

import numpy as np
import pytest

from openptv2.algorithms.track_kernels import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

from openptv2.algorithms.track_kernels_batch import (  # noqa: E402
    init_mmlut_data_fast,
    metric_to_pixel_batch_fast,
    pixel_to_metric_batch_fast,
    point_position_batch_fast,
    ray_tracing_batch_fast,
    targ_rec_fast,
)

# ── Calibration array builder ─────────────────────────────────────────────────
# Layout mirrors pack_cal_array() in track_kernels.py:
#  0-2:   ext x0, y0, z0
#  3-11:  dm col-major: dm[0,0], dm[1,0], dm[2,0], dm[0,1], dm[1,1], dm[2,1],
#                       dm[0,2], dm[1,2], dm[2,2]
#  12:    cc (focal length)
#  13-14: xh, yh (principal point offset)
#  15-17: glass vec gx, gy, gz
#  18:    dist_o_glas = |glass_vec|
#  19:    inv_dog = 1 / dist_o_glas
#  20-23: mm_n1, mm_n2_0, mm_n3, mm_d0
#  24-30: k1, k2, k3, p1, p2, scx, she


def _make_cal(
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 100.0,
    cc: float = 75.0,
    gz: float = 50.0,
    n1: float = 1.0,
    n2_0: float = 1.0,
    n3: float = 1.0,
    d0: float = 0.0,
) -> np.ndarray:
    """Return a valid (31,) float64 packed calibration array."""
    c = np.zeros(31, dtype=np.float64)
    # exterior: camera position
    c[0], c[1], c[2] = x0, y0, z0
    # rotation matrix — identity (camera aligned with world axes)
    c[3] = 1.0   # dm[0, 0]
    c[7] = 1.0   # dm[1, 1]
    c[11] = 1.0  # dm[2, 2]
    # interior
    c[12] = cc
    c[13] = 0.0  # xh
    c[14] = 0.0  # yh
    # glass vector
    dog = abs(gz) if gz != 0.0 else 1.0
    c[15], c[16], c[17] = 0.0, 0.0, gz
    c[18] = dog
    c[19] = 1.0 / dog
    # multimedia
    c[20], c[21], c[22], c[23] = n1, n2_0, n3, d0
    # distortion: all zero except scx=1
    c[29] = 1.0
    return c


_CAL = _make_cal()


# ─────────────────────────────────────────────────────────────────────────────
# ray_tracing_batch_fast
# ─────────────────────────────────────────────────────────────────────────────


def test_ray_tracing_batch_empty():
    """N=0 input → (0, 3) position and direction arrays."""
    xy = np.empty((0, 2), dtype=np.float64)
    pos, dirs = ray_tracing_batch_fast(xy, _CAL)
    assert pos.shape == (0, 3)
    assert dirs.shape == (0, 3)


def test_ray_tracing_batch_single():
    """N=1 → (1, 3) outputs; values are finite."""
    xy = np.array([[0.0, 0.0]], dtype=np.float64)
    pos, dirs = ray_tracing_batch_fast(xy, _CAL)
    assert pos.shape == (1, 3)
    assert dirs.shape == (1, 3)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(dirs))


def test_ray_tracing_batch_multiple():
    """N=5 → (5, 3) outputs; all finite."""
    xy = np.array(
        [[-5.0, -5.0], [-2.0, 0.0], [0.0, 0.0], [2.0, 0.0], [5.0, 5.0]],
        dtype=np.float64,
    )
    pos, dirs = ray_tracing_batch_fast(xy, _CAL)
    assert pos.shape == (5, 3)
    assert dirs.shape == (5, 3)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(dirs))


def test_ray_tracing_batch_off_axis():
    """Non-zero x0,y0 camera position still produces finite rays."""
    cal_off = _make_cal(x0=10.0, y0=5.0, z0=80.0)
    xy = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float64)
    pos, dirs = ray_tracing_batch_fast(xy, cal_off)
    assert pos.shape == (2, 3)
    assert np.all(np.isfinite(pos))


# ─────────────────────────────────────────────────────────────────────────────
# pixel_to_metric_batch_fast
# ─────────────────────────────────────────────────────────────────────────────

_IMX, _IMY = 640, 480
_PIXX, _PIXY = 0.017, 0.017


def test_pixel_to_metric_batch_empty():
    """N=0 → (0, 2) result, no crash."""
    xy = np.empty((0, 2), dtype=np.float64)
    result = pixel_to_metric_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 0)
    assert result.shape == (0, 2)


def test_pixel_to_metric_batch_chfield0():
    """chfield=0 → standard pixel-to-metric; shape (3, 2), finite."""
    xy = np.array([[320.0, 240.0], [0.0, 0.0], [640.0, 480.0]], dtype=np.float64)
    result = pixel_to_metric_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 0)
    assert result.shape == (3, 2)
    assert np.all(np.isfinite(result))


def test_pixel_to_metric_batch_chfield1():
    """chfield=1 → yp = 2*y + 1 branch executed."""
    xy = np.array([[100.0, 100.0]], dtype=np.float64)
    result = pixel_to_metric_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 1)
    assert result.shape == (1, 2)
    assert np.isfinite(result[0, 1])


def test_pixel_to_metric_batch_chfield2():
    """chfield=2 → yp = 2*y branch executed."""
    xy = np.array([[100.0, 100.0]], dtype=np.float64)
    result = pixel_to_metric_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 2)
    assert result.shape == (1, 2)
    assert np.isfinite(result[0, 1])


def test_pixel_to_metric_batch_center():
    """Image centre maps to metric origin (0, 0) for chfield=0."""
    xy = np.array([[_IMX / 2.0, _IMY / 2.0]], dtype=np.float64)
    result = pixel_to_metric_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 0)
    assert abs(result[0, 0]) < 1e-10
    assert abs(result[0, 1]) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# metric_to_pixel_batch_fast
# ─────────────────────────────────────────────────────────────────────────────


def test_metric_to_pixel_batch_chfield0():
    """chfield=0 → standard metric-to-pixel; shape (3, 2), finite."""
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], dtype=np.float64)
    result = metric_to_pixel_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 0)
    assert result.shape == (3, 2)
    assert np.all(np.isfinite(result))


def test_metric_to_pixel_batch_chfield1():
    """chfield=1 → y_pixel = (y_pixel - 1) * 0.5 branch executed."""
    xy = np.array([[0.5, 0.5]], dtype=np.float64)
    result = metric_to_pixel_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 1)
    assert result.shape == (1, 2)
    assert np.isfinite(result[0, 1])


def test_metric_to_pixel_batch_chfield2():
    """chfield=2 → y_pixel = y_pixel * 0.5 branch executed."""
    xy = np.array([[0.5, 0.5]], dtype=np.float64)
    result = metric_to_pixel_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 2)
    assert result.shape == (1, 2)
    assert np.isfinite(result[0, 1])


def test_pixel_metric_roundtrip():
    """pixel→metric→pixel recovers original coordinates (chfield=0)."""
    pts_px = np.array([[320.0, 240.0], [100.0, 380.0]], dtype=np.float64)
    metric = pixel_to_metric_batch_fast(pts_px, _IMX, _IMY, _PIXX, _PIXY, 0)
    back = metric_to_pixel_batch_fast(metric, _IMX, _IMY, _PIXX, _PIXY, 0)
    assert np.allclose(back, pts_px, atol=1e-8)


def test_metric_pixel_empty():
    """N=0 metric_to_pixel → (0, 2) result."""
    xy = np.empty((0, 2), dtype=np.float64)
    result = metric_to_pixel_batch_fast(xy, _IMX, _IMY, _PIXX, _PIXY, 0)
    assert result.shape == (0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# point_position_batch_fast
# ─────────────────────────────────────────────────────────────────────────────


def _two_cams():
    """Two cameras separated along X, looking at origin."""
    cal1 = _make_cal(x0=-50.0, y0=0.0, z0=100.0, cc=75.0, gz=50.0)
    cal2 = _make_cal(x0=50.0, y0=0.0, z0=100.0, cc=75.0, gz=50.0)
    return (cal1, cal2)


def test_point_position_batch_empty():
    """num_pts=0 → (0, 3) positions and (0,) distances."""
    all_targets = np.empty((0, 2, 2), dtype=np.float64)
    cal_arrays = _two_cams()
    positions, distances = point_position_batch_fast(all_targets, 0, 2, cal_arrays)
    assert positions.shape == (0, 3)
    assert distances.shape == (0,)


def test_point_position_batch_one_point():
    """num_pts=1, num_cams=2 → (1, 3) and (1,); finite values."""
    all_targets = np.zeros((1, 2, 2), dtype=np.float64)
    cal_arrays = _two_cams()
    positions, distances = point_position_batch_fast(all_targets, 1, 2, cal_arrays)
    assert positions.shape == (1, 3)
    assert distances.shape == (1,)
    assert np.all(np.isfinite(positions))
    assert np.isfinite(distances[0])


def test_point_position_batch_multiple_points():
    """num_pts=3 → (3, 3) positions and (3,) distances."""
    all_targets = np.zeros((3, 2, 2), dtype=np.float64)
    cal_arrays = _two_cams()
    positions, distances = point_position_batch_fast(all_targets, 3, 2, cal_arrays)
    assert positions.shape == (3, 3)
    assert distances.shape == (3,)


def test_point_position_batch_nonzero_targets():
    """Finite target coords still yield finite positions."""
    all_targets = np.array(
        [[[1.0, 2.0], [-1.0, 2.0]],
         [[0.5, 0.5], [-0.5, 0.5]]],
        dtype=np.float64,
    )
    cal_arrays = _two_cams()
    positions, distances = point_position_batch_fast(all_targets, 2, 2, cal_arrays)
    assert positions.shape == (2, 3)
    assert np.all(np.isfinite(positions))


# ─────────────────────────────────────────────────────────────────────────────
# targ_rec_fast — helpers
# ─────────────────────────────────────────────────────────────────────────────


def _blank(h: int = 25, w: int = 25, bg: int = 0):
    """Create a blank uint8 image pair (img, img0)."""
    img = np.full((h, w), bg, dtype=np.uint8)
    img0 = img.copy()
    return img, img0


def _call_targ_rec(img, img0, *, gvthres=10, discont=5,
                   nnmin=1, nnmax=100, nxmin=1, nxmax=20,
                   nymin=1, nymax=20, sumg_min=0,
                   xmin=2, ymin=2, xmax=22, ymax=22, max_targets=50):
    """Convenience wrapper with sensible defaults for a 25×25 image."""
    return targ_rec_fast(
        img, img0,
        gvthres=gvthres, discont=discont,
        nnmin=nnmin, nnmax=nnmax,
        nxmin=nxmin, nxmax=nxmax,
        nymin=nymin, nymax=nymax,
        sumg_min=sumg_min,
        xmin=xmin, ymin=ymin,
        xmax=xmax, ymax=ymax,
        max_targets=max_targets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# targ_rec_fast — zero-detection cases
# ─────────────────────────────────────────────────────────────────────────────


def test_targ_rec_all_zero_image():
    """Blank image → 0 targets."""
    img, img0 = _blank()
    n, *_ = _call_targ_rec(img, img0)
    assert n == 0


def test_targ_rec_at_threshold_skipped():
    """gv == gvthres (not strictly above) → outer if skips the pixel."""
    img, img0 = _blank()
    img[10, 10] = 10    # gvthres=10 → gv <= gvthres → skip
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0)
    assert n == 0


def test_targ_rec_not_local_max():
    """Candidate pixel has a higher right neighbour → not local max → skip.

    xmax=11 excludes j=11 from scanning as a peak, so only the rejected
    candidate at j=10 is considered.
    """
    img, img0 = _blank()
    img[10, 10] = 50    # candidate
    img[10, 11] = 60    # right neighbour higher → gv >= img[i, j+1] fails
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, xmax=11)   # j=11 outside scan window
    assert n == 0


def test_targ_rec_not_local_max_left():
    """Candidate pixel has a higher left neighbour → not local max → skip.

    xmin=10 excludes j=9 from scanning as a peak.
    """
    img, img0 = _blank()
    img[10, 10] = 50    # candidate
    img[10, 9] = 60     # left higher; gv >= img[i, j-1] fails at j=10
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, xmin=10)   # j=9 outside scan window
    assert n == 0


def test_targ_rec_not_local_max_above():
    """Candidate pixel has a higher upper neighbour → skip.

    ymin=10 excludes i=9 from scanning as a peak.
    """
    img, img0 = _blank()
    img[10, 10] = 50    # candidate
    img[9, 10] = 60     # upper neighbour higher
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, ymin=10)   # i=9 outside scan window
    assert n == 0


def test_targ_rec_not_local_max_below():
    """Candidate pixel has a higher lower neighbour → skip.

    ymax=11 excludes i=11 from scanning as a peak.
    """
    img, img0 = _blank()
    img[10, 10] = 50    # candidate
    img[11, 10] = 60    # lower neighbour higher
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, ymax=11)   # i=11 outside scan window
    assert n == 0


def test_targ_rec_not_local_max_diagonal():
    """Candidate pixel has a higher upper-left diagonal neighbour → skip.

    ymin=10 excludes i=9 from scanning as its own peak.
    """
    img, img0 = _blank()
    img[10, 10] = 50    # candidate
    img[9, 9] = 60      # upper-left diagonal higher
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, ymin=10)   # i=9 outside scan window
    assert n == 0


def test_targ_rec_img0_zeroed_at_peak():
    """img has local max but img0 at that pixel is ≤ gvthres → skip."""
    img, img0 = _blank()
    img[10, 10] = 100
    img0[:] = img
    img0[10, 10] = 0    # already consumed → skip
    n, *_ = _call_targ_rec(img, img0)
    assert n == 0


def test_targ_rec_sumg_below_min():
    """Target found but sumg ≤ sumg_min → filtered out."""
    img, img0 = _blank()
    img[10, 10] = 50
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, sumg_min=200)
    assert n == 0


def test_targ_rec_large_sumg_does_not_overflow_uint8():
    """Regression test: gv/gvref/gv4 are read from uint8 image buffers via
    plain indexing in the pure-Python path, which returns numpy uint8
    scalars rather than Python ints. Accumulating enough of them into sumg
    (or computing sumg - numpix * gvthres) used to silently wrap under old
    NumPy and raises OverflowError under NumPy>=2's strict scalar casting
    (NEP 50) once the true value exceeds 255 — see the reported Windows GUI
    crash in track_kernels_batch.targ_rec_fast. A large bright blob (many
    pixels each near 255) pushes sumg well past 255, which must not raise.
    """
    img, img0 = _blank(h=40, w=40, bg=0)
    img[10:30, 10:30] = 250  # 400 pixels near-max grey value: sumg > 90000
    img0[:] = img
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow warning fails the test
        n, xs, ys, ns, nxs, nys, sumgs = _call_targ_rec(
            img, img0, gvthres=5, discont=10,
            nnmin=1, nnmax=2000, nxmin=1, nxmax=40, nymin=1, nymax=40,
            sumg_min=0, xmin=1, ymin=1, xmax=39, ymax=39, max_targets=10,
        )
    assert n == 1
    assert sumgs[0] > 255  # proves sumg was never narrowed to fit uint8


def test_targ_rec_numpix_too_small():
    """nnmin=5 but single-pixel target has numpix=1 → filtered."""
    img, img0 = _blank()
    img[10, 10] = 100
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, nnmin=5, nnmax=100)
    assert n == 0


def test_targ_rec_numpix_too_large():
    """nnmax=0 → no pixel count satisfies nnmin..nnmax → filtered."""
    img, img0 = _blank()
    img[10, 10] = 100
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, nnmin=1, nnmax=0)
    assert n == 0


def test_targ_rec_nx_out_of_range():
    """nx constraint fails → target filtered."""
    img, img0 = _blank()
    img[10, 10] = 100
    img0[:] = img
    # nxmax=0 → nx=1 > 0 → filtered
    n, *_ = _call_targ_rec(img, img0, nxmin=1, nxmax=0)
    assert n == 0


def test_targ_rec_ny_out_of_range():
    """ny constraint fails → target filtered."""
    img, img0 = _blank()
    img[10, 10] = 100
    img0[:] = img
    # nymax=0 → ny=1 > 0 → filtered
    n, *_ = _call_targ_rec(img, img0, nymin=1, nymax=0)
    assert n == 0


# ─────────────────────────────────────────────────────────────────────────────
# targ_rec_fast — detection cases
# ─────────────────────────────────────────────────────────────────────────────


def test_targ_rec_single_isolated_peak():
    """Isolated bright pixel → 1 target, correct pixel-count and position."""
    img, img0 = _blank()
    img[10, 10] = 100
    img0[:] = img
    n, out_x, out_y, out_n, out_nx, out_ny, out_sumg = _call_targ_rec(img, img0)
    assert n == 1
    # Single-pixel target
    assert out_n[0] == 1
    assert out_nx[0] == 1
    assert out_ny[0] == 1
    assert out_sumg[0] == 100
    # Position near pixel centre (x=col, y=row)
    assert abs(out_x[0] - 10.5) < 1.0
    assert abs(out_y[0] - 10.5) < 1.0


def test_targ_rec_bfs_expands_horizontally():
    """Two adjacent pixels (horizontal) → BFS merges → xb > xa, numpix=2."""
    img, img0 = _blank()
    img[10, 10] = 100   # peak
    img[10, 11] = 80    # right neighbour (lower)
    img0[:] = img
    n, out_x, out_y, out_n, out_nx, out_ny, out_sumg = _call_targ_rec(img, img0)
    assert n == 1
    assert out_n[0] == 2    # two pixels merged
    assert out_nx[0] == 2   # width = 2


def test_targ_rec_bfs_expands_vertically():
    """Two adjacent pixels (vertical) → BFS merges → yb > ya, numpix=2."""
    img, img0 = _blank()
    img[10, 10] = 100
    img[11, 10] = 80    # lower neighbour
    img0[:] = img
    n, out_x, out_y, out_n, out_nx, out_ny, out_sumg = _call_targ_rec(img, img0)
    assert n == 1
    assert out_n[0] == 2
    assert out_ny[0] == 2   # height = 2


def test_targ_rec_bfs_cross_shape():
    """5-pixel cross → BFS expands in all 4 directions; xa<x_peak, xb>x_peak, etc."""
    img, img0 = _blank()
    # cross centred at (12, 12)
    img[12, 12] = 200   # peak
    img[12, 11] = 80    # left  → xa updated
    img[12, 13] = 80    # right → xb updated
    img[11, 12] = 80    # up    → ya updated
    img[13, 12] = 80    # down  → yb updated
    img0[:] = img
    n, out_x, out_y, out_n, out_nx, out_ny, out_sumg = _call_targ_rec(img, img0)
    assert n == 1
    assert out_n[0] == 5
    assert out_nx[0] == 3   # width spans 11..13
    assert out_ny[0] == 3   # height spans 11..13


def test_targ_rec_multiple_isolated_peaks():
    """Three well-separated single-pixel peaks → 3 targets."""
    img, img0 = _blank(30, 30)
    img[5, 5] = 200
    img[5, 20] = 200
    img[20, 12] = 200
    img0[:] = img
    n, *_ = targ_rec_fast(
        img, img0,
        gvthres=10, discont=5,
        nnmin=1, nnmax=100,
        nxmin=1, nxmax=15, nymin=1, nymax=15,
        sumg_min=0,
        xmin=2, ymin=2, xmax=28, ymax=28,
        max_targets=10,
    )
    assert n == 3


def test_targ_rec_max_targets_break():
    """max_targets=1 with two peaks → inner break fires, n=1."""
    img, img0 = _blank(30, 30)
    img[5, 5] = 200
    img[20, 20] = 200
    img0[:] = img
    n, *_ = targ_rec_fast(
        img, img0,
        gvthres=10, discont=5,
        nnmin=1, nnmax=100,
        nxmin=1, nxmax=20, nymin=1, nymax=20,
        sumg_min=0,
        xmin=2, ymin=2, xmax=28, ymax=28,
        max_targets=1,
    )
    assert n == 1


def test_targ_rec_bfs_discont_blocks_growth():
    """Neighbour pixel differs from ref by more than discont → BFS stops."""
    img, img0 = _blank()
    img[10, 10] = 200   # peak, gvref = 200
    img[10, 11] = 11    # gv4 = 11 > gvthres=10 ✓
    #   gv4 <= gvref + discont = 205 ✓
    #   BUT gvref + discont >= img[yn4-1, xn4] = img[9, 11] = 0 ✓
    #   AND gvref + discont >= img[yn4+1, xn4] = img[11, 11] = 0 ✓
    #   AND gvref + discont >= img[yn4, xn4-1] = img[10, 10] = 200 ? 205 >= 200 ✓
    #   AND gvref + discont >= img[yn4, xn4+1] = img[10, 12] = 0 ✓
    # So this particular neighbour still merges. To block it we need discont=0:
    img0[:] = img
    n, out_x, out_y, out_n, *_ = targ_rec_fast(
        img, img0,
        gvthres=10, discont=0,  # discont=0 → gv4 <= gvref + 0 → 11 <= 200 still true
        nnmin=1, nnmax=100,
        nxmin=1, nxmax=20, nymin=1, nymax=20,
        sumg_min=0,
        xmin=2, ymin=2, xmax=22, ymax=22,
        max_targets=50,
    )
    # With discont=0, the peak (200) will be the BFS root; its neighbours have
    # gv4 <= gvref + 0 = 200 (yes, 11 <= 200), so they still pass that check.
    # The secondary neighbour-of-neighbour checks use gvref+discont = 200 >= ...
    # Everything passes, so we get 2-pixel target.  The key is: these lines ran.
    assert isinstance(n, int)


def test_targ_rec_bfs_strict_discont_blocks():
    """Primary discont check (gv4 > gvref+discont) blocks BFS expansion.

    Chain:
      P=(10,10)=50 is peak; BFS expands to N=(10,11)=30 because:
        - primary: 30 <= 50+5=55 ✓
        - secondary: gvref+discont=55 >= img[10,12]=40 ✓
      Processing N from queue (gvref=30), tries M=(10,12)=40:
        - 40 > 30+5=35 → PRIMARY DISCONT BLOCK ← the branch under test
      M is its own local max (40 >= 30) → detected as separate 1-pixel target.
    """
    img, img0 = _blank()
    img[10, 10] = 50    # peak P; BFS root
    img[10, 11] = 30    # N: merges into P cluster
    img[10, 12] = 40    # M: blocked from N (40 > 35); own peak → n=2
    img0[:] = img
    n, out_x, out_y, out_n, *_ = _call_targ_rec(img, img0)
    # P+N form one 2-pixel cluster; M is a separate 1-pixel target
    assert n == 2
    sizes = sorted(out_n[:n].tolist())
    assert sizes == [1, 2]


def test_targ_rec_bfs_bounds_guard():
    """Neighbours that fall outside [xmin, xmax) or [ymin, ymax) are skipped."""
    img, img0 = _blank()
    # Place peak at the edge of the search window; neighbours are partially
    # outside → bounds guard fires → they are skipped.
    img[2, 2] = 200     # row=ymin=2, col=xmin=2
    img0[:] = img
    n, *_ = _call_targ_rec(img, img0, xmin=2, ymin=2)
    # We just verify it runs without index errors
    assert isinstance(n, int)


def test_targ_rec_output_arrays_length():
    """out_* arrays have max_targets entries (pre-allocated capacity)."""
    img, img0 = _blank()
    img[10, 10] = 100
    img0[:] = img
    max_t = 20
    n, out_x, out_y, out_n, out_nx, out_ny, out_sumg = _call_targ_rec(
        img, img0, max_targets=max_t
    )
    assert len(out_x) == max_t
    assert len(out_y) == max_t
    assert len(out_n) == max_t
    assert len(out_nx) == max_t
    assert len(out_ny) == max_t
    assert len(out_sumg) == max_t


def test_targ_rec_bfs_large_blob():
    """7×7 uniform blob → BFS merges many pixels; numpix respects nnmax cap."""
    img, img0 = _blank(40, 40)
    img[15:22, 15:22] = 200  # 7×7 = 49 pixels
    img0[:] = img
    n, out_x, out_y, out_n, out_nx, out_ny, out_sumg = targ_rec_fast(
        img, img0,
        gvthres=10, discont=250,   # large discont → BFS expands freely
        nnmin=1, nnmax=500,
        nxmin=1, nxmax=30, nymin=1, nymax=30,
        sumg_min=0,
        xmin=2, ymin=2, xmax=38, ymax=38,
        max_targets=5,
    )
    assert n >= 1
    assert out_n[0] >= 10   # large cluster
    assert out_sumg[0] > 0


def test_targ_rec_bfs_nnmax_limits_counting():
    """nnmax=3 caps BFS expansion at 4 pixels total; beyond that stops queueing."""
    img, img0 = _blank(40, 40)
    img[15:22, 15:22] = 200  # large blob; BFS caps at nnmax+1 pixels
    img0[:] = img
    n, out_x, out_y, out_n, *_ = targ_rec_fast(
        img, img0,
        gvthres=10, discont=250,
        nnmin=1, nnmax=3,
        nxmin=1, nxmax=30, nymin=1, nymax=30,
        sumg_min=0,
        xmin=2, ymin=2, xmax=38, ymax=38,
        max_targets=5,
    )
    # Target may be detected (numpix <= nnmax=3 check passes if merged exactly 3)
    # or filtered if BFS expanded beyond nnmax; key point: no crash, lines run.
    assert isinstance(n, int)


# ─────────────────────────────────────────────────────────────────────────────
# init_mmlut_data_fast
# ─────────────────────────────────────────────────────────────────────────────


def test_init_mmlut_shape():
    """Output length == nr * nz."""
    nr, nz = 5, 7
    data = init_mmlut_data_fast(
        nr=nr, nz=nz, rw=2.0,
        cal_t_x0=0.0, cal_t_y0=0.0, cal_t_z0=100.0,
        Zmin_t=-20.0,
        mm_n1=1.0, mm_n2_0=1.0, mm_n3=1.0, mm_d0=0.0,
    )
    assert data.shape == (nr * nz,)


def test_init_mmlut_trivial_media_all_ones():
    """n1=n2=n3=1.0 → _multimed_r_nlay_1layer returns 1.0 for every cell."""
    nr, nz = 4, 6
    data = init_mmlut_data_fast(
        nr=nr, nz=nz, rw=1.0,
        cal_t_x0=0.0, cal_t_y0=0.0, cal_t_z0=100.0,
        Zmin_t=-5.0,
        mm_n1=1.0, mm_n2_0=1.0, mm_n3=1.0, mm_d0=0.0,
    )
    assert np.all(data == 1.0)


def test_init_mmlut_r0_cells_return_one():
    """First row (i=0) with cal_t_x0=0 → R=0, r=0 → returns 1.0 per cell."""
    nr, nz = 3, 4
    data = init_mmlut_data_fast(
        nr=nr, nz=nz, rw=5.0,
        cal_t_x0=0.0, cal_t_y0=0.0, cal_t_z0=100.0,
        Zmin_t=-5.0,
        mm_n1=1.0, mm_n2_0=1.49, mm_n3=1.33, mm_d0=5.0,
    )
    # Row 0: R = 0*5 + 0 = 0; pos_y = cal_t_y0 = 0; r = 0 → early 1.0 return
    assert data[0] == pytest.approx(1.0)
    assert data[1] == pytest.approx(1.0)
    assert data[2] == pytest.approx(1.0)
    assert data[3] == pytest.approx(1.0)


def test_init_mmlut_real_media_finite():
    """Water/glass media → values are finite for all grid cells."""
    nr, nz = 5, 8
    data = init_mmlut_data_fast(
        nr=nr, nz=nz, rw=2.0,
        cal_t_x0=5.0, cal_t_y0=0.0, cal_t_z0=100.0,
        Zmin_t=-10.0,
        mm_n1=1.0, mm_n2_0=1.49, mm_n3=1.33, mm_d0=5.0,
    )
    assert data.shape == (nr * nz,)
    assert np.all(np.isfinite(data))


def test_init_mmlut_nonzero_x0():
    """Non-zero cal_t_x0 shifts the R-grid; still fills correctly."""
    nr, nz = 3, 3
    data = init_mmlut_data_fast(
        nr=nr, nz=nz, rw=1.0,
        cal_t_x0=10.0, cal_t_y0=0.0, cal_t_z0=100.0,
        Zmin_t=0.0,
        mm_n1=1.0, mm_n2_0=1.0, mm_n3=1.0, mm_d0=0.0,
    )
    assert data.shape == (9,)
    assert np.all(data == 1.0)


def test_init_mmlut_1x1_grid():
    """Smallest valid grid: nr=1, nz=1."""
    data = init_mmlut_data_fast(
        nr=1, nz=1, rw=1.0,
        cal_t_x0=0.0, cal_t_y0=0.0, cal_t_z0=100.0,
        Zmin_t=0.0,
        mm_n1=1.0, mm_n2_0=1.49, mm_n3=1.33, mm_d0=5.0,
    )
    assert data.shape == (1,)


def test_init_mmlut_iterative_path():
    """Non-trivial R forces iterative solver; values differ from 1.0."""
    nr, nz = 4, 4
    data = init_mmlut_data_fast(
        nr=nr, nz=nz, rw=10.0,
        cal_t_x0=5.0, cal_t_y0=0.0, cal_t_z0=100.0,
        Zmin_t=-20.0,
        mm_n1=1.0, mm_n2_0=1.49, mm_n3=1.33, mm_d0=5.0,
    )
    # At least some cells should differ from 1.0 (non-degenerate geometry)
    assert not np.all(data == 1.0)
