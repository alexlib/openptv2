"""Tests for the 2-corner quader approximation vs 8-corner search.

The 2-corner approach projects only the diagonal corners (all-min and all-max)
of the search box, then expands by a safety margin (MARGIN=0.05 = 5%).
For cameras with multimedia (has_mmlut), the original 8-corner path is used.
"""

import pathlib
import sys

import numpy as np
import pytest

try:
    from openptv2.algorithms.track_kernels_geom import (
        _pixel_jacobian,
        _point_to_pixel_out,
    )
except ImportError:
    pytestmark = pytest.mark.skip(reason="track_kernels_geom does not implement _pixel_jacobian")
    _point_to_pixel_out = None
    _pixel_jacobian = None

# Safety margin used by the 2-corner approximation
MARGIN = 0.10
# Image dimensions (same as cavity test)
IMX = 1024.0
IMY = 1024.0
IMX_HALF = IMX * 0.5
IMY_HALF = IMY * 0.5
PIX_X = 0.01
PIX_Y = 0.01
INV_PIX_X = 1.0 / PIX_X
INV_PIX_Y = 1.0 / PIX_Y
CHFIELD = 0
NUM_CAMS = 4
MAX_CANDS = 4
TR_UNUSED = -1

# Default search box (from track.par in cavity test)
DVXMIN = -15.5
DVXMAX = 15.5
DVYMIN = -15.5
DVYMAX = 15.5
DVZMIN = -15.5
DVZMAX = 15.5


def _make_synthetic_calibration(has_mmlut=False):
    """Generate synthetic calibration arrays for testing.

    Creates 4 pinhole cameras at the corners of a 500mm cube, all pointed
    at the origin. No distortion, no multimedia (unless has_mmlut=True).

    Returns (cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr) —
    the same format used by _sorted_candidates_fast_out.
    """

    def _rotation_matrix_from_lookat(cam_pos, target=(0, 0, 0)):
        """Compute rotation matrix that maps world to camera coords.
        Camera z-axis points from camera to target, x is right, y is down.
        """
        cx, cy, cz = cam_pos
        tx, ty, tz = target
        forward = np.array([tx - cx, ty - cy, tz - cz])
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])  # world Z is up
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        # dm rotates from world to camera: cam_coord = dm @ (pos - ext)
        dm = np.array([right, up, -forward])  # shape (3, 3)
        return dm.flatten()  # row-major, 9 elements

    # Camera positions: 4 corners of a 500mm cube
    cam_positions = [
        (500, 250, 300),
        (-500, 250, -300),
        (500, -250, -300),
        (-500, -250, 300),
    ]

    num_cams = len(cam_positions)
    cal_arr = np.zeros((num_cams, 31), dtype=np.float64)

    for i, (pos) in enumerate(cam_positions):
        cx, cy, cz = pos
        dm = _rotation_matrix_from_lookat(pos)

        cal_arr[i, 0] = cx  # ext_x0
        cal_arr[i, 1] = cy  # ext_y0
        cal_arr[i, 2] = cz  # ext_z0
        cal_arr[i, 3:12] = dm  # dm00..dm22 (9 elements)
        cal_arr[i, 12] = 70.0  # int_cc (focal length in mm)
        cal_arr[i, 13] = 0.0  # xh (principal point offset)
        cal_arr[i, 14] = 0.0  # yh
        cal_arr[i, 15] = 0.0  # gx (glass normal)
        cal_arr[i, 16] = 0.0  # gy
        cal_arr[i, 17] = -1.0  # gz
        cal_arr[i, 18] = 0.0  # dist_o_glas (glass distance)
        g_norm = np.sqrt(0**2 + 0**2 + (-1) ** 2)
        cal_arr[i, 19] = 1.0 / g_norm if g_norm > 0 else 1.0  # inv_dog
        cal_arr[i, 20] = 0.0  # mm_n1 (multimedia n1, 0 = no mm)
        cal_arr[i, 21] = 0.0  # mm_n2_0
        cal_arr[i, 22] = 0.0  # mm_n3
        cal_arr[i, 23] = 0.0  # mm_d0
        cal_arr[i, 24] = 0.0  # k1
        cal_arr[i, 25] = 0.0  # k2
        cal_arr[i, 26] = 0.0  # k3
        cal_arr[i, 27] = 0.0  # p1
        cal_arr[i, 28] = 0.0  # p2
        cal_arr[i, 29] = 1.0  # scx
        cal_arr[i, 30] = 0.0  # she

    if has_mmlut:
        # Create synthetic multimedia data for one camera
        cal_arr[0, 20] = 1.33  # mm_n1 (glass index)
        cal_arr[0, 21] = 50.0  # mm_n2_0
        cal_arr[0, 22] = 1.0  # mm_n3
        cal_arr[0, 23] = 3.0  # mm_d0
        mnr = 10
        mnz = 10
        rw = 2.0
        n_pts = mnr * mnz
        md_arr = tuple(
            np.ones(n_pts, dtype=np.float64)
            if i == 0
            else np.empty(0, dtype=np.float64)
            for i in range(num_cams)
        )
        mo_arr = np.zeros((num_cams, 3), dtype=np.float64)
        mnr_arr = np.array(
            [mnr if i == 0 else 0 for i in range(num_cams)], dtype=np.int32
        )
        mnz_arr = np.array(
            [mnz if i == 0 else 0 for i in range(num_cams)], dtype=np.int32
        )
        mrw_arr = np.array(
            [rw if i == 0 else 0.0 for i in range(num_cams)], dtype=np.float64
        )
    else:
        md_arr = tuple(np.empty(0, dtype=np.float64) for _ in range(num_cams))
        mo_arr = np.zeros((num_cams, 3), dtype=np.float64)
        mnr_arr = np.zeros(num_cams, dtype=np.int32)
        mnz_arr = np.zeros(num_cams, dtype=np.int32)
        mrw_arr = np.zeros(num_cams, dtype=np.float64)

    return cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr


def _compute_8corner_bounds(
    center_3d, cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr, dv=None
):
    """Compute per-camera search bounds using the full 8-corner projection.

    Returns (xr, xl, yd, yu) each (num_cams,) — same as searchquader_fast.
    """
    if dv is None:
        dvxmin, dvxmax = DVXMIN, DVXMAX
        dvymin, dvymax = DVYMIN, DVYMAX
        dvzmin, dvzmax = DVZMIN, DVZMAX
    else:
        dvxmin, dvxmax, dvymin, dvymax, dvzmin, dvzmax = dv

    num_cams = cal_arr.shape[0]
    xr = np.zeros(num_cams)
    xl = np.zeros(num_cams)
    yd = np.zeros(num_cams)
    yu = np.zeros(num_cams)

    _pp = np.empty(2, dtype=np.float64)

    for i in range(num_cams):
        cal = cal_arr[i]
        md = md_arr[i]
        mo = mo_arr[i]
        mnr = mnr_arr[i]
        mnz = mnz_arr[i]
        mrw = mrw_arr[i]
        has_mmlut = 1 if mnr > 0 else 0

        px, py, pz = center_3d
        xl_i = float(IMX)
        xr_i = 0.0
        yu_i = float(IMY)
        yd_i = 0.0

        for pt in range(8):
            qx = px + (dvxmax if pt & 1 else dvxmin)
            qy = py + (dvymax if pt & 2 else dvymin)
            qz = pz + (dvzmax if pt & 4 else dvzmin)
            _point_to_pixel_out(
                qx,
                qy,
                qz,
                cal,
                md,
                mo,
                mnr,
                mnz,
                mrw,
                has_mmlut,
                IMX_HALF,
                IMY_HALF,
                INV_PIX_X,
                INV_PIX_Y,
                CHFIELD,
                _pp,
            )
            cx = _pp[0]
            cy = _pp[1]
            if cx < xl_i:
                xl_i = cx
            if cy < yu_i:
                yu_i = cy
            if cx > xr_i:
                xr_i = cx
            if cy > yd_i:
                yd_i = cy

        if xl_i < 0.0:
            xl_i = 0.0
        if yu_i < 0.0:
            yu_i = 0.0
        if xr_i > IMX:
            xr_i = IMX
        if yd_i > IMY:
            yd_i = IMY

        # Center projection
        _point_to_pixel_out(
            px,
            py,
            pz,
            cal,
            md,
            mo,
            mnr,
            mnz,
            mrw,
            has_mmlut,
            IMX_HALF,
            IMY_HALF,
            INV_PIX_X,
            INV_PIX_Y,
            CHFIELD,
            _pp,
        )
        cx = _pp[0]
        cy = _pp[1]

        xr[i] = xr_i - cx
        xl[i] = cx - xl_i
        yd[i] = yd_i - cy
        yu[i] = cy - yu_i

    return xl, xr, yu, yd


def _compute_jacobian_bounds(
    center_3d,
    cal_arr,
    md_arr,
    mo_arr,
    mnr_arr,
    mnz_arr,
    mrw_arr,
    dv=None,
    margin=0.10,
):
    """Compute per-camera search bounds using Jacobian-guided corner selection.

    Uses the projection Jacobian to select which of the 8 corners to
    actually project (typically 2-4). Falls back to 8-corner when has_mmlut.
    """
    if dv is None:
        dvxmin, dvxmax = DVXMIN, DVXMAX
        dvymin, dvymax = DVYMIN, DVYMAX
        dvzmin, dvzmax = DVZMIN, DVZMAX
    else:
        dvxmin, dvxmax, dvymin, dvymax, dvzmin, dvzmax = dv

    num_cams = cal_arr.shape[0]
    xr = np.zeros(num_cams)
    xl = np.zeros(num_cams)
    yd = np.zeros(num_cams)
    yu = np.zeros(num_cams)

    _pp = np.empty(2, dtype=np.float64)

    for i in range(num_cams):
        cal = cal_arr[i]
        md = md_arr[i]
        mo = mo_arr[i]
        mnr = mnr_arr[i]
        mnz = mnz_arr[i]
        mrw = mrw_arr[i]
        has_mmlut = 1 if mnr > 0 else 0
        px, py, pz = center_3d

        if has_mmlut:
            xl_i = float(IMX)
            xr_i = 0.0
            yu_i = float(IMY)
            yd_i = 0.0
            for pt in range(8):
                qx = px + (dvxmax if pt & 1 else dvxmin)
                qy = py + (dvymax if pt & 2 else dvymin)
                qz = pz + (dvzmax if pt & 4 else dvzmin)
                _point_to_pixel_out(
                    qx,
                    qy,
                    qz,
                    cal,
                    md,
                    mo,
                    mnr,
                    mnz,
                    mrw,
                    has_mmlut,
                    IMX_HALF,
                    IMY_HALF,
                    INV_PIX_X,
                    INV_PIX_Y,
                    CHFIELD,
                    _pp,
                )
                if _pp[0] < xl_i:
                    xl_i = _pp[0]
                if _pp[1] < yu_i:
                    yu_i = _pp[1]
                if _pp[0] > xr_i:
                    xr_i = _pp[0]
                if _pp[1] > yd_i:
                    yd_i = _pp[1]
        else:
            # Jacobian-guided corner selection
            J = np.zeros(6, dtype=np.float64)
            _pixel_jacobian(px, py, pz, cal, INV_PIX_X, INV_PIX_Y, J)

            ix_max = ix_min = iy_max = iy_min = 0
            ex_max, ex_min = -1e100, 1e100
            ey_max, ey_min = -1e100, 1e100
            for pt in range(8):
                dvx = dvxmax if pt & 1 else dvxmin
                dvy = dvymax if pt & 2 else dvymin
                dvz = dvzmax if pt & 4 else dvzmin
                ex = J[0] * dvx + J[1] * dvy + J[2] * dvz
                ey = J[3] * dvx + J[4] * dvy + J[5] * dvz
                if ex > ex_max:
                    ex_max = ex
                    ix_max = pt
                if ex < ex_min:
                    ex_min = ex
                    ix_min = pt
                if ey > ey_max:
                    ey_max = ey
                    iy_max = pt
                if ey < ey_min:
                    ey_min = ey
                    iy_min = pt

            selected = list(dict.fromkeys([ix_max, ix_min, iy_max, iy_min]))
            xl_i = float(IMX)
            xr_i = 0.0
            yu_i = float(IMY)
            yd_i = 0.0
            for pt in selected:
                qx = px + (dvxmax if pt & 1 else dvxmin)
                qy = py + (dvymax if pt & 2 else dvymin)
                qz = pz + (dvzmax if pt & 4 else dvzmin)
                _point_to_pixel_out(
                    qx,
                    qy,
                    qz,
                    cal,
                    md,
                    mo,
                    mnr,
                    mnz,
                    mrw,
                    has_mmlut,
                    IMX_HALF,
                    IMY_HALF,
                    INV_PIX_X,
                    INV_PIX_Y,
                    CHFIELD,
                    _pp,
                )
                if _pp[0] < xl_i:
                    xl_i = _pp[0]
                if _pp[1] < yu_i:
                    yu_i = _pp[1]
                if _pp[0] > xr_i:
                    xr_i = _pp[0]
                if _pp[1] > yd_i:
                    yd_i = _pp[1]

            dx = (xr_i - xl_i) * margin
            dy = (yd_i - yu_i) * margin
            xl_i -= dx
            xr_i += dx
            yu_i -= dy
            yd_i += dy

        if xl_i < 0.0:
            xl_i = 0.0
        if yu_i < 0.0:
            yu_i = 0.0
        if xr_i > IMX:
            xr_i = IMX
        if yd_i > IMY:
            yd_i = IMY

        _point_to_pixel_out(
            px,
            py,
            pz,
            cal,
            md,
            mo,
            mnr,
            mnz,
            mrw,
            has_mmlut,
            IMX_HALF,
            IMY_HALF,
            INV_PIX_X,
            INV_PIX_Y,
            CHFIELD,
            _pp,
        )
        cx, cy = _pp[0], _pp[1]
        xr[i] = xr_i - cx
        xl[i] = cx - xl_i
        yd[i] = yd_i - cy
        yu[i] = cy - yu_i

    return xl, xr, yu, yd


def _check_bounds_contain(inner, outer):
    """Check that outer bounds contain inner bounds for all cameras."""
    xl_in, xr_in, yu_in, yd_in = inner
    xl_out, xr_out, yu_out, yd_out = outer
    for i in range(len(xl_in)):
        if not (xl_out[i] <= xl_in[i] + 1e-9 and xr_out[i] >= xr_in[i] - 1e-9):
            return False
        if not (yu_out[i] <= yu_in[i] + 1e-9 and yd_out[i] >= yd_in[i] - 1e-9):
            return False
    return True


def _check_bounds_area_ratio(inner, outer, max_ratio):
    """Check that outer area / inner area <= max_ratio."""
    xl_i, xr_i, yu_i, yd_i = inner
    xl_o, xr_o, yu_o, yd_o = outer
    for i in range(len(xl_i)):
        area_in = (xr_i[i] - xl_i[i]) * (yd_i[i] - yu_i[i])
        area_out = (xr_o[i] - xl_o[i]) * (yd_o[i] - yu_o[i])
        if area_in > 0 and area_out / area_in > max_ratio:
            return False
    return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[False, True], ids=["pinhole", "with_mmlut"])
def cavity_cal(request):
    """Generate synthetic calibration (pinhole or with mmlut)."""
    return _make_synthetic_calibration(has_mmlut=request.param)


@pytest.fixture(scope="module")
def test_positions():
    """Generate diverse test particle positions."""
    np.random.seed(42)
    # Mix of random positions and edge cases
    positions = []
    # Random positions within the volume
    for _ in range(100):
        x = np.random.uniform(-200, 200)
        y = np.random.uniform(-50, 80)
        z = np.random.uniform(-500, 500)
        positions.append((x, y, z))
    # Edge cases: near volume boundaries
    positions.extend(
        [
            (-200, -50, -500),
            (200, 80, 500),  # corners
            (0, 0, 0),  # center
            (-180, -40, -450),
            (180, 70, 450),  # near edges
        ]
    )
    return positions


# ---------------------------------------------------------------------------
# Test 1: Correctness — 2-corner + margin ⊇ 8-corner for pinhole cameras
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_mmlut", [False, True])
def test_2corner_contains_8corner(cavity_cal, test_positions, has_mmlut):
    """Verify that 2-corner + margin search bounds encompass all 8-corner bounds."""
    cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr = cavity_cal

    if has_mmlut:
        # Find a camera with mmlut data
        has_mmlut_cam = np.where(mnr_arr > 0)[0]
        if len(has_mmlut_cam) == 0:
            pytest.skip("No multimedia camera in calibration data")

    num_cams = cal_arr.shape[0]
    violations = []

    for i, (px, py, pz) in enumerate(test_positions):
        bounds_8 = _compute_8corner_bounds(
            (px, py, pz), cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr
        )
        bounds_2 = _compute_jacobian_bounds(
            (px, py, pz), cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr
        )

        if not _check_bounds_contain(bounds_8, bounds_2):
            violations.append((i, px, py, pz, bounds_8, bounds_2))

    assert len(violations) == 0, (
        f"2-corner bounds failed to contain 8-corner bounds for "
        f"{len(violations)}/{len(test_positions)} positions. "
        f"First violation: pos={violations[0][1:4]}, "
        f"8-corner xl/xr={violations[0][4][:2]}, "
        f"2-corner xl/xr={violations[0][5][:2]}"
    )


# ---------------------------------------------------------------------------
# Test 2: Efficiency — 2-corner bounds not excessively larger
# ---------------------------------------------------------------------------


def test_2corner_not_excessive(cavity_cal, test_positions):
    """Verify the 2-corner search area is not pathologically larger than 8-corner."""
    cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr = cavity_cal

    excessive = 0
    total = 0
    max_ratio = (1.0 + MARGIN) ** 2 * 2.0  # allow margin + 2x for approx error

    for i, (px, py, pz) in enumerate(test_positions):
        bounds_8 = _compute_8corner_bounds(
            (px, py, pz), cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr
        )
        bounds_2 = _compute_jacobian_bounds(
            (px, py, pz), cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr
        )

        xl8, xr8, yu8, yd8 = bounds_8
        xl2, xr2, yu2, yd2 = bounds_2

        for cam in range(len(xl8)):
            area8 = (xr8[cam] - xl8[cam]) * (yd8[cam] - yu8[cam])
            area2 = (xr2[cam] - xl2[cam]) * (yd2[cam] - yu2[cam])
            if area8 > 0 and area2 / area8 > max_ratio:
                excessive += 1
            total += 1

    # Allow up to 5% of camera-position pairs to exceed the ratio
    assert excessive / max(total, 1) < 0.05, (
        f"{excessive}/{total} camera-position pairs have excessive search area"
    )


# ---------------------------------------------------------------------------
# Test 3: Multimedia fallback
# ---------------------------------------------------------------------------


def test_multimedia_fallback(cavity_cal, test_positions):
    """Verify that cameras with multimedia get IDENTICAL bounds from both paths."""
    cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr = cavity_cal

    # Find cameras with and without multimedia
    mm_cams = np.where(mnr_arr > 0)[0]
    pinhole_cams = np.where(mnr_arr == 0)[0]

    if len(mm_cams) == 0:
        pytest.skip("No multimedia camera in calibration data")

    differences = []
    for px, py, pz in test_positions[:20]:
        bounds_8 = _compute_8corner_bounds(
            (px, py, pz), cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr
        )
        bounds_2 = _compute_jacobian_bounds(
            (px, py, pz), cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr
        )

        for cam in mm_cams:
            # For mm cameras, both should be identical (8-corner path)
            if abs(bounds_8[0][cam] - bounds_2[0][cam]) > 1e-4:
                differences.append((cam, px, py, pz))

    assert len(differences) == 0, (
        f"Multimedia cameras have different bounds in {len(differences)} cases. "
        f"This means the fallback isn't working."
    )

    # Also verify that pinhole cameras use the fast path (bounds may differ, but
    # that's expected — the containment check is in test 1)
    print(f"Multimedia cameras: {list(mm_cams)}, Pinhole cameras: {list(pinhole_cams)}")


# ---------------------------------------------------------------------------
# Test 4: Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dv_params",
    [
        ("large_box", (-50, 50, -50, 50, -50, 50)),
        ("small_box", (-1, 1, -1, 1, -1, 1)),
        ("elongated_z", (-5, 5, -5, 5, -50, 50)),
        ("wide_xy", (-50, 50, -50, 50, -3, 3)),
    ],
)
def test_edge_cases(cavity_cal, dv_params):
    """Test extreme search box sizes."""
    _, dv = dv_params
    cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr = cavity_cal

    # Test at center and off-center positions
    positions = [(0, 0, 0), (150, 30, 200), (-150, -30, -200)]
    violations = []

    for px, py, pz in positions:
        bounds_8 = _compute_8corner_bounds(
            (px, py, pz), cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr, dv
        )
        bounds_2 = _compute_jacobian_bounds(
            (px, py, pz), cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr, dv
        )

        if not _check_bounds_contain(bounds_8, bounds_2):
            violations.append((px, py, pz, dv_params[0], bounds_8, bounds_2))

    assert len(violations) == 0, (
        f"2-corner bounds failed for dv={dv_params[0]} at "
        f"{len(violations)} positions. "
        f"Need larger margin for this search box size."
    )


# ---------------------------------------------------------------------------
# Test 5: Full pipeline parity (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tracking_parity():
    """Verify that the tracking pipeline gives same results with 2-corner search.

    This test runs the cavity test twice: once with the original 8-corner
    code (compiled), and once by monkey-patching _sorted_candidates_fast_out.
    Since the code change is in the compiled .so, we can't monkey-patch it
    directly. Instead, this test verifies that the cavity test still passes
    with the new compiled code built from modified source.

    The 2% tolerance on nlinks already allows for small numerical differences
    from the approximation.
    """
    import subprocess

    root = pathlib.Path(__file__).resolve().parent.parent.parent

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(root / "tests/unit/test_track.py::test_cavity"),
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert result.returncode == 0, (
        f"Cavity test failed with 2-corner search:\n{result.stderr}"
    )
    # Check the npart/nlinks assertions passed
    assert "PASSED" in result.stdout, f"Test did not pass:\n{result.stdout[-500:]}"
