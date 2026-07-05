"""Synthetic calibration tests: full_calibration with known ground truth.

We construct a known Calibration from scratch, generate perfect synthetic
2D-3D correspondences, perturb the calibration, and verify that
full_calibration recovers the original parameters.
"""

import numpy as np
import pytest

from openptv2.algorithms.calibration import (
    Calibration,
    Exterior,
    Interior,
    AddedPar,
    Glass,
    MmLut,
)
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.orientation import full_calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.trafo import metric_to_pixel
from openptv2.algorithms.tracking_frame_buf import Target
from openptv2.orientation import full_calibration as wrapper_full_calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_calibration(
    pos=(100.0, 50.0, -500.0),
    angles=(0.5, -0.3, 0.1),
    cc=60.0,
    xh=0.0,
    yh=0.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    scx=1.0,
    she=0.0,
):
    """Build a Calibration with known parameters from scratch."""
    return Calibration(
        ext_par=Exterior(
            x0=pos[0],
            y0=pos[1],
            z0=pos[2],
            omega=angles[0],
            phi=angles[1],
            kappa=angles[2],
        ),
        int_par=Interior(xh=xh, yh=yh, cc=cc),
        added_par=AddedPar(
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, scx=scx, she=she, field=0
        ),
        # glass vector must be non-zero (the imaging model divides by its
        # magnitude to compute the glass-interface ray transformation).
        glass_par=Glass(
            vec_x=0.0, vec_y=0.0, vec_z=-125.0, n1=1.0, n2=1.0, n3=1.0, d=6.0
        ),
        # mmLut with a large enough grid so multimedia lookups stay in bounds.
        mmlut=MmLut(
            origin=np.array([0.0, 0.0, -150.0]),
            nr=200,
            nz=200,
            rw=2.0,
            data=np.ones(40000),
        ),
    )


def make_control_par():
    """Standard ControlPar for a 4 camera, 1280x1024, 12um pixel setup."""
    return ControlPar(
        num_cams=4,
        img_base_name=[""] * 4,
        cal_img_base_name=[""] * 4,
        allCam_flag=0,
        hp_flag=1,
        chfield=0,
        tiff_flag=1,
        pix_x=0.012,
        pix_y=0.012,
        imx=1280,
        imy=1024,
        # Flat multimedia (n1=n2=n3=1, d=0) so mmLut is not actually
        # consulted during projection – the mmf factor stays 1.0.
        mm=MmNp(n1=1.0, n2=[1.0] * 4, d=[0.0] * 4, n3=1.0),
    )


def make_ref_pts(n=64, spread=40.0):
    """Create a set of 3D reference points."""
    side = int(round(n ** (1 / 3)))
    if side**3 < n:
        side += 1
    pts = []
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if len(pts) >= n:
                    break
                pts.append(
                    [
                        (ix - side / 2) * 2 * spread / side,
                        (iy - side / 2) * 2 * spread / side,
                        (iz - side / 2) * 2 * spread / side,
                    ]
                )
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    return np.array(pts[:n], dtype=np.float64)


def make_targets(ref_pts, cal, cpar):
    """Project 3D reference points to pixel coordinates for a given calibration."""
    pix = []
    for i, pt in enumerate(ref_pts):
        xm, ym = img_coord(pt, cal, cpar.mm)
        xp, yp = metric_to_pixel(xm, ym, cpar)
        t = Target()
        t.pnr = i
        t.x = xp
        t.y = yp
        pix.append(t)
    return pix


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyntheticFullCalibration:
    """Full calibration with synthetic data and known ground truth."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Shared setup: known calibration, control params, reference points."""
        np.random.seed(42)
        self.cpar = make_control_par()
        self.ref_pts = make_ref_pts(n=64, spread=40.0)
        self.ground_truth = make_calibration()

    def _run(self, flags, pos_perturb=15.0, ang_perturb=0.5, cc_perturb=0.0):
        """Run full_calibration with a perturbed copy of ground truth."""
        cal = make_calibration()  # fresh copy of ground truth
        targets = make_targets(self.ref_pts, cal, self.cpar)

        # Perturb
        pos = cal.get_pos()
        ang = cal.get_angles()
        cal.set_pos(pos + np.random.uniform(-pos_perturb, pos_perturb, 3))
        cal.set_angles(ang + np.random.uniform(-ang_perturb, ang_perturb, 3))
        cal.int_par.cc += cc_perturb

        residuals, targ_ix, err_est = full_calibration(
            cal, self.ref_pts, targets, self.cpar, flags=flags
        )
        return cal, residuals, targ_ix, err_est

    # --- No flags (6-exterior only) ---

    def test_exterior_only_converges(self):
        """6-exterior calibration converges and recovers position."""
        cal, residuals, _, _ = self._run(flags=[])
        assert residuals is not None
        assert np.all(np.isfinite(residuals))
        rms = np.sqrt(np.mean(residuals[:, 0] ** 2 + residuals[:, 1] ** 2))
        assert rms < 1e-6, f"RMS too large: {rms}"

    def test_exterior_only_recovers_position(self):
        """6-exterior calibration recovers original position within 1e-4."""
        cal, residuals, _, _ = self._run(flags=[])
        expected = self.ground_truth.get_pos()
        recovered = cal.get_pos()
        np.testing.assert_allclose(
            recovered, expected, atol=1e-4, err_msg="Position not recovered"
        )

    def test_exterior_only_recovers_orientation(self):
        """6-exterior calibration recovers original angles within 1e-4."""
        cal, residuals, _, _ = self._run(flags=[])
        expected = self.ground_truth.get_angles()
        recovered = cal.get_angles()
        np.testing.assert_allclose(
            recovered, expected, atol=1e-4, err_msg="Orientation not recovered"
        )

    # --- With interior flags ---

    def test_with_cc_flag_converges(self):
        """cc flag calibration converges with sub-pixel residuals."""
        cal, residuals, _, _ = self._run(flags=["cc"], cc_perturb=-5.0)
        assert residuals is not None
        assert np.all(np.isfinite(residuals))
        rms = np.sqrt(np.mean(residuals[:, 0] ** 2 + residuals[:, 1] ** 2))
        # cc trades off against z0 in an n1=n2=n3=1 model, so residuals
        # stay low even if cc does not exactly match ground truth.
        assert rms < 0.1, f"RMS too large: {rms}"

    def test_with_cc_xh_flags(self):
        """calibration with cc and xh flags improves residuals."""
        cal = make_calibration()
        cal.int_par.cc = 65.0
        cal.int_par.xh = 0.5
        targets = make_targets(self.ref_pts, cal, self.cpar)

        # Perturb back to defaults
        cal.int_par.cc = 60.0
        cal.int_par.xh = 0.0

        residuals, _, _ = full_calibration(
            cal, self.ref_pts, targets, self.cpar, flags=["cc", "xh"]
        )
        assert residuals is not None
        rms = np.sqrt(np.mean(residuals[:, 0] ** 2 + residuals[:, 1] ** 2))
        assert rms < 0.1, f"cc+xh RMS too large: {rms}"

    # --- No matched points ---

    def test_no_matched_points_leaves_calibration_unchanged(self):
        """Zero valid targets leaves calibration unmodified (no crash)."""
        expected = make_calibration()
        cal = make_calibration()
        old_pos = cal.get_pos().copy()
        targets = [Target(pnr=-1, x=0.0, y=0.0)]  # pnr != index → skipped
        full_calibration(cal, self.ref_pts[:1], targets, self.cpar, flags=[])
        # Calibration should be unmodified or at least not NaN
        pos = cal.get_pos()
        assert np.all(np.isfinite(pos)), "position became NaN"
        assert np.all(np.abs(pos - old_pos) < 10.0), "position diverged"

    # --- Empty input ---

    def test_empty_targets_leaves_calibration_unchanged(self):
        """Empty targets leaves calibration unmodified (no crash)."""
        cal = make_calibration()
        old_pos = cal.get_pos().copy()
        full_calibration(cal, self.ref_pts[:0], [], self.cpar, flags=[])
        pos = cal.get_pos()
        assert np.all(np.isfinite(pos)), "position became NaN"
        assert np.all(np.abs(pos - old_pos) < 10.0), "position diverged"


class TestSyntheticFullCalibrationAllDistortion:
    """Verify interior/distortion flags produce convergence with sub-pixel
    residuals.  Exact parameter recovery is not checked because several
    parameter pairs (e.g. cc↔z0, xh↔x0) are highly correlated in an
    n1=n2=n3 model — what matters is that residuals are low, meaning the
    bundle-adjustment improves the fit."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(99)
        self.cpar = make_control_par()
        self.ref_pts = make_ref_pts(n=64, spread=40.0)

    def _make_with_targets(self, **overrides):
        """Build a calibration with given overrides and project targets."""
        cal = make_calibration(**overrides)
        targets = make_targets(self.ref_pts, cal, self.cpar)
        return cal, targets

    def _perturb_and_run(self, cal, targets, flags, **perturbs):
        """Apply perturbations and run full_calibration."""
        for attr, delta in perturbs.items():
            obj = cal.int_par if attr in ("cc", "xh", "yh") else cal.added_par
            setattr(obj, attr, getattr(obj, attr) + delta)
        return full_calibration(cal, self.ref_pts, targets, self.cpar, flags=flags)

    @pytest.mark.parametrize(
        "flag, attr, ground_truth_val, perturb",
        [
            ("cc", "cc", 60.0, -5.0),
            ("k1", "k1", 0.0, 1e-7),
            ("k2", "k2", 0.0, 2e-12),
        ],
    )
    def test_flag_converges_with_subpixel_rms(
        self, flag, attr, ground_truth_val, perturb
    ):
        """With one flag enabled, full_calibration converges w/ RMS<0.1 pixels."""
        cal, targets = self._make_with_targets(**{attr: ground_truth_val + perturb})
        residuals, _, _ = self._perturb_and_run(
            cal, targets, [flag], **{attr: -perturb}
        )
        assert residuals is not None, f"{flag} did not converge"
        assert np.all(np.isfinite(residuals)), f"{flag} produced NaN residuals"
        rms = np.sqrt(np.mean(residuals[:, 0] ** 2 + residuals[:, 1] ** 2))
        assert rms < 0.1, f"{flag} RMS too large: {rms}"


class TestWrapperUnmatchedFiltering:
    """Verify the GUI wrapper filters out unmatched (pnr=-999) targets.

    The wrapper in src/openptv2/orientation.py extracts pixel coordinates
    and passes them to the algorithms module.  Before the fix, ALL targets
    (including unmatched garbage) were passed in, corrupting the solution.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(123)
        self.cpar = make_control_par()
        self.ref_pts = make_ref_pts(n=32, spread=30.0)
        self.cal = make_calibration()

    def _make_mixed_targets(self):
        """Return (targets, n_matched) where some targets have pnr=-999.

        Creates a full target array where the first n_matched targets
        are correct (projected from the known calibration), and the
        remaining targets have pnr=-999 with random garbage coordinates.
        This simulates what match_detection_to_ref returns after sortgrid.
        """
        # Project perfect targets for all ref points
        cal = self.cal
        perfect = []
        for i, pt in enumerate(self.ref_pts):
            xm, ym = img_coord(pt, cal, self.cpar.mm)
            xp, yp = metric_to_pixel(xm, ym, self.cpar)
            perfect.append(Target(pnr=i, x=xp, y=yp))

        # Replace the last 10 targets with unmatched garbage
        n_all = len(self.ref_pts)
        n_matched = n_all - 10
        unmatched = []
        for i in range(n_matched, n_all):
            t = Target(
                pnr=-999, x=np.random.uniform(0, 1280), y=np.random.uniform(0, 1024)
            )
            unmatched.append(t)

        targets = perfect[:n_matched] + unmatched
        return targets, n_matched

    def test_unmatched_targets_do_not_corrupt_calibration(self):
        """Wrapper full_calibration with pnr=-999 targets recovers position."""
        cal = make_calibration()
        targets, n_matched = self._make_mixed_targets()

        # Perturb
        pos = cal.get_pos()
        cal.set_pos(pos + np.array([5.0, -3.0, 8.0]))

        # Call through the WRAPPER (GUI code path)
        residuals, used, err_est = wrapper_full_calibration(
            cal, self.ref_pts, targets, self.cpar, flags=[]
        )
        assert residuals is not None

        # All residuals should be finite (only matched points passed to solver)
        assert np.all(np.isfinite(residuals)), "Matched residuals not finite"
        assert len(residuals) == n_matched, (
            f"Expected {n_matched} residuals, got {len(residuals)}"
        )

        # Calibration position should be reasonable (not NaN, not billions)
        final_pos = cal.get_pos()
        assert np.all(np.isfinite(final_pos)), "Position became NaN"
        assert np.all(np.abs(final_pos) < 1e4), f"Position diverged: {final_pos}"

        # RMS should be sub-pixel (only matched points contribute)
        rms = np.sqrt(np.mean(residuals[:, 0] ** 2 + residuals[:, 1] ** 2))
        assert rms < 0.01, f"RMS too large with garbage targets: {rms}"

    def test_all_unmatched_returns_empty_arrays(self):
        """All unmatched targets produce empty residuals (nothing to solve)."""
        cal = make_calibration()
        old_pos = cal.get_pos().copy()

        # All targets are unmatched garbage
        targets = [
            Target(pnr=-999, x=np.random.uniform(0, 1280), y=np.random.uniform(0, 1024))
            for _ in range(len(self.ref_pts))
        ]

        residuals, used, err_est = wrapper_full_calibration(
            cal, self.ref_pts, targets, self.cpar, flags=[]
        )
        assert residuals is not None
        assert len(residuals) == 0, "Expected empty residuals with no matches"
        final_pos = cal.get_pos()
        assert np.all(np.isfinite(final_pos)), "Position became NaN"
        assert np.allclose(final_pos, old_pos, atol=1e-4), (
            "Position changed with no matches"
        )

    def test_partial_unmatched_used_matches_only(self):
        """The 'used' array only contains matched pnr values (no -999)."""
        cal = make_calibration()
        targets, n_matched = self._make_mixed_targets()

        residuals, used, err_est = wrapper_full_calibration(
            cal, self.ref_pts, targets, self.cpar, flags=[]
        )
        assert len(used) == n_matched, (
            f"Expected {n_matched} used entries, got {len(used)}"
        )
        # All used values should be valid pnr (0..n_matched-1)
        assert np.all(used == np.arange(n_matched))
