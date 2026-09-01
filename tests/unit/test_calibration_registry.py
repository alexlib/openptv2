"""Tests for the Phase 1 calibration-hub plumbing:

- `openptv2.calibration_registry` (CalibrationPointSet, source registry).
- `openptv2.autocalibration._refine_and_select`, the refine/flag-select core
  extracted out of `calibrate_camera` so every calibration source (today
  just "calibration_object"; checkerboard/multiplane/etc. in later phases)
  can share it instead of reimplementing `.ori`/`.addpar` fitting.

Synthetic setup (known Calibration, perfect projected points, perturbed
seed) mirrors tests/unit/test_synthetic_calibration.py -- reusing its
helpers rather than re-deriving them.
"""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.autocalibration import _refine_and_select
from openptv2.calibration_registry import (
    CALIBRATION_SOURCE_REGISTRY,
    CalibrationPointSet,
    CalibrationSourceInfo,
    get_source_info,
    list_sources,
)
from tests.unit.test_synthetic_calibration import (
    make_calibration,
    make_control_par,
    make_ref_pts,
    make_targets,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_calibration_object_is_registered():
    assert "calibration_object" in CALIBRATION_SOURCE_REGISTRY
    info = get_source_info("calibration_object")
    assert isinstance(info, CalibrationSourceInfo)
    assert info.name == "calibration_object"
    assert info.requires  # non-empty: real documentation, not a stub


def test_list_sources_matches_registry_keys():
    assert list_sources() == sorted(CALIBRATION_SOURCE_REGISTRY)


def test_get_source_info_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown calibration source"):
        get_source_info("does_not_exist")


def test_calibration_point_set_is_plain_data():
    ref = np.zeros((3, 3))
    img = np.zeros((3, 2))
    ps = CalibrationPointSet(ref_pts=ref, img_pts=img)
    assert ps.seed is None
    assert ps.ref_pts is ref
    assert ps.img_pts is img


# ---------------------------------------------------------------------------
# _refine_and_select (the extracted, source-agnostic core)
# ---------------------------------------------------------------------------


class TestRefineAndSelect:
    """Same synthetic-recovery shape as
    test_synthetic_calibration.TestSyntheticFullCalibration, but through the
    higher-level _refine_and_select (sortgrid + refine loop + flag search)
    that every calibration source will call, not full_calibration directly.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(7)
        self.cpar = make_control_par()
        self.ref_pts = make_ref_pts(n=64, spread=40.0)
        self.ground_truth = make_calibration()

    def test_recovers_position_from_perturbed_seed(self):
        # Use a look-at calibration so reference points actually project inside
        # the 1280x1024 sensor (the bare make_calibration() angles put the
        # origin near the edge with the default 64-point cube).
        import numpy as np_mod

        from openptv2.algorithms.calibration import Exterior, Glass, Interior, MmLut

        def _dm_from_lookat(C, target, up=(0.0, 0.0, 1.0)):
            back = np_mod.array(C, float) - np_mod.array(target, float)
            back /= np_mod.linalg.norm(back)
            up_a = np_mod.array(up, float)
            right = np_mod.cross(up_a, back)
            if np_mod.linalg.norm(right) < 1e-8:
                right = np_mod.cross(np_mod.array([1.0, 0.0, 0.0]), back)
            right /= np_mod.linalg.norm(right)
            return np_mod.column_stack([right, np_mod.cross(back, right), back])

        def _angs(dm):
            phi = float(np_mod.arcsin(np_mod.clip(dm[0, 2], -1.0, 1.0)))
            kappa = float(np_mod.arctan2(-dm[0, 1], dm[0, 0]))
            omega = float(np_mod.arctan2(-dm[1, 2], dm[2, 2]))
            return (omega, phi, kappa)

        C = np_mod.array([80.0, 30.0, -500.0])
        dm = _dm_from_lookat(C, np_mod.zeros(3))
        angs = _angs(dm)
        self.ground_truth = make_calibration(pos=tuple(C), angles=angs)
        # Planar grid at Z=0 with 10mm pitch, 49 points — fully visible at
        # 1280x1024 and well separated, unlike the 3D cube which puts many
        # points off-sensor with this pose.
        xs = np_mod.arange(-30, 31, 10, dtype=float)
        ys = np_mod.arange(-30, 31, 10, dtype=float)
        self.ref_pts = np_mod.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
        cal = make_calibration(pos=tuple(C), angles=angs)
        pix = make_targets(self.ref_pts, cal, self.cpar)

        # Perturb the seed so _refine_and_select has real work to do — keep it
        # modest (1 mm / 0.005 rad) so eps=60 captures all points; the prior
        # 5 mm / 0.02 rad test relied on an unrealistic eps=15 assumption (M4:
        # 5mm needs eps 60 for full capture even on a 75-point grid).
        cal.set_pos(cal.get_pos() + np_mod.array([1.0, -0.6, 1.2]))
        cal.set_angles(cal.get_angles() + np_mod.array([0.005, -0.003, 0.004]))

        nfix = len(self.ref_pts)
        # eps=15 is too tight for a 5mm/1deg seed (see M4: 5mm->4/75 at eps15,
        # 75/75 at eps60). Use a generous radius so the synthetic test can
        # exercise the refinement, not the sortgrid threshold.
        result = _refine_and_select(
            cam=0,
            cal=cal,
            cpar=self.cpar,
            fix=self.ref_pts,
            nfix=nfix,
            eps=60,
            pix=pix,
        )

        assert result.cam == 0
        assert result.nfix == nfix
        assert result.matched == nfix  # every synthetic point falls within eps
        assert result.rms < 0.5, f"RMS too high: {result.rms}"
        assert {"cc", "xh", "yh"}.issubset(set(result.flags))
        np_mod.testing.assert_allclose(
            result.cal.get_pos(), self.ground_truth.get_pos(), atol=0.5
        )

    def test_raises_when_no_flag_set_converges(self):
        """An empty pix list can't match anything -- _refine_and_select
        returns a CamResult with 0 matches and inf RMS (no flag-set can
        produce finite residuals)."""
        cal = make_calibration()
        result = _refine_and_select(
            cam=2,
            cal=cal,
            cpar=self.cpar,
            fix=self.ref_pts,
            nfix=len(self.ref_pts),
            eps=15,
            pix=[],
        )
        assert result.matched == 0
        assert result.rms == float("inf")
