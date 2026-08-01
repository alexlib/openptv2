"""
Pure-Python line coverage tests for openptv2.algorithms.tracking_run.

Run via the coverage recipe:
    COVERAGE_FILE=/tmp/.cov_trun uv run pytest tests/unit/test_tracking_run_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Skip when the compiled .so is active
# ---------------------------------------------------------------------------
from openptv2.algorithms.tracking_run import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

from openptv2.algorithms.parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
    convert_track_par_to_tuple,
)
from openptv2.algorithms.tracking_run import TrackingRun, tr_new

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_FRAMEBUF = "openptv2.algorithms.tracking_run.FrameBuf"
_VOLDIM = "openptv2.algorithms.multimed.volumedimension"
_INIT_MMLUT = "openptv2.algorithms.multimed.init_mmlut"

# volumedimension returns (xmax, xmin, ymax, ymin, zmax, zmin)
_VOL_RETURN = (10.0, -10.0, 5.0, -5.0, 8.0, -8.0)


# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------


def _seq(num_cams=1):
    return SequencePar(
        num_cams=num_cams, img_base_name=["img"] * num_cams, first=1, last=10
    )


def _tpar():
    tp = TrackPar()
    tp.dvxmin = -2.0
    tp.dvxmax = 2.0
    tp.dvymin = -2.0
    tp.dvymax = 2.0
    tp.dvzmin = -2.0
    tp.dvzmax = 2.0
    tp.dangle = 0.1
    tp.dacc = 0.1
    tp.add = 1
    return tp


def _vpar():
    return VolumePar()


def _cpar(num_cams=1):
    return ControlPar(num_cams=num_cams)


def _cal(initialized=True):
    c = MagicMock()
    c.mmlut.is_initialized = initialized
    return c


def _make_run(cal_list=None, num_cams=1):
    """Build a TrackingRun with all heavy deps mocked."""
    if cal_list is None:
        cal_list = [_cal(initialized=True)]
    seq = _seq(num_cams)
    tpar_tuple = convert_track_par_to_tuple(_tpar())
    vpar = _vpar()
    cpar = _cpar(num_cams)
    with (
        patch(_FRAMEBUF) as fb,
        patch(_VOLDIM, return_value=_VOL_RETURN),
        patch(_INIT_MMLUT),
    ):
        fb.return_value = MagicMock()
        run = TrackingRun(
            seq_par=seq,
            tpar=tpar_tuple,
            vpar=vpar,
            cpar=cpar,
            buf_len=4,
            max_targets=100,
            corres_file_base="corr",
            linkage_file_base="link",
            prio_file_base="prio",
            cal=cal_list,
            flatten_tol=0.1,
        )
    return run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsCompiled:
    def test_returns_bool(self):
        result = _is_compiled()
        assert isinstance(result, bool)

    def test_is_false_in_pure_python(self):
        assert _is_compiled() is False


class TestTrackingRunPostInit:
    def test_basic_construction(self):
        run = _make_run()
        assert run.npart == 0
        assert run.nlinks == 0

    def test_lmax_computed(self):
        """lmax = norm of (dvxmin-dvxmax, dvymin-dvymax, dvzmin-dvzmax)."""
        run = _make_run()
        import numpy as np

        expected = np.linalg.norm([-4.0, -4.0, -4.0])
        assert abs(run.lmax - expected) < 1e-10

    def test_ymin_ymax_from_volumedimension(self):
        """ymax and ymin come from volumedimension return value (index 2, 3)."""
        run = _make_run()
        # _VOL_RETURN = (xmax=10, xmin=-10, ymax=5, ymin=-5, zmax=8, zmin=-8)
        assert run.ymax == 5.0
        assert run.ymin == -5.0

    def test_vpar_xlayers_updated(self):
        """vpar.X_lay[0]/[1] and Zmax_lay[1]/Zmin_lay[0] are set."""
        cal = [_cal(initialized=True)]
        seq = _seq(1)
        tpar_tuple = convert_track_par_to_tuple(_tpar())
        vpar = _vpar()
        cpar = _cpar(1)
        with (
            patch(_FRAMEBUF) as fb,
            patch(_VOLDIM, return_value=_VOL_RETURN),
            patch(_INIT_MMLUT),
        ):
            fb.return_value = MagicMock()
            run = TrackingRun(
                seq, tpar_tuple, vpar, cpar, 4, 100, "c", "l", "p", cal, 0.1
            )
        assert run.vpar.X_lay[1] == 10.0  # xmax
        assert run.vpar.X_lay[0] == -10.0  # xmin
        assert run.vpar.Zmax_lay[1] == 8.0
        assert run.vpar.Zmin_lay[0] == -8.0

    def test_init_mmlut_called_for_uninitialized_cal(self):
        """init_mmlut must be called when is_initialized is False."""
        cal_uninit = _cal(initialized=False)
        seq = _seq(1)
        tpar_tuple = convert_track_par_to_tuple(_tpar())
        vpar = _vpar()
        cpar = _cpar(1)
        with (
            patch(_FRAMEBUF) as fb,
            patch(_VOLDIM, return_value=_VOL_RETURN),
            patch(_INIT_MMLUT) as mock_im,
        ):
            fb.return_value = MagicMock()
            TrackingRun(
                seq, tpar_tuple, vpar, cpar, 4, 100, "c", "l", "p", [cal_uninit], 0.1
            )
            assert mock_im.called

    def test_init_mmlut_not_called_for_initialized_cal(self):
        """init_mmlut must NOT be called when is_initialized is True."""
        cal_init = _cal(initialized=True)
        seq = _seq(1)
        tpar_tuple = convert_track_par_to_tuple(_tpar())
        vpar = _vpar()
        cpar = _cpar(1)
        with (
            patch(_FRAMEBUF) as fb,
            patch(_VOLDIM, return_value=_VOL_RETURN),
            patch(_INIT_MMLUT) as mock_im,
        ):
            fb.return_value = MagicMock()
            TrackingRun(
                seq, tpar_tuple, vpar, cpar, 4, 100, "c", "l", "p", [cal_init], 0.1
            )
            mock_im.assert_not_called()

    def test_empty_cal_list(self):
        """Empty cal list: no iteration, no crash."""
        run = _make_run(cal_list=[])
        assert hasattr(run, "fb")

    def test_mixed_cal_list(self):
        """Only uninitialized cals get init_mmlut called."""
        cal_a = _cal(initialized=True)
        cal_b = _cal(initialized=False)
        cal_c = _cal(initialized=False)
        seq = _seq(1)
        tpar_tuple = convert_track_par_to_tuple(_tpar())
        vpar = _vpar()
        cpar = _cpar(1)
        with (
            patch(_FRAMEBUF) as fb,
            patch(_VOLDIM, return_value=_VOL_RETURN),
            patch(_INIT_MMLUT) as mock_im,
        ):
            fb.return_value = MagicMock()
            TrackingRun(
                seq,
                tpar_tuple,
                vpar,
                cpar,
                4,
                100,
                "c",
                "l",
                "p",
                [cal_a, cal_b, cal_c],
                0.1,
            )
            assert mock_im.call_count == 2

    def test_framebuf_constructed_with_correct_args(self):
        """FrameBuf receives buf_len, num_cams, max_targets, and file bases."""
        seq = _seq(2)
        tpar_tuple = convert_track_par_to_tuple(_tpar())
        vpar = _vpar()
        cpar = _cpar(2)
        with (
            patch(_FRAMEBUF) as MockFB,
            patch(_VOLDIM, return_value=_VOL_RETURN),
            patch(_INIT_MMLUT),
        ):
            MockFB.return_value = MagicMock()
            TrackingRun(
                seq, tpar_tuple, vpar, cpar, 5, 200, "corr_", "link_", "prio_", [], 0.0
            )
            MockFB.assert_called_once_with(
                5, 2, 200, "corr_", "link_", "prio_", seq.img_base_name
            )


class TestTrNew:
    """Tests for tr_new — exercises all str/object branches."""

    def _run_tr_new(self, seq_par, tpar, vpar, cpar, cal=None):
        if cal is None:
            cal = [_cal(initialized=True)]
        with (
            patch(_FRAMEBUF) as fb,
            patch(_VOLDIM, return_value=_VOL_RETURN),
            patch(_INIT_MMLUT),
        ):
            fb.return_value = MagicMock()
            return tr_new(
                seq_par,
                tpar,
                vpar,
                cpar,
                buf_len=4,
                max_targets=50,
                corres_file_base="c",
                linkage_file_base="l",
                prio_file_base="p",
                cal=cal,
                flatten_tol=0.0,
            )

    def test_all_objects(self):
        """All params as objects — no from_file calls."""
        tpar_tuple = convert_track_par_to_tuple(_tpar())
        run = self._run_tr_new(_seq(), tpar_tuple, _vpar(), _cpar())
        assert isinstance(run, TrackingRun)

    def test_trackpar_object_converted(self):
        """TrackPar object is converted to tuple by tr_new."""
        tp = _tpar()
        run = self._run_tr_new(_seq(), tp, _vpar(), _cpar())
        assert isinstance(run, TrackingRun)

    def test_cpar_as_string(self):
        """cpar as a str triggers ControlPar.from_file."""
        cpar_obj = _cpar(1)
        with patch.object(ControlPar, "from_file", return_value=cpar_obj) as mock_cf:
            run = self._run_tr_new(_seq(), _tpar(), _vpar(), "/fake/ptv.par")
            mock_cf.assert_called_once_with("/fake/ptv.par")
        assert isinstance(run, TrackingRun)

    def test_seq_par_as_string(self):
        """seq_par as a str triggers SequencePar.from_file."""
        cpar_obj = _cpar(1)
        seq_obj = _seq(1)
        with (
            patch.object(ControlPar, "from_file", return_value=cpar_obj),
            patch.object(SequencePar, "from_file", return_value=seq_obj) as mock_sf,
        ):
            run = self._run_tr_new("/fake/seq.par", _tpar(), _vpar(), "/fake/ptv.par")
            mock_sf.assert_called_once()
        assert isinstance(run, TrackingRun)

    def test_tpar_as_string(self):
        """tpar as a str triggers TrackPar.from_file then conversion."""
        tp = _tpar()
        with patch.object(TrackPar, "from_file", return_value=tp) as mock_tf:
            run = self._run_tr_new(_seq(), "/fake/track.par", _vpar(), _cpar())
            mock_tf.assert_called_once_with("/fake/track.par")
        assert isinstance(run, TrackingRun)

    def test_vpar_as_string(self):
        """vpar as a str triggers VolumePar.from_file."""
        vp = _vpar()
        with patch.object(VolumePar, "from_file", return_value=vp) as mock_vf:
            run = self._run_tr_new(_seq(), _tpar(), "/fake/criteria.par", _cpar())
            mock_vf.assert_called_once_with("/fake/criteria.par")
        assert isinstance(run, TrackingRun)

    def test_all_strings(self):
        """All four params as strings — all from_file methods called."""
        cpar_obj = _cpar(1)
        seq_obj = _seq(1)
        tp = _tpar()
        vp = _vpar()
        with (
            patch.object(ControlPar, "from_file", return_value=cpar_obj),
            patch.object(SequencePar, "from_file", return_value=seq_obj),
            patch.object(TrackPar, "from_file", return_value=tp),
            patch.object(VolumePar, "from_file", return_value=vp),
        ):
            run = self._run_tr_new(
                "/fake/seq.par",
                "/fake/track.par",
                "/fake/criteria.par",
                "/fake/ptv.par",
            )
        assert isinstance(run, TrackingRun)

    def test_returns_tracking_run_instance(self):
        """tr_new always returns a TrackingRun."""
        run = self._run_tr_new(_seq(), _tpar(), _vpar(), _cpar())
        assert type(run).__name__ == "TrackingRun"

    def test_tpar_tuple_passthrough(self):
        """A pre-converted tuple is not re-converted (isinstance check skips it)."""
        tp = _tpar()
        tpar_tuple = convert_track_par_to_tuple(tp)
        # Pass the tuple directly — should not hit convert_track_par_to_tuple again
        run = self._run_tr_new(_seq(), tpar_tuple, _vpar(), _cpar())
        assert isinstance(run, TrackingRun)
