"""Tests for algorithms.batch — the pure-Python batch runner.

Compares results from:
  1. algorithms.batch          (pure Python, no Cython)
  2. gui.pyptv.pyptv_batch     (Cython / optv engine)
  3. res_orig/ reference files (shipped with test data)

Strategy: heavy sequence-loop tests process only 1 frame to keep runtime
under ~60 s.  Tracking-only tests copy pre-existing res_orig/ into res/ so
the tracker can run without repeating the sequence loop.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from .conftest import FIXTURES

TEST_DATA_DIR = FIXTURES / "test_cavity"
YAML_FILE = TEST_DATA_DIR / "parameters_Run1.yaml"

# Single frame used by most tests — keeps Numba JIT + segmentation cost low.
FRAME = 10001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_rt_is(path: Path):
    """Read an rt_is.* file → (num_points, ndarray shape (N, 8))."""
    if not path.exists():
        return 0, np.empty((0, 8))
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return 0, np.empty((0, 8))
    n = int(lines[0].strip())
    if n == 0 or len(lines) <= 1:
        return n, np.empty((0, 8))
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 8:
            rows.append([float(v) for v in parts[:8]])
    return n, np.array(rows) if rows else np.empty((0, 8))


def _copy_test_env(dest: Path, *, copy_res_orig: bool = False) -> Path:
    """Clone test_cavity into *dest* (img/ is symlinked, not copied).

    If *copy_res_orig* is True the shipped ``res_orig/`` is copied into
    ``res/`` so tracking-only tests have pre-existing correspondence files.
    """
    dest.mkdir(parents=True, exist_ok=True)

    for name in ("img", "img_orig"):
        src = TEST_DATA_DIR / name
        if src.exists():
            link = dest / name
            if not link.exists():
                link.symlink_to(src.resolve())

    for item in TEST_DATA_DIR.iterdir():
        if item.is_dir() and item.name not in (
            "img", "img_orig", "res", "res_orig", "res_optv", "__pycache__",
        ):
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dest / item.name)

    res = dest / "res"
    res.mkdir(exist_ok=True)
    if copy_res_orig:
        src_res = TEST_DATA_DIR / "res_orig"
        if src_res.exists():
            for f in src_res.iterdir():
                shutil.copy2(f, res / f.name)

    yaml_files = list(dest.glob("parameters*.yaml"))
    assert yaml_files, f"No YAML file found in {dest}"
    return yaml_files[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def py_batch_dir(tmp_path):
    """Temp dir for pure-Python batch (no pre-existing res/)."""
    yaml_file = _copy_test_env(tmp_path / "py")
    return tmp_path / "py", yaml_file


@pytest.fixture
def py_batch_dir_with_res(tmp_path):
    """Temp dir with res_orig/ copied into res/ (for tracking-only tests)."""
    yaml_file = _copy_test_env(tmp_path / "py_res", copy_res_orig=True)
    return tmp_path / "py_res", yaml_file


# ---------------------------------------------------------------------------
# Unit tests — parameter building (fast, no I/O)
# ---------------------------------------------------------------------------

class TestParameterBuilding:
    """Verify YAML → pure-Python parameter objects."""

    @pytest.fixture(autouse=True)
    def _load_yaml(self):
        with open(YAML_FILE) as f:
            self.params = yaml.safe_load(f)

    def test_build_control_par(self):
        from algorithms.batch import _build_control_par

        cpar = _build_control_par(self.params["ptv"], self.params["num_cams"])
        assert cpar.num_cams == 4
        assert cpar.imx == self.params["ptv"]["imx"]
        assert cpar.imy == self.params["ptv"]["imy"]
        assert cpar.pix_x == self.params["ptv"]["pix_x"]
        assert len(cpar.img_base_name) == 4

    def test_build_sequence_par(self):
        from algorithms.batch import _build_sequence_par

        spar = _build_sequence_par(self.params["sequence"], self.params["num_cams"])
        assert spar.first == self.params["sequence"]["first"]
        assert spar.last == self.params["sequence"]["last"]
        assert len(spar.img_base_name) == 4

    def test_build_volume_par(self):
        from algorithms.batch import _build_volume_par

        vpar = _build_volume_par(self.params["criteria"])
        assert vpar.cn == self.params["criteria"]["cn"]
        assert vpar.corrmin == self.params["criteria"]["corrmin"]

    def test_build_track_par(self):
        from algorithms.batch import _build_track_par

        tp = _build_track_par(self.params["track"])
        assert tp.dvxmin == self.params["track"]["dvxmin"]
        assert tp.dangle == self.params["track"]["angle"]

    def test_build_target_par(self):
        from algorithms.batch import _build_target_par

        tp = _build_target_par(self.params["targ_rec"], self.params["num_cams"])
        assert len(tp.gvthresh) == 4
        assert tp.nnmin == self.params["targ_rec"]["nnmin"]

    def test_read_calibrations(self):
        from algorithms.batch import _read_calibrations_py

        original_cwd = os.getcwd()
        try:
            os.chdir(TEST_DATA_DIR)
            cals = _read_calibrations_py(
                self.params["cal_ori"], self.params["num_cams"]
            )
            assert len(cals) == 4
            for cal in cals:
                assert hasattr(cal, "ext_par") or hasattr(cal, "ext")
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Integration — pure-Python batch (1 frame to stay fast)
# ---------------------------------------------------------------------------

class TestPythonBatch:
    """Run algorithms.batch on test_cavity and validate output."""

    @pytest.mark.slow
    def test_sequence_produces_rt_is(self, py_batch_dir):
        """Sequence on a single frame must produce an rt_is file."""
        from algorithms.batch import run_batch

        work_dir, yaml_file = py_batch_dir
        run_batch(yaml_file, FRAME, FRAME, mode="sequence")

        rt = work_dir / "res" / f"rt_is.{FRAME}"
        assert rt.exists(), f"Missing {rt}"
        n, _ = _read_rt_is(rt)
        assert n > 0, f"Frame {FRAME}: zero correspondences"

    @pytest.mark.slow
    def test_tracking_on_preexisting_res(self, py_batch_dir_with_res):
        """Tracking-only using pre-existing res_orig/ data."""
        from algorithms.batch import run_batch

        work_dir, yaml_file = py_batch_dir_with_res
        tracker = run_batch(yaml_file, 10001, 10004, mode="tracking")

        ptv = work_dir / "res" / "ptv_is.10001"
        assert ptv.exists(), f"Missing {ptv}"
        assert tracker is not None

    @pytest.mark.slow
    def test_observer_on_preexisting_res(self, py_batch_dir_with_res):
        """Observer should collect events during tracking-only run."""
        from algorithms.batch import run_batch
        from algorithms.track import TrackingObserver

        work_dir, yaml_file = py_batch_dir_with_res
        obs = TrackingObserver()
        run_batch(yaml_file, 10001, 10004, mode="tracking", observer=obs)

        assert len(obs.events) > 0, "Observer should have recorded events"


# ---------------------------------------------------------------------------
# Comparison — Python batch vs shipped reference (res_orig/)
# ---------------------------------------------------------------------------

class TestPythonVsReference:
    """Compare pure-Python batch output against res_orig/ reference."""

    @pytest.mark.slow
    def test_counts_and_positions_vs_reference(self, py_batch_dir):
        """Single-frame sequence output should be close to res_orig/."""
        from algorithms.batch import run_batch

        work_dir, yaml_file = py_batch_dir
        run_batch(yaml_file, FRAME, FRAME, mode="sequence")

        ref_dir = TEST_DATA_DIR / "res_orig"
        res_dir = work_dir / "res"

        n_ref, ref_data = _read_rt_is(ref_dir / f"rt_is.{FRAME}")
        n_py, py_data = _read_rt_is(res_dir / f"rt_is.{FRAME}")

        # Count within ±50 %
        ratio = n_py / n_ref if n_ref > 0 else 1.0
        assert 0.5 < ratio < 1.5, (
            f"Python={n_py} vs ref={n_ref} (ratio {ratio:.2f})"
        )

        if n_ref > 0 and n_py > 0:
            n = min(n_ref, n_py)
            ref_xyz = ref_data[:n, 1:4][np.lexsort(ref_data[:n, 1:4].T)]
            py_xyz = py_data[:n, 1:4][np.lexsort(py_data[:n, 1:4].T)]
            med = np.median(np.linalg.norm(ref_xyz - py_xyz, axis=1))
            assert med < 5.0, f"median 3-D distance = {med:.3f} mm"


# ---------------------------------------------------------------------------
# Comparison — Python batch vs Cython batch (pyptv_batch)
# ---------------------------------------------------------------------------

class TestPythonVsCython:
    """Compare algorithms.batch vs gui.pyptv.pyptv_batch (single frame)."""

    @staticmethod
    def _optv_available() -> bool:
        try:
            from optv.tracker import Tracker  # noqa: F401
            return True
        except ImportError:
            return False

    @pytest.mark.slow
    def test_sequence_parity(self, tmp_path):
        """Single-frame sequence: Python batch vs Cython batch."""
        if not self._optv_available():
            pytest.skip("optv (Cython) not installed")

        # --- Cython ---
        cy_dir = tmp_path / "cython"
        cy_yaml = _copy_test_env(cy_dir)

        from openptv2.engine import set_engine
        set_engine("optv")

        from gui.pyptv import pyptv_batch
        pyptv_batch.main(cy_yaml, FRAME, FRAME, mode="sequence")

        # --- Python ---
        py_dir = tmp_path / "python"
        py_yaml = _copy_test_env(py_dir)

        from algorithms.batch import run_batch as py_run_batch
        py_run_batch(py_yaml, FRAME, FRAME, mode="sequence")

        # --- Compare ---
        n_cy, cy_data = _read_rt_is(cy_dir / "res" / f"rt_is.{FRAME}")
        n_py, py_data = _read_rt_is(py_dir / "res" / f"rt_is.{FRAME}")

        if n_cy == 0:
            pytest.skip("Cython produced 0 correspondences")

        ratio = n_py / n_cy
        print(f"Cython={n_cy}, Python={n_py}, ratio={ratio:.2f}")
        assert 0.5 < ratio < 1.5

        n = min(n_cy, n_py)
        if n > 0:
            cy_xyz = cy_data[:n, 1:4][np.lexsort(cy_data[:n, 1:4].T)]
            py_xyz = py_data[:n, 1:4][np.lexsort(py_data[:n, 1:4].T)]
            med = np.median(np.linalg.norm(cy_xyz - py_xyz, axis=1))
            print(f"median 3-D diff = {med:.4f} mm")
            assert med < 5.0

    @pytest.mark.slow
    def test_tracking_parity(self, tmp_path):
        """Tracking-only using the same res_orig/ data with both engines."""
        if not self._optv_available():
            pytest.skip("optv (Cython) not installed")

        # --- Cython ---
        cy_dir = tmp_path / "cy_track"
        cy_yaml = _copy_test_env(cy_dir, copy_res_orig=True)

        from openptv2.engine import set_engine
        set_engine("optv")

        from gui.pyptv import pyptv_batch
        pyptv_batch.main(cy_yaml, 10001, 10004, mode="tracking")

        # --- Python ---
        py_dir = tmp_path / "py_track"
        py_yaml = _copy_test_env(py_dir, copy_res_orig=True)

        from algorithms.batch import run_batch as py_run_batch
        py_run_batch(py_yaml, 10001, 10004, mode="tracking")

        # Both should produce ptv_is files
        for frame in range(10001, 10005):
            cy_ptv = cy_dir / "res" / f"ptv_is.{frame}"
            py_ptv = py_dir / "res" / f"ptv_is.{frame}"
            if cy_ptv.exists():
                assert py_ptv.exists(), (
                    f"Cython produced ptv_is.{frame} but Python did not"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
