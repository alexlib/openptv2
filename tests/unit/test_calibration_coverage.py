"""Pure-Python coverage tests for openptv2.algorithms.calibration.

Run with:
    COVERAGE_FILE=/tmp/.cov_cal uv run pytest tests/unit/test_calibration_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

# Skip in compiled mode — cfuncs are not exported and coverage is meaningless.
from openptv2.algorithms.calibration import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

from openptv2.algorithms.calibration import (
    AddedPar,
    Calibration,
    Exterior,
    Glass,
    Interior,
    MmLut,
)

# ---------------------------------------------------------------------------
# Paths to calibration test data
# ---------------------------------------------------------------------------
CAL_DIR = Path(__file__).parent.parent.parent / "test_data" / "calibration"
ORI_FILE = CAL_DIR / "cam1.tif.ori"
ADD_FILE = CAL_DIR / "cam1.tif.addpar"
ORI2_FILE = CAL_DIR / "cam2.tif.ori"
ADD2_FILE = CAL_DIR / "cam2.tif.addpar"


# ===========================================================================
# Exterior
# ===========================================================================


class TestExterior:
    def test_default_construction(self):
        e = Exterior()
        assert e.x0 == 0.0
        assert e.y0 == 0.0
        assert e.z0 == 0.0
        assert e.omega == 0.0
        assert e.phi == 0.0
        assert e.kappa == 0.0
        assert e.dm.shape == (3, 3)
        assert np.allclose(e.dm, np.eye(3))

    def test_custom_construction(self):
        e = Exterior(x0=1.0, y0=2.0, z0=3.0, omega=0.1, phi=0.2, kappa=0.3)
        assert e.x0 == 1.0
        assert e.y0 == 2.0
        assert e.z0 == 3.0

    def test_compute_rotation_matrix_identity(self):
        """All-zero angles → identity matrix."""
        e = Exterior()
        dm = e.compute_rotation_matrix()
        assert np.allclose(dm, np.eye(3))
        assert np.allclose(e.dm, np.eye(3))

    def test_compute_rotation_matrix_kappa90(self):
        """phi=0, omega=0, kappa=90° → known rotation."""
        e = Exterior(kappa=math.pi / 2)
        dm = e.compute_rotation_matrix()
        # cp*ck = 1*0=0, -cp*sk = -1, sp=0
        assert abs(dm[0, 0]) < 1e-10
        assert abs(dm[0, 1] - (-1.0)) < 1e-10
        assert abs(dm[0, 2]) < 1e-10

    def test_compute_rotation_matrix_phi90(self):
        """phi=90° → sp=1."""
        e = Exterior(phi=math.pi / 2)
        dm = e.compute_rotation_matrix()
        assert abs(dm[0, 2] - 1.0) < 1e-10  # sp

    def test_compute_rotation_matrix_omega90(self):
        """omega=90° → so=1, co=0."""
        e = Exterior(omega=math.pi / 2)
        dm = e.compute_rotation_matrix()
        assert abs(dm[1, 2] - (-1.0)) < 1e-10  # -so*cp

    def test_compute_rotation_matrix_returns_ndarray(self):
        e = Exterior(omega=0.1, phi=0.2, kappa=0.3)
        dm = e.compute_rotation_matrix()
        assert isinstance(dm, np.ndarray)
        assert dm.dtype == np.float64
        assert dm.shape == (3, 3)

    def test_mutation_updates_dm(self):
        e = Exterior()
        e.compute_rotation_matrix()
        old = e.dm.copy()
        e.phi = 0.5
        e.compute_rotation_matrix()
        assert not np.allclose(e.dm, old)


# ===========================================================================
# Interior
# ===========================================================================


class TestInterior:
    def test_default(self):
        i = Interior()
        assert i.xh == 0.0
        assert i.yh == 0.0
        assert i.cc == 0.0

    def test_custom(self):
        i = Interior(xh=1.5, yh=-2.3, cc=100.0)
        assert i.xh == 1.5
        assert i.yh == -2.3
        assert i.cc == 100.0


# ===========================================================================
# Glass
# ===========================================================================


class TestGlass:
    def test_default(self):
        g = Glass()
        assert g.vec_x == 0.0
        assert g.vec_y == 0.0
        assert g.vec_z == 1.0

    def test_custom(self):
        g = Glass(vec_x=0.1, vec_y=0.2, vec_z=0.9)
        assert g.vec_x == 0.1

    def test_sanitize_nonzero_does_nothing(self):
        g = Glass(vec_x=0.0, vec_y=0.0, vec_z=1.0)
        g.sanitize()
        assert g.vec_z == 1.0

    def test_sanitize_zero_vector_warns_and_fixes(self):
        g = Glass(vec_x=0.0, vec_y=0.0, vec_z=0.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g.sanitize()
        assert len(caught) == 1
        assert "zero" in str(caught[0].message).lower()
        assert g.vec_z == 1.0
        assert g.vec_x == 0.0
        assert g.vec_y == 0.0

    def test_is_zero_true(self):
        g = Glass(vec_x=0.0, vec_y=0.0, vec_z=0.0)
        assert g.is_zero() is True

    def test_is_zero_false(self):
        g = Glass(vec_x=0.0, vec_y=0.0, vec_z=1.0)
        assert g.is_zero() is False

    def test_is_zero_tiny_nonzero(self):
        g = Glass(vec_x=1e-13, vec_y=0.0, vec_z=0.0)
        assert g.is_zero() is True

    def test_sanitize_tiny_vector(self):
        g = Glass(vec_x=1e-15, vec_y=0.0, vec_z=0.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            g.sanitize()
        assert g.vec_z == 1.0


# ===========================================================================
# AddedPar
# ===========================================================================


class TestAddedPar:
    def test_defaults(self):
        ap = AddedPar()
        assert ap.k1 == 0.0
        assert ap.k2 == 0.0
        assert ap.k3 == 0.0
        assert ap.p1 == 0.0
        assert ap.p2 == 0.0
        assert ap.scx == 1.0
        assert ap.she == 0.0
        assert ap.field == 0

    def test_custom(self):
        ap = AddedPar(k1=1e-4, k2=2e-8, k3=3e-12, p1=1e-5, p2=2e-5, scx=0.99, she=0.01)
        assert ap.k1 == 1e-4
        assert ap.scx == 0.99


# ===========================================================================
# MmLut
# ===========================================================================


class TestMmLut:
    def test_defaults(self):
        mm = MmLut()
        assert mm.nr == 0
        assert mm.nz == 0
        assert mm.rw == 2
        assert mm.data is None
        assert not mm.is_initialized

    def test_is_initialized(self):
        mm = MmLut()
        mm.data = np.zeros(10)
        assert mm.is_initialized

    def test_origin_default(self):
        mm = MmLut()
        assert np.allclose(mm.origin, np.zeros(3))


# ===========================================================================
# Calibration — construction
# ===========================================================================


class TestCalibrationConstruction:
    def test_default(self):
        cal = Calibration()
        assert isinstance(cal.ext_par, Exterior)
        assert isinstance(cal.int_par, Interior)
        assert isinstance(cal.glass_par, Glass)
        assert isinstance(cal.added_par, AddedPar)
        assert isinstance(cal.mmlut, MmLut)

    def test_with_dataclass_args(self):
        ext = Exterior(x0=1.0, y0=2.0, z0=3.0)
        int_p = Interior(xh=0.1, yh=0.2, cc=50.0)
        glass = Glass(vec_z=1.0)
        added = AddedPar(k1=1e-4)
        mm = MmLut()
        cal = Calibration(ext_par=ext, int_par=int_p, glass_par=glass, added_par=added, mmlut=mm)
        assert cal.ext_par.x0 == 1.0
        assert cal.int_par.cc == 50.0
        assert cal.added_par.k1 == 1e-4

    def test_with_pos_and_angs(self):
        cal = Calibration(
            pos=np.array([10.0, 20.0, 30.0]),
            angs=np.array([0.1, 0.2, 0.3]),
        )
        assert cal.ext_par.x0 == 10.0
        assert cal.ext_par.y0 == 20.0
        assert cal.ext_par.z0 == 30.0
        assert abs(cal.ext_par.omega - 0.1) < 1e-12

    def test_with_primary_point(self):
        cal = Calibration(prim_point=np.array([1.0, 2.0, 100.0]))
        assert cal.int_par.xh == 1.0
        assert cal.int_par.yh == 2.0
        assert cal.int_par.cc == 100.0

    def test_with_radial_distortion(self):
        cal = Calibration(rad_dist=np.array([1e-4, 2e-8, 3e-12]))
        assert cal.added_par.k1 == 1e-4

    def test_with_decentering(self):
        cal = Calibration(decent=np.array([1e-5, 2e-5]))
        assert cal.added_par.p1 == 1e-5
        assert cal.added_par.p2 == 2e-5

    def test_with_affine(self):
        cal = Calibration(affine=np.array([0.99, 0.001]))
        assert cal.added_par.scx == 0.99
        assert cal.added_par.she == 0.001

    def test_with_glass(self):
        cal = Calibration(glass=np.array([0.0, 0.0, 150.0]))
        assert cal.glass_par.vec_z == 150.0

    def test_with_cal_copy(self):
        original = Calibration()
        original.ext_par.x0 = 99.0
        cal = Calibration(cal=original)
        assert cal.ext_par.x0 == 99.0

    def test_legacy_positional_remap(self):
        """Non-Exterior first arg triggers legacy-positional remap."""
        pos = np.array([1.0, 2.0, 3.0])
        angs = np.array([0.0, 0.0, 0.0])
        prim = np.array([0.1, 0.2, 100.0])
        rad = np.array([1e-4, 0.0, 0.0])
        decent = np.array([0.0, 0.0])
        affine = np.array([1.0, 0.0])
        glass = np.array([0.0, 0.0, 1.0])
        cal = Calibration(pos, angs, prim, rad, decent, affine, glass)
        assert cal.ext_par.x0 == 1.0
        assert cal.int_par.cc == 100.0
        assert cal.added_par.k1 == 1e-4

    def test_rotation_matrix_computed_on_init(self):
        cal = Calibration()
        # dm should be identity for zero angles
        assert np.allclose(cal.ext_par.dm, np.eye(3))

    def test_glass_sanitized_on_init(self):
        """Calibration with zero glass vector triggers sanitize warning."""
        glass = Glass(vec_x=0.0, vec_y=0.0, vec_z=0.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal = Calibration(glass_par=glass)
        assert len(caught) >= 1


# ===========================================================================
# Calibration — set/get methods
# ===========================================================================


class TestCalibrationSetGet:
    def setup_method(self):
        self.cal = Calibration()

    def test_get_set_pos(self):
        self.cal.set_pos([10.0, 20.0, 30.0])
        pos = self.cal.get_pos()
        assert np.allclose(pos, [10.0, 20.0, 30.0])

    def test_set_pos_wrong_length(self):
        with pytest.raises(ValueError):
            self.cal.set_pos([1.0, 2.0])

    def test_get_set_angles(self):
        self.cal.set_angles([0.1, 0.2, 0.3])
        angs = self.cal.get_angles()
        assert np.allclose(angs, [0.1, 0.2, 0.3])

    def test_set_angles_wrong_length(self):
        with pytest.raises(ValueError):
            self.cal.set_angles([0.1, 0.2])

    def test_set_angles_updates_rotation_matrix(self):
        self.cal.set_angles([0.0, 0.0, math.pi / 2])
        dm = self.cal.get_rotation_matrix()
        assert abs(dm[0, 1] - (-1.0)) < 1e-10

    def test_get_set_primary_point(self):
        self.cal.set_primary_point([1.5, -0.5, 75.0])
        pp = self.cal.get_primary_point()
        assert np.allclose(pp, [1.5, -0.5, 75.0])

    def test_set_primary_point_wrong_length(self):
        with pytest.raises(ValueError):
            self.cal.set_primary_point([1.0, 2.0])

    def test_get_set_radial_distortion(self):
        self.cal.set_radial_distortion([1e-4, 2e-8, 3e-12])
        rd = self.cal.get_radial_distortion()
        assert np.allclose(rd, [1e-4, 2e-8, 3e-12])

    def test_set_radial_distortion_wrong_length(self):
        with pytest.raises(ValueError):
            self.cal.set_radial_distortion([1e-4, 2e-8])

    def test_get_set_decentering(self):
        self.cal.set_decentering([1e-5, 2e-5])
        d = self.cal.get_decentering()
        assert np.allclose(d, [1e-5, 2e-5])

    def test_set_decentering_wrong_length(self):
        with pytest.raises(ValueError):
            self.cal.set_decentering([1e-5])

    def test_get_set_affine(self):
        self.cal.set_affine_trans([0.99, 0.001])
        a = self.cal.get_affine()
        assert np.allclose(a, [0.99, 0.001])

    def test_set_affine_wrong_length(self):
        with pytest.raises(ValueError):
            self.cal.set_affine_trans([0.99])

    def test_get_set_glass_vec(self):
        self.cal.set_glass_vec([0.0, 0.0, 150.0])
        gv = self.cal.get_glass_vec()
        assert np.allclose(gv, [0.0, 0.0, 150.0])

    def test_set_glass_vec_wrong_length(self):
        with pytest.raises(ValueError):
            self.cal.set_glass_vec([0.0, 1.0])

    def test_set_glass_vec_zero_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.cal.set_glass_vec([0.0, 0.0, 0.0])
        assert len(caught) >= 1

    def test_get_rotation_matrix_returns_copy(self):
        dm = self.cal.get_rotation_matrix()
        dm[0, 0] = 999.0  # mutate the copy
        assert self.cal.ext_par.dm[0, 0] != 999.0


# ===========================================================================
# Calibration.write (bytes path handling)
# ===========================================================================


class TestCalibrationWrite:
    def test_write_accepts_bytes_ori(self, tmp_path):
        cal = Calibration()
        ori = str(tmp_path / "out.ori").encode()
        cal.write(ori)
        assert (tmp_path / "out.ori").exists()

    def test_write_accepts_bytes_add(self, tmp_path):
        cal = Calibration()
        ori = str(tmp_path / "out.ori")
        add = str(tmp_path / "out.addpar").encode()
        cal.write(ori, add)
        assert (tmp_path / "out.addpar").exists()

    def test_write_accepts_str_paths(self, tmp_path):
        cal = Calibration()
        ori = str(tmp_path / "out.ori")
        add = str(tmp_path / "out.addpar")
        cal.write(ori, add)
        assert (tmp_path / "out.ori").exists()
        assert (tmp_path / "out.addpar").exists()


# ===========================================================================
# Calibration.to_file / _from_file_class  (round-trip)
# ===========================================================================


class TestCalibrationToFile:
    def test_to_file_ori_only(self, tmp_path):
        cal = Calibration()
        cal.set_pos([1.0, 2.0, 3.0])
        cal.set_angles([0.0, 0.0, 0.0])
        cal.set_primary_point([0.1, 0.2, 100.0])
        cal.set_glass_vec([0.0, 0.0, 1.0])
        ori = tmp_path / "out.ori"
        cal.to_file(str(ori))
        assert ori.exists()
        text = ori.read_text()
        assert "1.00000000" in text

    def test_to_file_with_add(self, tmp_path):
        cal = Calibration()
        cal.set_radial_distortion([1e-4, 2e-8, 3e-12])
        ori = tmp_path / "out.ori"
        add = tmp_path / "out.addpar"
        cal.to_file(str(ori), str(add))
        assert add.exists()
        content = add.read_text()
        assert "0.00010000" in content

    def test_round_trip(self, tmp_path):
        cal = Calibration()
        cal.set_pos([10.0, 20.0, 30.0])
        cal.set_angles([0.1, 0.2, 0.3])
        cal.set_primary_point([1.5, -0.5, 75.0])
        cal.set_glass_vec([0.0, 0.0, 150.0])
        cal.set_radial_distortion([1e-4, 2e-8, 3e-12])
        ori = tmp_path / "rt.ori"
        add = tmp_path / "rt.addpar"
        cal.to_file(str(ori), str(add))
        cal2 = Calibration.from_file(str(ori), str(add))
        assert abs(cal2.ext_par.x0 - 10.0) < 1e-4
        assert abs(cal2.int_par.cc - 75.0) < 1e-3
        assert abs(cal2.glass_par.vec_z - 150.0) < 1e-6

    def test_rotation_matrix_written_and_read(self, tmp_path):
        """Rotation matrix rows appear in output file."""
        cal = Calibration()
        cal.set_angles([0.1, 0.2, 0.3])
        ori = tmp_path / "out.ori"
        cal.to_file(str(ori))
        text = ori.read_text()
        # At least one non-identity value should appear
        assert any(v in text for v in ["0.9", "0.8", "0.7"])


# ===========================================================================
# Calibration._from_file_class
# ===========================================================================


class TestCalibrationFromFile:
    def test_from_file_real_data(self):
        cal = Calibration.from_file(str(ORI_FILE))
        # cam1.tif.ori: x0~105.26, y0~102.75, z0~403.88
        assert abs(cal.ext_par.x0 - 105.26320000) < 1e-5
        assert abs(cal.ext_par.y0 - 102.74580000) < 1e-5
        assert abs(cal.ext_par.z0 - 403.88220000) < 1e-5

    def test_from_file_interior(self):
        cal = Calibration.from_file(str(ORI_FILE))
        assert abs(cal.int_par.xh - (-2.4742)) < 1e-3
        assert abs(cal.int_par.yh - 3.2567) < 1e-3
        assert abs(cal.int_par.cc - 100.0) < 1e-3

    def test_from_file_glass(self):
        cal = Calibration.from_file(str(ORI_FILE))
        assert abs(cal.glass_par.vec_z - 150.0) < 1e-6

    def test_from_file_with_addpar(self):
        cal = Calibration.from_file(str(ORI_FILE), str(ADD_FILE))
        # cam1.tif.addpar: all zeros except scx=1.0
        assert cal.added_par.k1 == 0.0
        assert cal.added_par.scx == 1.0

    def test_from_file_missing_ori_raises(self):
        with pytest.raises(FileNotFoundError):
            Calibration.from_file("/nonexistent/path/cam.ori")

    def test_from_file_bytes_path(self):
        cal = Calibration.from_file(str(ORI_FILE).encode())
        assert abs(cal.ext_par.x0 - 105.26320000) < 1e-5

    def test_from_file_add_bytes_path(self):
        cal = Calibration.from_file(
            str(ORI_FILE).encode(),
            str(ADD_FILE).encode(),
        )
        assert cal.added_par.scx == 1.0

    def test_from_file_fallback_add(self, tmp_path):
        """When add_file doesn't exist, fallback is used."""
        nonexistent = tmp_path / "missing.addpar"
        cal = Calibration.from_file(
            str(ORI_FILE),
            str(nonexistent),
            str(ADD_FILE),
        )
        assert cal.added_par.scx == 1.0

    def test_from_file_fallback_bytes(self, tmp_path):
        nonexistent = str(tmp_path / "missing.addpar").encode()
        cal = Calibration.from_file(
            str(ORI_FILE).encode(),
            nonexistent,
            str(ADD_FILE).encode(),
        )
        assert cal.added_par.scx == 1.0

    def test_from_file_cam2(self):
        cal = Calibration.from_file(str(ORI2_FILE), str(ADD2_FILE))
        assert cal.ext_par is not None
        assert cal.glass_par.vec_z != 0.0

    def test_from_file_instance_method(self):
        """from_file called on an instance updates in place."""
        cal = Calibration()
        cal.from_file(str(ORI_FILE))
        assert abs(cal.ext_par.x0 - 105.26320000) < 1e-5

    def test_from_file_instance_method_with_add(self):
        cal = Calibration()
        cal.from_file(str(ORI_FILE), str(ADD_FILE))
        assert cal.added_par.scx == 1.0

    def test_from_file_split_exterior_two_lines(self, tmp_path):
        """Parse 3+3 split-across-lines exterior format."""
        content = (
            "105.26320000 102.74580000 403.88220000\n"
            "-0.23832910  0.24428100  0.05525770\n"
            "\n"
            "0.9688305 -0.0535899  0.2418587\n"
            "-0.0033422  0.9734041  0.2290704\n"
            "-0.2477021 -0.2227388  0.9428845\n"
            "\n"
            "-2.4742   3.2567\n"
            "100.0000\n"
            "\n"
            "0.000100000000000    0.000010000000000   150.000000000000000\n"
        )
        ori = tmp_path / "split.ori"
        ori.write_text(content)
        cal = Calibration.from_file(str(ori))
        assert abs(cal.ext_par.x0 - 105.26320000) < 1e-5
        assert abs(cal.int_par.cc - 100.0) < 1e-3

    def test_from_file_interior_two_line_format(self, tmp_path):
        """Parse interior where xh yh are on one line, cc on next."""
        content = (
            "105.26320000 102.74580000 403.88220000\n"
            "-0.23832910  0.24428100  0.05525770\n"
            "\n"
            "0.9688305 -0.0535899  0.2418587\n"
            "-0.0033422  0.9734041  0.2290704\n"
            "-0.2477021 -0.2227388  0.9428845\n"
            "\n"
            "-2.4742   3.2567\n"
            "100.0000\n"
            "\n"
            "0.000100000000000    0.000010000000000   150.000000000000000\n"
        )
        ori = tmp_path / "int2.ori"
        ori.write_text(content)
        cal = Calibration.from_file(str(ori))
        assert abs(cal.int_par.xh - (-2.4742)) < 1e-3
        assert abs(cal.int_par.cc - 100.0) < 1e-3

    def test_from_file_addpar_too_short_ignored(self, tmp_path):
        """add_file with < 7 values → defaults kept."""
        ori = ORI_FILE
        add = tmp_path / "short.addpar"
        add.write_text("0.001 0.002 0.003\n")  # only 3 values
        cal = Calibration.from_file(str(ori), str(add))
        assert cal.added_par.k1 == 0.0
        assert cal.added_par.scx == 1.0  # unchanged default

    def test_from_file_fallback_too_short(self, tmp_path):
        """Fallback add_file with < 7 values → defaults kept."""
        nonexistent = tmp_path / "missing.addpar"
        fallback = tmp_path / "fb_short.addpar"
        fallback.write_text("0.001 0.002\n")
        cal = Calibration.from_file(str(ORI_FILE), str(nonexistent), str(fallback))
        assert cal.added_par.scx == 1.0

    def test_from_file_invalid_exterior_raises(self, tmp_path):
        """File with 1-value exterior line raises ValueError."""
        bad = tmp_path / "bad.ori"
        bad.write_text("105.26\n-0.23\n\n1 0 0\n0 1 0\n0 0 1\n\n0 0\n100\n\n0 0 1\n")
        with pytest.raises((ValueError, IndexError)):
            Calibration.from_file(str(bad))

    def test_from_file_six_values_on_one_line(self, tmp_path):
        """Exterior all on one line."""
        content = (
            "105.263 102.746 403.882 -0.238 0.244 0.055\n"
            "\n"
            "0.9688305 -0.0535899  0.2418587\n"
            "-0.0033422  0.9734041  0.2290704\n"
            "-0.2477021 -0.2227388  0.9428845\n"
            "\n"
            "-2.4742   3.2567   100.0000\n"
            "\n"
            "0.0001    0.00001   150.0\n"
        )
        ori = tmp_path / "oneline.ori"
        ori.write_text(content)
        cal = Calibration.from_file(str(ori))
        assert abs(cal.ext_par.x0 - 105.263) < 1e-3
        assert abs(cal.int_par.cc - 100.0) < 1e-3

    def test_from_file_interior_three_values_on_one_line(self, tmp_path):
        """Interior all on one line."""
        content = (
            "105.263 102.746 403.882 -0.238 0.244 0.055\n"
            "\n"
            "0.9688305 -0.0535899  0.2418587\n"
            "-0.0033422  0.9734041  0.2290704\n"
            "-0.2477021 -0.2227388  0.9428845\n"
            "\n"
            "-2.4742   3.2567   100.0000\n"
            "\n"
            "0.0001    0.00001   150.0\n"
        )
        ori = tmp_path / "int3.ori"
        ori.write_text(content)
        cal = Calibration.from_file(str(ori))
        assert abs(cal.int_par.xh - (-2.4742)) < 1e-3


# ===========================================================================
# hybrid_from_file descriptor
# ===========================================================================


class TestHybridFromFile:
    def test_classmethod_returns_new_calibration(self):
        cal = Calibration.from_file(str(ORI_FILE))
        assert isinstance(cal, Calibration)

    def test_instance_method_updates_in_place(self):
        cal = Calibration()
        old_x0 = cal.ext_par.x0
        result = cal.from_file(str(ORI_FILE))
        assert result is cal
        assert cal.ext_par.x0 != old_x0
        assert abs(cal.ext_par.x0 - 105.26320000) < 1e-5

    # (removed test_instance_method_with_add_file — byte-identical to
    #  test_from_file_instance_method_with_add above)


# ===========================================================================
# is_compiled
# ===========================================================================


def test_is_compiled_returns_bool():
    from openptv2.algorithms.calibration import is_compiled
    result = is_compiled()
    assert isinstance(result, bool)
    assert result is False  # pure-Python mode


# ===========================================================================
# Missing-line coverage — edge cases
# ===========================================================================


class TestMissingLineCoverage:
    def test_post_init_called_directly(self):
        """Explicitly call __post_init__ to hit lines 304-305."""
        cal = Calibration()
        cal.__post_init__()
        # Should not raise and rotation matrix should still be valid
        assert np.allclose(cal.ext_par.dm, np.eye(3))

    def test_post_init_ext_par_none(self):
        """__post_init__ with ext_par=None hits the False branch of 304."""
        cal = Calibration()
        cal.ext_par = None
        cal.__post_init__()  # Should not raise

    def test_interior_single_value_raises(self, tmp_path):
        """Interior with only 1 value triggers line-500 ValueError."""
        content = (
            "105.263 102.746 403.882 -0.238 0.244 0.055\n"
            "\n"
            "0.9688305 -0.0535899  0.2418587\n"
            "-0.0033422  0.9734041  0.2290704\n"
            "-0.2477021 -0.2227388  0.9428845\n"
            "\n"
            "100.0000\n"
            "\n"
            "0.0001    0.00001   150.0\n"
        )
        ori = tmp_path / "bad_interior.ori"
        ori.write_text(content)
        with pytest.raises(ValueError, match="Invalid interior"):
            Calibration.from_file(str(ori))

    def test_glass_too_few_values_raises(self, tmp_path):
        """Glass line with fewer than 3 values triggers line-514 ValueError."""
        content = (
            "105.263 102.746 403.882 -0.238 0.244 0.055\n"
            "\n"
            "0.9688305 -0.0535899  0.2418587\n"
            "-0.0033422  0.9734041  0.2290704\n"
            "-0.2477021 -0.2227388  0.9428845\n"
            "\n"
            "-2.4742   3.2567   100.0000\n"
            "\n"
            "150.0\n"
        )
        ori = tmp_path / "bad_glass.ori"
        ori.write_text(content)
        with pytest.raises(ValueError, match="Invalid glass"):
            Calibration.from_file(str(ori))

    def test_add_file_missing_no_fallback(self, tmp_path):
        """add_file missing, add_fallback=None — hits branch 532->547."""
        nonexistent = tmp_path / "missing.addpar"
        cal = Calibration.from_file(str(ORI_FILE), str(nonexistent))
        # Defaults should be kept
        assert cal.added_par.scx == 1.0

    def test_add_fallback_also_missing(self, tmp_path):
        """add_file missing, fallback provided but also missing — hits 534->547."""
        nonexistent_add = tmp_path / "missing.addpar"
        nonexistent_fb = tmp_path / "missing_fb.addpar"
        cal = Calibration.from_file(
            str(ORI_FILE),
            str(nonexistent_add),
            str(nonexistent_fb),
        )
        assert cal.added_par.scx == 1.0

    def test_init_ext_par_none_branch(self):
        """Force self.ext_par to be None before the line-299 check."""
        # The only way to get there is if ext_par ends up None; we do this
        # by subclassing and calling super().__init__ with a patched path.
        # Simpler: just verify that setting ext_par=None inside init still works.
        # Provide cal= with an object that has ext_par=None
        class FakeCal:
            ext_par = None
            int_par = Interior()
            glass_par = Glass()
            added_par = AddedPar()
            mmlut = MmLut()

        cal = Calibration(cal=FakeCal())
        # ext_par is None — no crash
        assert cal.ext_par is None


# ===========================================================================
# Sym camera files (extra coverage via different ORI files)
# ===========================================================================


class TestSymCamFiles:
    @pytest.mark.parametrize("fname", [
        "sym_cam1.tif.ori",
        "sym_cam2.tif.ori",
        "sym_cam3.tif.ori",
        "sym_cam4.tif.ori",
    ])
    def test_load_sym_cam(self, fname):
        ori = CAL_DIR / fname
        cal = Calibration.from_file(str(ori))
        assert isinstance(cal, Calibration)
        assert cal.ext_par is not None
        assert cal.int_par is not None
