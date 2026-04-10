import ctypes
import importlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from algorithms.calibration import Calibration, Exterior, Interior
from algorithms.correspondences import _find_candidates_vectorized
from algorithms.epi import Coord2d_dtype
from algorithms.find_candidate import find_candidate
from algorithms.parameters import ControlPar, MultimediaPar, VolumePar
from algorithms.track import candsearch_in_pix, candsearch_in_pix_rest
from algorithms.tracking_frame_buf import Target


ROOT = Path(__file__).resolve().parents[2]
BUILD_LIB = ROOT / "bindings" / "build"
OPTv_DIR = ROOT / "bindings" / "optv"


class CTarget(ctypes.Structure):
    _fields_ = [
        ("pnr", ctypes.c_int),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("n", ctypes.c_int),
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("sumg", ctypes.c_int),
        ("tnr", ctypes.c_int),
    ]


class CCandidate(ctypes.Structure):
    _fields_ = [
        ("pnr", ctypes.c_int),
        ("tol", ctypes.c_double),
        ("corr", ctypes.c_double),
    ]


class CCoord2d(ctypes.Structure):
    _fields_ = [
        ("pnr", ctypes.c_int),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
    ]


class CExterior(ctypes.Structure):
    _fields_ = [
        ("x0", ctypes.c_double),
        ("y0", ctypes.c_double),
        ("z0", ctypes.c_double),
        ("omega", ctypes.c_double),
        ("phi", ctypes.c_double),
        ("kappa", ctypes.c_double),
        ("dm", ctypes.c_double * 9),
    ]


class CInterior(ctypes.Structure):
    _fields_ = [
        ("xh", ctypes.c_double),
        ("yh", ctypes.c_double),
        ("cc", ctypes.c_double),
    ]


class CGlass(ctypes.Structure):
    _fields_ = [
        ("vec_x", ctypes.c_double),
        ("vec_y", ctypes.c_double),
        ("vec_z", ctypes.c_double),
        ("n1", ctypes.c_double),
        ("n2", ctypes.c_double),
        ("n3", ctypes.c_double),
        ("d", ctypes.c_double),
    ]


class CAp52(ctypes.Structure):
    _fields_ = [
        ("k1", ctypes.c_double),
        ("k2", ctypes.c_double),
        ("k3", ctypes.c_double),
        ("p1", ctypes.c_double),
        ("p2", ctypes.c_double),
        ("scx", ctypes.c_double),
        ("she", ctypes.c_double),
        ("field", ctypes.c_int),
    ]


class CMmLut(ctypes.Structure):
    _fields_ = [
        ("origin", ctypes.c_double * 3),
        ("nr", ctypes.c_int),
        ("nz", ctypes.c_int),
        ("rw", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_double)),
    ]


class CCalibration(ctypes.Structure):
    _fields_ = [
        ("ext_par", CExterior),
        ("int_par", CInterior),
        ("glass_par", CGlass),
        ("added_par", CAp52),
        ("mmlut", CMmLut),
    ]


class CVolumePar(ctypes.Structure):
    _fields_ = [
        ("X_lay", ctypes.c_double * 2),
        ("Zmin_lay", ctypes.c_double * 2),
        ("Zmax_lay", ctypes.c_double * 2),
        ("cn", ctypes.c_double),
        ("cnx", ctypes.c_double),
        ("cny", ctypes.c_double),
        ("csumg", ctypes.c_double),
        ("eps0", ctypes.c_double),
        ("corrmin", ctypes.c_double),
    ]


class CMmNp(ctypes.Structure):
    _fields_ = [
        ("nlay", ctypes.c_int),
        ("n1", ctypes.c_double),
        ("n2", ctypes.c_double * 3),
        ("d", ctypes.c_double * 3),
        ("n3", ctypes.c_double),
    ]


class CControlPar(ctypes.Structure):
    _fields_ = [
        ("num_cams", ctypes.c_int),
        ("img_base_name", ctypes.POINTER(ctypes.c_char_p)),
        ("cal_img_base_name", ctypes.POINTER(ctypes.c_char_p)),
        ("hp_flag", ctypes.c_int),
        ("allCam_flag", ctypes.c_int),
        ("tiff_flag", ctypes.c_int),
        ("imx", ctypes.c_int),
        ("imy", ctypes.c_int),
        ("pix_x", ctypes.c_double),
        ("pix_y", ctypes.c_double),
        ("chfield", ctypes.c_int),
        ("mm", ctypes.POINTER(CMmNp)),
    ]


def _build_glob(pattern: str) -> list[Path]:
    return sorted(BUILD_LIB.glob(pattern))


def _importable_module_path(module_name: str) -> Path | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    return Path(module_file)


@lru_cache(maxsize=1)
def _tracker_lib() -> ctypes.CDLL | None:
    module_path = _importable_module_path("optv.tracker")
    if module_path is not None:
        return ctypes.CDLL(str(module_path))

    direct_matches = sorted(OPTv_DIR.glob("tracker*.so"))
    if direct_matches:
        return ctypes.CDLL(str(direct_matches[0]))

    build_matches = _build_glob("lib.*/optv/tracker*.so")
    if build_matches:
        return ctypes.CDLL(str(build_matches[0]))
    return None


@lru_cache(maxsize=1)
def _correspondences_lib() -> ctypes.CDLL | None:
    module_path = _importable_module_path("optv.correspondences")
    if module_path is not None:
        return ctypes.CDLL(str(module_path))

    direct_matches = sorted(OPTv_DIR.glob("correspondences*.so"))
    if direct_matches:
        return ctypes.CDLL(str(direct_matches[0]))

    build_matches = _build_glob("lib.*/optv/correspondences*.so")
    if build_matches:
        return ctypes.CDLL(str(build_matches[0]))
    return None


def _native_available() -> bool:
    return _tracker_lib() is not None and _correspondences_lib() is not None


skip_no_native = pytest.mark.skipif(
    not _native_available(), reason="built native optv modules not available"
)


@lru_cache(maxsize=1)
def _candsearch_in_pix_fn():
    fn = _tracker_lib().candsearch_in_pix
    fn.argtypes = [
        ctypes.POINTER(CTarget),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(CControlPar),
    ]
    fn.restype = ctypes.c_int
    return fn


@lru_cache(maxsize=1)
def _candsearch_in_pix_rest_fn():
    fn = _tracker_lib().candsearch_in_pix_rest
    fn.argtypes = [
        ctypes.POINTER(CTarget),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(CControlPar),
    ]
    fn.restype = ctypes.c_int
    return fn


@lru_cache(maxsize=1)
def _find_candidate_fn():
    fn = _correspondences_lib().find_candidate
    fn.argtypes = [
        ctypes.POINTER(CCoord2d),
        ctypes.POINTER(CTarget),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(CCandidate),
        ctypes.POINTER(CVolumePar),
        ctypes.POINTER(CControlPar),
        ctypes.POINTER(CCalibration),
    ]
    fn.restype = ctypes.c_int
    return fn


def _native_control_par(imx: int = 1280, imy: int = 1024) -> tuple[CControlPar, CMmNp]:
    mm = CMmNp(
        nlay=1,
        n1=1.0,
        n2=(ctypes.c_double * 3)(1.49, 0.0, 0.0),
        d=(ctypes.c_double * 3)(5.0, 0.0, 0.0),
        n3=1.33,
    )
    cpar = CControlPar(
        num_cams=4,
        img_base_name=None,
        cal_img_base_name=None,
        hp_flag=1,
        allCam_flag=0,
        tiff_flag=1,
        imx=imx,
        imy=imy,
        pix_x=0.02,
        pix_y=0.02,
        chfield=0,
        mm=ctypes.pointer(mm),
    )
    return cpar, mm


def _python_control_par(imx: int = 1280, imy: int = 1024) -> ControlPar:
    return ControlPar(
        num_cams=4,
        imx=imx,
        imy=imy,
        pix_x=0.02,
        pix_y=0.02,
        chfield=0,
        mm=MultimediaPar(nlay=1, n1=1.0, n2=[1.49], d=[5.0], n3=1.33),
        all_cam_flag=0,
    )


def _python_volume_par() -> VolumePar:
    return VolumePar(
        x_lay=[-250.0, 250.0],
        z_min_lay=[-100.0, -100.0],
        z_max_lay=[100.0, 100.0],
        cn=0.01,
        cnx=0.3,
        cny=0.3,
        csumg=0.01,
        eps0=1.0,
        corrmin=33.0,
    )


def _native_volume_par() -> CVolumePar:
    return CVolumePar(
        X_lay=(ctypes.c_double * 2)(-250.0, 250.0),
        Zmin_lay=(ctypes.c_double * 2)(-100.0, -100.0),
        Zmax_lay=(ctypes.c_double * 2)(100.0, 100.0),
        cn=0.01,
        cnx=0.3,
        cny=0.3,
        csumg=0.01,
        eps0=1.0,
        corrmin=33.0,
    )


def _python_calibration() -> Calibration:
    ext = np.array(
        (0.0, 0.0, 100.0, 0.0, 0.0, 0.0, np.eye(3)),
        dtype=Exterior.dtype,
    ).view(np.recarray)

    int_par = np.array((0.0, 0.0, 100.0), dtype=Interior.dtype).view(np.recarray)

    return Calibration(
        ext_par=ext,
        int_par=int_par,
        glass_par=np.array([0.0, 0.0, 50.0], dtype=np.float64),
        added_par=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    )


def _native_calibration() -> CCalibration:
    dm = (ctypes.c_double * 9)(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )
    mmlut_origin = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
    return CCalibration(
        ext_par=CExterior(0.0, 0.0, 100.0, 0.0, 0.0, 0.0, dm),
        int_par=CInterior(0.0, 0.0, 100.0),
        glass_par=CGlass(0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0),
        added_par=CAp52(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0),
        mmlut=CMmLut(mmlut_origin, 0, 0, 0, None),
    )


def _tracking_targets_case_one() -> list[tuple[int, float, float, int, int, int, int, int]]:
    return [
        (0, 0.0, -0.2, 5, 1, 2, 10, -999),
        (6, 0.2, 0.2, 10, 8, 1, 20, -999),
        (3, 0.2, 0.3, 10, 3, 3, 30, -999),
        (4, 0.2, 1.0, 10, 3, 3, 40, -999),
        (1, -0.7, 1.2, 10, 3, 3, 50, -999),
        (7, 1.2, 1.3, 10, 3, 3, 60, -999),
        (5, 10.4, 2.1, 10, 3, 3, 70, -999),
    ]


def _tracking_targets_case_rest() -> list[tuple[int, float, float, int, int, int, int, int]]:
    return [
        (0, 0.0, -0.2, 5, 1, 2, 10, 0),
        (6, 100.0, 100.0, 10, 8, 1, 20, -1),
        (3, 102.0, 102.0, 10, 3, 3, 30, -1),
        (4, 103.0, 103.0, 10, 3, 3, 40, 2),
        (1, -0.7, 1.2, 10, 3, 3, 50, 5),
        (7, 1.2, 1.3, 10, 3, 3, 60, 7),
        (5, 1200.0, 201.1, 10, 3, 3, 70, 11),
    ]


def _epi_targets_case() -> tuple[np.ndarray, list[Target], list[tuple[int, float, float, int, int, int, int, int]]]:
    crd = np.array(
        [
            (6, 0.1, 0.1),
            (3, 0.2, 0.8),
            (4, 0.4, -1.1),
            (1, 0.7, -0.1),
            (2, 1.2, 0.3),
            (0, 0.0, 0.0),
            (5, 10.4, 0.1),
        ],
        dtype=Coord2d_dtype,
    ).view(np.recarray)

    pix_rows = [
        (0, 0.0, -0.2, 5, 1, 2, 10, -999),
        (6, 0.2, 0.0, 10, 8, 1, 20, -999),
        (3, 0.2, 0.8, 10, 3, 3, 30, -999),
        (4, 0.4, -1.1, 10, 3, 3, 40, -999),
        (1, 0.7, -0.1, 10, 3, 3, 50, -999),
        (2, 1.2, 0.3, 10, 3, 3, 60, -999),
        (5, 10.4, 0.1, 10, 3, 3, 70, -999),
    ]

    pix = [
        Target(pnr=pnr, x=x, y=y, n=n, nx=nx, ny=ny, sumg=sumg, tnr=tnr)
        for pnr, x, y, n, nx, ny, sumg, tnr in pix_rows
    ]
    return crd, pix, pix_rows


def _native_targets(rows: list[tuple[int, float, float, int, int, int, int, int]]):
    return (CTarget * len(rows))(*(CTarget(*row) for row in rows))


def _native_tracking_search(rows, center, bounds):
    cpar, mm = _native_control_par()
    p = (ctypes.c_int * 4)(-999, -999, -999, -999)
    targets = _native_targets(rows)
    count = _candsearch_in_pix_fn()(
        targets,
        len(rows),
        center[0],
        center[1],
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        p,
        ctypes.byref(cpar),
    )
    return count, list(p)


def _native_tracking_search_rest(rows, center, bounds):
    cpar, mm = _native_control_par()
    p = (ctypes.c_int * 4)(-999, -999, -999, -999)
    targets = _native_targets(rows)
    count = _candsearch_in_pix_rest_fn()(
        targets,
        len(rows),
        center[0],
        center[1],
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        p,
        ctypes.byref(cpar),
    )
    return count, list(p)


def _native_find_candidate():
    crd, _, pix_rows = _epi_targets_case()
    native_crd = (CCoord2d * len(crd))(
        *(CCoord2d(int(row.pnr), float(row.x), float(row.y)) for row in crd)
    )
    native_pix = _native_targets(pix_rows)
    native_cands = (CCandidate * len(pix_rows))()
    vpar = _native_volume_par()
    cpar, mm = _native_control_par()
    cal = _native_calibration()

    count = _find_candidate_fn()(
        native_crd,
        native_pix,
        len(pix_rows),
        -10.0,
        -10.0,
        10.0,
        10.0,
        10,
        3,
        3,
        100,
        native_cands,
        ctypes.byref(vpar),
        ctypes.byref(cpar),
        ctypes.byref(cal),
    )
    return count, [native_cands[i] for i in range(count)]


@skip_no_native
class TestSmallCaseNativeParity:
    def test_candsearch_in_pix_matches_native(self):
        rows = _tracking_targets_case_one()
        center = (0.2, 0.2)
        bounds = (0.1, 0.1, 0.1, 0.1)

        native_count, native_indices = _native_tracking_search(rows, center, bounds)
        py_indices = candsearch_in_pix(
            [
                Target(pnr=pnr, x=x, y=y, n=n, nx=nx, ny=ny, sumg=sumg, tnr=tnr)
                for pnr, x, y, n, nx, ny, sumg, tnr in rows
            ],
            len(rows),
            center[0],
            center[1],
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            _python_control_par(),
        )

        assert native_count == 2
        assert native_indices[:native_count] == [1, 2]
        assert py_indices[:native_count] == native_indices[:native_count]

        wide_bounds = (10.2, 10.2, 10.2, 10.2)
        native_count, native_indices = _native_tracking_search(rows, (0.5, 0.3), wide_bounds)
        py_indices = candsearch_in_pix(
            [
                Target(pnr=pnr, x=x, y=y, n=n, nx=nx, ny=ny, sumg=sumg, tnr=tnr)
                for pnr, x, y, n, nx, ny, sumg, tnr in rows
            ],
            len(rows),
            0.5,
            0.3,
            wide_bounds[0],
            wide_bounds[1],
            wide_bounds[2],
            wide_bounds[3],
            _python_control_par(),
        )

        assert native_count == 4
        assert py_indices[:native_count] == native_indices[:native_count]

    def test_candsearch_in_pix_rest_matches_native(self):
        rows = _tracking_targets_case_rest()
        center = (98.9, 98.9)
        bounds = (3.0, 3.0, 3.0, 3.0)

        native_count, native_indices = _native_tracking_search_rest(rows, center, bounds)
        p = [-1, -1, -1, -1]
        py_count = candsearch_in_pix_rest(
            [
                Target(pnr=pnr, x=x, y=y, n=n, nx=nx, ny=ny, sumg=sumg, tnr=tnr)
                for pnr, x, y, n, nx, ny, sumg, tnr in rows
            ],
            len(rows),
            center[0],
            center[1],
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            p,
            _python_control_par(),
        )

        assert native_count == 1
        assert native_indices[0] == 1
        assert py_count == native_count
        assert p[0] == native_indices[0]

    def test_find_candidate_matches_native(self):
        crd, pix, _ = _epi_targets_case()
        native_count, native_cands = _native_find_candidate()
        py_cands = find_candidate(
            crd,
            pix,
            len(pix),
            -10.0,
            -10.0,
            10.0,
            10.0,
            10,
            3,
            3,
            100,
            _python_volume_par(),
            _python_control_par(),
            _python_calibration(),
        )

        assert native_count == 5
        assert len(py_cands) == native_count
        assert int(py_cands[0].pnr) == 0
        assert float(py_cands[0].tol) == pytest.approx(0.0, abs=1e-12)

        for index, native_cand in enumerate(native_cands):
            assert int(py_cands[index].pnr) == native_cand.pnr
            assert float(py_cands[index].tol) == pytest.approx(native_cand.tol, abs=1e-6)
            assert float(py_cands[index].corr) == pytest.approx(native_cand.corr, abs=1e-6)

    def test_vectorized_find_candidate_matches_native(self):
        crd, pix, _ = _epi_targets_case()
        native_count, native_cands = _native_find_candidate()
        py_cands = _find_candidates_vectorized(
            crd.x,
            crd.y,
            crd.pnr.astype(np.int64),
            len(crd),
            np.array([target.n for target in pix], dtype=np.int64),
            np.array([target.nx for target in pix], dtype=np.int64),
            np.array([target.ny for target in pix], dtype=np.int64),
            np.array([target.sumg for target in pix], dtype=np.int64),
            -10.0,
            -10.0,
            10.0,
            10.0,
            10,
            3,
            3,
            100,
            1.0,
            0.01,
            0.3,
            0.3,
            0.01,
        )

        assert len(py_cands) == native_count
        for index, native_cand in enumerate(native_cands):
            py_pnr, py_tol, py_corr = py_cands[index]
            assert py_pnr == native_cand.pnr
            assert py_tol == pytest.approx(native_cand.tol, abs=1e-6)
            assert py_corr == pytest.approx(native_cand.corr, abs=1e-6)