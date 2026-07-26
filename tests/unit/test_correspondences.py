"""Tests for correspondence matching, translated from C check_correspondences.c."""

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.correspondences import (
    NMAX,
    NTupel,
    allocate_adjacency_arrays,
    consistent_pair_matching,
    correct_frame,
    correspondences,
    four_camera_matching,
    match_pairs,
    quicksort_coord2d_x,
    quicksort_target_y,
    three_camera_matching,
)
from openptv2.algorithms.epi import Coord2d
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.parameters import ControlPar, VolumePar
from openptv2.algorithms.tracking_frame_buf import Frame, Target
from openptv2.algorithms.trafo import metric_to_pixel


def read_all_calibration(cpar):
    calib = []
    added_name = "test_data/calibration/cam1.tif.addpar"
    for cam in range(cpar.num_cams):
        ori_name = f"test_data/calibration/sym_cam{cam + 1}.tif.ori"
        calib.append(Calibration.from_file(ori_name, added_name))
    return calib


def generate_test_set(calib, cpar, vpar):
    """Generate data for 16 targets on 4 cameras (4x4 grid, 10mm apart).

    Matches C generate_test_set exactly.
    """
    frm = Frame(cpar.num_cams, 16)

    for cam in range(cpar.num_cams):
        frm.num_targets[cam] = 16

        for cpt_horz in range(4):
            for cpt_vert in range(4):
                cpt_ix = cpt_horz * 4 + cpt_vert
                if cam % 2:
                    cpt_ix = 15 - cpt_ix

                targ = frm.targets[cam][cpt_ix]
                targ.pnr = cpt_ix

                tmp = np.array([cpt_vert * 10.0, cpt_horz * 10.0, 0.0])
                x, y = img_coord(tmp, calib[cam], cpar.mm)
                x, y = metric_to_pixel(x, y, cpar)

                targ.x = x
                targ.y = y
                targ.n = 25
                targ.nx = 5
                targ.ny = 5
                targ.sumg = 10

    return frm


class TestSorting:
    def test_qs_target_y(self):
        targets = [
            Target(pnr=0, x=0.0, y=-0.2, n=5, nx=1, ny=2, sumg=10, tnr=-999),
            Target(pnr=6, x=0.2, y=0.0, n=10, nx=8, ny=1, sumg=20, tnr=-999),
            Target(pnr=3, x=0.2, y=0.8, n=10, nx=3, ny=3, sumg=30, tnr=-999),
            Target(pnr=4, x=0.4, y=-1.1, n=10, nx=3, ny=3, sumg=40, tnr=-999),
            Target(pnr=1, x=0.7, y=-0.1, n=10, nx=3, ny=3, sumg=50, tnr=-999),
            Target(pnr=7, x=1.2, y=0.3, n=10, nx=3, ny=3, sumg=60, tnr=-999),
            Target(pnr=5, x=10.4, y=0.1, n=10, nx=3, ny=3, sumg=70, tnr=-999),
        ]
        quicksort_target_y(targets)
        assert abs(targets[0].y - (-1.1)) < 1e-6
        assert abs(targets[1].y - (-0.2)) < 1e-6
        assert abs(targets[6].y - 0.8) < 1e-6

    def test_quicksort_coord2d_x(self):
        crds = [
            Coord2d(pnr=0, x=0.0, y=0.0),
            Coord2d(pnr=6, x=0.1, y=0.1),
            Coord2d(pnr=3, x=0.2, y=-0.8),
            Coord2d(pnr=4, x=-0.4, y=-1.1),
            Coord2d(pnr=1, x=0.7, y=-0.1),
            Coord2d(pnr=7, x=1.2, y=0.3),
            Coord2d(pnr=5, x=10.4, y=0.1),
        ]
        quicksort_coord2d_x(crds)
        assert abs(crds[0].x - (-0.4)) < 1e-6
        assert abs(crds[1].x - 0.0) < 1e-6
        assert abs(crds[6].x - 10.4) < 1e-6

    def test_quicksort_con(self):
        cons = [
            NTupel(p=[0, 1, 2, 3], corr=0.1),
            NTupel(p=[0, 1, 2, 3], corr=0.2),
            NTupel(p=[0, 1, 2, 3], corr=0.15),
        ]
        cons.sort(key=lambda t: -t.corr)
        assert abs(cons[0].corr - 0.2) < 1e-6
        assert abs(cons[2].corr - 0.1) < 1e-6


class TestPairwiseMatching:
    def test_pairwise_matching(self):
        cpar = ControlPar.from_yaml("test_data/parameters.yaml")
        vpar = VolumePar.from_yaml("test_data/parameters.yaml")

        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar, vpar)
        corrected = correct_frame(frm, calib, cpar, 0.0001)

        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = allocate_adjacency_arrays(
            cpar.num_cams, frm.num_targets
        )
        match_pairs(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr, corrected, frm, vpar, cpar, calib
        )

        for cam in range(cpar.num_cams - 1):
            for subcam in range(cam + 1, cpar.num_cams):
                for part in range(frm.num_targets[cam]):
                    if (subcam - cam) % 2 == 0:
                        correct_pnr = corrected[cam][p1_arr[cam, subcam, part]].pnr
                    else:
                        correct_pnr = 15 - corrected[cam][p1_arr[cam, subcam, part]].pnr

                    found = False
                    for cand_idx in range(n_arr[cam, subcam, part]):
                        p2_idx = p2_arr[cam, subcam, part, cand_idx]
                        if p2_idx < 0 or p2_idx >= len(corrected[subcam]):
                            continue
                        if corrected[subcam][p2_idx].pnr == correct_pnr:
                            found = True
                            break
                    assert found, (
                        f"cam={cam} subcam={subcam} part={part}: "
                        f"expected pnr={correct_pnr} not found in candidates"
                    )


class TestFourCameraMatching:
    def test_four_camera_matching(self):
        cpar = ControlPar.from_yaml("test_data/parameters.yaml")
        vpar = VolumePar.from_yaml("test_data/parameters.yaml")

        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar, vpar)
        corrected = correct_frame(frm, calib, cpar, 0.0001)

        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = allocate_adjacency_arrays(
            cpar.num_cams, frm.num_targets
        )
        match_pairs(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr, corrected, frm, vpar, cpar, calib
        )

        import numpy as np
        scratch_p = np.full((16, 4), -1, dtype=np.int32)
        scratch_corr = np.zeros(16, dtype=np.float64)
        matched = four_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr, 16, 1.0,
            scratch_p, scratch_corr, 16,
        )
        assert matched == 16


class TestThreeCameraMatching:
    def test_three_camera_matching(self):
        cpar = ControlPar.from_yaml("test_data/parameters.yaml")
        vpar = VolumePar.from_yaml("test_data/parameters.yaml")

        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar, vpar)

        # Darken cam 2 (index 1) below acceptance
        for part in range(frm.num_targets[1]):
            targ = frm.targets[1][part]
            targ.n = 0
            targ.nx = 0
            targ.ny = 0
            targ.sumg = 0

        corrected = correct_frame(frm, calib, cpar, 0.0001)
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = allocate_adjacency_arrays(
            cpar.num_cams, frm.num_targets
        )
        match_pairs(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr, corrected, frm, vpar, cpar, calib
        )

        import numpy as np
        scratch_p = np.full((4 * 16, 4), -1, dtype=np.int32)
        scratch_corr = np.zeros(4 * 16, dtype=np.float64)
        tusage = np.zeros((cpar.num_cams, NMAX), dtype=np.int32)

        matched = three_camera_matching(
            p1_arr,
            n_arr,
            p2_arr,
            corr_arr,
            dist_arr,
            4,
            frm.num_targets,
            100000.0,
            scratch_p,
            scratch_corr,
            4 * 16,
            tusage,
        )
        assert matched == 16


class TestTwoCameraMatching:
    def test_two_camera_matching(self):
        cpar = ControlPar.from_yaml("test_data/parameters.yaml")
        vpar = VolumePar.from_yaml("test_data/parameters.yaml")

        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001
        vpar.Zmin_lay[0] = -1
        vpar.Zmin_lay[1] = -1
        vpar.Zmax_lay[0] = 1
        vpar.Zmax_lay[1] = 1

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar, vpar)

        cpar.num_cams = 2
        corrected = correct_frame(frm, calib, cpar, 0.0001)
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = allocate_adjacency_arrays(
            cpar.num_cams, frm.num_targets
        )
        match_pairs(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr, corrected, frm, vpar, cpar, calib
        )

        import numpy as np
        scratch_p = np.full((4 * 16, 2), -1, dtype=np.int32)
        scratch_corr = np.zeros(4 * 16, dtype=np.float64)
        tusage = np.zeros((cpar.num_cams, NMAX), dtype=np.int32)

        matched = consistent_pair_matching(
            p1_arr,
            n_arr,
            p2_arr,
            corr_arr,
            dist_arr,
            2,
            frm.num_targets,
            10000.0,
            scratch_p,
            scratch_corr,
            4 * 16,
            tusage,
        )
        assert matched == 16


class TestFullCorrespondences:
    def test_correspondences(self):
        cpar = ControlPar.from_yaml("test_data/parameters.yaml")
        vpar = VolumePar.from_yaml("test_data/parameters.yaml")

        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar, vpar)
        corrected = correct_frame(frm, calib, cpar, 0.0001)

        con, match_counts = correspondences(frm, corrected, vpar, cpar, calib)

        assert match_counts[0] == 16
        assert match_counts[1] == 0
        assert match_counts[2] == 0
        assert match_counts[3] == 16


def test_get_by_pnrs_uses_coord_unused_sentinel():
    """Missing pnrs must be filled with COORD_UNUSED (-1e10), the sentinel
    point_position triangulation skips — NOT PT_UNUSED (-999), which would be
    triangulated as a real ray and produce out-of-volume 3D garbage for any
    non-quadruplet (triplet/pair) correspondence.

    Regression: existing tests only exercise 4-camera quads, so this mismatch
    slipped through while GUI sequence/determination produced garbage 3D.
    """
    from openptv2.algorithms.constants import COORD_UNUSED
    from openptv2.algorithms.epi import Coord2d
    from openptv2.correspondences import MatchedCoords

    mc = MatchedCoords.__new__(MatchedCoords)
    mc._corrected = [Coord2d(x=1.0, y=2.0, pnr=5)]
    out = mc.get_by_pnrs(np.array([5, -1], dtype=np.int32))
    assert out[0, 0] == 1.0 and out[0, 1] == 2.0
    assert out[1, 0] == COORD_UNUSED and out[1, 1] == COORD_UNUSED


@pytest.mark.integration
def test_determination_3d_cloud_is_physically_bounded():
    """End-to-end image->3D on real 4-camera data must not explode.

    This is the test that was MISSING: prior tests only built synthetic
    4-camera quadruplets, so a triplet/pair (which carry a COORD_UNUSED
    sentinel for the absent camera) was never triangulated. With the wrong
    sentinel (-999) those absent cameras were treated as real rays and
    produced 3D coordinates in the tens of thousands (|X|~35000, |Z|~45000),
    while every unit test still passed.

    The volume here is X in [-40,40], Z in [-20,20]. A correct cloud sits
    near that (|X|,|Z| < ~70); the bug produced |X|,|Z| in the tens of
    thousands. A bound of 500 cleanly separates the two.
    """
    import os
    from pathlib import Path

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.parameters import ControlPar, VolumePar
    from openptv2.correspondences import MatchedCoords, correspondences
    from openptv2.orientation import point_positions
    # gui.ptv pulls in skimage (a GUI-optional dep not installed in the
    # cibuildwheel test env); skip cleanly there instead of erroring.
    pytest.importorskip("skimage")
    from openptv2.gui.ptv import read_targets

    ds = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity"
    yaml = ds / "parameters_Run1.yaml"
    if not yaml.exists() or not (ds / "img" / "cam1.10001_targets").exists():
        pytest.skip("test_cavity fixture not available")

    cwd = os.getcwd()
    os.chdir(ds)
    try:
        cpar = ControlPar.from_yaml(str(yaml))
        vpar = VolumePar.from_yaml(str(yaml))
        nc = cpar.num_cams
        cals = []
        for i in range(nc):
            c = Calibration()
            c.from_file(f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar")
            cals.append(c)

        detections, corrected = [], []
        for i in range(nc):
            t = read_targets(f"img/cam{i+1}", 10001)
            if len(t) > 0:
                t.sort_y()
            detections.append(t)
            corrected.append(MatchedCoords(t, cpar, cals[i]))

        _, sorted_corresp, _ = correspondences(detections, corrected, cals, vpar, cpar)
        concat = np.concatenate(sorted_corresp, axis=1)
        flat = np.array(
            [c.get_by_pnrs(concat[cam]) for cam, c in enumerate(corrected)]
        )
        pos, _ = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)
    finally:
        os.chdir(cwd)

    assert len(pos) > 0
    x, z = pos[:, 0], pos[:, 2]
    # The bug drove these into the tens of thousands; correct output is ~70.
    assert np.abs(x).max() < 500.0, f"X exploded: max|X|={np.abs(x).max():.1f}"
    assert np.abs(z).max() < 500.0, f"Z exploded: max|Z|={np.abs(z).max():.1f}"
    # Most points should land near the measurement volume, not scattered far.
    near = (np.abs(x) < 80.0) & (np.abs(z) < 60.0)
    assert near.mean() > 0.5, f"only {near.mean():.0%} of cloud near volume"
