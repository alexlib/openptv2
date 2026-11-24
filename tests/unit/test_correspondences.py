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
            p1_arr,
            n_arr,
            p2_arr,
            corr_arr,
            dist_arr,
            16,
            1.0,
            scratch_p,
            scratch_corr,
            16,
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


class TestNTupelIdentityTranslation:
    """NTupel.p[cam] is an INDEX into corrected[cam] (x-sorted) for every
    camera, not a pnr -- despite find_candidate's cand_pnr output/docstring
    suggesting otherwise. The only existing correspondence test above checks
    match COUNTS, never per-camera IDENTITY, so it cannot catch a caller that
    treats NTupel.p as an already-correct pnr: match_counts stays right
    either way, only which particles get matched together changes.

    These tests build a random (non-grid) scatter of particles with pnr
    assigned in Y-SORTED order -- what real detection actually produces
    (gui/ptv.py's _detect calls targs.sort_y() before MatchedCoords assigns
    pnr) -- and verify every camera's component of every matched NTupel
    identifies the SAME true particle once correctly translated via
    corrected[cam][p[cam]].pnr. This is the exact scenario a partial/wrong
    translation (e.g. only translating camera 0) silently gets wrong: it
    still finds the right NUMBER of quads, just pairs up the wrong
    particles across cameras.
    """

    NUM_CAMS = 4

    def _random_scatter_frame(self, cals, cpar, n=40, seed=0):
        """n random 3D points, each camera's targets pnr-assigned in
        y-sorted order (the real-detection convention), returns
        (frm, true_pid_by_pnr) where true_pid_by_pnr[cam][pnr] is the
        ground-truth particle id of that camera's pnr-th target."""
        rng = np.random.default_rng(seed)
        positions = rng.uniform(-21.0, 21.0, (n, 3))

        frm = Frame(cpar.num_cams, n)
        true_pid_by_pnr = [dict() for _ in range(cpar.num_cams)]
        for cam in range(cpar.num_cams):
            frm.num_targets[cam] = n
            proj = []
            for pid in range(n):
                mx, my = img_coord(positions[pid], cals[cam], cpar.mm)
                x, y = metric_to_pixel(mx, my, cpar)
                proj.append((pid, x, y))
            proj.sort(key=lambda t: t[2])  # y-sort, matching targs.sort_y()
            for pnr, (pid, x, y) in enumerate(proj):
                targ = frm.targets[cam][pnr]
                targ.pnr = pnr
                targ.x = x
                targ.y = y
                targ.n = 100
                targ.nx = 10
                targ.ny = 10
                targ.sumg = 1000
                true_pid_by_pnr[cam][pnr] = pid
        return frm, true_pid_by_pnr

    def test_quad_p_translates_to_one_consistent_particle_per_camera(self):
        cpar = ControlPar.from_yaml("test_data/tracking_synthetic/parameters_Run1.yaml")
        vpar = VolumePar.from_yaml("test_data/tracking_synthetic/parameters_Run1.yaml")
        calib = [
            Calibration.from_file(
                f"test_data/tracking_synthetic/cal/cam{c + 1}.tif.ori",
                f"test_data/tracking_synthetic/cal/cam{c + 1}.tif.addpar",
            )
            for c in range(self.NUM_CAMS)
        ]

        frm, true_pid_by_pnr = self._random_scatter_frame(calib, cpar, n=40, seed=0)
        corrected = correct_frame(frm, calib, cpar, 0.0001)
        con, match_counts = correspondences(frm, corrected, vpar, cpar, calib)

        assert match_counts[0] > 0, (
            "expected at least some quads on a well-separated scatter"
        )

        n_checked = 0
        for tup in con[: match_counts[0]]:  # quads are listed first
            pids = []
            for cam in range(self.NUM_CAMS):
                idx = tup.p[cam]
                assert idx >= 0, "a quad must have all four cameras populated"
                # THE fix under test: idx is a corrected[cam] INDEX, not a pnr.
                pnr = corrected[cam][idx].pnr
                pids.append(true_pid_by_pnr[cam][pnr])
            assert len(set(pids)) == 1, (
                f"quad cameras disagree on particle identity: {pids} "
                f"(raw NTupel.p={list(tup.p)}) -- NTupel.p[cam] was used "
                f"as a pnr directly instead of translated via "
                f"corrected[cam][p[cam]].pnr"
            )
            n_checked += 1
        assert n_checked == match_counts[0]

    def test_naive_pnr_usage_would_have_failed_this_scatter(self):
        """Documents the footgun this class guards against: using
        NTupel.p[cam] directly as a pnr (skipping the corrected[cam][idx].pnr
        translation) scrambles identities on a real (non-grid) scatter, even
        though match_counts alone looks identical either way."""
        cpar = ControlPar.from_yaml("test_data/tracking_synthetic/parameters_Run1.yaml")
        vpar = VolumePar.from_yaml("test_data/tracking_synthetic/parameters_Run1.yaml")
        calib = [
            Calibration.from_file(
                f"test_data/tracking_synthetic/cal/cam{c + 1}.tif.ori",
                f"test_data/tracking_synthetic/cal/cam{c + 1}.tif.addpar",
            )
            for c in range(self.NUM_CAMS)
        ]

        frm, true_pid_by_pnr = self._random_scatter_frame(calib, cpar, n=40, seed=0)
        corrected = correct_frame(frm, calib, cpar, 0.0001)
        con, match_counts = correspondences(frm, corrected, vpar, cpar, calib)
        assert match_counts[0] > 0

        naive_correct = 0
        for tup in con[: match_counts[0]]:
            pids = [
                true_pid_by_pnr[cam].get(tup.p[cam])  # WRONG: no translation
                for cam in range(self.NUM_CAMS)
            ]
            if len(set(pids)) == 1:
                naive_correct += 1

        assert naive_correct < match_counts[0], (
            "expected the untranslated interpretation to disagree on at "
            "least some quads on a random scatter -- if this starts "
            "passing, NTupel.p's semantics may have changed and this "
            "guard (and its docstring) should be revisited"
        )


class TestProductionCorrespondencesWrapperIdentity:
    """openptv2.correspondences.correspondences() (NOT
    openptv2.algorithms.correspondences.correspondences -- this is the
    higher-level wrapper gui/ptv.py's real sequence loop actually calls) is
    the production path a real experiment runs. It performs the
    corrected[cam][geo_id].pnr translation internally for every camera, so
    it should NOT exhibit the raw-NTupel.p footgun TestNTupelIdentityTranslation
    guards against at the low level. This is the end-to-end regression proof
    for that: real detection order (y-sorted -> pnr = i), a random 3D
    scatter, checked against ground truth.
    """

    def test_wrapper_translates_every_camera_correctly(self):
        from openptv2.algorithms.tracking_frame_buf import Target as AlgoTarget
        from openptv2.correspondences import MatchedCoords
        from openptv2.correspondences import correspondences as prod_correspondences

        num_cams = 4
        cpar = ControlPar.from_yaml("test_data/tracking_synthetic/parameters_Run1.yaml")
        vpar = VolumePar.from_yaml("test_data/tracking_synthetic/parameters_Run1.yaml")
        calib = [
            Calibration.from_file(
                f"test_data/tracking_synthetic/cal/cam{c + 1}.tif.ori",
                f"test_data/tracking_synthetic/cal/cam{c + 1}.tif.addpar",
            )
            for c in range(num_cams)
        ]

        rng = np.random.default_rng(1)
        n = 40
        positions = rng.uniform(-21.0, 21.0, (n, 3))

        # Mimic real detection: per-camera list, y-sorted, THEN MatchedCoords
        # assigns pnr = i in that (y-sorted) order -- see gui/ptv.py's
        # _detect (targs.sort_y()) followed by MatchedCoords(reset_numbers=True).
        img_pts = []
        true_pid_by_pnr = []
        for cam in range(num_cams):
            proj = []
            for pid in range(n):
                mx, my = img_coord(positions[pid], calib[cam], cpar.mm)
                x, y = metric_to_pixel(mx, my, cpar)
                proj.append((pid, x, y))
            proj.sort(key=lambda t: t[2])
            targs = [
                AlgoTarget(pnr=0, x=x, y=y, n=100, nx=10, ny=10, sumg=1000, tnr=0)
                for pid, x, y in proj
            ]
            img_pts.append(targs)
            true_pid_by_pnr.append([pid for pid, x, y in proj])

        flat_coords = [
            MatchedCoords(targs, cpar, calib[cam]) for cam, targs in enumerate(img_pts)
        ]

        sorted_pos, sorted_corresp, num_targs = prod_correspondences(
            img_pts, flat_coords, calib, vpar, cpar
        )

        quad_corresp = sorted_corresp[0]  # clique_type 0 == quads
        total = quad_corresp.shape[1]
        assert total > 0, "expected at least some quads on a well-separated scatter"

        for i in range(total):
            pids = []
            for cam in range(num_cams):
                pnr = int(quad_corresp[cam, i])
                assert pnr >= 0, "a quad must have all four cameras populated"
                pids.append(true_pid_by_pnr[cam][pnr])
            assert len(set(pids)) == 1, (
                f"production wrapper quad disagrees on particle identity: "
                f"{pids} (quad {i})"
            )


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
    from openptv2.storage import RunStore, find_existing_store

    ds = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity"
    yaml = ds / "parameters_Run1.yaml"
    store_path = find_existing_store(ds)
    if not yaml.exists() or store_path is None:
        pytest.skip("test_cavity RunStore fixture not available")

    store = RunStore(store_path, mode="r")
    cwd = os.getcwd()
    os.chdir(ds)
    try:
        cpar = ControlPar.from_yaml(str(yaml))
        vpar = VolumePar.from_yaml(str(yaml))
        nc = cpar.num_cams
        cals = []
        for i in range(nc):
            c = Calibration()
            c.from_file(f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar")
            cals.append(c)

        detections, corrected = [], []
        for i in range(nc):
            t = read_targets(f"img/cam{i + 1}", 10001, store=store, cam_idx=i)
            if len(t) > 0:
                t.sort_y()
            detections.append(t)
            corrected.append(MatchedCoords(t, cpar, cals[i]))

        _, sorted_corresp, _ = correspondences(detections, corrected, cals, vpar, cpar)
        concat = np.concatenate(sorted_corresp, axis=1)
        flat = np.array([c.get_by_pnrs(concat[cam]) for cam, c in enumerate(corrected)])
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


class TestCandidateBuffer:
    """The candidate buffer used to be 4 * NMAX rows and a full buffer silently dropped every candidate of the targets
    visited last (those at the end of the camera-1 order), so with a wide epipolar band (criteria.eps0) and dense
    seeding the last particles were never matched and no warning said so. Now a stage that fills the buffer runs again on
    a buffer twice as large, up to a hard cap that warns; the target-usage table follows the number of targets."""

    NUM_CAMS = 4

    def _frame(self, n, eps0=None, seed=1):
        import openptv2.algorithms.correspondences as corr_mod  # noqa: F401  (module attributes are patched by the tests)

        cpar = ControlPar.from_yaml("test_data/parameters.yaml")
        vpar = VolumePar.from_yaml("test_data/parameters.yaml")
        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001
        if eps0 is not None:
            vpar.eps0 = eps0
        calib = read_all_calibration(cpar)
        frm, pid_by_pnr = TestNTupelIdentityTranslation()._random_scatter_frame(calib, cpar, n=n, seed=seed)
        corrected = correct_frame(frm, calib, cpar, 0.0001)
        return frm, corrected, vpar, cpar, calib, pid_by_pnr

    @staticmethod
    def _signature(con, match_counts):
        return list(match_counts), [(tuple(int(x) for x in t.p), round(float(t.corr), 9)) for t in con]

    def _pids(self, con, corrected, pid_by_pnr, k):
        """True particle of every camera of the first k tuples (all cameras must agree)."""
        out = []
        for tup in con[:k]:
            pids = {pid_by_pnr[c][corrected[c][tup.p[c]].pnr] for c in range(self.NUM_CAMS)}
            assert len(pids) == 1, f"cameras disagree on the particle: {pids}"
            out.append(pids.pop())
        return out

    @pytest.mark.filterwarnings("ignore:.*epipolar band.*too wide:RuntimeWarning")
    def test_wide_band_is_not_truncated_at_the_default_buffer(self):
        # 150 particles and a 1 mm band: more candidates than the 80960 rows of the old buffer. The old code returned
        # 66 quads + 93 triplets (mostly wrong); the right answer is every particle as a quad.
        frm, corrected, vpar, cpar, calib, pid_by_pnr = self._frame(150, eps0=1.0)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert list(mc) == [150, 0, 0, 150]
        assert sorted(self._pids(con, corrected, pid_by_pnr, 150)) == list(range(150))

    def test_a_tiny_initial_buffer_gives_the_same_matches(self, monkeypatch):
        import openptv2.algorithms.correspondences as corr_mod

        frm, corrected, vpar, cpar, calib, _ = self._frame(60, eps0=0.5)
        reference = self._signature(*correspondences(frm, corrected, vpar, cpar, calib))
        assert reference[0][0] >= 40                     # the reference is not a trivial case
        monkeypatch.setattr(corr_mod, "CANDIDATE_ROWS_PER_CAM", 4)       # 16 rows: overflows at once, many times
        assert self._signature(*correspondences(frm, corrected, vpar, cpar, calib)) == reference

    def test_target_tables_follow_the_number_of_targets(self, monkeypatch):
        import openptv2.algorithms.correspondences as corr_mod

        frm, corrected, vpar, cpar, calib, _ = self._frame(60, eps0=0.5)
        reference = self._signature(*correspondences(frm, corrected, vpar, cpar, calib))
        monkeypatch.setattr(corr_mod, "NMAX", 8)         # fewer slots than targets: used to skip / overrun every target past 8
        monkeypatch.setattr(corr_mod, "CANDIDATE_ROWS_PER_CAM", 8)
        assert self._signature(*correspondences(frm, corrected, vpar, cpar, calib)) == reference

    def test_reaching_the_cap_warns_instead_of_failing_silently(self, monkeypatch):
        import openptv2.algorithms.correspondences as corr_mod

        frm, corrected, vpar, cpar, calib, _ = self._frame(60, eps0=0.5)
        monkeypatch.setattr(corr_mod, "CANDIDATE_ROWS_PER_CAM", 4)
        monkeypatch.setattr(corr_mod, "MAX_CANDIDATE_ROWS", 16)
        with pytest.warns(RuntimeWarning, match="candidate buffer reached its cap of 16 rows"):
            con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[3] == len(con) <= 16                    # what fitted is still returned, consistently

    def test_stage_runner_doubles_until_the_stage_fits(self):
        import openptv2.algorithms.correspondences as corr_mod

        calls = []

        def stage(scratch_p, scratch_corr, scratch_size):
            calls.append(scratch_size)
            return min(100, scratch_size)               # wants 100 rows, stops when full

        matched, p, c, size = corr_mod._run_matching_stage(
            stage, (), (), np.full((8, 4), -1, np.int32), np.zeros(8), 8, 4, "test stage"
        )
        assert (matched, size, calls) == (100, 128, [8, 16, 32, 64, 128])
        assert p.shape == (128, 4) and c.shape == (128,)

    def test_stage_runner_passes_the_other_arguments_in_order(self):
        import openptv2.algorithms.correspondences as corr_mod

        seen = {}

        def stage(a, b, scratch_p, scratch_corr, scratch_size, tail):
            seen.update(a=a, b=b, size=scratch_size, tail=tail)
            return 3

        out = corr_mod._run_matching_stage(stage, ("A", "B"), ("T",), np.zeros((8, 4), np.int32), np.zeros(8), 8, 4, "x")
        assert out[0] == 3 and seen == {"a": "A", "b": "B", "size": 8, "tail": "T"}


class TestCandidateBufferEstimates:
    """The buffer is sized from the data: an upper bound from the candidate counts (never grow past what a stage can
    write), a projection from how far a full stage got (one rerun instead of one per doubling), and a warning when
    the candidates are absurdly many per target."""

    def _stage_needing(self, rows, calls):
        def stage(scratch_p, scratch_corr, scratch_size):
            calls.append(scratch_size)
            return min(rows, scratch_size)

        return stage

    def _run(self, calls, rows, start=8, **kw):
        import openptv2.algorithms.correspondences as corr_mod

        return corr_mod._run_matching_stage(
            self._stage_needing(rows, calls), (), (), np.full((start, 4), -1, np.int32), np.zeros(start), start, 4, "t", **kw
        )

    def test_growth_never_passes_the_bound(self):
        calls = []
        matched, _, _, size = self._run(calls, rows=100, bound=100)
        assert matched == 100 and calls == [8, 16, 32, 64, 101]          # bound + 1: room for a stage that writes the bound

    def test_a_stage_that_writes_exactly_its_bound_still_ends(self):
        calls = []
        matched, _, _, size = self._run(calls, rows=64, bound=64)
        assert matched == 64 and size == 65 and calls == [8, 16, 32, 64, 65]

    def test_progress_projects_the_size_in_one_step(self):
        calls = []
        matched, _, _, size = self._run(calls, rows=100, progress=lambda p, s: 0.1)       # "10 % done": needs ~80, with margin 121
        assert matched == 100 and calls == [8, 121]

    def test_projection_is_limited_per_step_and_doubling_is_the_floor(self):
        import openptv2.algorithms.correspondences as corr_mod

        calls = []
        self._run(calls, rows=100_000, progress=lambda p, s: 1e-9)             # absurd projection: limited per step
        assert calls[1] == 8 * corr_mod.MAX_GROWTH_PER_STEP
        calls = []
        self._run(calls, rows=20, progress=lambda p, s: 0.99)                    # nearly done: still at least doubling
        assert calls == [8, 16, 32]

    def test_bounds_equal_a_brute_force_count(self):
        import openptv2.algorithms.correspondences as corr_mod

        rng = np.random.default_rng(3)
        nc, nt = 4, np.array([7, 9, 5, 8])
        n = rng.integers(0, 6, (nc, nc, 9)).astype(np.int32)
        quads = sum(int(n[0, 1, i]) * int(n[0, 2, i]) * int(n[0, 3, i]) for i in range(nt[0]))
        trips = sum(int(n[a, b, i]) * int(n[a, c, i]) for a in range(2) for b in range(a + 1, 3) for c in range(b + 1, 4)
                    for i in range(nt[a]) if b < nc - 1)
        pairs = sum(int(n[a, b, i]) for a in range(3) for b in range(a + 1, 4) for i in range(nt[a]))
        assert corr_mod._bound_quads(n, nt) == quads
        assert corr_mod._bound_triplets(n, nt, nc) == trips
        assert corr_mod._bound_pairs(n, nt, nc) == pairs

    def test_progress_readers_name_the_target_a_full_buffer_reached(self):
        import openptv2.algorithms.correspondences as corr_mod

        tc = [100, 100, 100, 100]
        p = np.full((5, 4), -1, np.int32)
        p[4] = [39, 7, 8, 9]                                                     # last quad belongs to target 39 of 100
        assert corr_mod._progress_four(p, 5, tc, 4) == pytest.approx(0.395)
        p3 = np.full((5, 4), -2, np.int32)
        p3[4] = [-2, 49, 3, 4]                                                   # block a=1 (the second of two), target 49
        assert corr_mod._progress_three(p3, 5, tc, 4) == pytest.approx((100 + 49.5) / 200)
        p2 = np.full((5, 4), -2, np.int32)
        p2[4] = [-2, 9, 20, -2]                                                  # camera pair (1, 2), target 9: pairs (0,1)(0,2)(0,3) done
        assert corr_mod._progress_pair(p2, 5, tc, 4) == pytest.approx((300 + 9.5) / 600)

    def test_stage_bounds_hold_on_real_frames_and_the_jump_saves_reruns(self, monkeypatch):
        import openptv2.algorithms.correspondences as corr_mod

        seen = []
        orig = corr_mod._run_matching_stage

        def spy(stage, ab, aa, p, c, size, nc, name, **kw):
            out = orig(stage, ab, aa, p, c, size, nc, name, **kw)
            seen.append((name, out[0], kw.get("bound")))
            return out

        attempts = []
        orig_four = corr_mod._stage_four

        def counting(*a):
            attempts.append(a[-1])
            return orig_four(*a)

        monkeypatch.setattr(corr_mod, "_run_matching_stage", spy)
        monkeypatch.setattr(corr_mod, "_stage_four", counting)
        frm, corrected, vpar, cpar, calib, _ = TestCandidateBuffer()._frame(150, eps0=1.5)
        correspondences(frm, corrected, vpar, cpar, calib)
        assert seen and all(m <= b for _, m, b in seen), seen                 # every stage wrote no more than its bound
        quads = [m for n, m, _ in seen if n.startswith("four")][0]
        assert quads > 500_000
        assert len(attempts) <= 3, attempts                                       # doubling alone needed 5 attempts for this frame

    def test_wide_band_warns_once_and_normal_data_does_not(self, monkeypatch):
        import warnings

        import openptv2.algorithms.correspondences as corr_mod

        monkeypatch.setattr(corr_mod, "_wide_band_warned", False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                                         # a normal band: no warning at all
            frm, corrected, vpar, cpar, calib, _ = TestCandidateBuffer()._frame(150, eps0=0.2)
            correspondences(frm, corrected, vpar, cpar, calib)
        frm, corrected, vpar, cpar, calib, _ = TestCandidateBuffer()._frame(150, eps0=1.0)
        with pytest.warns(RuntimeWarning, match=r"epipolar band.*too wide"):
            correspondences(frm, corrected, vpar, cpar, calib)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                                         # once per process
            correspondences(frm, corrected, vpar, cpar, calib)


def _take_best_reference(src_p, src_corr, dst_p, dst_corr, num_cams, num_cands, tusage, dst_offset):
    """The loop of take_best_candidates as it was written before the typed order array (plain Python)."""
    order = np.argsort(src_corr[:num_cands])[::-1]
    taken = 0
    for idx in range(num_cands):
        cand = int(order[idx])
        if any(src_p[cand, c] > -1 and tusage[c, src_p[cand, c]] > 0 for c in range(num_cams)):
            continue
        for c in range(num_cams):
            if src_p[cand, c] > -1:
                tusage[c, src_p[cand, c]] += 1
            dst_p[dst_offset + taken, c] = src_p[cand, c]
        dst_corr[dst_offset + taken] = src_corr[cand]
        taken += 1
    return taken


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_take_best_candidates_is_unchanged_by_the_typed_order(seed):
    from openptv2.algorithms.correspondences import take_best_candidates

    rng = np.random.default_rng(seed)
    n, nc = 5000, 4
    src_p = rng.integers(-1, 300, (n, nc)).astype(np.int32)
    src_corr = np.round(rng.random(n), 2)                        # many exact ties: the order among them must not change
    out = []
    for fn in (take_best_candidates, _take_best_reference):
        dst_p = np.full((n + 10, nc), -1, np.int32)
        dst_corr = np.zeros(n + 10)
        tus = np.zeros((nc, 400), np.int32)
        k = fn(src_p, src_corr, dst_p, dst_corr, nc, n, tus, 3)
        out.append((k, dst_p.copy(), dst_corr.copy(), tus.copy()))
    assert out[0][0] == out[1][0] > 50
    for a, b in zip(out[0][1:], out[1][1:]):
        np.testing.assert_array_equal(a, b)
