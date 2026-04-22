"""Tests for correspondence matching, translated from C check_correspondences.c."""

import numpy as np
import pytest

from algorithms.calibration import Calibration
from algorithms.parameters import ControlPar, VolumePar
from algorithms.tracking_frame_buf import Target, Frame
from algorithms.epi import Coord2d
from algorithms.correspondences import (
    NTupel, Correspond, quicksort_target_y, quicksort_coord2d_x,
    safely_allocate_adjacency_lists, match_pairs, four_camera_matching,
    three_camera_matching, consistent_pair_matching, take_best_candidates,
    correct_frame, correspondences, NMAX,
)
from algorithms.epi import MAXCAND
from algorithms.imgcoord import img_coord
from algorithms.trafo import metric_to_pixel


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
        cpar = ControlPar.from_file("test_data/parameters/ptv.par")
        vpar = VolumePar.from_file("test_data/parameters/criteria.par")

        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar, vpar)
        corrected = correct_frame(frm, calib, cpar, 0.0001)

        lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
        match_pairs(lists, corrected, frm, vpar, cpar, calib)

        for cam in range(cpar.num_cams - 1):
            for subcam in range(cam + 1, cpar.num_cams):
                for part in range(frm.num_targets[cam]):
                    if (subcam - cam) % 2 == 0:
                        correct_pnr = corrected[cam][lists[cam][subcam][part].p1].pnr
                    else:
                        correct_pnr = 15 - corrected[cam][lists[cam][subcam][part].p1].pnr

                    found = False
                    for cand_idx in range(MAXCAND):
                        p2_idx = lists[cam][subcam][part].p2[cand_idx]
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
        cpar = ControlPar.from_file("test_data/parameters/ptv.par")
        vpar = VolumePar.from_file("test_data/parameters/criteria.par")

        cpar.mm.n2[0] = 1.0001
        cpar.mm.n3 = 1.0001

        calib = read_all_calibration(cpar)
        frm = generate_test_set(calib, cpar, vpar)
        corrected = correct_frame(frm, calib, cpar, 0.0001)

        lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
        match_pairs(lists, corrected, frm, vpar, cpar, calib)

        con = [NTupel() for _ in range(16)]
        matched = four_camera_matching(lists, 16, 1.0, con, 16)
        assert matched == 16


class TestThreeCameraMatching:
    def test_three_camera_matching(self):
        cpar = ControlPar.from_file("test_data/parameters/ptv.par")
        vpar = VolumePar.from_file("test_data/parameters/criteria.par")

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
        lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
        match_pairs(lists, corrected, frm, vpar, cpar, calib)

        con = [NTupel() for _ in range(4 * 16)]
        tusage = [[0] * NMAX for _ in range(cpar.num_cams)]

        matched = three_camera_matching(lists, 4, frm.num_targets,
            100000.0, con, 4 * 16, tusage)
        assert matched == 16


class TestTwoCameraMatching:
    def test_two_camera_matching(self):
        cpar = ControlPar.from_file("test_data/parameters/ptv.par")
        vpar = VolumePar.from_file("test_data/parameters/criteria.par")

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
        lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
        match_pairs(lists, corrected, frm, vpar, cpar, calib)

        con = [NTupel() for _ in range(4 * 16)]
        tusage = [[0] * NMAX for _ in range(cpar.num_cams)]

        matched = consistent_pair_matching(lists, 2, frm.num_targets,
            10000.0, con, 4 * 16, tusage)
        assert matched == 16


class TestFullCorrespondences:
    def test_correspondences(self):
        cpar = ControlPar.from_file("test_data/parameters/ptv.par")
        vpar = VolumePar.from_file("test_data/parameters/criteria.par")

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
