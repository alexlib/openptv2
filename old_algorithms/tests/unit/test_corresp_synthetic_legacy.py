"""
Diagnostic tests for correspondences using the same synthetic 4x4 grid setup
as the C (check_correspondences.c) and Cython (test_corresp.py) tests.

Setup: 4 symmetric cameras, 16 targets on a 4x4 grid (10 mm apart) at z=0,
no multimedia (n1=n2=n3≈1.0).  The correct answer is 16 quadruplets.

Each test isolates one stage of the pipeline so failures pinpoint exactly
where the Python code diverges from C/Cython.
"""

import numpy as np
import pytest

from algorithms.calibration import Calibration
from algorithms.correspondences import (
    MatchedCoords,
    correspondences,
    four_camera_matching,
    match_pairs,
    safely_allocate_adjacency_lists,
    safely_allocate_target_usage_marks,
    take_best_candidates,
    three_camera_matching,
    consistent_pair_matching,
)
from algorithms.epi import epi_mm
from algorithms.imgcoord import image_coordinates
from algorithms.parameters import ControlPar, MultimediaPar, VolumePar
from algorithms.tracking_frame_buf import Frame, Target, n_tupel_dtype
from algorithms.trafo import metric_to_pixel

# ---------------------------------------------------------------------------
# Fixtures: replicate the C / Cython synthetic scene exactly
# ---------------------------------------------------------------------------

CAL_DIR = "test_data/calibration"
CORRESP_DIR = "test_data/corresp"
NUM_CAMS = 4
NUM_PTS = 16  # 4×4 grid


def _build_cpar() -> ControlPar:
    """Build control parameters matching test_data/corresp/control.par
    but with near-unity refractive indices (no multimedia)."""
    mm = MultimediaPar(n1=1.0, n2=[1.0001], n3=1.0001, d=[1.0])
    return ControlPar(
        num_cams=4,
        imx=1280,
        imy=1024,
        pix_x=0.017,
        pix_y=0.017,
        chfield=0,
        mm=mm,
        all_cam_flag=0,
    )


def _build_vpar() -> VolumePar:
    """Build volume parameters matching test_data/corresp/criteria.par."""
    return VolumePar(
        x_lay=[-250.0, 250.0],
        z_min_lay=[-100.0, -100.0],
        z_max_lay=[100.0, 100.0],
        cnx=0.3,
        cny=0.3,
        cn=0.01,
        csumg=0.01,
        corrmin=33.0,
        eps0=1.0,
    )


def _load_calibrations() -> list:
    """Load the 4 symmetric calibration files."""
    cals = []
    for c in range(NUM_CAMS):
        cal = Calibration()
        cal.from_file(
            f"{CAL_DIR}/sym_cam{c+1}.tif.ori",
            f"{CAL_DIR}/cam1.tif.addpar",
        )
        cals.append(cal)
    return cals


def _generate_targets(cals, cpar):
    """Generate the same synthetic 4×4 grid targets as in C/Cython tests.

    Returns (frame, corrected_list, img_pts_per_cam).
    """
    mm = cpar.mm
    frm = Frame(NUM_CAMS)
    corrected = [None] * NUM_CAMS

    for cam in range(NUM_CAMS):
        frm.num_targets[cam] = NUM_PTS
        targs = [None] * NUM_PTS

        for row in range(4):
            for col in range(4):
                targ_ix = row * 4 + col
                # Avoid symmetric case — same as C and Cython tests
                if cam % 2:
                    targ_ix = 15 - targ_ix

                pos3d = 10.0 * np.array([[col, row, 0]], dtype=np.float64)
                pos2d = image_coordinates(pos3d, cals[cam], mm)
                px, py = metric_to_pixel(pos2d[0, 0], pos2d[0, 1], cpar)

                t = Target()
                t.pnr = targ_ix
                t.x = px
                t.y = py
                t.n = 25
                t.nx = 5
                t.ny = 5
                t.sumg = 10
                t.tnr = -1
                targs[targ_ix] = t

        # Store in frame
        for i, t in enumerate(targs):
            while i >= len(frm.targets[cam]):
                frm.targets[cam].append(Target())
            frm.targets[cam][i].pnr = t.pnr
            frm.targets[cam][i].x = t.x
            frm.targets[cam][i].y = t.y
            frm.targets[cam][i].n = t.n
            frm.targets[cam][i].nx = t.nx
            frm.targets[cam][i].ny = t.ny
            frm.targets[cam][i].sumg = t.sumg
            frm.targets[cam][i].tnr = t.tnr

        mc = MatchedCoords(targs, cpar, cals[cam])
        corrected[cam] = mc.buf

    return frm, corrected


# ---------------------------------------------------------------------------
# Optv (Cython) helpers — skip if not installed
# ---------------------------------------------------------------------------

def _optv_available():
    try:
        import optv  # noqa: F401
        return True
    except ImportError:
        return False


def _optv_generate_targets(cpar):
    """Generate the same 4×4 grid through Cython."""
    from optv.calibration import Calibration as OptvCal
    from optv.correspondences import MatchedCoords as OptvMC
    from optv.imgcoord import image_coordinates as optv_img_coord
    from optv.parameters import ControlParams as OptvCP, VolumeParams as OptvVP
    from optv.tracking_framebuf import TargetArray as OptvTA
    from optv.transforms import convert_arr_metric_to_pixel as optv_m2p

    optv_cpar = OptvCP(4)
    optv_cpar.read_control_par(f"{CORRESP_DIR}/control.par")
    optv_cpar.get_multimedia_params().set_layers([1.0001], [1.0])
    optv_cpar.get_multimedia_params().set_n3(1.0001)

    optv_cals = []
    optv_img_pts = []
    optv_corrected = []

    for c in range(NUM_CAMS):
        cal = OptvCal()
        cal.from_file(
            f"{CAL_DIR}/sym_cam{c+1}.tif.ori".encode(),
            f"{CAL_DIR}/cam1.tif.addpar".encode(),
        )
        optv_cals.append(cal)

        targs = OptvTA(NUM_PTS)
        for row in range(4):
            for col in range(4):
                targ_ix = row * 4 + col
                if c % 2:
                    targ_ix = 15 - targ_ix
                targ = targs[targ_ix]

                pos3d = 10.0 * np.array([[col, row, 0]], dtype=np.float64)
                pos2d = optv_img_coord(pos3d, cal, optv_cpar.get_multimedia_params())
                targ.set_pos(optv_m2p(pos2d, optv_cpar)[0])
                targ.set_pnr(targ_ix)
                targ.set_pixel_counts(25, 5, 5)
                targ.set_sum_grey_value(10)

        optv_img_pts.append(targs)
        mc = OptvMC(targs, optv_cpar, cal)
        optv_corrected.append(mc)

    return optv_cpar, optv_cals, optv_img_pts, optv_corrected


# ===================================================================
# STAGE 0: Verify target generation matches Cython
# ===================================================================

class TestStage0_TargetGeneration:
    """Verify that Python target pixel coords match Cython."""

    @pytest.mark.skipif(not _optv_available(), reason="optv not installed")
    def test_target_pixel_positions_match_cython(self):
        cpar = _build_cpar()
        cals = _load_calibrations()
        frm, corrected = _generate_targets(cals, cpar)

        _, optv_cals, optv_img_pts, _ = _optv_generate_targets(cpar)

        for cam in range(NUM_CAMS):
            for i in range(NUM_PTS):
                py_t = frm.targets[cam][i]
                optv_t = optv_img_pts[cam][i]
                optv_pos = optv_t.pos()

                assert abs(py_t.x - optv_pos[0]) < 0.01, (
                    f"cam{cam} targ{i}: py x={py_t.x:.4f}, optv x={optv_pos[0]:.4f}"
                )
                assert abs(py_t.y - optv_pos[1]) < 0.01, (
                    f"cam{cam} targ{i}: py y={py_t.y:.4f}, optv y={optv_pos[1]:.4f}"
                )


# ===================================================================
# STAGE 1: Verify MatchedCoords (flat coords + x-sort) match Cython
# ===================================================================

class TestStage1_MatchedCoords:
    """Verify corrected (flat) coordinates and sort order match Cython."""

    @pytest.mark.skipif(not _optv_available(), reason="optv not installed")
    def test_corrected_coords_match_cython(self):
        cpar = _build_cpar()
        cals = _load_calibrations()
        frm, corrected = _generate_targets(cals, cpar)

        _, _, _, optv_corrected = _optv_generate_targets(cpar)

        for cam in range(NUM_CAMS):
            py_pos = np.column_stack([corrected[cam].x, corrected[cam].y])
            py_pnr = corrected[cam].pnr.astype(np.int_)
            optv_pos, optv_pnr = optv_corrected[cam].as_arrays()

            np.testing.assert_array_equal(
                py_pnr, optv_pnr,
                err_msg=f"cam{cam} pnr order mismatch",
            )
            np.testing.assert_allclose(
                py_pos, optv_pos, atol=1e-4,
                err_msg=f"cam{cam} corrected positions mismatch",
            )


# ===================================================================
# STAGE 2: Verify match_pairs (adjacency lists)
# ===================================================================

class TestStage2_MatchPairs:
    """Verify pairwise candidate lists match Cython."""

    def test_match_pairs_finds_candidates_for_all_targets(self):
        """Every target should have at least one candidate in each pair."""
        cpar = _build_cpar()
        cals = _load_calibrations()
        vpar = _build_vpar()
        frm, corrected = _generate_targets(cals, cpar)

        corr_list = safely_allocate_adjacency_lists(NUM_CAMS, frm.num_targets)
        match_pairs(corr_list, corrected, frm, vpar, cpar, cals)

        for i1 in range(NUM_CAMS - 1):
            for i2 in range(i1 + 1, NUM_CAMS):
                for part in range(frm.num_targets[i1]):
                    n = corr_list[i1][i2][part].n
                    assert n > 0, (
                        f"pair ({i1},{i2}) target {part}: no candidates found"
                    )

    def test_match_pairs_correct_candidate_identity(self):
        """For each target in cam i1, the true match in cam i2 must appear
        among the candidates (p2 is a sorted-array index, so we look up
        corrected[i2][p2].pnr)."""
        cpar = _build_cpar()
        cals = _load_calibrations()
        vpar = _build_vpar()
        frm, corrected = _generate_targets(cals, cpar)

        corr_list = safely_allocate_adjacency_lists(NUM_CAMS, frm.num_targets)
        match_pairs(corr_list, corrected, frm, vpar, cpar, cals)

        for i1 in range(NUM_CAMS - 1):
            for i2 in range(i1 + 1, NUM_CAMS):
                for part in range(frm.num_targets[i1]):
                    p1 = corr_list[i1][i2][part].p1
                    # The pnr of the source target
                    src_pnr = corrected[i1][p1].pnr

                    # What pnr should we find in i2?
                    # Even cams: same pnr. Odd-parity cams: 15 - pnr.
                    if (i2 - i1) % 2 == 0:
                        expected_pnr = src_pnr
                    else:
                        expected_pnr = 15 - src_pnr

                    # Check all candidates
                    found = False
                    n = corr_list[i1][i2][part].n
                    for c in range(n):
                        p2_idx = int(corr_list[i1][i2][part].p2[c])
                        cand_pnr = corrected[i2][p2_idx].pnr
                        if cand_pnr == expected_pnr:
                            found = True
                            break

                    assert found, (
                        f"pair ({i1},{i2}) part {part}: src_pnr={src_pnr}, "
                        f"expected_pnr={expected_pnr} not in candidates "
                        f"(n={n}, p2s={[int(corr_list[i1][i2][part].p2[c]) for c in range(n)]}, "
                        f"pnrs={[int(corrected[i2][int(corr_list[i1][i2][part].p2[c])].pnr) for c in range(n)]})"
                    )


# ===================================================================
# STAGE 3: four_camera_matching
# ===================================================================

class TestStage3_FourCameraMatching:
    """Verify four-camera matching finds all 16 quadruplets."""

    def test_four_camera_matching_finds_all_correct_matches(self):
        """four_camera_matching must find at least 16 valid quadruplets.

        With multiple candidates per pair, more than 16 preliminary matches
        are expected.  The C test limited the scratch buffer to 16 which
        masked this.  The important thing is that the correct 16 are among
        them — verified by take_best_candidates (tested in Stage 5).
        """
        cpar = _build_cpar()
        cals = _load_calibrations()
        vpar = _build_vpar()
        frm, corrected = _generate_targets(cals, cpar)

        corr_list = safely_allocate_adjacency_lists(NUM_CAMS, frm.num_targets)
        match_pairs(corr_list, corrected, frm, vpar, cpar, cals)

        con0 = np.recarray((4 * NUM_PTS,), dtype=n_tupel_dtype)
        con0.p = 0
        con0.corr = 0.0

        # Use accept_corr=1.0 like the C test
        matched = four_camera_matching(
            corr_list, frm.num_targets[0], 1.0, con0, 4 * NUM_PTS
        )
        assert matched >= NUM_PTS, (
            f"Expected >= {NUM_PTS} quadruplets, got {matched}"
        )


# ===================================================================
# STAGE 4: three_camera_matching
# ===================================================================

class TestStage4_ThreeCameraMatching:
    """Verify three-camera matching with one camera darkened."""

    def test_three_camera_matching_16_triplets(self):
        cpar = _build_cpar()
        cals = _load_calibrations()
        vpar = _build_vpar()
        frm, corrected = _generate_targets(cals, cpar)

        # Darken camera 2 (index 1) — same as C test
        for i in range(frm.num_targets[1]):
            frm.targets[1][i].n = 0
            frm.targets[1][i].nx = 0
            frm.targets[1][i].ny = 0
            frm.targets[1][i].sumg = 0

        corr_list = safely_allocate_adjacency_lists(NUM_CAMS, frm.num_targets)
        match_pairs(corr_list, corrected, frm, vpar, cpar, cals)

        con0 = np.recarray((4 * NUM_PTS,), dtype=n_tupel_dtype)
        con0.p = 0
        con0.corr = 0.0
        tim = safely_allocate_target_usage_marks(NUM_CAMS)

        # High accept_corr as in C test
        matched = three_camera_matching(
            corr_list, NUM_CAMS, frm.num_targets, 100000.0,
            con0, 4 * NUM_PTS, tim,
        )
        assert matched == NUM_PTS, (
            f"Expected {NUM_PTS} triplets, got {matched}"
        )


# ===================================================================
# STAGE 5: Full correspondences()
# ===================================================================

class TestStage5_FullCorrespondences:
    """Full pipeline: must produce exactly 16 quadruplets."""

    def test_correspondences_16_quadruplets(self):
        cpar = _build_cpar()
        cals = _load_calibrations()
        vpar = _build_vpar()
        frm, corrected = _generate_targets(cals, cpar)

        match_counts = [0] * 4
        con = correspondences(frm, corrected, vpar, cpar, cals, match_counts)

        assert match_counts[0] == NUM_PTS, (
            f"Expected {NUM_PTS} quadruplets, got {match_counts[0]} "
            f"(match_counts={match_counts})"
        )
        assert match_counts[1] == 0, f"Unexpected triplets: {match_counts[1]}"
        assert match_counts[2] == 0, f"Unexpected pairs: {match_counts[2]}"
        assert match_counts[3] == NUM_PTS, (
            f"Total should be {NUM_PTS}, got {match_counts[3]}"
        )

    def test_frame_soa_fields_populated(self):
        """correspondences() must populate frm.num_parts, corres_nr, corres_p."""
        cpar = _build_cpar()
        cals = _load_calibrations()
        vpar = _build_vpar()
        frm, corrected = _generate_targets(cals, cpar)

        match_counts = [0] * 4
        correspondences(frm, corrected, vpar, cpar, cals, match_counts)

        # num_parts must equal total matches
        assert frm.num_parts == NUM_PTS, (
            f"frm.num_parts={frm.num_parts}, expected {NUM_PTS}"
        )
        # corres_nr must be set for each particle
        for i in range(NUM_PTS):
            assert frm.corres_nr[i] == i, (
                f"corres_nr[{i}]={frm.corres_nr[i]}, expected {i}"
            )
        # corres_p must hold valid target indices (all >= 0, no CORRES_NONE=-1)
        # since it's a 4-cam test all 4 cameras should have matches
        for i in range(NUM_PTS):
            for j in range(NUM_CAMS):
                assert frm.corres_p[i, j] >= 0, (
                    f"corres_p[{i},{j}]={frm.corres_p[i,j]}: "
                    f"expected valid target index, got CORRES_NONE"
                )
                # Index must be within valid range for this camera
                assert frm.corres_p[i, j] < frm.num_targets[j], (
                    f"corres_p[{i},{j}]={frm.corres_p[i,j]} >= "
                    f"num_targets[{j}]={frm.num_targets[j]}"
                )
        # targets must have their tnr set to the particle index
        # (checked via corres_p back-reference)
        for i in range(NUM_PTS):
            for j in range(NUM_CAMS):
                tix = frm.corres_p[i, j]
                tnr = frm.targets[j][tix].tnr
                assert tnr == i, (
                    f"targets[{j}][{tix}].tnr={tnr}, expected {i}"
                )

    @pytest.mark.skipif(not _optv_available(), reason="optv not installed")
    def test_correspondences_match_cython(self):
        """Python match_counts must equal Cython match_counts."""
        from optv.correspondences import correspondences as optv_corr
        from optv.parameters import VolumeParams as OptvVP

        cpar = _build_cpar()
        cals = _load_calibrations()
        vpar = _build_vpar()
        frm, corrected = _generate_targets(cals, cpar)

        optv_cpar, optv_cals, optv_img_pts, optv_corrected = (
            _optv_generate_targets(cpar)
        )

        # Python
        py_counts = [0] * 4
        correspondences(frm, corrected, vpar, cpar, cals, py_counts)

        # Cython
        optv_vpar = OptvVP()
        optv_vpar.read_volume_par(f"{CORRESP_DIR}/criteria.par")

        _, _, optv_total = optv_corr(
            optv_img_pts, optv_corrected, optv_cals, optv_vpar, optv_cpar,
        )

        assert py_counts[3] == optv_total, (
            f"Python total={py_counts[3]}, Cython total={optv_total}"
        )
        assert py_counts[0] == NUM_PTS, (
            f"Python quadruplets={py_counts[0]}, expected {NUM_PTS}"
        )


# ===================================================================
# STAGE 2b: Verify p2 values are sorted-array indices (not pnr)
# ===================================================================

class TestStage2b_P2Semantics:
    """The p2 field must hold sorted-array indices, not particle numbers.

    In the C code: cand[count].pnr = j  (loop index into sorted crd array).
    If the Python code stores crd[j].pnr instead, cross-referencing in
    four_camera_matching will silently use wrong rows.
    """

    def test_p2_is_sorted_index_not_pnr(self):
        cpar = _build_cpar()
        cals = _load_calibrations()
        vpar = _build_vpar()
        frm, corrected = _generate_targets(cals, cpar)

        corr_list = safely_allocate_adjacency_lists(NUM_CAMS, frm.num_targets)
        match_pairs(corr_list, corrected, frm, vpar, cpar, cals)

        for i1 in range(NUM_CAMS - 1):
            for i2 in range(i1 + 1, NUM_CAMS):
                max_idx = frm.num_targets[i2]
                for part in range(frm.num_targets[i1]):
                    n = corr_list[i1][i2][part].n
                    for c in range(n):
                        p2 = int(corr_list[i1][i2][part].p2[c])
                        assert 0 <= p2 < max_idx, (
                            f"pair ({i1},{i2}) part {part} cand {c}: "
                            f"p2={p2} out of range [0, {max_idx}). "
                            f"p2 may be pnr instead of sorted-array index."
                        )
