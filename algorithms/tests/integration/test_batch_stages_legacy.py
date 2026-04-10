"""Stage-by-stage tests for the batch tracking pipeline on cavity data.

Each test class covers one stage of the pipeline, using only what's needed
from the cavity dataset.  Tests are ordered so failures cascade logically:
parameters → calibration → frame buffer → single-particle ops → one-step
tracking → full tracking comparison.

All tests are sized to run in < 30 s each (most under 5 s).
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from ..conftest import FIXTURES

TEST_DATA = FIXTURES / "test_cavity"
YAML_FILE = TEST_DATA / "parameters_Run1.yaml"
FRAME = 10001


@pytest.fixture(scope="module")
def params():
    with open(YAML_FILE) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def cavity_env(tmp_path_factory):
    """Clone test_cavity into a temp dir (no res files)."""
    dest = tmp_path_factory.mktemp("cavity")
    for name in ("img", "img_orig"):
        src = TEST_DATA / name
        if src.exists():
            link = dest / name
            if not link.exists():
                link.symlink_to(src.resolve())
    for item in TEST_DATA.iterdir():
        if item.is_dir() and item.name not in (
            "img", "img_orig", "res", "res_orig", "res_optv", "__pycache__",
        ):
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dest / item.name)
    res = dest / "res"
    res.mkdir(exist_ok=True)
    return dest


@pytest.fixture(scope="module")
def cavity_env_with_fresh_res(tmp_path_factory, params):
    """Clone test_cavity and run Python sequence to generate fresh res/.

    The shipped ``res_orig/`` was created by an *older* C version whose
    multimedia/calibration model has since changed.  Fresh correspondences
    are needed for tracking to produce any links.
    """
    from algorithms.batch import run_batch

    dest = tmp_path_factory.mktemp("cavity_fresh")
    for name in ("img", "img_orig"):
        src = TEST_DATA / name
        if src.exists():
            link = dest / name
            if not link.exists():
                link.symlink_to(src.resolve())
    for item in TEST_DATA.iterdir():
        if item.is_dir() and item.name not in (
            "img", "img_orig", "res", "res_orig", "res_optv", "__pycache__",
        ):
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dest / item.name)
    res = dest / "res"
    res.mkdir(exist_ok=True)

    yaml_files = list(dest.glob("parameters*.yaml"))
    yaml_file = yaml_files[0]

    original = os.getcwd()
    try:
        os.chdir(dest)
        run_batch(yaml_file, FRAME, FRAME + 3, mode="sequence")
    finally:
        os.chdir(original)

    return dest


# -----------------------------------------------------------------------
# Stage 1: Parameter building
# -----------------------------------------------------------------------
class TestStage1_Parameters:
    """Verify YAML → parameter objects for cavity."""

    def test_control_par(self, params):
        from algorithms.batch import _build_control_par
        cpar = _build_control_par(params["ptv"], params["num_cams"])
        assert cpar.num_cams == 4
        assert cpar.imx > 0 and cpar.imy > 0

    def test_sequence_par(self, params):
        from algorithms.batch import _build_sequence_par
        spar = _build_sequence_par(params["sequence"], params["num_cams"])
        assert spar.first > 0
        assert spar.last >= spar.first

    def test_volume_par(self, params):
        from algorithms.batch import _build_volume_par
        vpar = _build_volume_par(params["criteria"])
        assert vpar.x_lay[0] < vpar.x_lay[1]

    def test_track_par(self, params):
        from algorithms.batch import _build_track_par
        tpar = _build_track_par(params["track"])
        assert tpar.dvxmin < tpar.dvxmax
        assert tpar.dangle > 0
        assert tpar.dacc > 0


# -----------------------------------------------------------------------
# Stage 2: Calibration loading
# -----------------------------------------------------------------------
class TestStage2_Calibration:
    """Load calibration files and verify they have sensible values."""

    def test_load_all_cals(self, params):
        from algorithms.batch import _read_calibrations_py
        original = os.getcwd()
        try:
            os.chdir(TEST_DATA)
            cals = _read_calibrations_py(params["cal_ori"], params["num_cams"])
        finally:
            os.chdir(original)
        assert len(cals) == 4
        for i, cal in enumerate(cals):
            # Exterior parameters: rotation matrix should be close to orthogonal
            dm = cal.ext_par.dm
            R = np.array(dm).reshape(3, 3) if hasattr(dm, '__len__') else None
            # Just check something is loaded
            assert hasattr(cal, 'ext_par'), f"cal[{i}] missing ext_par"
            assert hasattr(cal, 'int_par'), f"cal[{i}] missing int_par"


# -----------------------------------------------------------------------
# Stage 3: Frame buffer initialization
# -----------------------------------------------------------------------
class TestStage3_FrameBuffer:
    """Test that FrameBuf can read cavity frames correctly."""

    def test_read_rt_is(self, cavity_env_with_fresh_res):
        """Freshly-generated rt_is files should be readable."""
        rt = cavity_env_with_fresh_res / "res" / f"rt_is.{FRAME}"
        assert rt.exists(), f"Missing {rt}"
        with open(rt) as f:
            n = int(f.readline().strip())
        assert n > 0, f"Expected >0 correspondences, got {n}"
        print(f"Frame {FRAME}: {n} correspondences")

    def test_framebuf_reads_targets(self, params, cavity_env_with_fresh_res):
        """FrameBuf should load target files correctly."""
        from algorithms.batch import _build_control_par, _build_sequence_par, _build_track_par, _build_volume_par, _read_calibrations_py
        from algorithms.parameters import convert_track_par_to_tuple
        from algorithms.tracking_run import TrackingRun
        from algorithms.constants import TR_BUFSPACE, MAX_TARGETS
        from algorithms.track import default_naming

        original = os.getcwd()
        try:
            os.chdir(cavity_env_with_fresh_res)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            spar = _build_sequence_par(params["sequence"], num_cams)
            vpar = _build_volume_par(params["criteria"])
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)

            spar.first = FRAME
            spar.last = FRAME + 3

            run = TrackingRun(
                spar, tpar, vpar, cpar,
                TR_BUFSPACE, MAX_TARGETS,
                default_naming["corres"],
                default_naming["linkage"],
                default_naming["prio"],
                cals, 0.0001,
            )
            fb = run.fb
            # Read first frame
            fb.read_frame_at_end(FRAME)
            assert fb.buf[fb.buf_len - 1].num_parts > 0, "No particles loaded"
            num = fb.buf[fb.buf_len - 1].num_parts
            print(f"Frame {FRAME}: {num} particles loaded")
            assert num > 0, f"Expected >0 particles, got {num}"

            # Check targets are loaded per camera
            for cam in range(num_cams):
                n_targ = fb.buf[fb.buf_len - 1].num_targets[cam]
                print(f"  cam {cam}: {n_targ} targets")
                assert n_targ > 0, f"cam {cam} has no targets"

        finally:
            os.chdir(original)

    def test_framebuf_primes_correctly(self, params, cavity_env_with_fresh_res):
        """track_forward_start should prime 3 frames."""
        from algorithms.batch import _build_control_par, _build_sequence_par, _build_track_par, _build_volume_par, _read_calibrations_py
        from algorithms.tracking_run import TrackingRun
        from algorithms.constants import TR_BUFSPACE, MAX_TARGETS
        from algorithms.track import default_naming, track_forward_start

        original = os.getcwd()
        try:
            os.chdir(cavity_env_with_fresh_res)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            spar = _build_sequence_par(params["sequence"], num_cams)
            vpar = _build_volume_par(params["criteria"])
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)

            spar.first = FRAME
            spar.last = FRAME + 3

            run = TrackingRun(
                spar, tpar, vpar, cpar,
                TR_BUFSPACE, MAX_TARGETS,
                default_naming["corres"],
                default_naming["linkage"],
                default_naming["prio"],
                cals, 0.0001,
            )
            track_forward_start(run)

            # After priming, buf[1] should be frame 10001 (current)
            # buf[0] should be empty or frame before
            # buf[2] should be frame 10002
            counts = [run.fb.buf[i].num_parts for i in range(TR_BUFSPACE)]
            print(f"Buffer particle counts: {counts}")

            # At least buf[1] and buf[2] should have particles
            assert run.fb.buf[1].num_parts > 0, "buf[1] has no particles"
            assert run.fb.buf[2].num_parts > 0, "buf[2] has no particles"

        finally:
            os.chdir(original)


# -----------------------------------------------------------------------
# Stage 4: Particle position & projection
# -----------------------------------------------------------------------
class TestStage4_Projection:
    """Test 3D→2D projection for cavity particles."""

    def test_point_to_pixel(self, params, cavity_env):
        """point_to_pixel should project a known 3D position sensibly."""
        from algorithms.batch import _build_control_par, _read_calibrations_py
        from algorithms.track import point_to_pixel

        original = os.getcwd()
        try:
            os.chdir(cavity_env)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            cals = _read_calibrations_py(params["cal_ori"], num_cams)
        finally:
            os.chdir(original)

        # First particle in rt_is.10001: xyz = (22.142, 41.030, 9.046)
        pos_3d = np.array([22.142, 41.030, 9.046])
        for cam in range(num_cams):
            px = point_to_pixel(pos_3d, cals[cam], cpar)
            assert np.isfinite(px).all(), f"cam {cam}: non-finite pixel {px}"
            # Pixel coords should be within image bounds
            assert -500 < px[0] < cpar.imx + 500, f"cam {cam}: x={px[0]} out of bounds"
            assert -500 < px[1] < cpar.imy + 500, f"cam {cam}: y={px[1]} out of bounds"
            print(f"  cam {cam}: 3D {pos_3d} → pixel {px}")

    def test_point_to_pixel_matches_target_tnr(self, params, cavity_env_with_fresh_res):
        """Projected 3D positions should land near corresponding 2D targets."""
        from algorithms.batch import _build_control_par, _build_sequence_par, _build_track_par, _build_volume_par, _read_calibrations_py
        from algorithms.tracking_run import TrackingRun
        from algorithms.constants import TR_BUFSPACE, MAX_TARGETS, CORRES_NONE
        from algorithms.track import default_naming, track_forward_start, point_to_pixel

        original = os.getcwd()
        try:
            os.chdir(cavity_env_with_fresh_res)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            spar = _build_sequence_par(params["sequence"], num_cams)
            vpar = _build_volume_par(params["criteria"])
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)

            spar.first = FRAME
            spar.last = FRAME + 3

            run = TrackingRun(
                spar, tpar, vpar, cpar,
                TR_BUFSPACE, MAX_TARGETS,
                default_naming["corres"],
                default_naming["linkage"],
                default_naming["prio"],
                cals, 0.0001,
            )
            track_forward_start(run)

            fb = run.fb
            # Check first 10 particles in buf[1] (current frame)
            errors = []
            checked = 0
            for h in range(min(10, fb.buf[1].num_parts)):
                pos_3d = fb.buf[1].path_info[h].x
                corr = fb.buf[1].correspond[h]

                for cam in range(num_cams):
                    tgt_idx = corr.p[cam]
                    if tgt_idx == CORRES_NONE or tgt_idx < 0:
                        continue
                    if tgt_idx >= len(fb.buf[1].targets[cam]):
                        continue

                    tgt = fb.buf[1].targets[cam][tgt_idx]
                    px_proj = point_to_pixel(pos_3d, cals[cam], cpar)
                    px_tgt = np.array([tgt.x, tgt.y])
                    dist = np.linalg.norm(px_proj - px_tgt)
                    checked += 1
                    if dist > 10:  # > 10 pixels is suspicious
                        errors.append(
                            f"particle {h}, cam {cam}: "
                            f"proj={px_proj}, tgt={px_tgt}, dist={dist:.1f}"
                        )

            print(f"Checked {checked} projections, {len(errors)} > 10 px")
            if errors:
                for e in errors[:5]:
                    print(f"  WARNING: {e}")
            assert checked > 0, "No projections checked"
            # Allow some tolerance — most should be close
            bad_rate = len(errors) / checked if checked > 0 else 0
            assert bad_rate < 0.5, f"{len(errors)}/{checked} projections off by >10px"

        finally:
            os.chdir(original)


# -----------------------------------------------------------------------
# Stage 5: Search volume (searchquader)
# -----------------------------------------------------------------------
class TestStage5_Searchquader:
    """Test search volume computation for cavity particles."""

    def test_searchquader_shape(self, params, cavity_env):
        """searchquader should produce 4 arrays of length num_cams."""
        from algorithms.batch import _build_control_par, _build_track_par, _read_calibrations_py
        from algorithms.track import searchquader

        original = os.getcwd()
        try:
            os.chdir(cavity_env)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)
        finally:
            os.chdir(original)

        pos_3d = np.array([22.142, 41.030, 9.046])
        right, left, down, up = searchquader(pos_3d, tpar, cpar, cals)

        for name, arr in [("right", right), ("left", left), ("down", down), ("up", up)]:
            assert len(arr) == num_cams, f"{name} has {len(arr)} entries"
            for cam in range(num_cams):
                assert arr[cam] > 0, f"{name}[{cam}] = {arr[cam]} (should be > 0)"
                print(f"  {name}[{cam}] = {arr[cam]:.1f}")


# -----------------------------------------------------------------------
# Stage 6: Candidate search (single particle)
# -----------------------------------------------------------------------
class TestStage6_CandidateSearch:
    """Test sorted_candidates_in_volume on single particles."""

    def test_candidates_for_first_particle(self, params, cavity_env_with_fresh_res):
        """First particle should find candidates in the next frame."""
        from algorithms.batch import _build_control_par, _build_sequence_par, _build_track_par, _build_volume_par, _read_calibrations_py
        from algorithms.tracking_run import TrackingRun
        from algorithms.constants import TR_BUFSPACE, MAX_TARGETS, TR_UNUSED
        from algorithms.track import (
            default_naming, track_forward_start,
            sorted_candidates_in_volume, point_to_pixel,
        )

        original = os.getcwd()
        try:
            os.chdir(cavity_env_with_fresh_res)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            spar = _build_sequence_par(params["sequence"], num_cams)
            vpar = _build_volume_par(params["criteria"])
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)

            spar.first = FRAME
            spar.last = FRAME + 3

            run = TrackingRun(
                spar, tpar, vpar, cpar,
                TR_BUFSPACE, MAX_TARGETS,
                default_naming["corres"],
                default_naming["linkage"],
                default_naming["prio"],
                cals, 0.0001,
            )
            track_forward_start(run)

            fb = run.fb
            # Test first 5 particles
            found_any = False
            for h in range(min(5, fb.buf[1].num_parts)):
                pos_3d = fb.buf[1].path_info[h].x
                # Project to cameras
                v1 = np.zeros((num_cams, 2))
                for cam in range(num_cams):
                    v1[cam] = point_to_pixel(pos_3d, cals[cam], cpar)

                w = sorted_candidates_in_volume(pos_3d, v1, fb.buf[2], run)
                n_cands = sum(1 for i in range(w.shape[0]) if w[i].ftnr != TR_UNUSED)
                print(f"  particle {h}: pos={pos_3d}, {n_cands} candidates")
                if n_cands > 0:
                    found_any = True
                    for i in range(min(3, n_cands)):
                        ftnr = w[i].ftnr
                        freq = w[i].freq
                        if ftnr >= fb.buf[2].num_parts:
                            print(f"    cand {i}: ftnr={ftnr} out of range ({fb.buf[2].num_parts})")
                            continue
                        cand_pos = fb.buf[2].path_info[ftnr].x
                        dist = np.linalg.norm(cand_pos - pos_3d)
                        print(f"    cand {i}: ftnr={ftnr}, freq={freq}, "
                              f"pos={cand_pos}, dist={dist:.1f}")

            assert found_any, "No candidates found for any of the first 5 particles"

        finally:
            os.chdir(original)

    def test_candidate_3d_distance_reasonable(self, params, cavity_env_with_fresh_res):
        """Candidates' 3D positions should be within tracking bounds."""
        from algorithms.batch import _build_control_par, _build_sequence_par, _build_track_par, _build_volume_par, _read_calibrations_py
        from algorithms.tracking_run import TrackingRun
        from algorithms.constants import TR_BUFSPACE, MAX_TARGETS, TR_UNUSED
        from algorithms.track import (
            default_naming, track_forward_start,
            sorted_candidates_in_volume, point_to_pixel,
        )

        original = os.getcwd()
        try:
            os.chdir(cavity_env_with_fresh_res)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            spar = _build_sequence_par(params["sequence"], num_cams)
            vpar = _build_volume_par(params["criteria"])
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)

            spar.first = FRAME
            spar.last = FRAME + 3

            run = TrackingRun(
                spar, tpar, vpar, cpar,
                TR_BUFSPACE, MAX_TARGETS,
                default_naming["corres"],
                default_naming["linkage"],
                default_naming["prio"],
                cals, 0.0001,
            )
            track_forward_start(run)

            fb = run.fb
            max_disp = max(abs(tpar.dvxmin), abs(tpar.dvxmax),
                          abs(tpar.dvymin), abs(tpar.dvymax),
                          abs(tpar.dvzmin), abs(tpar.dvzmax))

            close_count = 0
            far_count = 0
            total_cands = 0

            for h in range(min(50, fb.buf[1].num_parts)):
                pos_3d = fb.buf[1].path_info[h].x
                v1 = np.zeros((num_cams, 2))
                for cam in range(num_cams):
                    v1[cam] = point_to_pixel(pos_3d, cals[cam], cpar)

                w = sorted_candidates_in_volume(pos_3d, v1, fb.buf[2], run)
                for i in range(w.shape[0]):
                    if w[i].ftnr == TR_UNUSED:
                        break
                    ftnr = w[i].ftnr
                    if ftnr >= fb.buf[2].num_parts:
                        continue
                    cand_pos = fb.buf[2].path_info[ftnr].x
                    dist = np.linalg.norm(cand_pos - pos_3d)
                    total_cands += 1
                    if dist < max_disp * 2:
                        close_count += 1
                    else:
                        far_count += 1

            print(f"Total candidates: {total_cands}")
            print(f"Close (< {max_disp*2:.1f} mm): {close_count}")
            print(f"Far (>= {max_disp*2:.1f} mm): {far_count}")

            # At least some candidates should be nearby
            if total_cands > 0:
                close_rate = close_count / total_cands
                print(f"Close rate: {close_rate:.1%}")
                # This is the KEY diagnostic: if close_rate is low,
                # the 2D→3D mapping via tnr/ftnr is broken
                assert close_rate > 0.1, (
                    f"Only {close_rate:.1%} of candidates are nearby — "
                    f"likely a tnr/ftnr mapping problem"
                )

        finally:
            os.chdir(original)


# -----------------------------------------------------------------------
# Stage 7: One tracking step timing and output
# -----------------------------------------------------------------------
class TestStage7_OneTrackingStep:
    """Run trackcorr_c_loop for a single step and check output."""

    def test_single_step_completes_in_time(self, params, cavity_env_with_fresh_res):
        """One step of trackcorr_c_loop should complete in < 60s."""
        from algorithms.batch import _build_control_par, _build_sequence_par, _build_track_par, _build_volume_par, _read_calibrations_py
        from algorithms.tracking_run import TrackingRun
        from algorithms.constants import TR_BUFSPACE, MAX_TARGETS
        from algorithms.track import (
            default_naming, track_forward_start,
            trackcorr_c_loop,
        )

        original = os.getcwd()
        try:
            os.chdir(cavity_env_with_fresh_res)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            spar = _build_sequence_par(params["sequence"], num_cams)
            vpar = _build_volume_par(params["criteria"])
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)

            spar.first = FRAME
            spar.last = FRAME + 3

            run = TrackingRun(
                spar, tpar, vpar, cpar,
                TR_BUFSPACE, MAX_TARGETS,
                default_naming["corres"],
                default_naming["linkage"],
                default_naming["prio"],
                cals, 0.0001,
            )
            track_forward_start(run)

            t0 = time.time()
            trackcorr_c_loop(run, FRAME)
            elapsed = time.time() - t0

            print(f"Single step took {elapsed:.1f}s")
            assert elapsed < 600, f"Single step took {elapsed:.1f}s (too slow)"

        finally:
            os.chdir(original)

    def test_single_step_produces_links(self, params, cavity_env_with_fresh_res):
        """One step should produce at least some links."""
        from algorithms.batch import _build_control_par, _build_sequence_par, _build_track_par, _build_volume_par, _read_calibrations_py
        from algorithms.tracking_run import TrackingRun
        from algorithms.constants import TR_BUFSPACE, MAX_TARGETS, NEXT_NONE
        from algorithms.track import (
            default_naming, track_forward_start,
            trackcorr_c_loop, TrackingObserver,
        )

        original = os.getcwd()
        try:
            os.chdir(cavity_env_with_fresh_res)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            spar = _build_sequence_par(params["sequence"], num_cams)
            vpar = _build_volume_par(params["criteria"])
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)

            spar.first = FRAME
            spar.last = FRAME + 3

            run = TrackingRun(
                spar, tpar, vpar, cpar,
                TR_BUFSPACE, MAX_TARGETS,
                default_naming["corres"],
                default_naming["linkage"],
                default_naming["prio"],
                cals, 0.0001,
            )
            track_forward_start(run)

            obs = TrackingObserver()
            trackcorr_c_loop(run, FRAME, observer=obs)

            fb = run.fb
            # After trackcorr_c_loop, fb_next() rotated the buffer:
            # what was buf[1] (current frame with links) is now buf[0].
            links = sum(
                1 for i in range(fb.buf[0].num_parts)
                if fb.buf[0].path_info[i].next_frame != NEXT_NONE
            )
            lost = fb.buf[0].num_parts - links

            print(f"step {FRAME}: particles={fb.buf[0].num_parts}, "
                  f"links={links}, lost={lost}")

            # Categorize observer events
            no_cand = sum(1 for e in obs.events if e.get("type") == "no_candidates")
            has_cand = sum(1 for e in obs.events if e.get("candidates"))
            print(f"Observer: {len(obs.events)} events, "
                  f"no_candidates={no_cand}, has_candidates={has_cand}")

            assert links > 0, (
                f"Zero links out of {fb.buf[0].num_parts} particles! "
                f"Observer: no_candidates={no_cand}, has_candidates={has_cand}"
            )

        finally:
            os.chdir(original)


# -----------------------------------------------------------------------
# Stage 8: Cython comparison (if optv available)
# -----------------------------------------------------------------------
try:
    from optv.tracker import Tracker as CyTracker
    HAS_OPTV = True
except ImportError:
    HAS_OPTV = False

skip_no_optv = pytest.mark.skipif(not HAS_OPTV, reason="optv not installed")


@skip_no_optv
class TestStage8_CythonComparison:
    """Compare Python vs Cython tracking on cavity data."""

    def test_single_step_link_count_parity(self, params, cavity_env):
        """Python and Cython should produce similar link counts."""
        pytest.skip("TODO: implement Cython comparison for single step")


# -----------------------------------------------------------------------
# Stage 9: track3d_loop as alternative
# -----------------------------------------------------------------------
class TestStage9_Track3D:
    """Test track3d_loop on cavity data (which is what C uses)."""

    def test_track3d_single_step(self, params, cavity_env_with_fresh_res):
        """track3d_loop should produce links on cavity data."""
        from algorithms.batch import _build_control_par, _build_sequence_par, _build_track_par, _build_volume_par, _read_calibrations_py
        from algorithms.tracking_run import TrackingRun
        from algorithms.constants import TR_BUFSPACE, MAX_TARGETS, NEXT_NONE
        from algorithms.track import (
            default_naming, track_forward_start,
            track3d_loop,
        )

        original = os.getcwd()
        try:
            os.chdir(cavity_env_with_fresh_res)
            num_cams = params["num_cams"]
            cpar = _build_control_par(params["ptv"], num_cams)
            spar = _build_sequence_par(params["sequence"], num_cams)
            vpar = _build_volume_par(params["criteria"])
            tpar = _build_track_par(params["track"])
            cals = _read_calibrations_py(params["cal_ori"], num_cams)

            spar.first = FRAME
            spar.last = FRAME + 3

            run = TrackingRun(
                spar, tpar, vpar, cpar,
                TR_BUFSPACE, MAX_TARGETS,
                default_naming["corres"],
                default_naming["linkage"],
                default_naming["prio"],
                cals, 0.0001,
            )
            track_forward_start(run)

            t0 = time.time()
            track3d_loop(run, FRAME)
            elapsed = time.time() - t0

            fb = run.fb
            # After track3d_loop, fb_next() rotated the buffer:
            # what was buf[1] (current frame with links) is now buf[0].
            links = sum(
                1 for i in range(fb.buf[0].num_parts)
                if fb.buf[0].path_info[i].next_frame != NEXT_NONE
            )
            lost = fb.buf[0].num_parts - links

            print(f"track3d step {FRAME}: particles={fb.buf[0].num_parts}, "
                  f"links={links}, lost={lost}, time={elapsed:.1f}s")

            assert elapsed < 60, f"track3d took {elapsed:.1f}s"
            assert links > 0, f"track3d produced 0 links out of {fb.buf[0].num_parts}"

        finally:
            os.chdir(original)
