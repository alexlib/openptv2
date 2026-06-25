"""
Full batch pipeline test using pure Python algorithms.

Mirrors gui/tests/test_pyptv_batch.py but uses algorithms/ directly,
with all three bug fixes applied and detailed logging at every step.

Pipeline:
  1. Copy img_orig/ images → img/
  2. Detection → write _targets → compare to img_orig/_targets (reference)
  3. Correspondences → write rt_is → compare to res_orig/rt_is (reference)
  4. Tracking → write ptv_is → log results

Usage:
    uv run pytest algorithms/tests/test_batch_python_pipeline.py -v -s
"""

import math
import shutil
import numpy as np
import yaml
from pathlib import Path

import pytest

from algorithms.parameters import (
    ControlPar, VolumePar, TargetPar, SequencePar,
    TrackPar, MmNp, convert_track_par_to_tuple,
)
from algorithms.calibration import Calibration
from algorithms.segmentation import targ_rec
from algorithms.image_processing import prepare_image
from algorithms.correspondences import (
    correspondences as algo_correspondences, correct_frame,
)
from algorithms.epi import Coord2d
from algorithms.tracking_frame_buf import (
    Frame, Target, read_targets, write_targets,
)
from algorithms.orientation import point_positions as algo_point_positions
from algorithms.track import (
    track_forward_start, trackcorr_c_loop, trackcorr_c_finish,
    trackback_c, angle_acc,
)
from algorithms.tracking_run import TrackingRun


CAVITY_DIR = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
YAML_PATH = CAVITY_DIR / "parameters_Run1.yaml"


# ---------------------------------------------------------------------------
# Parameter builders (YAML → algorithms dataclasses, no compat layer)
# ---------------------------------------------------------------------------

def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _build_cpar(params):
    ptv = params["ptv"]
    num_cams = params.get("num_cams", 4)
    mm = MmNp(
        nlay=1,
        n1=ptv.get("mmp_n1", 1.0),
        n2=[ptv.get("mmp_n2", 1.0)],
        d=[ptv.get("mmp_d", 0.0)],
        n3=ptv.get("mmp_n3", 1.0),
    )
    return ControlPar(
        num_cams=num_cams,
        imx=ptv["imx"], imy=ptv["imy"],
        pix_x=ptv["pix_x"], pix_y=ptv["pix_y"],
        hp_flag=1 if ptv.get("hp_flag", True) else 0,
        all_cam_flag=1 if ptv.get("allcam_flag", False) else 0,
        tiff_flag=1 if ptv.get("tiff_flag", True) else 0,
        chfield=ptv.get("chfield", 0),
        mm=mm,
    )


def _build_vpar(params):
    crit = params.get("criteria", {})
    return VolumePar(
        X_lay=crit.get("X_lay", [-100, 100]),
        Zmin_lay=crit.get("Zmin_lay", [-100, -100]),
        Zmax_lay=crit.get("Zmax_lay", [100, 100]),
        cn=crit.get("cn", 0), cnx=crit.get("cnx", 0),
        cny=crit.get("cny", 0), csumg=crit.get("csumg", 0),
        eps0=crit.get("eps0", 0), corrmin=crit.get("corrmin", 0),
    )


def _build_tpar(params):
    tr = params.get("targ_rec", {})
    gvthres = tr.get("gvthres", None)
    if gvthres is None:
        dp = params.get("detect_plate", {})
        gvthres = [dp.get(f"gvth_{i+1}", 40) for i in range(4)]
    return TargetPar(
        gvthres=gvthres,
        discont=tr.get("disco", tr.get("discont", 5)),
        nnmin=tr.get("nnmin", 1), nnmax=tr.get("nnmax", 500),
        nxmin=tr.get("nxmin", 1), nxmax=tr.get("nxmax", 50),
        nymin=tr.get("nymin", 1), nymax=tr.get("nymax", 50),
        sumg_min=tr.get("sumg_min", 10),
        cr_sz=tr.get("cr_sz", 3),
    )


def _build_track_par(params):
    tk = params.get("track", {})
    return TrackPar(
        dvxmin=tk.get("dvxmin", -10), dvxmax=tk.get("dvxmax", 10),
        dvymin=tk.get("dvymin", -10), dvymax=tk.get("dvymax", 10),
        dvzmin=tk.get("dvzmin", -10), dvzmax=tk.get("dvzmax", 10),
        dangle=tk.get("angle", tk.get("dangle", 100)),
        dacc=tk.get("dacc", 2.0),
        add=1 if tk.get("flagNewParticles", False) else 0,
    )


def _build_spar(params, exp_dir):
    seq = params.get("sequence", {})
    num_cams = params.get("num_cams", 4)
    base = seq.get("base_name", [f"img/cam{i+1}.%d" for i in range(num_cams)])
    img_base = [str(exp_dir / b) for b in base]
    return SequencePar(
        num_cams=num_cams, img_base_name=img_base,
        first=seq.get("first", 10001), last=seq.get("last", 10004),
    )


def _load_calibrations(params, exp_dir):
    cal_ori = params.get("cal_ori", {})
    cals = []
    for ori in cal_ori.get("img_ori", []):
        ori_path = str(exp_dir / ori)
        add_path = ori_path.replace(".ori", ".addpar")
        cals.append(Calibration.from_file(ori_path, add_path))
    return cals


# ---------------------------------------------------------------------------
# Detection with bug fix #2: reassign pnr after y-sort
# ---------------------------------------------------------------------------

HIGHPASS_DIM = 25


def _detect_targets(img, tpar, cam_idx, cpar):
    hp = prepare_image(
        img, dim_lp=HIGHPASS_DIM, imx=cpar.imx, imy=cpar.imy,
        filter_hp=0,
    )
    targets = targ_rec(
        img=hp,
        gvthres=int(tpar.gvthres[cam_idx]),
        discont=tpar.discont,
        nnmin=tpar.nnmin, nnmax=tpar.nnmax,
        nxmin=tpar.nxmin, nxmax=tpar.nxmax,
        nymin=tpar.nymin, nymax=tpar.nymax,
        sumg_min=tpar.sumg_min,
        xmin=1, xmax=cpar.imx - 1,
        ymin=1, ymax=cpar.imy - 1,
    )
    # BUG FIX #2: sort by y, then reassign pnr = array position
    targets.sort(key=lambda t: t.y)
    for j, t in enumerate(targets):
        t.pnr = j
    return targets


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cavity_work(tmp_path):
    """Set up working copy of cavity dataset with images from img_orig/."""
    if not YAML_PATH.exists():
        pytest.skip("Cavity test data not available")

    img_orig = CAVITY_DIR / "img_orig"
    if not img_orig.exists():
        pytest.skip("img_orig/ not found — original images required")

    work = tmp_path / "cavity"
    work.mkdir()

    # Copy everything EXCEPT img/ and res/ (we'll create those fresh)
    for item in CAVITY_DIR.iterdir():
        if item.name in ("img", "res"):
            continue
        if item.name.startswith("tmp") and item.suffix == ".yaml":
            continue
        if item.is_dir():
            shutil.copytree(item, work / item.name)
        else:
            shutil.copy2(item, work / item.name)

    # Copy only images (not _targets) from img_orig/ → img/
    img_dir = work / "img"
    img_dir.mkdir()
    for f in img_orig.iterdir():
        if "_targets" not in f.name:
            shutil.copy2(f, img_dir / f.name)

    # Create empty res/
    (work / "res").mkdir()

    params = _load_yaml(work / "parameters_Run1.yaml")
    cpar = _build_cpar(params)
    vpar = _build_vpar(params)
    tpar = _build_tpar(params)
    track_par = _build_track_par(params)
    spar = _build_spar(params, work)
    cals = _load_calibrations(params, work)
    num_cams = params.get("num_cams", 4)

    seq = params.get("sequence", {})
    base_names = seq.get("base_name", [f"img/cam{i+1}.%d" for i in range(num_cams)])

    return {
        "params": params, "cpar": cpar, "vpar": vpar, "tpar": tpar,
        "track_par": track_par, "spar": spar, "cals": cals,
        "work_dir": work, "num_cams": num_cams,
        "base_names": base_names,
        "img_orig": img_orig, "res_orig": CAVITY_DIR / "res_orig",
    }


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_batch_pipeline(cavity_work, monkeypatch):
    """Full pipeline: detect → compare _targets → correspond → compare rt_is → track."""
    import os
    from skimage.io import imread
    from skimage.util import img_as_ubyte
    from skimage.color import rgb2gray

    cfg = cavity_work
    cpar, vpar, tpar = cfg["cpar"], cfg["vpar"], cfg["tpar"]
    track_par, spar, cals = cfg["track_par"], cfg["spar"], cfg["cals"]
    num_cams = cfg["num_cams"]
    work_dir = cfg["work_dir"]
    base_names = cfg["base_names"]
    img_orig = cfg["img_orig"]
    res_orig = cfg["res_orig"]
    params = cfg["params"]

    monkeypatch.chdir(work_dir)

    # =======================================================================
    # LOG: Parameters
    # =======================================================================
    print("\n" + "=" * 70)
    print("PARAMETERS")
    print("=" * 70)
    print(f"  num_cams    = {num_cams}")
    print(f"  image_size  = {cpar.imx} x {cpar.imy}")
    print(f"  pixel_size  = {cpar.pix_x} x {cpar.pix_y}")
    print(f"  hp_flag     = {cpar.hp_flag}")
    print(f"  multimedia  = n1={cpar.mm.n1}, n2={cpar.mm.n2}, "
          f"n3={cpar.mm.n3}, d={cpar.mm.d}")
    print(f"  velocity    = x[{track_par.dvxmin}, {track_par.dvxmax}] "
          f"y[{track_par.dvymin}, {track_par.dvymax}] "
          f"z[{track_par.dvzmin}, {track_par.dvzmax}]")
    print(f"  dacc={track_par.dacc}  dangle={track_par.dangle}  add={track_par.add}")
    print(f"  volume      = X_lay={vpar.X_lay} Zmin={vpar.Zmin_lay} "
          f"Zmax={vpar.Zmax_lay}")
    print(f"  corrmin     = {vpar.corrmin}")
    print(f"  target      = gvthres={tpar.gvthres} discont={tpar.discont}")
    print(f"  frames      = {spar.first} to {spar.last}")
    for i, c in enumerate(cals):
        print(f"  cal[{i}] pos  = ({c.ext_par.x0:.1f}, {c.ext_par.y0:.1f}, "
              f"{c.ext_par.z0:.1f})")

    # =======================================================================
    # STAGE 1: DETECTION — detect targets, write _targets, compare to ref
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: TARGET DETECTION")
    print("=" * 70)

    all_detections = {}
    detection_mismatches = []

    for frame in range(spar.first, spar.last + 1):
        frame_dets = []
        for i_cam in range(num_cams):
            img_path = work_dir / (base_names[i_cam] % frame)
            if not img_path.exists():
                pytest.skip(f"Image not found: {img_path}")

            img = imread(str(img_path))
            if img.ndim > 2:
                img = rgb2gray(img)
            if img.dtype != np.uint8:
                img = img_as_ubyte(img)

            targets = _detect_targets(img, tpar, i_cam, cpar)
            frame_dets.append(targets)

        counts = [len(d) for d in frame_dets]
        print(f"  Frame {frame}: detected {counts}, total={sum(counts)}")
        all_detections[frame] = frame_dets

        # Compare to reference _targets in img_orig/
        for i_cam in range(num_cams):
            cam_name = f"cam{i_cam + 1}"
            ref_targets_path = img_orig / f"{cam_name}.{frame}_targets"
            if not ref_targets_path.exists():
                print(f"    cam{i_cam}: no reference _targets to compare")
                continue

            ref_tgts = read_targets(str(img_orig / f"{cam_name}."), frame)
            n_det = len(frame_dets[i_cam])
            n_ref = len(ref_tgts)

            pct_diff = 100 * abs(n_det - n_ref) / max(n_ref, 1)
            # Match targets by position (both sorted by y)
            matched_pos = 0
            if n_det > 0 and n_ref > 0:
                det_xy = np.array([[t.x, t.y] for t in frame_dets[i_cam]])
                ref_xy = np.array([[t.x, t.y] for t in ref_tgts])
                for ri in range(n_ref):
                    d = np.sqrt((det_xy[:, 0] - ref_xy[ri, 0])**2 +
                                (det_xy[:, 1] - ref_xy[ri, 1])**2)
                    if d.min() < 2.0:
                        matched_pos += 1

            if pct_diff < 5:
                print(f"    cam{i_cam}: CLOSE det={n_det} ref={n_ref} "
                      f"({pct_diff:.1f}% diff), "
                      f"pos_match={matched_pos}/{n_ref}")
            else:
                print(f"    cam{i_cam}: MISMATCH det={n_det} ref={n_ref} "
                      f"({pct_diff:.1f}% diff), "
                      f"pos_match={matched_pos}/{n_ref}")
                detection_mismatches.append(
                    (frame, i_cam, n_det, n_ref, pct_diff, matched_pos))

    if detection_mismatches:
        print(f"\n  Detection mismatches (>5%): {len(detection_mismatches)}")
        for m in detection_mismatches:
            print(f"    frame={m[0]} cam={m[1]}: det={m[2]} ref={m[3]} "
                  f"({m[4]:.1f}%) pos_match={m[5]}")

    # =======================================================================
    # STAGE 2: CORRESPONDENCES — build Frame, correct, match, write rt_is
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: STEREO CORRESPONDENCES")
    print("=" * 70)

    all_positions = {}
    all_corresp_tgt = {}
    rt_is_mismatches = []

    for frame in range(spar.first, spar.last + 1):
        dets = all_detections[frame]

        max_tgt = max(len(d) for d in dets) + 10
        frm = Frame(num_cams=num_cams, max_targets=max_tgt)
        for i_cam in range(num_cams):
            frm.num_targets[i_cam] = len(dets[i_cam])
            for j, t in enumerate(dets[i_cam]):
                frm.targets[i_cam][j] = t

        corr = correct_frame(frm, cals, cpar, tol=1e-5)

        con, match_counts = algo_correspondences(frm, corr, vpar, cpar, cals)
        n_quads, n_trips, n_pairs, n_total = match_counts
        print(f"  Frame {frame}: {n_total} correspondences "
              f"(quads={n_quads}, triplets={n_trips}, pairs={n_pairs})")

        if n_total > 0:
            valid = con[:n_total]
            corresp_corr = np.array([nt.p[:num_cams] for nt in valid]).T

            # BUG FIX #3: store target pnr, not corrected-list indices
            corresp_tgt = np.array([
                [corr[cam][nt.p[cam]].pnr if nt.p[cam] >= 0 else -1
                 for cam in range(num_cams)]
                for nt in valid
            ]).T
        else:
            corresp_corr = np.zeros((num_cams, 0), dtype=int)
            corresp_tgt = np.zeros((num_cams, 0), dtype=int)

        # 3D point positions
        if corresp_corr.shape[1] > 0:
            flat_arr = np.array([
                [[corr[cam][int(corresp_corr[cam, i])].x,
                  corr[cam][int(corresp_corr[cam, i])].y]
                 if corresp_corr[cam, i] >= 0 else [-999.0, -999.0]
                 for i in range(corresp_corr.shape[1])]
                for cam in range(num_cams)
            ])
            flat_arr = flat_arr.transpose(1, 0, 2)  # (N, num_cams, 2)
            pos, rcm = algo_point_positions(flat_arr, cpar, cals, vpar)
        else:
            pos = np.zeros((0, 3))

        all_positions[frame] = pos
        all_corresp_tgt[frame] = corresp_tgt

        # Write rt_is
        rt_is_path = work_dir / "res" / f"rt_is.{frame}"
        if num_cams < 4:
            print_corresp = -1 * np.ones((4, corresp_tgt.shape[1]), dtype=int)
            print_corresp[:num_cams, :] = corresp_tgt
        else:
            print_corresp = corresp_tgt
        with open(rt_is_path, "w") as f:
            f.write(f"{pos.shape[0]}\n")
            for pix, pt in enumerate(pos):
                pt_args = (pix + 1,) + tuple(pt) + tuple(print_corresp[:, pix])
                f.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)

        # Write _targets AFTER correspondences (tnr is now set)
        for i_cam in range(num_cams):
            file_base = base_names[i_cam].replace("%d", "")
            n_tgt = frm.num_targets[i_cam]
            tgt_list = [frm.targets[i_cam][j] for j in range(n_tgt)]
            write_targets(tgt_list, n_tgt, str(work_dir / file_base), frame)

        # ---------------------------------------------------------------
        # Compare rt_is to res_orig/
        # ---------------------------------------------------------------
        ref_rt_is = res_orig / f"rt_is.{frame}"
        if ref_rt_is.exists():
            with open(ref_rt_is) as f:
                ref_lines = f.readlines()
            n_ref = int(ref_lines[0].strip())
            ref_pos = np.array([
                [float(x) for x in ref_lines[i + 1].split()[1:4]]
                for i in range(n_ref)
            ])

            print(f"    rt_is compare: python={pos.shape[0]} vs ref={n_ref} "
                  f"(delta={pos.shape[0] - n_ref})")

            if pos.shape[0] > 0 and n_ref > 0:
                # For each ref point, find nearest python point
                match_1mm = match_5mm = 0
                dists = []
                for i in range(n_ref):
                    d = np.linalg.norm(pos - ref_pos[i], axis=1)
                    min_d = d.min()
                    dists.append(min_d)
                    if min_d < 1.0:
                        match_1mm += 1
                    if min_d < 5.0:
                        match_5mm += 1

                dists = np.array(dists)
                print(f"    ref→python match: {match_1mm}/{n_ref} <1mm, "
                      f"{match_5mm}/{n_ref} <5mm, "
                      f"median={np.median(dists):.2f}mm")
                if match_5mm < n_ref * 0.5:
                    rt_is_mismatches.append(
                        (frame, pos.shape[0], n_ref, match_5mm, np.median(dists)))
        else:
            print(f"    no reference rt_is for frame {frame}")

        # Log tnr assignment
        for i_cam in range(num_cams):
            n_tgt = frm.num_targets[i_cam]
            tnr_set = sum(1 for j in range(n_tgt)
                         if frm.targets[i_cam][j].tnr >= 0)
            print(f"    cam{i_cam}: {tnr_set}/{n_tgt} targets mapped to 3D")

    if rt_is_mismatches:
        print(f"\n  rt_is mismatches: {len(rt_is_mismatches)}")
        for m in rt_is_mismatches:
            print(f"    frame={m[0]}: python={m[1]} ref={m[2]} "
                  f"matched={m[3]} median_dist={m[4]:.3f}")

    # =======================================================================
    # STAGE 3: TRACKING — forward + backward
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: TRACKING")
    print("=" * 70)

    tpar_tuple = convert_track_par_to_tuple(track_par)
    run = TrackingRun(
        seq_par=spar,
        tpar=tpar_tuple,
        vpar=vpar,
        cpar=cpar,
        buf_len=4,
        max_targets=10000,
        corres_file_base=str(work_dir / "res" / "rt_is"),
        linkage_file_base=str(work_dir / "res" / "ptv_is"),
        prio_file_base=str(work_dir / "res" / "added"),
        cal=cals,
        flatten_tol=0.0001,
    )
    print(f"  TrackingRun: lmax={run.lmax:.2f}")

    track_forward_start(run)
    print(f"  track_forward_start: loaded frames into buffer")

    # Projection sanity check
    from algorithms.track import point_to_pixel
    fb = run.fb
    curr_slot = fb.buf[1]
    offsets = []
    n_check = min(10, curr_slot.num_parts)
    for h in range(n_check):
        pos_3d = curr_slot.path_x[h]
        for cam in range(num_cams):
            px, py = point_to_pixel(pos_3d, cals[cam], cpar)
            for j in range(curr_slot.num_targets[cam]):
                if curr_slot.targets[cam][j].tnr == h:
                    t = curr_slot.targets[cam][j]
                    offsets.append(math.sqrt((px - t.x)**2 + (py - t.y)**2))
                    break
    if offsets:
        print(f"  projection check: median={np.median(offsets):.1f}px, "
              f"max={max(offsets):.1f}px (n={len(offsets)})")
        assert np.median(offsets) < 50, (
            f"Projection mismatch too large: median={np.median(offsets):.1f}px "
            f"(expect <50px). Likely index mapping bug.")

    # Forward tracking
    for step in range(spar.first, spar.last + 1):
        trackcorr_c_loop(run, step)
        print(f"  step {step}: npart={run.npart}, nlinks={run.nlinks}")

    trackcorr_c_finish(run, spar.last)
    print(f"  forward done: npart={run.npart}, nlinks={run.nlinks}")

    # Backward tracking
    trackback_c(run)
    print(f"  backward done: npart={run.npart}, nlinks={run.nlinks}")

    # =======================================================================
    # STAGE 4: ANALYZE ptv_is — linking statistics
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: TRACKING RESULTS")
    print("=" * 70)

    total_particles = 0
    total_fwd = 0
    total_bwd = 0

    for frame in range(spar.first, spar.last + 1):
        ptv_path = work_dir / "res" / f"ptv_is.{frame}"
        if not ptv_path.exists():
            print(f"  Frame {frame}: no ptv_is file")
            continue

        with open(ptv_path) as f:
            lines = f.readlines()
        n = int(lines[0].strip())
        total_particles += n

        linked_fwd = linked_bwd = isolated = 0
        for i in range(1, n + 1):
            parts = lines[i].split()
            prev_idx, next_idx = int(parts[0]), int(parts[1])
            if next_idx >= 0:
                linked_fwd += 1
            if prev_idx >= 0:
                linked_bwd += 1
            if prev_idx < 0 and next_idx < 0:
                isolated += 1

        total_fwd += linked_fwd
        total_bwd += linked_bwd

        pct_fwd = 100 * linked_fwd / n if n > 0 else 0
        pct_bwd = 100 * linked_bwd / n if n > 0 else 0
        print(f"  Frame {frame}: {n} particles, "
              f"fwd={linked_fwd} ({pct_fwd:.0f}%), "
              f"bwd={linked_bwd} ({pct_bwd:.0f}%), "
              f"isolated={isolated}")

    # =======================================================================
    # STAGE 5: DISPLACEMENT ANALYSIS for linked particles
    # =======================================================================
    print("\n" + "=" * 70)
    print("STAGE 5: DISPLACEMENT ANALYSIS")
    print("=" * 70)

    mid_frame = spar.first + 1
    ptv_path = work_dir / "res" / f"ptv_is.{mid_frame}"
    rt_path = work_dir / "res" / f"rt_is.{mid_frame}"
    rt_next_path = work_dir / "res" / f"rt_is.{mid_frame + 1}"

    if ptv_path.exists() and rt_path.exists() and rt_next_path.exists():
        with open(rt_path) as f:
            rl = f.readlines()
        n_curr = int(rl[0].strip())
        pos_curr = np.array([[float(x) for x in rl[i+1].split()[1:4]]
                             for i in range(n_curr)])

        with open(rt_next_path) as f:
            rl2 = f.readlines()
        n_next = int(rl2[0].strip())
        pos_next = np.array([[float(x) for x in rl2[i+1].split()[1:4]]
                             for i in range(n_next)])

        with open(ptv_path) as f:
            pl = f.readlines()
        n_ptv = int(pl[0].strip())

        linked_disps = []
        for i in range(n_ptv):
            parts = pl[i+1].split()
            next_idx = int(parts[1])
            if 0 <= next_idx < n_next:
                d = np.linalg.norm(pos_next[next_idx] - pos_curr[i])
                linked_disps.append(d)

        if linked_disps:
            ld = np.array(linked_disps)
            print(f"  Frame {mid_frame}: {len(ld)} linked particles")
            print(f"    displacement: min={ld.min():.3f} "
                  f"med={np.median(ld):.3f} max={ld.max():.3f} mm")
            assert np.median(ld) < 10.0, (
                f"Median displacement {np.median(ld):.1f}mm unreasonably large")

    # =======================================================================
    # SUMMARY
    # =======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_frames = spar.last - spar.first + 1
    avg_particles = total_particles / n_frames if n_frames > 0 else 0
    avg_fwd = total_fwd / n_frames if n_frames > 0 else 0
    avg_bwd = total_bwd / n_frames if n_frames > 0 else 0

    print(f"  Frames processed: {n_frames}")
    print(f"  Average particles/frame: {avg_particles:.0f}")
    print(f"  Average fwd links/frame: {avg_fwd:.0f}")
    print(f"  Average bwd links/frame: {avg_bwd:.0f}")
    print(f"  Detection mismatches: {len(detection_mismatches)}")
    print(f"  rt_is mismatches: {len(rt_is_mismatches)}")

    assert total_fwd > 0, "No forward links — tracking failed"
    print(f"\nAverage over sequence, particles: {avg_particles:.1f}, "
          f"links: {avg_fwd:.1f}, lost: {avg_particles - avg_fwd:.1f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
