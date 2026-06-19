"""
Full pipeline diagnostic: images → detection → correspondences → tracking.

Uses algorithms/ directly (no compat wrappers, no TrackingRun).
Logs every stage with counts and parameters so we can see exactly where
particles are lost.

Usage:
    uv run pytest algorithms/tests/test_full_tracking_diagnostic.py -v -s
"""

import math
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field

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
from algorithms.trafo import pixel_to_metric, dist_to_flat
from algorithms.orientation import point_positions as algo_point_positions
from algorithms.track import (
    track_forward_start, trackcorr_c_loop, trackcorr_c_finish,
    trackback_c, angle_acc,
)
from algorithms.tracking_run import TrackingRun


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

CAVITY_DIR = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
YAML_PATH = CAVITY_DIR / "parameters_Run1.yaml"


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _build_cpar(params):
    """Build ControlPar directly from YAML dict — no compat layer."""
    ptv = params["ptv"]
    num_cams = params.get("num_cams", 4)

    mm = MmNp(
        nlay=1,
        n1=ptv.get("mmp_n1", 1.0),
        n2=[ptv.get("mmp_n2", 1.0)],
        d=[ptv.get("mmp_d", 0.0)],
        n3=ptv.get("mmp_n3", 1.0),
    )
    cpar = ControlPar(
        num_cams=num_cams,
        imx=ptv["imx"], imy=ptv["imy"],
        pix_x=ptv["pix_x"], pix_y=ptv["pix_y"],
        hp_flag=1 if ptv.get("hp_flag", True) else 0,
        all_cam_flag=1 if ptv.get("allcam_flag", False) else 0,
        tiff_flag=1 if ptv.get("tiff_flag", True) else 0,
        chfield=ptv.get("chfield", 0),
        mm=mm,
    )
    return cpar


def _build_vpar(params):
    crit = params.get("criteria", {})
    return VolumePar(
        X_lay=crit.get("X_lay", [-100, 100]),
        Zmin_lay=crit.get("Zmin_lay", [-100, -100]),
        Zmax_lay=crit.get("Zmax_lay", [100, 100]),
        cn=crit.get("cn", 0),
        cnx=crit.get("cnx", 0),
        cny=crit.get("cny", 0),
        csumg=crit.get("csumg", 0),
        eps0=crit.get("eps0", 0),
        corrmin=crit.get("corrmin", 0),
    )


def _build_tpar(params):
    tr = params.get("targ_rec", {})
    # gvthres can be in targ_rec directly or in detect_plate
    gvthres = tr.get("gvthres", None)
    if gvthres is None:
        dp = params.get("detect_plate", {})
        gvthres = [
            dp.get("gvth_1", 40), dp.get("gvth_2", 40),
            dp.get("gvth_3", 40), dp.get("gvth_4", 40),
        ]
    return TargetPar(
        gvthres=gvthres,
        discont=tr.get("disco", tr.get("discont", 5)),
        nnmin=tr.get("nnmin", 1),
        nnmax=tr.get("nnmax", 500),
        nxmin=tr.get("nxmin", 1),
        nxmax=tr.get("nxmax", 50),
        nymin=tr.get("nymin", 1),
        nymax=tr.get("nymax", 50),
        sumg_min=tr.get("sumg_min", 10),
        cr_sz=tr.get("cr_sz", 3),
    )


def _build_track_par(params):
    tk = params.get("track", {})
    return TrackPar(
        dvxmin=tk.get("dvxmin", -10),
        dvxmax=tk.get("dvxmax", 10),
        dvymin=tk.get("dvymin", -10),
        dvymax=tk.get("dvymax", 10),
        dvzmin=tk.get("dvzmin", -10),
        dvzmax=tk.get("dvzmax", 10),
        dangle=tk.get("angle", tk.get("dangle", 100)),
        dacc=tk.get("dacc", 2.0),
        add=1 if tk.get("flagNewParticles", False) else 0,
    )


def _build_spar(params, exp_dir):
    seq = params.get("sequence", {})
    num_cams = params.get("num_cams", 4)
    base = seq.get("base_name", ["img/cam%d." % (i+1) for i in range(num_cams)])
    img_base = [str(exp_dir / b) for b in base]
    return SequencePar(
        num_cams=num_cams,
        img_base_name=img_base,
        first=seq.get("first", 10000),
        last=seq.get("last", 10004),
    )


def _load_calibrations(params, exp_dir):
    cal_ori = params.get("cal_ori", {})
    ori_files = cal_ori.get("img_ori", [])
    cals = []
    for ori in ori_files:
        ori_path = str(exp_dir / ori)
        add_path = ori_path.replace(".ori", ".addpar")
        c = Calibration.from_file(ori_path, add_path)
        cals.append(c)
    return cals


def _detect_targets(img, tpar, cam_idx, cpar):
    """Run highpass + target recognition on one image. Returns list of Target."""
    hp = prepare_image(
        img, dim_lp=1, imx=cpar.imx, imy=cpar.imy,
        filter_hp=1 if cpar.hp_flag else 0,
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
    targets.sort(key=lambda t: t.y)
    for j, t in enumerate(targets):
        t.pnr = j
    return targets


def _correct_targets(targets, cpar, cal):
    """Pixel → metric → flat correction. Returns x-sorted list of Coord2d."""
    corrected = []
    ap = cal.added_par
    ip = cal.int_par
    for t in targets:
        mx, my = pixel_to_metric(t.x, t.y, cpar.imx, cpar.imy,
                                 cpar.pix_x, cpar.pix_y, cpar.chfield)
        fx, fy = dist_to_flat(mx, my, ip.xh, ip.yh,
                              ap.k1, ap.k2, ap.k3, ap.p1, ap.p2,
                              ap.scx, ap.she, tol=1e-5)
        corrected.append(Coord2d(x=fx, y=fy, pnr=t.pnr))
    corrected.sort(key=lambda c: c.x)
    return corrected


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

@pytest.fixture
def cavity_setup(tmp_path):
    """Load cavity parameters and calibrations."""
    import shutil
    import os

    if not YAML_PATH.exists():
        pytest.skip("Cavity test data not available")

    work = tmp_path / "cavity"
    shutil.copytree(CAVITY_DIR, work)

    params = _load_yaml(work / "parameters_Run1.yaml")
    cpar = _build_cpar(params)
    vpar = _build_vpar(params)
    tpar = _build_tpar(params)
    track_par = _build_track_par(params)
    spar = _build_spar(params, work)
    cals = _load_calibrations(params, work)

    # Ensure res/ directory exists and is empty
    res_dir = work / "res"
    res_dir.mkdir(exist_ok=True)
    for f in res_dir.glob("*"):
        if f.is_file():
            f.unlink()

    original = os.getcwd()
    try:
        yield {
            "params": params, "cpar": cpar, "vpar": vpar, "tpar": tpar,
            "track_par": track_par, "spar": spar, "cals": cals,
            "work_dir": work, "num_cams": params.get("num_cams", 4),
        }
    finally:
        os.chdir(original)


def test_full_pipeline_diagnostic(cavity_setup):
    """Run the complete pipeline with detailed logging at every stage."""
    import os
    from skimage.io import imread
    from skimage.util import img_as_ubyte
    from skimage.color import rgb2gray

    cfg = cavity_setup
    cpar = cfg["cpar"]
    vpar = cfg["vpar"]
    tpar = cfg["tpar"]
    track_par = cfg["track_par"]
    spar = cfg["spar"]
    cals = cfg["cals"]
    num_cams = cfg["num_cams"]
    work_dir = cfg["work_dir"]
    params = cfg["params"]

    os.chdir(work_dir)

    # ---------------------------------------------------------------
    # LOG: Parameters
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PARAMETERS")
    print("=" * 70)
    print(f"  num_cams    = {num_cams}")
    print(f"  image_size  = {cpar.imx} x {cpar.imy}")
    print(f"  pixel_size  = {cpar.pix_x} x {cpar.pix_y}")
    print(f"  hp_flag     = {cpar.hp_flag}")
    print(f"  multimedia  = n1={cpar.mm.n1}, n2={cpar.mm.n2}, "
          f"n3={cpar.mm.n3}, d={cpar.mm.d}, nlay={cpar.mm.nlay}")
    print(f"  velocity    = x[{track_par.dvxmin}, {track_par.dvxmax}] "
          f"y[{track_par.dvymin}, {track_par.dvymax}] "
          f"z[{track_par.dvzmin}, {track_par.dvzmax}]")
    print(f"  dacc={track_par.dacc}  dangle={track_par.dangle}  "
          f"add={track_par.add}")
    print(f"  volume      = X_lay={vpar.X_lay} Zmin={vpar.Zmin_lay} "
          f"Zmax={vpar.Zmax_lay}")
    print(f"  volume crit = cn={vpar.cn} cnx={vpar.cnx} cny={vpar.cny}")
    print(f"  target      = gvthres={tpar.gvthres} discont={tpar.discont}")
    print(f"                nn=[{tpar.nnmin},{tpar.nnmax}] "
          f"nx=[{tpar.nxmin},{tpar.nxmax}] ny=[{tpar.nymin},{tpar.nymax}]")
    print(f"  frames      = {spar.first} to {spar.last}")
    for i, c in enumerate(cals):
        pos = np.array([c.ext_par.x0, c.ext_par.y0, c.ext_par.z0])
        print(f"  cal[{i}] pos  = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    # ---------------------------------------------------------------
    # STAGE 1: Detection per frame
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: TARGET DETECTION")
    print("=" * 70)

    all_detections = {}   # frame -> [list_of_targets_per_cam]
    all_corrected = {}    # frame -> [list_of_Coord2d_per_cam]

    seq = params.get("sequence", {})
    base_names = seq.get("base_name", [f"img/cam{i+1}.%d" for i in range(num_cams)])

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
        print(f"  Frame {frame}: detected {counts} targets per camera, "
              f"total={sum(counts)}")

        # Build Frame object and use correct_frame for proper correction
        max_tgt = max(len(d) for d in frame_dets) + 10
        frm = Frame(num_cams=num_cams, max_targets=max_tgt)
        for i_cam in range(num_cams):
            frm.num_targets[i_cam] = len(frame_dets[i_cam])
            for j, t in enumerate(frame_dets[i_cam]):
                frm.targets[i_cam][j] = t

        frame_corr = correct_frame(frm, cals, cpar, tol=1e-5)

        if frame == spar.first:
            for i_cam in range(num_cams):
                if len(frame_corr[i_cam]) > 0:
                    c0 = frame_corr[i_cam][0]
                    cn = frame_corr[i_cam][-1]
                    print(f"    cam{i_cam}: corrected x-sorted: "
                          f"[0]=({c0.x:.2f},{c0.y:.2f} pnr={c0.pnr}) "
                          f"[-1]=({cn.x:.2f},{cn.y:.2f} pnr={cn.pnr})")

        all_detections[frame] = frame_dets
        all_corrected[frame] = frame_corr

    # ---------------------------------------------------------------
    # STAGE 2: Correspondences per frame
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2: STEREO CORRESPONDENCES")
    print("=" * 70)

    all_positions = {}  # frame -> ndarray (N, 3)
    all_corresp = {}    # frame -> ndarray (num_cams, N)

    # Diagnostic: check epi_mm for first frame, first target in cam0→cam1
    from algorithms.epi import epi_mm, find_candidate
    first_corr = all_corrected[spar.first]
    if len(first_corr[0]) > 0:
        c = first_corr[0][0]
        xmin, ymin, xmax, ymax = epi_mm(
            c.x, c.y, cals[0], cals[1], cpar.mm, vpar)
        print(f"  epi_mm cam0→cam1 for target ({c.x:.2f},{c.y:.2f}):")
        print(f"    band: x=[{xmin:.2f},{xmax:.2f}] y=[{ymin:.2f},{ymax:.2f}]")
        band_w = xmax - xmin
        band_h = ymax - ymin
        print(f"    size: {band_w:.2f} x {band_h:.2f}")

        # Count how many cam1 targets fall in this band
        in_band = 0
        for t in first_corr[1]:
            if xmin <= t.x <= xmax and ymin <= t.y <= ymax:
                in_band += 1
        print(f"    cam1 targets in band: {in_band}/{len(first_corr[1])}")

    for frame in range(spar.first, spar.last + 1):
        dets = all_detections[frame]
        corr = all_corrected[frame]

        # Rebuild Frame object (correspondences modifies targets[].tnr)
        max_tgt = max(len(d) for d in dets) + 10
        frm = Frame(num_cams=num_cams, max_targets=max_tgt)
        for i_cam in range(num_cams):
            frm.num_targets[i_cam] = len(dets[i_cam])
            for j, t in enumerate(dets[i_cam]):
                frm.targets[i_cam][j] = t

        con, match_counts = algo_correspondences(frm, corr, vpar, cpar, cals)

        n_quads, n_trips, n_pairs, n_total = match_counts
        print(f"  Frame {frame}: {n_total} correspondences "
              f"(quads={n_quads}, triplets={n_trips}, pairs={n_pairs})")

        if n_total > 0:
            valid = con[:n_total]
            # corresp_corr: indices into corrected list (for 3D triangulation)
            corresp_corr = np.array([nt.p[:num_cams] for nt in valid]).T
            # corresp_tgt: target pnr values (for rt_is file — tracker
            # uses these to index into targ_x/targ_y arrays)
            corresp_tgt = np.array([
                [corr[cam][nt.p[cam]].pnr if nt.p[cam] >= 0 else -1
                 for cam in range(num_cams)]
                for nt in valid
            ]).T
        else:
            corresp_corr = np.zeros((num_cams, 0), dtype=int)
            corresp_tgt = np.zeros((num_cams, 0), dtype=int)

        # 3D point positions (use corrected-list indices)
        if corresp_corr.shape[1] > 0:
            flat = np.array([
                [corr[cam][int(corresp_corr[cam, i])] if corresp_corr[cam, i] >= 0
                 else Coord2d(x=-999, y=-999, pnr=-1)
                 for i in range(corresp_corr.shape[1])]
                for cam in range(num_cams)
            ])
            flat_arr = np.array([
                [[c.x, c.y] for c in cam_coords]
                for cam_coords in flat
            ])
            flat_arr = flat_arr.transpose(1, 0, 2)  # (N, num_cams, 2)
            pos, rcm = algo_point_positions(flat_arr, cpar, cals, vpar)
        else:
            pos = np.zeros((0, 3))

        all_positions[frame] = pos
        all_corresp[frame] = corresp_tgt

        # Write rt_is — use target pnr values (not corrected-list indices)
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

        # Log tnr assignment from correspondences
        if frame == spar.first:
            for i_cam in range(num_cams):
                n_tgt = frm.num_targets[i_cam]
                tnr_vals = [frm.targets[i_cam][j].tnr for j in range(n_tgt)]
                tnr_set = sum(1 for v in tnr_vals if v >= 0)
                tnr_neg = sum(1 for v in tnr_vals if v < 0)
                print(f"    cam{i_cam}: {tnr_set}/{n_tgt} targets have "
                      f"tnr>=0 (mapped to 3D particle), {tnr_neg} unmatched")
                if tnr_set > 0:
                    valid_tnrs = sorted([v for v in tnr_vals if v >= 0])
                    print(f"      tnr range: {valid_tnrs[0]}..{valid_tnrs[-1]}, "
                          f"max_tnr={max(valid_tnrs)}")


        # Write _targets files for tracking — MUST use frm.targets (not dets)
        # because correspondences() sets targets[cam][pnr].tnr = corr_index,
        # which the tracker needs to map 2D targets → 3D particles.
        for i_cam in range(num_cams):
            file_base = base_names[i_cam].replace("%d", "")
            n_tgt = frm.num_targets[i_cam]
            tgt_list = [frm.targets[i_cam][j] for j in range(n_tgt)]
            write_targets(
                tgt_list, n_tgt,
                str(work_dir / file_base), frame,
            )

    # ---------------------------------------------------------------
    # STAGE 3: 3D NEAREST-NEIGHBOR ANALYSIS (pre-tracking diagnostic)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 3: 3D NEAREST-NEIGHBOR ANALYSIS (before tracking)")
    print("=" * 70)

    frames = sorted(all_positions.keys())
    dvx = (track_par.dvxmin, track_par.dvxmax)
    dvy = (track_par.dvymin, track_par.dvymax)
    dvz = (track_par.dvzmin, track_par.dvzmax)

    for i in range(len(frames) - 1):
        f_curr = frames[i]
        f_next = frames[i + 1]
        pos_curr = all_positions[f_curr]
        pos_next = all_positions[f_next]

        if len(pos_curr) == 0 or len(pos_next) == 0:
            print(f"  {f_curr}→{f_next}: skipped (empty)")
            continue

        nn_counts = []
        in_bounds = 0
        for j in range(len(pos_curr)):
            disp = pos_next - pos_curr[j]
            mask = (
                (dvx[0] < disp[:, 0]) & (disp[:, 0] < dvx[1]) &
                (dvy[0] < disp[:, 1]) & (disp[:, 1] < dvy[1]) &
                (dvz[0] < disp[:, 2]) & (disp[:, 2] < dvz[1])
            )
            nn_counts.append(mask.sum())
            if mask.sum() > 0:
                in_bounds += 1

        nn_arr = np.array(nn_counts)
        print(f"  {f_curr}→{f_next}: {len(pos_curr)} particles, "
              f"{in_bounds} ({100*in_bounds/len(pos_curr):.0f}%) have >=1 "
              f"neighbor in bounds")
        if len(nn_arr) > 0:
            print(f"    neighbors in bounds: "
                  f"min={nn_arr.min()} median={int(np.median(nn_arr))} "
                  f"max={nn_arr.max()} mean={nn_arr.mean():.1f}")

        # 4-frame check: for particles with candidates in f_next,
        # check if those candidates also have candidates in f_next+1
        if i + 2 < len(frames):
            f_nn = frames[i + 2]
            pos_nn = all_positions[f_nn]
            confirmed = 0
            no_confirm = 0

            for j in range(len(pos_curr)):
                disp = pos_next - pos_curr[j]
                cand_mask = (
                    (dvx[0] < disp[:, 0]) & (disp[:, 0] < dvx[1]) &
                    (dvy[0] < disp[:, 1]) & (disp[:, 1] < dvy[1]) &
                    (dvz[0] < disp[:, 2]) & (disp[:, 2] < dvz[1])
                )
                if cand_mask.sum() == 0:
                    continue

                # For best candidate (nearest), check confirmation in f_nn
                cand_dists = np.linalg.norm(disp[cand_mask], axis=1)
                best_cand_idx = np.where(cand_mask)[0][np.argmin(cand_dists)]
                best_cand = pos_next[best_cand_idx]

                # Predict: linear extrapolation from curr→best_cand→???
                predicted = 2 * best_cand - pos_curr[j]
                disp2 = pos_nn - predicted
                conf_mask = (
                    (dvx[0] < disp2[:, 0]) & (disp2[:, 0] < dvx[1]) &
                    (dvy[0] < disp2[:, 1]) & (disp2[:, 1] < dvy[1]) &
                    (dvz[0] < disp2[:, 2]) & (disp2[:, 2] < dvz[1])
                )
                if conf_mask.sum() > 0:
                    confirmed += 1
                else:
                    no_confirm += 1

            total_with_cands = confirmed + no_confirm
            if total_with_cands > 0:
                print(f"    4-frame check ({f_curr}→{f_next}→{f_nn}): "
                      f"{confirmed}/{total_with_cands} "
                      f"({100*confirmed/total_with_cands:.0f}%) confirmed")

        # acc/angle analysis for the best candidate pairs
        if i + 2 < len(frames):
            f_nn = frames[i + 2]
            pos_nn = all_positions[f_nn]
            acc_ok = 0
            acc_fail = 0
            acc_vals = []
            angle_vals = []

            for j in range(len(pos_curr)):
                disp = pos_next - pos_curr[j]
                cand_mask = (
                    (dvx[0] < disp[:, 0]) & (disp[:, 0] < dvx[1]) &
                    (dvy[0] < disp[:, 1]) & (disp[:, 1] < dvy[1]) &
                    (dvz[0] < disp[:, 2]) & (disp[:, 2] < dvz[1])
                )
                if cand_mask.sum() == 0:
                    continue

                cand_indices = np.where(cand_mask)[0]
                cand_dists = np.linalg.norm(disp[cand_mask], axis=1)
                best_idx = cand_indices[np.argmin(cand_dists)]
                best_cand = pos_next[best_idx]

                predicted = 2 * best_cand - pos_curr[j]
                disp2 = pos_nn - predicted
                conf_mask = (
                    (dvx[0] < disp2[:, 0]) & (disp2[:, 0] < dvx[1]) &
                    (dvy[0] < disp2[:, 1]) & (disp2[:, 1] < dvy[1]) &
                    (dvz[0] < disp2[:, 2]) & (disp2[:, 2] < dvz[1])
                )
                if conf_mask.sum() == 0:
                    continue

                conf_indices = np.where(conf_mask)[0]
                conf_dists = np.linalg.norm(disp2[conf_mask], axis=1)
                best_conf_idx = conf_indices[np.argmin(conf_dists)]
                best_conf = pos_nn[best_conf_idx]

                ang, acc = angle_acc(pos_curr[j], best_cand, best_conf)
                acc_vals.append(acc)
                angle_vals.append(ang)

                passes = ((acc < track_par.dacc and ang < track_par.dangle)
                          or acc < track_par.dacc * 0.1)
                if passes:
                    acc_ok += 1
                else:
                    acc_fail += 1

            total_checked = acc_ok + acc_fail
            if total_checked > 0:
                print(f"    acc/angle check: {acc_ok}/{total_checked} "
                      f"({100*acc_ok/total_checked:.0f}%) pass "
                      f"(dacc={track_par.dacc}, dangle={track_par.dangle})")
                if acc_vals:
                    print(f"    acc  : min={min(acc_vals):.3f} "
                          f"med={np.median(acc_vals):.3f} "
                          f"max={max(acc_vals):.3f}")
                    print(f"    angle: min={min(angle_vals):.1f} "
                          f"med={np.median(angle_vals):.1f} "
                          f"max={max(angle_vals):.1f}")

    # ---------------------------------------------------------------
    # STAGE 4: ACTUAL TRACKING
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 4: TRACKING (via TrackingRun)")
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

    print(f"  TrackingRun created: lmax={run.lmax:.2f}")

    track_forward_start(run)
    print(f"  track_forward_start done")

    # --- DIAGNOSTIC: quick projection sanity check ---
    from algorithms.track import point_to_pixel
    fb = run.fb
    curr_slot = fb.buf[1]
    next_slot = fb.buf[2]
    print(f"  frame buffer: curr={curr_slot.num_parts} particles, "
          f"next={next_slot.num_parts} particles")

    # Verify projection consistency for first 5 particles
    offsets = []
    n_check = min(5, curr_slot.num_parts)
    for h in range(n_check):
        pos = curr_slot.path_x[h]
        for cam in range(num_cams):
            px, py = point_to_pixel(pos, cals[cam], cpar)
            for j in range(curr_slot.num_targets[cam]):
                if curr_slot.targets[cam][j].tnr == h:
                    t = curr_slot.targets[cam][j]
                    offsets.append(math.sqrt((px - t.x)**2 + (py - t.y)**2))
                    break
    if offsets:
        print(f"  projection check (3D→2D vs target): "
              f"median={np.median(offsets):.1f}px, "
              f"max={max(offsets):.1f}px (n={len(offsets)})")

    # Count valid tnr in next frame
    for cam in range(num_cams):
        n_tgt = next_slot.num_targets[cam]
        valid_tnr = sum(1 for j in range(n_tgt) if next_slot.targ_tnr[cam][j] >= 0)
        print(f"  next frame cam{cam}: {valid_tnr}/{n_tgt} targets with valid tnr")

    for step in range(spar.first, spar.last + 1):
        trackcorr_c_loop(run, step)
        print(f"  step {step}: npart={run.npart}, nlinks={run.nlinks}")

    trackcorr_c_finish(run, spar.last)
    print(f"  forward tracking: npart={run.npart}, nlinks={run.nlinks}")

    trackback_c(run)
    print(f"  after backward: npart={run.npart}, nlinks={run.nlinks}")

    # ---------------------------------------------------------------
    # STAGE 5: ANALYZE TRACKING OUTPUT
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 5: TRACKING RESULTS ANALYSIS")
    print("=" * 70)

    for frame in range(spar.first, spar.last + 1):
        ptv_path = work_dir / "res" / f"ptv_is.{frame}"
        rt_path = work_dir / "res" / f"rt_is.{frame}"

        if not ptv_path.exists():
            print(f"  Frame {frame}: no ptv_is file")
            continue

        with open(ptv_path) as f:
            lines = f.readlines()
        n = int(lines[0].strip())

        linked_fwd = 0
        linked_bwd = 0
        unlinked = 0

        for i in range(1, n + 1):
            parts = lines[i].split()
            prev_idx = int(parts[0])
            next_idx = int(parts[1])
            if next_idx >= 0:
                linked_fwd += 1
            if prev_idx >= 0:
                linked_bwd += 1
            if prev_idx < 0 and next_idx < 0:
                unlinked += 1

        pct_fwd = 100 * linked_fwd / n if n > 0 else 0
        pct_bwd = 100 * linked_bwd / n if n > 0 else 0
        print(f"  Frame {frame}: {n} particles, "
              f"fwd_linked={linked_fwd} ({pct_fwd:.0f}%), "
              f"bwd_linked={linked_bwd} ({pct_bwd:.0f}%), "
              f"isolated={unlinked}")

    # ---------------------------------------------------------------
    # STAGE 6: COMPARE LINKED vs UNLINKED — WHY DID THEY FAIL?
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 6: LINKED vs UNLINKED — DISPLACEMENT ANALYSIS")
    print("=" * 70)

    # Pick a middle frame for analysis
    mid_frame = spar.first + 1
    ptv_path = work_dir / "res" / f"ptv_is.{mid_frame}"
    rt_path = work_dir / "res" / f"rt_is.{mid_frame}"
    rt_next_path = work_dir / "res" / f"rt_is.{mid_frame + 1}"

    if ptv_path.exists() and rt_path.exists() and rt_next_path.exists():
        # Read positions for current and next frame
        with open(rt_path) as f:
            rt_lines = f.readlines()
        n_curr = int(rt_lines[0].strip())
        pos_curr = np.array([
            [float(x) for x in rt_lines[i+1].split()[1:4]]
            for i in range(n_curr)
        ])

        with open(rt_next_path) as f:
            rt_lines_next = f.readlines()
        n_next = int(rt_lines_next[0].strip())
        pos_next_rt = np.array([
            [float(x) for x in rt_lines_next[i+1].split()[1:4]]
            for i in range(n_next)
        ])

        # Read ptv_is links
        with open(ptv_path) as f:
            ptv_lines = f.readlines()
        n_ptv = int(ptv_lines[0].strip())

        linked_disps = []
        unlinked_nn_dists = []
        for i in range(n_ptv):
            parts = ptv_lines[i+1].split()
            next_idx = int(parts[1])
            if next_idx >= 0 and next_idx < n_next:
                d = np.linalg.norm(pos_next_rt[next_idx] - pos_curr[i])
                linked_disps.append(d)
            elif next_idx < 0:
                dists = np.linalg.norm(pos_next_rt - pos_curr[i], axis=1)
                unlinked_nn_dists.append(dists.min())

        if linked_disps:
            ld = np.array(linked_disps)
            print(f"  Frame {mid_frame} linked particles ({len(ld)}):")
            print(f"    displacement: min={ld.min():.3f} "
                  f"med={np.median(ld):.3f} max={ld.max():.3f}")

        if unlinked_nn_dists:
            ud = np.array(unlinked_nn_dists)
            print(f"  Frame {mid_frame} unlinked particles ({len(ud)}):")
            print(f"    NN dist to next frame: min={ud.min():.3f} "
                  f"med={np.median(ud):.3f} max={ud.max():.3f}")
            in_bounds = (ud < min(abs(track_par.dvxmax), abs(track_par.dvymax),
                                  abs(track_par.dvzmax))).sum()
            print(f"    within velocity bounds: {in_bounds}/{len(ud)} "
                  f"({100*in_bounds/len(ud):.0f}%)")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
