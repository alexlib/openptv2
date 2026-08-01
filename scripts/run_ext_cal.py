import sys
from pathlib import Path

import numpy as np

from openptv2.algorithms.sortgrid import read_calblock, sortgrid
from openptv2.algorithms.tracking_frame_buf import read_targets
from openptv2.autocalibration import (
    CamResult,
    _load_dataset_params,
    _matched_pairs,
    cam_files,
    resolve_calblock,
    rms_px,
    save_overlay,
    target_base,
)
from openptv2.calibration import Calibration
from openptv2.orientation import external_calibration, full_calibration


def calibrate_ext_only(dataset_dir: str):
    base = Path(dataset_dir).resolve()
    calblock = resolve_calblock(base)
    fix, nfix = read_calblock(str(calblock))

    dp = _load_dataset_params(base, calblock)
    cpar, num_cams = dp.cpar, dp.num_cams

    out = base / "cal" / "ext_cal_only"
    out.mkdir(parents=True, exist_ok=True)

    print(
        "Calibrating only exterior orientation (6-DOF) for "
        f"{num_cams} cameras in {base}..."
    )

    R_B1_to_B2 = None
    R_B2_to_B1 = None
    t_B2_to_B1 = None

    for cam in range(num_cams):
        ids = dp.ids_per_cam[cam]
        fix4 = np.asarray([fix[i - 1] for i in ids], dtype=float)
        pix4 = dp.clicks_per_cam[cam]

        import glob

        target_files = glob.glob(str(base / "img" / f"cam{cam + 1}.*_targets"))
        if not target_files:
            print(
                f"cam{cam + 1}: no target files found matching cam{cam + 1}.*_targets"
            )
            continue

        target_file = target_files[0]
        # remove _targets from the end because read_targets will append it
        file_base_str = target_file[:-8]
        pix = read_targets(file_base_str, 0)
        if not pix:
            # Auto-detect targets from the calibration image
            from openptv2.autocalibration import _tpar_from_dataset

            tpar = _tpar_from_dataset(base)
            img, _, _ = cam_files(base, cam)
            if tpar is not None and img.exists():
                from imageio.v3 import imread
                from skimage.color import rgb2gray
                from skimage.util import img_as_ubyte

                from openptv2.algorithms.tracking_frame_buf import write_targets
                from openptv2.image_processing import preprocess_image
                from openptv2.segmentation import target_recognition

                raw_img = imread(img)
                if raw_img.ndim == 3:
                    raw_img = rgb2gray(raw_img)
                raw_img = img_as_ubyte(raw_img)
                hp_img = preprocess_image(raw_img, cpar.hp_flag or 1, cpar, 25)
                detected = target_recognition(hp_img, tpar, cam, cpar)
                if detected:
                    write_targets(
                        detected, len(detected), str(target_base(base, cam)), 0
                    )
                    pix = read_targets(str(target_base(base, cam)), 0)

        if not pix:
            print(f"cam{cam + 1}: no detected targets found. skipping.")
            continue

        img, ori, addpar = cam_files(base, cam)
        c = Calibration.from_file(str(ori), str(addpar))

        # Save old pose for transformation
        R_old = c.get_rotation_matrix()
        pos_old = c.get_pos()

        # Apply transformation if we have it
        if (
            R_B1_to_B2 is not None
            and R_B2_to_B1 is not None
            and t_B2_to_B1 is not None
        ):
            R_new = R_old @ R_B1_to_B2
            pos_new = R_B2_to_B1 @ pos_old + t_B2_to_B1
            c.set_rotation_matrix(R_new)
            c.set_pos(pos_new)
            print(f"cam{cam + 1}: Applied rig transformation from cam1.")

        if not external_calibration(c, fix4, pix4, cpar):
            print(
                "cam"
                f"{cam + 1}: external_calibration did not converge. "
                "Using initial guess."
            )
        else:
            print(f"cam{cam + 1}: After external_calibration, pos={c.get_pos()}")

        # If this is cam1, calculate the transformation for the rest
        if cam == 0:
            R_new = c.get_rotation_matrix()
            pos_new = c.get_pos()
            R_B1_to_B2 = R_old.T @ R_new
            R_B2_to_B1 = R_B1_to_B2.T
            t_B2_to_B1 = pos_new - R_B2_to_B1 @ pos_old

        # Progressive refinement of exterior orientation
        for current_eps in [15, 10, 5, 2]:
            sorted_pix = sortgrid(c, cpar, nfix, fix, len(pix), current_eps, pix)
            n_matched = sum(1 for t in sorted_pix if t.pnr >= 0)

            for _ in range(5):
                try:
                    full_calibration(c, fix, sorted_pix, cpar, [])
                except (ValueError, RuntimeError):
                    break
                sp = sortgrid(c, cpar, nfix, fix, len(pix), current_eps, pix)
                n = sum(1 for t in sp if t.pnr >= 0)
                sorted_pix = sp
                if n <= n_matched:
                    n_matched = n
                    break
                n_matched = n

        ref, det, rep = _matched_pairs(c, cpar, fix, sorted_pix)
        r = rms_px(det, rep)
        if np.isnan(r):
            print(f"cam{cam + 1}: RMS is NaN. Resetting to inf")
            r = float("inf")
        print(f"cam{cam + 1}: matched {n_matched}/{nfix} targets. RMS={r:6.3f}px")

        if r < 10.0:
            c.write(str(ori), str(addpar))
            print(f"cam{cam + 1}: saved updated orientation.")
        else:
            print(f"cam{cam + 1}: RMS too high ({r:6.3f}px). Skipping save.")

        res = CamResult(cam, n_matched, nfix, r, [], c, ref, det, rep)
        save_overlay(res, base, out)


if __name__ == "__main__":
    calibrate_ext_only(sys.argv[1])
