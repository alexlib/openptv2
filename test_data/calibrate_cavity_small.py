from pathlib import Path

import numpy as np
import pandas as pd

from openptv2.calibration import Calibration
from openptv2.orientation import external_calibration
from openptv2.parameters import ControlParams, MultimediaParams


def main():
    test_dir = Path("test_data/test_cavity_small")

    # 1. Load ground truth datasets
    df_particles = pd.read_csv(test_dir / "ground_truth/particles.csv")
    df_projections = pd.read_csv(test_dir / "ground_truth/projections.csv")

    # 2. Control parameters for 256x256 crop
    cpar = ControlParams(num_cams=4)
    cpar.set_image_size((256, 256))
    cpar.set_pixel_size((0.012, 0.012))

    # Multimedia parameters
    mm = MultimediaParams(n1=1.0, n2=[1.46], n3=1.33, d=[6.0])
    cpar.mm = mm

    for cam_idx in range(4):
        cam_num = cam_idx + 1
        print(f"\n--- Refining Calibration for Cam {cam_num} ---")

        # Load initial calibration
        cal = Calibration()
        cal.from_file(
            str(test_dir / f"cal/cam{cam_num}.tif.ori"),
            str(test_dir / f"cal/cam{cam_num}.tif.addpar"),
        )

        # 3. Collect 3D coordinates and 2D projections (in pixels!)
        ref_pts = []
        img_pts = []

        for idx, row in df_particles.iterrows():
            frame = row["frame"]
            pid = row["particle_id"]

            # Find matching projection for this camera and particle
            df_match = df_projections[
                (df_projections["frame"] == frame)
                & (df_projections["cam"] == cam_num)
                & (df_projections["particle_id"] == pid)
            ]

            if not df_match.empty:
                px = df_match.iloc[0]["x_px_crop"]
                py = df_match.iloc[0]["y_px_crop"]

                ref_pts.append([row["X"], row["Y"], row["Z"]])
                img_pts.append([px, py])

        ref_pts = np.array(ref_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float64)

        print(f"Collected {len(ref_pts)} physical 3D-2D correspondences.")

        if len(ref_pts) < 10:
            print("Not enough points to refine calibration.")
            continue

        # 4. Perform external calibration optimization (camera resectioning)
        try:
            success = external_calibration(cal, ref_pts, img_pts, cpar)
        except Exception as e:
            print(f"External calibration failed: {e}")
            success = False

        print(f"Optimization Convergence: {success}")

        # Save refined orientation
        cal.write(
            str(test_dir / f"cal/cam{cam_num}.tif.ori"),
            str(test_dir / f"cal/cam{cam_num}.tif.addpar"),
        )
        print(f"Saved refined calibration files for Cam {cam_num}.")


if __name__ == "__main__":
    main()
