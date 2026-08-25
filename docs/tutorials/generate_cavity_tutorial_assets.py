"""
Generate tutorial screenshots and animated GIF for test_cavity dataset.
Demonstrates the 4 optimization steps:
1. Target-plate autocalibration
2. Initial stereo-correspondences
3. Tracer self-calibration (shaking) & kinematic parameter tuning
4. 3D Lagrangian tracking
"""

import os
import shutil
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.parameters import ControlPar, SequencePar
from openptv2.autocalibration import calibrate_dataset, cam_files, tracer_self_calibrate
from openptv2.batch.pyptv_batch import build_processing_experiment, run_batch
from openptv2.gui.ptv import _open_run_store
from openptv2.tracking_framebuf import read_targets
from openptv2.transforms import convert_arr_metric_to_pixel


def generate_cavity_assets():
    base_dir = Path("C:/Users/alex/projects/openptv2").resolve()
    orig_cavity_dir = base_dir / "test_data/test_cavity"
    img_out_dir = base_dir / "docs/tutorials/images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cavity_dir = Path(tmp_dir) / "test_cavity"
        shutil.copytree(orig_cavity_dir, cavity_dir)
        yaml_file = cavity_dir / "parameters.yaml"

        spar = SequencePar.from_yaml(yaml_file)
        first, last = spar.first, spar.last

        print("=" * 70)
        print("STEP 1: Run Target-Plate Autocalibration")
        print("=" * 70)
        cal_res = calibrate_dataset(cavity_dir, write=True)
        mean_rms = np.mean([r.rms for r in cal_res if r.rms < float("inf")])
        print(f"Calibrated 4 cameras. Mean Reprojection RMS: {mean_rms:.3f} px")

        print("\n" + "=" * 70)
        print("STEP 2: Run Sequence & Initial Correspondences")
        print("=" * 70)
        prev_cwd = Path.cwd()
        os.chdir(cavity_dir)
        try:
            run_batch(yaml_file, first, last, mode="both")
        finally:
            os.chdir(prev_cwd)

        print("\n" + "=" * 70)
        print("STEP 3: Run Tracer Self-Calibration (Shaking) & Kinematic Warmup")
        print("=" * 70)
        cpar = ControlPar.from_yaml(yaml_file)
        cals = []
        for c in range(cpar.num_cams):
            _, ori_p, add_p = cam_files(cavity_dir, c)
            cals.append(Calibration.from_file(str(ori_p), str(add_p)))

        new_cals, info = tracer_self_calibrate(
            cavity_dir,
            cpar,
            cals,
            tol_px=2.0,
            max_particles=300,
            iters=3,
            hold_cam=1,
        )
        rcm_red = (1.0 - info['rcm_after'] / info['rcm_before']) * 100.0 if info.get('rcm_before') else 0.0
        print(f"Shaking RCM: {info['rcm_before']:.2f} um -> {info['rcm_after']:.2f} um ({rcm_red:.1f}% reduction)")

        # Save refined calibrations
        for cam_idx, cal in enumerate(new_cals, 1):
            cal.write(
                str(cavity_dir / f"cal/cam{cam_idx}.tif.ori"),
                str(cavity_dir / f"cal/cam{cam_idx}.tif.addpar")
            )

        # Re-run sequence with refined calibration
        os.chdir(cavity_dir)
        try:
            run_batch(yaml_file, first, last, mode="sequence")
        finally:
            os.chdir(prev_cwd)

        # Kinematic parameter tuning
        import yaml
        with open(yaml_file, "r") as f:
            ydata = yaml.safe_load(f)
        if "tracking" not in ydata:
            ydata["tracking"] = {}
        ydata["tracking"].update({
            "dvxmin": -1.0,
            "dvxmax": 1.0,
            "dvymin": -1.0,
            "dvymax": 1.0,
            "dvzmin": -1.0,
            "dvzmax": 1.0,
            "dacc": 1.0,
            "dangle": 120.0,
            "plugin_name": "default"
        })
        if "plugins" not in ydata:
            ydata["plugins"] = {}
        ydata["plugins"]["selected_tracking"] = "priority_segment_3d"
        with open(yaml_file, "w") as f:
            yaml.safe_dump(ydata, f)
        print("Tuned kinematic parameters: search box = [-1.0, 1.0] mm, dacc = 1.0 mm, dangle = 120 deg")

        print("\n" + "=" * 70)
        print("STEP 4: Run Tracking with Tuned Kinematics")
        print("=" * 70)
        os.chdir(cavity_dir)
        try:
            run_batch(yaml_file, first, last, mode="tracking")
        finally:
            os.chdir(prev_cwd)

        # Load reconstructed trajectories from Zarr or text files
        proc_exp = build_processing_experiment(yaml_file, first, last)
        store = _open_run_store(proc_exp.exp_path)

        trajectories = []
        point_by_frame = {}

        if store is not None:
            for frame_idx, frame_num in enumerate(range(first, last + 1)):
                pos, links, _ = store.read_frame_data(frame_num)
                point_by_frame[frame_num] = {
                    "pos": pos,
                    "links": links
                }
                print(f"Frame {frame_num}: {len(pos)} particles, {np.sum(links >= 0)} forward links")
        else:
            for frame_idx, frame_num in enumerate(range(first, last + 1)):
                ptv_path = cavity_dir / f"res/ptv_is.{frame_num}"
                if ptv_path.exists():
                    data = np.loadtxt(ptv_path, skiprows=1)
                    if data.ndim == 1 and len(data) > 0:
                        data = data.reshape(1, -1)
                    if len(data) > 0:
                        pos = data[:, 1:4]
                        prev_l = data[:, 0].astype(int)
                        next_l = data[:, 4].astype(int)
                        point_by_frame[frame_num] = {
                            "pos": pos,
                            "links": next_l,
                            "prev": prev_l
                        }
                        print(f"Frame {frame_num}: {len(pos)} particles, {np.sum(next_l >= 0)} forward links")

        # Reconstruct continuous tracks
        for start_idx, start_pos in enumerate(point_by_frame[first]["pos"]):
            track = [start_pos]
            curr_link = point_by_frame[first]["links"][start_idx]
            curr_f = first + 1
            while curr_link >= 0 and curr_f <= last:
                if curr_f in point_by_frame and curr_link < len(point_by_frame[curr_f]["pos"]):
                    next_pos = point_by_frame[curr_f]["pos"][curr_link]
                    track.append(next_pos)
                    curr_link = point_by_frame[curr_f]["links"][curr_link]
                    curr_f += 1
                else:
                    break
            if len(track) >= 2:
                trajectories.append(np.array(track))

        print(f"Constructed {len(trajectories)} multi-frame trajectories (>=2 frames).")

        # -------------------------------------------------------------
        # Visualization 1: 3D Trajectory Plot (Snapshot)
        # -------------------------------------------------------------
        print("Rendering 3D Trajectory Snapshot...")
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        # Compute velocities for colormap
        vel_mags = [np.linalg.norm(t[-1] - t[0]) / (len(t) - 1) for t in trajectories]
        norm = plt.Normalize(vmin=np.percentile(vel_mags, 5), vmax=np.percentile(vel_mags, 95))
        cmap = plt.cm.viridis

        for traj, v in zip(trajectories, vel_mags):
            color = cmap(norm(v))
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=1.2, alpha=0.7)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=8, alpha=0.8)

        ax.set_xlabel("X [mm]", fontsize=11, fontweight="bold")
        ax.set_ylabel("Y [mm]", fontsize=11, fontweight="bold")
        ax.set_zlabel("Z [mm]", fontsize=11, fontweight="bold")
        ax.set_title(f"OpenPTV2: Cavity Flow 3D Trajectories ({len(trajectories)} Tracks)", fontsize=13, fontweight="bold", pad=15)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.08)
        cbar.set_label("Mean Velocity Magnitude [mm/frame]", fontsize=10, fontweight="bold")

        ax.view_init(elev=28, azim=45)
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        traj_3d_png = img_out_dir / "test_cavity_trajectories_3d.png"
        plt.savefig(traj_3d_png, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {traj_3d_png}")

        # -------------------------------------------------------------
        # Visualization 2: Multi-Camera Reprojections Overlay
        # -------------------------------------------------------------
        print("Rendering 4-Camera Reprojection Snapshot...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
        axes = axes.flatten()

        frame_num = first
        frame_xyz = point_by_frame[frame_num]["pos"]

        for cam_i, ax in enumerate(axes):
            cam_num = cam_i + 1
            img_path = cavity_dir / f"img/cam{cam_num}.{frame_num}"
            if img_path.exists():
                try:
                    img_data = plt.imread(img_path)
                    ax.imshow(img_data, cmap="gray", origin="upper", aspect="equal")
                except Exception:
                    ax.imshow(np.zeros((cpar.imy, cpar.imx)), cmap="gray", origin="upper")
            else:
                ax.imshow(np.zeros((cpar.imy, cpar.imx)), cmap="gray", origin="upper")

            # 2D detected targets
            targs = read_targets(str(cavity_dir / "img"), cam_i, frame_num)
            tx = [t.x() for t in targs]
            ty = [t.y() for t in targs]
            ax.scatter(tx, ty, s=12, facecolors="none", edgecolors="#00ffff", linewidth=0.8, alpha=0.6, label="2D Detections")

            # Reprojected 3D correspondences
            cal = new_cals[cam_i]
            reproj_pts = []
            for p in frame_xyz:
                rx, ry = img_coord(p, cal, cpar.mm)
                reproj_pts.append([rx, ry])
            reproj_metric = np.array(reproj_pts)
            reproj_pix = convert_arr_metric_to_pixel(reproj_metric, cpar)

            ax.scatter(reproj_pix[:, 0], reproj_pix[:, 1], s=10, c="#ff3366", marker="x", linewidth=0.8, alpha=0.7, label="3D Reprojection")

            ax.set_title(f"Camera {cam_num} (Frame {frame_num}) - {len(targs)} Detections", fontsize=11, fontweight="bold")
            ax.set_xlim(0, cpar.imx)
            ax.set_ylim(cpar.imy, 0)
            ax.axis("off")
            if cam_i == 0:
                ax.legend(loc="upper right", framealpha=0.8, fontsize=9)

        plt.suptitle("Multi-Camera 2D Target Detections vs 3D Stereo Reprojections", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        reproj_png = img_out_dir / "test_cavity_multicam_reprojection.png"
        plt.savefig(reproj_png, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {reproj_png}")

        # -------------------------------------------------------------
        # Visualization 3: Animated 3D Rotating GIF
        # -------------------------------------------------------------
        print("Rendering 3D Rotating Trajectory Animation (GIF)...")
        fig = plt.figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        for traj, v in zip(trajectories, vel_mags):
            color = cmap(norm(v))
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=1.2, alpha=0.75)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=6, alpha=0.85)

        ax.set_xlabel("X [mm]", fontsize=9, fontweight="bold")
        ax.set_ylabel("Y [mm]", fontsize=9, fontweight="bold")
        ax.set_zlabel("Z [mm]", fontsize=9, fontweight="bold")
        ax.set_title("OpenPTV2 Cavity Flow - 3D Trajectories", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        frames_gif = []
        for azim in range(0, 360, 10):
            ax.view_init(elev=25, azim=azim)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames_gif.append(rgba)

        plt.close()

        gif_path = img_out_dir / "test_cavity_trajectories.gif"
        imageio.mimsave(gif_path, frames_gif, duration=0.1, loop=0)
        print(f"Saved Animated GIF: {gif_path}")

        # Copy the gif and screenshots to docs/images/ for README inclusion
        docs_img_dir = base_dir / "docs/images"
        docs_img_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gif_path, docs_img_dir / "test_cavity_trajectories.gif")
        shutil.copyfile(traj_3d_png, docs_img_dir / "test_cavity_trajectories_3d.png")
        shutil.copyfile(reproj_png, docs_img_dir / "test_cavity_multicam_reprojection.png")
        print(f"Copied all assets to: {docs_img_dir}")

        print("\nAll assets generated successfully!")

if __name__ == "__main__":
    generate_cavity_assets()
