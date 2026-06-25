import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from pathlib import Path

# Add project root to python path to import openptv2 and gui modules
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from openptv2.imgcoord import image_coordinates
from openptv2.transforms import convert_arr_metric_to_pixel
from openptv2.tracking_framebuf import read_targets
from gui.pyptv.experiment import Experiment
from gui.pyptv.ptv import py_start_proc_c
from flowtracks.io import trajectories_ptvis

def main():
    print("Generating visual tutorial snapshots...")
    
    # 1. Define paths
    original_cwd = Path.cwd()
    images_dir = original_cwd / "docs/tutorials/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    cavity_dir = original_cwd / "test_data/test_cavity"
    yaml_file = cavity_dir / "parameters_Run1.yaml"
    ptv_is_pattern = str(cavity_dir / "res/ptv_is.%d")
    
    # 2. Load trajectories using flowtracks
    print("Loading trajectories...")
    dataset = trajectories_ptvis(ptv_is_pattern, first=10001, last=10004)
    print(f"Loaded {len(dataset)} trajectories.")
    
    # Filter trajectories with length >= 3 for better visualization
    trajs = [t for t in dataset if len(t) >= 3]
    print(f"Found {len(trajs)} trajectories with length >= 3.")
    
    # 3. Load calibrations and control parameters (Change CWD to cavity_dir first)
    print("Loading parameters and calibrations...")
    os.chdir(cavity_dir)
    try:
        experiment = Experiment()
        experiment.pm.from_yaml("parameters_Run1.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(experiment.pm)
    finally:
        os.chdir(original_cwd)
    
    # 4. Generate Snapshot 1: 3D Trajectories (Matplotlib)
    print("Generating 3D trajectory plot...")
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Select the first 25 trajectories to plot
    plot_trajs = trajs[:25]
    
    # Use a premium qualitative colormap for distinct trajectories
    cmap = plt.colormaps.get_cmap("tab20")
    num_colors = 20
    
    for idx, traj in enumerate(plot_trajs):
        # Convert position from meters (flowtracks standard) to mm (OpenPTV2 standard)
        pos = np.array(traj.pos()) * 1000.0
        
        ax.plot(
            pos[:, 0], pos[:, 1], pos[:, 2],
            marker='o', markersize=4, linestyle='-', linewidth=2,
            color=cmap(idx % num_colors), label=f"Path {traj.trajid}"
        )
        
    ax.set_xlabel('X [mm]', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y [mm]', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Z [mm]', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('3D Particle Trajectories - Cavity Flow Dataset', fontsize=14, fontweight='bold', pad=15)
    
    # Make the grid styling clean and modern
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Add a legend but limit entries to make it clean
    if len(plot_trajs) <= 10:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    output_3d_path = images_dir / "trajectory_3d.png"
    plt.savefig(output_3d_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved 3D trajectory plot to: {output_3d_path}")
    
    # 5. Generate Snapshot 2: 2D Projection with overlays on Camera 1 Frame 10001
    print("Generating 2D camera overlay plot...")
    cam_idx = 0  # Camera 1
    frame_number = 10001
    
    img_path = cavity_dir / f"img/cam1.{frame_number}"
    if not img_path.exists():
        print(f"Error: Camera image {img_path} not found.")
        return
        
    # Read raw TIFF image
    image = skimage.io.imread(img_path)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot image in grayscale with some contrast enhancement
    ax.imshow(image, cmap='gray', origin='upper')
    
    # Load targets (detected particles) for Camera 1, Frame 10001
    # NOTE: The file_base path must end with a dot so read_targets forms "cam1.10001_targets"
    targets = read_targets(str(cavity_dir / "img/cam1."), frame_number)
    print(f"Loaded {len(targets)} detected targets.")
    
    # Plot detected targets as blue '+' signs
    target_x = [t.x() for t in targets]
    target_y = [t.y() for t in targets]
    ax.scatter(
        target_x, target_y, 
        color='#00BFFF', marker='+', s=40, linewidths=1.2, 
        label='Detected Particle Targets', alpha=0.85
    )
    
    # Project 3D trajectories to Camera 1 2D image coordinates and plot them
    projected_count = 0
    for idx, traj in enumerate(plot_trajs):
        # Convert to mm
        pos_3d_mm = np.array(traj.pos()) * 1000.0
        
        # Project 3D mm coords to 2D sensor metric coords
        projected = image_coordinates(pos_3d_mm, cals[cam_idx], cpar.get_multimedia_params())
        
        # Convert sensor metric coords to pixel coordinates
        pixels = convert_arr_metric_to_pixel(projected, cpar)
        
        # Draw projected trajectory line
        ax.plot(
            pixels[:, 0], pixels[:, 1], 
            linestyle='-', linewidth=1.5, color='#FF4500', 
            alpha=0.8
        )
        # Draw head/endpoints
        ax.scatter(
            pixels[-1, 0], pixels[-1, 1], 
            color='#FFD700', marker='o', s=15, edgecolors='black', 
            linewidths=0.5, alpha=0.9
        )
        projected_count += 1
        
    # Dummy handle for trajectories in legend
    ax.plot([], [], color='#FF4500', linestyle='-', linewidth=1.5, label='Projected 3D Trajectories')
    ax.scatter([], [], color='#FFD700', marker='o', s=25, edgecolors='black', label='Trajectory Endpoints')
    
    imx, imy = cpar.get_image_size()
    ax.set_xlim(0, imx)
    ax.set_ylim(imy, 0)  # Invert y-axis to match image/pixel space coordinates
    ax.set_xlabel('X [pixels]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y [pixels]', fontsize=11, fontweight='bold')
    ax.set_title(f'2D Particle Detection & Trajectory Overlays (Camera 1, Frame {frame_number})', fontsize=13, fontweight='bold', pad=15)
    
    # Clean up axes style
    ax.legend(loc='lower left', framealpha=0.9, facecolor='#ffffff', edgecolor='#cccccc')
    ax.grid(False) # No grid for image overlays
    
    plt.tight_layout()
    output_2d_path = images_dir / "camera_projection_2d.png"
    plt.savefig(output_2d_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved 2D projection plot to: {output_2d_path}")
    print("All tutorial snapshots generated successfully!")

if __name__ == "__main__":
    main()
