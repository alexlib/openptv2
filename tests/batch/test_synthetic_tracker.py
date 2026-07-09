import os
import shutil
import numpy as np
from pathlib import Path
from openptv2.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.parameters import ControlParams, MultimediaParams
from openptv2.gui.parameter_manager import ParameterManager
from openptv2.batch import pyptv_batch

def _write_ori_file(path: Path, pos, angles, focal_len=100.0):
    """Writes a clean synthetic .ori file."""
    omega, phi, kappa = angles
    # For synthetic, we compute a clean rotation matrix from omega, phi, kappa
    cal = Calibration()
    cal.set_pos(pos)
    cal.set_angles(angles)
    cal.set_primary_point([0.0, 0.0, focal_len])
    cal.set_glass_vec([0.0, 0.0, 50.0]) # Glass window interface at Z = 50 mm
    
    # Format the .ori file content exactly as expected by the C reader
    r = cal.ext_par
    content = f"""{r.x0:11.4f} {r.y0:11.4f} {r.z0:11.4f}
{r.omega:14.7f} {r.phi:14.7f} {r.kappa:14.7f}

{r.dm[0, 0]:14.7f} {r.dm[0, 1]:14.7f} {r.dm[0, 2]:14.7f}
{r.dm[1, 0]:14.7f} {r.dm[1, 1]:14.7f} {r.dm[1, 2]:14.7f}
{r.dm[2, 0]:14.7f} {r.dm[2, 1]:14.7f} {r.dm[2, 2]:14.7f}

{cal.int_par.xh:11.4f} {cal.int_par.yh:11.4f}
{cal.int_par.cc:11.4f}

{cal.glass_par.vec_x:11.4f} {cal.glass_par.vec_y:11.4f} {cal.glass_par.vec_z:11.4f}
"""
    path.write_text(content, encoding="utf-8")
    
    # Write a clean, zero-distortion .addpar file
    addpar_path = path.with_suffix(".addpar")
    addpar_content = "0.0 0.0 0.0 0.0 0.0 1.0 0.0\n"
    addpar_path.write_text(addpar_content, encoding="utf-8")
    return cal

def test_fully_verifiable_synthetic_tracker(tmp_path):
    """
    Creates a perfectly clean, symmetric synthetic environment with 4 cameras,
    projects a set of controlled Lagrangian trajectories to 2D targets, writes the
    synthetic data, runs tracking, and asserts 100% correct track recovery.
    """
    # 1. Directory Setup
    test_dir = tmp_path / "test_synthetic"
    test_dir.mkdir()
    (test_dir / "cal").mkdir()
    (test_dir / "img").mkdir()
    (test_dir / "res").mkdir()
    
    # 2. Setup Parameters and Control Files
    cpar = ControlParams(num_cams=4)
    cpar.set_image_size((1024, 1024))
    cpar.set_pixel_size((0.01, 0.01))
    
    # Air=1.0, Glass(5mm)=1.46, Water=1.33
    mpar = MultimediaParams(n1=1.0, n2=[1.46], n3=1.33, d=[5.0])
    # MultimediaParams is now an alias for the algorithms MmNp class, so mpar is
    # already the raw multimedia-params object (no wrapper to unwrap).
    raw_mm = mpar
    
    # Cameras placement (symmetric square looking down from Z=500 to water domain Z=0)
    positions = [
        [100.0, 100.0, 500.0],
        [-100.0, 100.0, 500.0],
        [-100.0, -100.0, 500.0],
        [100.0, -100.0, 500.0]
    ]
    angles = [
        [0.2, -0.2, -0.78],
        [0.2, 0.2, 0.78],
        [-0.2, 0.2, 2.35],
        [-0.2, -0.2, -2.35]
    ]
    
    cals = []
    for i in range(4):
        p = test_dir / "cal" / f"cam{i+1}.tif.ori"
        cals.append(_write_ori_file(p, positions[i], angles[i]))
        
    # Load baseline parameter yaml file to ensure all required fields are present
    pm = ParameterManager()
    pm.from_yaml(Path('/home/user/Documents/GitHub/openptv2/test_data/test_cavity/parameters_Run1.yaml'))
    
    # Overwrite relevant values for our synthetic environment
    pm.parameters['ptv'] = {
        'mmp_n1': 1.0, 'mmp_n2': 1.46, 'mmp_n3': 1.33, 'mmp_d': 5.0,
        'mmp_nlay': 1, 'imx': 1024, 'imy': 1024,
        'pix_x': 0.01, 'pix_y': 0.01,
        'mmp_gvec_x': 0.0, 'mmp_gvec_y': 0.0, 'mmp_gvec_z': 1.0,
        'num_cams': 4,
        'img_name': [f"img/cam{i+1}.tif" for i in range(4)],
        'img_cal': [f"cal/cam{i+1}.tif" for i in range(4)],
        'allcam_flag': 1, 'tiff_flag': 1, 'chfield': 0, 'hp_flag': 1, 'splitter': 0
    }
    pm.parameters['pft_version'] = {
        'Existing_Target': 1,
    }
    pm.parameters['criteria']['eps0'] = 0.15
    pm.parameters['track'] = {
        'dvxmin': -15.5, 'dvxmax': 15.5,
        'dvymin': -15.5, 'dvymax': 15.5,
        'dvzmin': -15.5, 'dvzmax': 15.5,
        'dacc': 5.5, 'angle': 120.0,
        'flagNewParticles': 1, 'track_mode': 1
    }
    pm.parameters['sequence'] = {
        'base_name': [f"img/cam{i+1}." for i in range(4)],
        'first': 10001,
        'last': 10005
    }
    
    yaml_file = test_dir / "parameters.yaml"
    pm.to_yaml(yaml_file)
    
    # 3. Generate Known Lagrangian Trajectories
    frames = [10001, 10002, 10003, 10004, 10005]
    # We define 4 clean physical trajectories moving in water (Z < 50)
    traj_defs = [
        # Traj 1: Stationary particle (slow)
        lambda f: np.array([0.0, 0.0, 10.0]),
        # Traj 2: Constant slow velocity along X (1 mm/frame)
        lambda f: np.array([-2.0 + (f - 10001) * 1.0, 5.0, 15.0]),
        # Traj 3: Constant fast velocity along Y (-3.5 mm/frame)
        lambda f: np.array([12.0, 6.0 - (f - 10001) * 3.5, 20.0]),
        # Traj 4: Accelerating particle along Z
        lambda f: np.array([-10.0, -10.0, 5.0 + 0.5 * (f - 10001)**2])
    ]
    
    num_particles = len(traj_defs)
    
    # Frame-by-frame data generation
    for f_idx, f in enumerate(frames):
        pts_3d = []
        for t_idx, traj in enumerate(traj_defs):
            pts_3d.append(traj(f))
            
        pts_3d = np.array(pts_3d)
        
        # Project 3D points to 2D targets on all 4 cameras
        cam_targets = {i: [] for i in range(4)}
        for p_idx, pt in enumerate(pts_3d):
            for i in range(4):
                x_met, y_met = img_coord(pt, cals[i], raw_mm)
                # Convert metric flat coordinates to pixel coordinates
                x_pix = 1024 / 2.0 + x_met / 0.01
                y_pix = 1024 / 2.0 - y_met / 0.01
                # Format target: pnr, x_pix, y_pix, nx, ny, npix, sumg, tnr
                cam_targets[i].append((p_idx, x_pix, y_pix, 5, 5, 25, 120, p_idx))
                
        # Write cam*_targets files for this frame
        for i in range(4):
            targets_file = test_dir / "img" / f"cam{i+1}.{f}_targets"
            target_arr = np.array(cam_targets[i])
            np.savetxt(
                targets_file,
                target_arr,
                fmt="%4d %9.4f %9.4f %5d %5d %5d %5d %5d",
                header=f"{num_particles}",
                comments=""
            )
            
        # Write corresponding rt_is file
        rt_is_file = test_dir / "res" / f"rt_is.{f}"
        with open(rt_is_file, "w", encoding="utf-8") as fh:
            fh.write(f"{num_particles}\n")
            for p_idx, pt in enumerate(pts_3d):
                # Format: index, x, y, z, pnr_cam1, pnr_cam2, pnr_cam3, pnr_cam4
                fh.write(f"{p_idx+1:4d} {pt[0]:9.3f} {pt[1]:9.3f} {pt[2]:9.3f} {p_idx:4d} {p_idx:4d} {p_idx:4d} {p_idx:4d}\n")
                
    # 4. Execute pyptv-batch Tracking on the synthetic dataset
    print("\n--- Running Tracking on Symmetric Synthetic Dataset ---")
    pyptv_batch.main(str(yaml_file), 10001, 10005, mode="tracking")
    
    # 5. Read output ptv_is linkage files and verify 100% Correct Trajectory Matching
    # In PTV tracking with a 3-frame buffer, a 5-frame sequence (10001-10005) 
    # produces tracking linkages for frames 10001, 10002, and 10003.
    tracked_frames = [10001, 10002, 10003]
    reconstructed = []
    
    for f in tracked_frames:
        ptv_file = test_dir / "res" / f"ptv_is.{f}"
        assert ptv_file.exists(), f"Tracking output {ptv_file.name} was not created!"
        lines = ptv_file.read_text().strip().splitlines()
        n = int(lines[0])
        assert n == num_particles, f"Expected {num_particles} tracked particles in frame {f}, but got {n}"
        
        frame_links = []
        for line in lines[1:]:
            parts = line.split()
            frame_links.append((int(parts[0]), int(parts[1])))
        reconstructed.append(frame_links)
        
    # Verify that each trajectory matches perfectly
    # - Particle 0: stationary
    # - Particle 1: slow X
    # - Particle 2: fast Y
    # - Particle 3: accelerating Z
    for p_idx in range(num_particles):
        # Frame 10001: prev should be -1 (start of track), next should be p_idx (linked forward)
        assert reconstructed[0][p_idx][0] == -1, f"Particle {p_idx} start error"
        assert reconstructed[0][p_idx][1] == p_idx, f"Particle {p_idx} linkage error at 10001"
        
        # Frame 10002: prev should be p_idx, next should be p_idx (middle of track)
        assert reconstructed[1][p_idx][0] == p_idx, f"Particle {p_idx} back-linkage error at frame 10002"
        assert reconstructed[1][p_idx][1] == p_idx, f"Particle {p_idx} forward-linkage error at frame 10002"
            
        # Frame 10003: prev should be p_idx, next should be p_idx (middle/forward of track to frame 10004)
        assert reconstructed[2][p_idx][0] == p_idx, f"Particle {p_idx} back-linkage error at frame 10003"
        assert reconstructed[2][p_idx][1] == p_idx, f"Particle {p_idx} forward-linkage error at frame 10003"
    print("\n--- ASSERTION SUCCESS: 100% of Synthetic Trajectories Tracked Perfectly! ---")
