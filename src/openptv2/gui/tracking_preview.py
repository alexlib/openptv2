"""Tracking preview functionality for the 'Tracking with display' feature."""

from typing import Dict

from openptv2 import (
    Calibration,
    Tracker,
    convert_optv_calibrations,
    get_control_par,
    get_sequence_par,
    get_track_par_tuple,
    get_volume_par,
)


def run_tracking_preview(main_gui, num_frames: int = 5) -> Dict:
    """
    Run tracking preview for a specified number of frames.

    Args:
        main_gui: The MainGUI instance
        num_frames: Number of frames to track and preview (default: 5)

    Returns:
        Dictionary containing:
        - 'frames': List of frame data for each processed frame
        - 'statistics': Per-frame statistics
        - 'tracking_run': The TrackingRun object for accessing final state
    """
    # Get current parameters from the main GUI
    pm = main_gui.exp1.pm
    params = pm.parameters

    # Convert parameters to the expected types
    cpar = get_control_par(params)
    cpar.num_cams = params.get("num_cams", 4)
    spar = get_sequence_par(params)
    vpar = get_volume_par(params)
    tpar_tuple = get_track_par_tuple(params)

    # Get calibration parameters
    calib_params = params.get("calibration")

    # Get calibration objects
    cals = []
    if calib_params and isinstance(calib_params, list):
        for cal_dict in calib_params:
            cal = Calibration()
            # Assuming calib_params is a list of dicts with calibration data
            # This may need adjustment based on actual calibration structure
            for key, value in cal_dict.items():
                if hasattr(cal, key):
                    setattr(cal, key, value)
            cals.append(cal)
    else:
        # Handle case where calib_params might be a single dict or other format
        try:
            import ptv

            _, _, _, _, _, _, cals = ptv.py_start_proc_c(main_gui.exp1.pm)
            cals = convert_optv_calibrations(cals)
        except Exception:
            # Fallback: create dummy calibrations
            cals = [Calibration() for _ in range(cpar.num_cams)]

    # Initialize the tracker
    tracker = Tracker(cpar, vpar, tpar_tuple, spar, cals)

    # Restart to initialize frame buffer
    tracker.restart()

    # Storage for frame-by-frame data
    frames_data = []
    statistics = []

    # Check track_mode from parameters
    track_mode = params.get("track", {}).get("track_mode", 0)

    # Track for the specified number of frames
    for frame_idx in range(num_frames):
        step_success = (
            tracker.step_forward_3d() if track_mode == 1 else tracker.step_forward()
        )
        if not step_success:
            break  # No more frames to process

        # Get current state from the tracker
        tracker._get_current_state()

        # Extract detailed information for visualization
        frame_data = extract_frame_details(tracker.run_info, frame_idx)
        frames_data.append(frame_data)

        # Calculate statistics for this frame
        stats = calculate_frame_statistics(tracker.run_info, frame_idx)
        statistics.append(stats)

    # Finalize the tracking run
    tracker.finalize()

    return {
        "frames": frames_data,
        "statistics": statistics,
        "tracking_run": tracker.run_info,
        "tracker": tracker,
    }


def extract_frame_details(run_info, frame_num: int) -> Dict:
    """
    Extract detailed information from a frame for visualization.

    Args:
        run_info: The TrackingRun object
        frame_num: The frame number to extract data for

    Returns:
        Dictionary containing particle positions, detections, and link information
    """
    fb = run_info.fb

    # Get the current frame buffer (index 1 is current, 0 is previous, 2 is next)
    # After step_forward(), buf[1] contains the current frame data
    current_buf = fb.buf[1]

    # Extract 3D particle positions
    particles_3d = []
    for i in range(current_buf.num_parts):
        path_info = current_buf.path_info[i]
        particles_3d.append(
            {
                "id": i,
                "position": path_info.x.copy(),  # [x, y, z] in metric coordinates
                "tnr": -1,  # Will be updated from target data if available
                "in_list": path_info.inlist > 0,
                "prev_frame": path_info.prev_frame,
                "next_frame": path_info.next_frame,
            }
        )

    # Update TNR values from target data if available
    for cam_idx in range(run_info.cpar.num_cams):
        targets = current_buf.targets[cam_idx]
        for j in range(current_buf.num_targets[cam_idx]):
            target = targets[j]
            # Find which particle this target belongs to (if any)
            for pid, pdata in enumerate(particles_3d):
                # Simple approach: if TNR matches particle ID, or we can do spatial matching
                if target.tnr >= 0 and target.tnr < len(particles_3d):
                    # Update the particle's TNR
                    particles_3d[target.tnr]["tnr"] = target.tnr
                    break

    # Extract 2D detections per camera
    detections_2d = []
    for cam_idx in range(run_info.cpar.num_cams):
        camera_detections = []
        targets = current_buf.targets[cam_idx]
        for j in range(current_buf.num_targets[cam_idx]):
            target = targets[j]
            # Find which particle this target belongs to (if any)
            particle_id = -1
            if 0 <= target.tnr < len(particles_3d):
                particle_id = target.tnr

            camera_detections.append(
                {
                    "x": float(target.x),
                    "y": float(target.y),
                    "tnr": int(target.tnr),
                    "particle_id": particle_id,
                    "is_linked": target.tnr >= 0
                    and 0 <= target.tnr < len(particles_3d),
                }
            )
        detections_2d.append(camera_detections)

    # Extract link information (which particles connect to which)
    links = []
    for i in range(current_buf.num_parts):
        path_info = current_buf.path_info[i]
        if path_info.prev_frame >= 0 and path_info.prev_frame < len(particles_3d):
            links.append(
                {
                    "from_particle": i,
                    "to_particle": path_info.prev_frame,
                    "type": "backward",
                }
            )
        if path_info.next_frame >= 0 and path_info.next_frame < len(particles_3d):
            links.append(
                {
                    "from_particle": i,
                    "to_particle": path_info.next_frame,
                    "type": "forward",
                }
            )

    return {
        "frame_number": frame_num + run_info.seq_par.first,
        "particles_3d": particles_3d,
        "detections_2d": detections_2d,
        "links": links,
        "num_particles": current_buf.num_parts,
        "num_detections_per_cam": [
            current_buf.num_targets[cam] for cam in range(run_info.cpar.num_cams)
        ],
    }


def calculate_frame_statistics(run_info, frame_num: int) -> Dict:
    """
    Calculate statistics for a frame.

    Args:
        run_info: The TrackingRun object
        frame_num: The frame number to calculate statistics for

    Returns:
        Dictionary containing various tracking statistics
    """
    fb = run_info.fb
    current_buf = fb.buf[1]

    # Count linked vs unlinked particles
    linked_count = 0
    for i in range(current_buf.num_parts):
        path_info = current_buf.path_info[i]
        if path_info.inlist > 0:  # Particle is in a track
            linked_count += 1

    unlinked_count = current_buf.num_parts - linked_count

    # Count total links (avoid double counting)
    total_links = 0
    link_map = set()  # To avoid counting the same link twice
    for i in range(current_buf.num_parts):
        path_info = current_buf.path_info[i]
        if path_info.prev_frame >= 0 and path_info.prev_frame < current_buf.num_parts:
            link_key = tuple(sorted([i, path_info.prev_frame]))
            if link_key not in link_map:
                link_map.add(link_key)
                total_links += 1
        if path_info.next_frame >= 0 and path_info.next_frame < current_buf.num_parts:
            link_key = tuple(sorted([i, path_info.next_frame]))
            if link_key not in link_map:
                link_map.add(link_key)
                total_links += 1

    return {
        "frame_number": frame_num + run_info.seq_par.first,
        "num_particles": current_buf.num_parts,
        "num_linked_particles": linked_count,
        "num_unlinked_particles": unlinked_count,
        "num_links": total_links,
        "linking_ratio": linked_count / max(current_buf.num_parts, 1),
    }


def apply_tracking_parameters(main_gui, tracking_params_dict: dict) -> None:
    """
    Apply tracking parameters to the main GUI's parameter manager.

    Args:
        main_gui: The MainGUI instance
        tracking_params_dict: Dictionary of tracking parameters to apply
    """
    # Update the tracking parameters in the parameter manager
    current_params = main_gui.exp1.pm.get_parameter("tracking")
    if current_params is None:
        current_params = {}

    # Update only the provided parameters
    current_params.update(tracking_params_dict)
    main_gui.exp1.pm.parameters["tracking"] = current_params

    # Also update the active parameter set if it exists
    if main_gui.exp1.active_params is not None:
        main_gui.exp1.active_params.parameters["tracking"] = current_params
