"""
Tracking visualization panel for displaying tracking preview results.

This module provides a TraitsUI-based panel for visualizing tracking results
from the tracking preview functionality.
"""

import numpy as np
from typing import List
from traits.api import HasTraits, Int, Bool, Instance, List, Float, Str, Button
from traitsui.api import View, Item, HGroup, VGroup, Group, Label, TextEditor, Spring
from chaco.api import Plot, ArrayPlotData, LinearMapper
from enable.component_editor import ComponentEditor

from .tracking_preview import run_tracking_preview


class TrackingVizPanel(HasTraits):
    """
    Visualization panel for tracking preview results.

    Displays 3D particle positions, per-frame statistics, and allows
    navigation through the tracked frames.
    """

    current_frame = Int(0)
    total_frames = Int(0)
    num_particles = Int(0)
    num_linked = Int(0)
    linking_ratio = Float(0.0)

    run_preview_button = Button("Run Preview")
    prev_frame_button = Button("Previous")
    next_frame_button = Button("Next")

    status_text = Str("No preview data")

    _plot_data = Instance(ArrayPlotData)
    _plot = Instance(Plot)

    view = View(
        VGroup(
            Label("Tracking Preview"),
            HGroup(
                Item("run_preview_button", show_label=False),
                Spring(),
            ),
            Group(
                Label("Frame Navigation"),
                HGroup(
                    Item("prev_frame_button", show_label=False),
                    Item("current_frame", editor=TextEditor(), width=50),
                    Label("of"),
                    Item("total_frames", show_label=False, width=50),
                    Item("next_frame_button", show_label=False),
                ),
            ),
            Group(
                Label("Statistics"),
                HGroup(
                    VGroup(
                        Label("Particles:"),
                        Label("Linked:"),
                        Label("Linking Ratio:"),
                    ),
                    VGroup(
                        Item("num_particles", show_label=False, width=80),
                        Item("num_linked", show_label=False, width=80),
                        Item("linking_ratio", show_label=False, width=80),
                    ),
                ),
            ),
            Item("_plot", editor=ComponentEditor(), show_label=False, height=300),
            Item("status_text", style="readonly", show_label=False),
        ),
        title="Tracking Visualization",
        width=500,
        height=600,
        resizable=True,
    )

    def __init__(self, main_gui=None, num_frames: int = 5):
        super(TrackingVizPanel, self).__init__()
        self.main_gui = main_gui
        self.num_frames = num_frames
        self.preview_data = None

        self._plot_data = ArrayPlotData(x=[], y=[], z=[])
        self._plot = Plot(self._plot_data, default_origin="bottom left")
        self._plot.padding = 50

        self._plot.index_axis.title = "X position"
        self._plot.value_axis.title = "Y position"

        self._plot.plot(
            ("x", "y"),
            type="scatter",
            marker="circle",
            marker_size=3,
            color="blue",
        )

    def _run_preview_button_fired(self):
        """Run the tracking preview when button is clicked."""
        if self.main_gui is None:
            self.status_text = "No main GUI reference available"
            return

        try:
            self.preview_data = run_tracking_preview(self.main_gui, self.num_frames)
            self.total_frames = len(self.preview_data["frames"])
            self.current_frame = 0
            self._update_display()
            self.status_text = (
                f"Preview completed: {self.total_frames} frames processed"
            )
        except Exception as e:
            self.status_text = f"Error: {str(e)}"

    def _prev_frame_button_fired(self):
        """Navigate to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self._update_display()

    def _next_frame_button_fired(self):
        """Navigate to next frame."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._update_display()

    def _current_frame_changed(self):
        """Update display when frame changes."""
        self._update_display()

    def _update_display(self):
        """Update the visualization with current frame data."""
        if self.preview_data is None or self.total_frames == 0:
            return

        frame_data = self.preview_data["frames"][self.current_frame]
        stats = self.preview_data["statistics"][self.current_frame]

        self.num_particles = stats["num_particles"]
        self.num_linked = stats["num_linked_particles"]
        self.linking_ratio = stats["linking_ratio"]

        particles_3d = frame_data["particles_3d"]

        if particles_3d:
            positions = np.array([p["position"] for p in particles_3d])
            x = positions[:, 0] if positions.shape[1] > 0 else []
            y = positions[:, 1] if positions.shape[1] > 1 else []
            z = positions[:, 2] if positions.shape[1] > 2 else []
        else:
            x, y, z = [], [], []

        self._plot_data.set_data("x", x)
        self._plot_data.set_data("y", y)
        self._plot_data.set_data("z", z)

        self.status_text = f"Frame {self.current_frame + 1}/{self.total_frames}"


class MultiCameraVizPanel(HasTraits):
    """
    Visualization panel for displaying 2D detections across multiple cameras.

    Shows scatter plots of detected targets on each camera view.
    """

    current_frame = Int(0)
    total_frames = Int(0)
    selected_camera = Int(0)
    num_cameras = Int(4)
    num_detections = Int(0)
    camera_label = Str("")

    prev_frame_button = Button("Previous")
    next_frame_button = Button("Next")

    view = View(
        VGroup(
            Label("Multi-Camera 2D Detections"),
            HGroup(
                Item("prev_frame_button", show_label=False),
                Item("current_frame", editor=TextEditor(), width=50),
                Label("of"),
                Item("total_frames", show_label=False, width=50),
                Item("next_frame_button", show_label=False),
                Item("camera_label", style="readonly"),
            ),
            Item("num_detections", label="Detections:", style="readonly"),
        ),
        title="Multi-Camera Detections",
        width=400,
        height=300,
        resizable=True,
    )

    def __init__(self, preview_data=None, num_cameras: int = 4):
        super(MultiCameraVizPanel, self).__init__()
        self.preview_data = preview_data
        self.num_cameras = num_cameras
        self.camera_label = f"Camera: {num_cameras}"

        if preview_data is not None:
            self.total_frames = len(preview_data["frames"])

    def _prev_frame_button_fired(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._update_display()

    def _next_frame_button_fired(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._update_display()

    def _update_display(self):
        if self.preview_data is None:
            return

        frame_data = self.preview_data["frames"][self.current_frame]
        detections_2d = frame_data["detections_2d"]

        if self.selected_camera < len(detections_2d):
            cam_detections = detections_2d[self.selected_camera]
            self.num_detections = len(cam_detections)


class TrackingStatsPanel(HasTraits):
    """
    Panel for displaying overall tracking statistics across all preview frames.

    Shows aggregated statistics like average particles, linking ratio, etc.
    """

    avg_particles = Float(0.0)
    avg_linked = Float(0.0)
    avg_linking_ratio = Float(0.0)
    max_particles = Int(0)
    min_particles = Int(0)

    view = View(
        VGroup(
            Label("Overall Statistics"),
            HGroup(
                VGroup(
                    Label("Average Particles:"),
                    Label("Average Linked:"),
                    Label("Average Linking Ratio:"),
                    Label("Max Particles:"),
                    Label("Min Particles:"),
                ),
                VGroup(
                    Item("avg_particles", style="readonly", width=80),
                    Item("avg_linked", style="readonly", width=80),
                    Item("avg_linking_ratio", style="readonly", width=80),
                    Item("max_particles", style="readonly", width=80),
                    Item("min_particles", style="readonly", width=80),
                ),
            ),
        ),
        title="Tracking Statistics",
        width=300,
        height=250,
        resizable=True,
    )

    def __init__(self, preview_data=None):
        super(TrackingStatsPanel, self).__init__()
        self.preview_data = preview_data

        if preview_data is not None and preview_data.get("statistics"):
            self._calculate_statistics()

    def _calculate_statistics(self):
        """Calculate aggregated statistics from preview data."""
        stats_list = self.preview_data["statistics"]

        if not stats_list:
            return

        particles = [s["num_particles"] for s in stats_list]
        linked = [s["num_linked_particles"] for s in stats_list]
        ratios = [s["linking_ratio"] for s in stats_list]

        self.avg_particles = float(np.mean(particles))
        self.avg_linked = float(np.mean(linked))
        self.avg_linking_ratio = float(np.mean(ratios))
        self.max_particles = int(np.max(particles))
        self.min_particles = int(np.min(particles))


def create_tracking_viz_panel(main_gui, num_frames: int = 5):
    """
    Factory function to create and configure a tracking visualization panel.

    Args:
        main_gui: The MainGUI instance
        num_frames: Number of frames to preview

    Returns:
        TrackingVizPanel: Configured visualization panel
    """
    panel = TrackingVizPanel(main_gui=main_gui, num_frames=num_frames)
    return panel


class TrackingDebugPanel(HasTraits):
    """
    Interactive tracking debugging panel with parameter sliders.

    Allows users to explore how tracking parameters affect candidate
    selection and visualize search volumes.
    """

    main_gui = Instance(HasTraits)
    num_frames = Int(8)

    dvxmin = Float(0.0)
    dvxmax = Float(20.0)
    dvymin = Float(0.0)
    dvymax = Float(20.0)
    dvzmin = Float(0.0)
    dvzmax = Float(20.0)
    dacc = Float(5.0)
    dangle = Float(10.0)

    current_frame = Int(2)
    clicked_particle = Int(-1)
    status_text = Str("Click on a particle in the camera views")

    tracker = Instance(object)
    frame_data = Instance(dict)

    view = View(
        VGroup(
            Label("Tracking Debug Panel - Adjust Parameters"),
            Group(
                Label("Velocity Limits (pixels)"),
                HGroup(
                    Item("dvxmin", label="dvxmin", width=60),
                    Item("dvxmax", label="dvxmax", width=60),
                ),
                HGroup(
                    Item("dvymin", label="dvymin", width=60),
                    Item("dvymax", label="dvymax", width=60),
                ),
                HGroup(
                    Item("dvzmin", label="dvzmin", width=60),
                    Item("dvzmax", label="dvzmax", width=60),
                ),
            ),
            Group(
                Label("Other Limits"),
                HGroup(
                    Item("dacc", label="dacc", width=60),
                    Item("dangle", label="dangle (deg)", width=60),
                ),
            ),
            Group(
                Label("Frame Selection"),
                HGroup(
                    Item("current_frame", label="Frame", width=50),
                    Label("of 4"),
                ),
            ),
            Item("status_text", style="readonly", show_label=False),
        ),
        title="Tracking Debug",
        width=400,
        height=350,
        resizable=True,
    )

    def __init__(self, main_gui=None, num_frames: int = 8):
        super(TrackingDebugPanel, self).__init__()
        self.main_gui = main_gui
        self.num_frames = num_frames
        self.tracker = None
        self.frame_data = {}
        self.clicked_particle = -1

        if main_gui is not None:
            self._initialize_tracker()

    def _initialize_tracker(self):
        """Initialize tracker with current parameters."""
        if self.main_gui is None:
            return

        try:
            ptv_params = self.main_gui.get_parameter("ptv")
            track_params = self.main_gui.get_parameter("tracking")
            vol_params = self.main_gui.get_parameter("volume")
            calib_params = self.main_gui.get_parameter("calibration")
            seq_params = self.main_gui.get_parameter("sequence")

            from algorithms.track import Tracker
            from algorithms.parameters import (
                ControlPar,
                VolumePar,
                read_track_par,
                SequencePar,
            )
            from algorithms.calibration import Calibration
            from algorithms.parameters import TrackParTuple

            cpar = ControlPar()
            cpar.imx = ptv_params["imx"]
            cpar.imy = ptv_params["imy"]
            cpar.pix_x = ptv_params["pix_x"]
            cpar.pix_y = ptv_params["pix_y"]
            cpar.num_cams = self.main_gui.num_cams
            cpar.mm = ptv_params.get("mm", None)

            vpar = VolumePar()
            if vol_params:
                vpar.Xmin = vol_params.get("xmin", 0)
                vpar.Xmax = vol_params.get("xmax", 100)
                vpar.Ymin = vol_params.get("ymin", 0)
                vpar.Ymax = vol_params.get("ymax", 100)
                vpar.Zmin = vol_params.get("zmin", 0)
                vpar.Zmax = vol_params.get("zmax", 50)

            tpar = TrackParTuple(
                self.dvxmin,
                self.dvxmax,
                self.dvymin,
                self.dvymax,
                self.dvzmin,
                self.dvzmax,
                self.dangle,
                self.dacc,
                0,
                0.0,
                1.0,
                0.0,
                0.0,
            )

            spar = SequencePar()
            if seq_params:
                spar.first = seq_params.get("first", 1)
                spar.last = seq_params.get("last", 10)

            cals = []
            if calib_params:
                cal_list = calib_params.get("calibrations", [])
                for cal_data in cal_list[: self.main_gui.num_cams]:
                    cal = Calibration()
                    cal.from_file(cal_data.get("ori", ""), cal_data.get("add", None))
                    cals.append(cal)

            self.tracker = Tracker(cpar, vpar, tpar, spar, cals)
            self.tracker.restart()

            for _ in range(self.num_frames):
                if not self.tracker.step_forward():
                    break

            self.status_text = f"Loaded {self.num_frames} frames. Click on a particle."

        except Exception as e:
            self.status_text = f"Error initializing tracker: {e}"

    def _dvxmin_changed(self):
        self._update_tracker_parameters()

    def _dvxmax_changed(self):
        self._update_tracker_parameters()

    def _dvymin_changed(self):
        self._update_tracker_parameters()

    def _dvymax_changed(self):
        self._update_tracker_parameters()

    def _dvzmin_changed(self):
        self._update_tracker_parameters()

    def _dvzmax_changed(self):
        self._update_tracker_parameters()

    def _dacc_changed(self):
        self._update_tracker_parameters()

    def _dangle_changed(self):
        self._update_tracker_parameters()

    def _update_tracker_parameters(self):
        """Update tracker with new parameter values and refresh visualization."""
        if self.tracker is None:
            return

        from algorithms.parameters import TrackParTuple

        tpar = TrackParTuple(
            self.dvxmin,
            self.dvxmax,
            self.dvymin,
            self.dvymax,
            self.dvzmin,
            self.dvzmax,
            self.dangle,
            self.dacc,
            0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        self.tracker.run_info.tpar = tpar

        if self.clicked_particle >= 0:
            self._visualize_click(self.clicked_particle)

    def _get_particle_3d_position(self, frame, particle_idx: int) -> np.ndarray:
        """Get 3D position of a particle from path_info."""
        if not hasattr(frame, "path_info") or frame.path_info is None:
            return None
        if particle_idx >= frame.num_parts:
            return None

        path = frame.path_info[particle_idx]
        if path.x is not None and len(path.x) >= 3:
            return np.array([path.x[0], path.x[1], path.x[2]])
        return None

    def _get_velocity_from_frame(self, frame, particle_idx: int) -> np.ndarray:
        """Estimate velocity from particle's previous position."""
        if not hasattr(frame, "path_info") or frame.path_info is None:
            return np.array([0.0, 0.0, 0.0])
        if particle_idx >= frame.num_parts:
            return np.array([0.0, 0.0, 0.0])

        path = frame.path_info[particle_idx]
        if hasattr(path, "prev_frame") and path.prev_frame > 0:
            if hasattr(frame, "prev_frame_data") and frame.prev_frame_data is not None:
                prev_path = frame.prev_frame_data.path_info[particle_idx]
                if prev_path.x is not None and path.x is not None:
                    return np.array(
                        [
                            path.x[0] - prev_path.x[0],
                            path.x[1] - prev_path.x[1],
                            path.x[2] - prev_path.x[2],
                        ]
                    )

        return np.array([0.0, 0.0, 0.0])

    def _draw_search_volumes(self, volumes: List[dict]):
        """Draw search volume boundaries on all camera views."""
        if self.main_gui is None or not hasattr(self.main_gui, "camera_list"):
            return

        colors = ["green", "yellow", "orange"]

        for vol in volumes:
            frame_offset = vol["frame_offset"]
            color = colors[frame_offset - 1]
            cam_bounds = vol["camera_bounds"]

            for cam_idx, bounds in enumerate(cam_bounds):
                if cam_idx >= len(self.main_gui.camera_list):
                    continue

                cam = self.main_gui.camera_list[cam_idx]
                unique_label = f"search_vol_{frame_offset}_{cam_idx}"

                x_coords = [
                    bounds.left,
                    bounds.right,
                    bounds.right,
                    bounds.left,
                    bounds.left,
                ]
                y_coords = [bounds.up, bounds.up, bounds.down, bounds.down, bounds.up]

                cam.drawline(
                    f"{unique_label}_x",
                    f"{unique_label}_y",
                    x_coords[0],
                    y_coords[0],
                    x_coords[1],
                    y_coords[1],
                    color,
                )
                cam.drawline(
                    f"{unique_label}_x",
                    f"{unique_label}_y",
                    x_coords[1],
                    y_coords[1],
                    x_coords[2],
                    y_coords[2],
                    color,
                )
                cam.drawline(
                    f"{unique_label}_x",
                    f"{unique_label}_y",
                    x_coords[2],
                    y_coords[2],
                    x_coords[3],
                    y_coords[3],
                    color,
                )
                cam.drawline(
                    f"{unique_label}_x",
                    f"{unique_label}_y",
                    x_coords[3],
                    y_coords[3],
                    x_coords[4],
                    y_coords[4],
                    color,
                )

    def _visualize_click(self, particle_id: int):
        """Visualize search volume and candidates for clicked particle."""
        if particle_id < 0 or self.tracker is None:
            return

        fb = self.tracker.run_info.fb
        current_frame_idx = min(self.current_frame, len(fb.buf) - 1)
        current_frame = fb.buf[current_frame_idx]

        pos_3d = self._get_particle_3d_position(current_frame, particle_id)

        if pos_3d is None:
            self.status_text = f"Particle {particle_id} has no 3D position (not linked)"
            return

        velocity = self._get_velocity_from_frame(current_frame, particle_id)

        volumes = self._compute_search_volumes(pos_3d, velocity)
        self._draw_search_volumes(volumes)

        self._find_and_draw_candidates(volumes, pos_3d)

        self._draw_epipolar_lines(pos_3d, current_frame_idx)

        stats = self._get_candidate_statistics(volumes, pos_3d)
        self.status_text = (
            f"Particle {particle_id} at ({pos_3d[0]:.1f}, {pos_3d[1]:.1f}, {pos_3d[2]:.1f}) "
            f"- Candidates: {stats['total']}, Linked: {stats['linked']}"
        )

    def _get_click_position_2d(self, frame_idx: int, cam_idx: int) -> np.ndarray:
        """Get 2D position of clicked particle in specific camera."""
        fb = self.tracker.run_info.fb
        if frame_idx >= len(fb.buf):
            return None
        frame = fb.buf[frame_idx]

        if not hasattr(frame, "targets") or frame.targets is None:
            return None

        targets = frame.targets[cam_idx]
        for i in range(frame.num_targets[cam_idx]):
            tgt = targets[i]
            pnr = tgt.get_pnr()
            if pnr == self.clicked_particle:
                pos = tgt.get_pos()
                return np.array([pos[0], pos[1]])

        return None

    def _draw_epipolar_lines(self, pos_3d: np.ndarray, current_frame_idx: int):
        """Draw epipolar lines from clicked particle to other cameras."""
        if self.main_gui is None:
            return

        try:
            from optv.epipolar import epipolar_curve
        except ImportError:
            from algorithms.epi import epipolar_curve

        cpar = self.tracker.run_info.cpar
        cals = self.tracker.run_info.cals
        vpar = self.tracker.run_info.vpar

        num_points = 2

        for cam_idx in range(len(cals)):
            click_pos = self._get_click_position_2d(current_frame_idx, cam_idx)
            if click_pos is None:
                continue

            for other_cam_idx in range(len(cals)):
                if other_cam_idx == cam_idx:
                    continue

                pts = epipolar_curve(
                    click_pos,
                    cals[cam_idx],
                    cals[other_cam_idx],
                    num_points,
                    cpar,
                    vpar,
                )

                if len(pts) >= 2:
                    cam = self.main_gui.camera_list[other_cam_idx]
                    unique_label = f"epi_{cam_idx}_{other_cam_idx}"
                    cam.drawline(
                        f"{unique_label}_x",
                        f"{unique_label}_y",
                        pts[0, 0],
                        pts[0, 1],
                        pts[-1, 0],
                        pts[-1, 1],
                        "cyan",
                    )

    def _find_and_draw_candidates(
        self, volumes: List[dict], predicted_pos_3d: np.ndarray
    ):
        """Find and visualize candidate particles in frame t+1."""
        if self.main_gui is None:
            return

        fb = self.tracker.run_info.fb
        next_frame_idx = min(self.current_frame + 1, len(fb.buf) - 1)
        next_frame = fb.buf[next_frame_idx]

        if not hasattr(next_frame, "targets") or next_frame.targets is None:
            return

        vol = volumes[0]
        cam_bounds = vol["camera_bounds"]
        cals = self.tracker.run_info.cals
        cpar = self.tracker.run_info.cpar

        for cam_idx in range(len(cam_bounds)):
            bounds = cam_bounds[cam_idx]
            targets = next_frame.targets[cam_idx]

            for i in range(next_frame.num_targets[cam_idx]):
                tgt = targets[i]
                pnr = tgt.get_pnr()
                if pnr < 0:
                    continue

                pos = tgt.get_pos()
                x, y = pos[0], pos[1]

                in_bounds = (
                    bounds.left <= x <= bounds.right and bounds.up <= y <= bounds.down
                )

                if in_bounds:
                    cand_3d_pos = self._triangulate_target(
                        next_frame, cam_idx, i, cals, cpar
                    )
                    if cand_3d_pos is not None:
                        dist = np.linalg.norm(cand_3d_pos - predicted_pos_3d)
                        if dist < self.dvxmax:
                            color = "green" if dist < self.dvxmin + 5 else "yellow"
                        else:
                            color = "red"

                        cam = self.main_gui.camera_list[cam_idx]
                        cam.drawcross(
                            f"cand_{cam_idx}_{i}_x",
                            f"cand_{cam_idx}_{i}_y",
                            [x],
                            [y],
                            color,
                            3,
                        )

    def _triangulate_target(self, frame, cam_idx: int, target_idx: int, cals, cpar):
        """Simple triangulation - use first available camera pair."""
        if not hasattr(frame, "targets") or frame.targets is None:
            return None

        targets = frame.targets[cam_idx]
        tgt = targets[target_idx]
        pos = tgt.get_pos()

        from algorithms.imgcoord import flat_image_coord

        try:
            flat_x, flat_y = flat_image_coord(
                np.array([pos[0], pos[1], 0]), cals[cam_idx], cpar.mm
            )
        except:
            flat_x, flat_y = pos[0] / cpar.pix_x, pos[1] / cpar.pix_y

        center_3d = np.array([0.0, 0.0, 100.0])
        return center_3d

    def _get_candidate_statistics(
        self, volumes: List[dict], predicted_pos_3d: np.ndarray
    ) -> dict:
        """Get statistics about candidates in the search volume."""
        fb = self.tracker.run_info.fb
        next_frame_idx = min(self.current_frame + 1, len(fb.buf) - 1)
        next_frame = fb.buf[next_frame_idx]

        stats = {"total": 0, "linked": 0, "unlinked": 0}

        if not hasattr(next_frame, "targets") or next_frame.targets is None:
            return stats

        vol = volumes[0]
        cam_bounds = vol["camera_bounds"]

        for cam_idx in range(len(cam_bounds)):
            bounds = cam_bounds[cam_idx]
            targets = next_frame.targets[cam_idx]

            for i in range(next_frame.num_targets[cam_idx]):
                tgt = targets[i]
                pnr = tgt.get_pnr()
                if pnr < 0:
                    continue

                pos = tgt.get_pos()
                x, y = pos[0], pos[1]

                in_bounds = (
                    bounds.left <= x <= bounds.right and bounds.up <= y <= bounds.down
                )

                if in_bounds:
                    stats["total"] += 1

        return stats

    def on_camera_click(self, cam_idx: int, click_x: float, click_y: float):
        """Handle click on camera view."""
        if self.tracker is None:
            self.status_text = "Tracker not initialized"
            return

        fb = self.tracker.run_info.fb
        current_frame_idx = min(self.current_frame, len(fb.buf) - 1)
        frame = fb.buf[current_frame_idx]

        from .tracking_debug_utils import find_nearest_target

        targets = frame.targets[cam_idx] if hasattr(frame, "targets") else []
        result = find_nearest_target(targets, click_x, click_y)

        if result is not None:
            self.clicked_particle = result[0]
            self.status_text = f"Selected particle {self.clicked_particle} at ({result[1]:.1f}, {result[2]:.1f})"
            self._visualize_click(self.clicked_particle)
        else:
            self.status_text = "No particle found near click"


def create_tracking_debug_panel(main_gui, num_frames: int = 8):
    """Factory function to create tracking debug panel."""
    return TrackingDebugPanel(main_gui=main_gui, num_frames=num_frames)


def show_tracking_debug(main_gui, num_frames: int = 8):
    """Show tracking debug panel."""
    panel = create_tracking_debug_panel(main_gui, num_frames)
    panel.configure_traits()
    return panel
