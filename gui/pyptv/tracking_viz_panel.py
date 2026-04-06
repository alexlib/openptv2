"""
Tracking visualization panel for displaying tracking preview results.

This module provides a TraitsUI-based panel for visualizing tracking results
from the tracking preview functionality.
"""

import numpy as np
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


def show_tracking_preview(main_gui, num_frames: int = 5):
    """
    Run tracking preview and display the visualization panel.

    Args:
        main_gui: The MainGUI instance
        num_frames: Number of frames to preview

    Returns:
        TrackingVizPanel: The visualization panel with results
    """
    panel = create_tracking_viz_panel(main_gui, num_frames)
    panel.configure_traits()
    return panel
