"""Unit tests for MyPTV 3D and 2D tracking plugins."""

import numpy as np
import pytest
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker, Tracking as MyPTV3DTrackingPlugin
from openptv2.plugins.myptv_2d_tracking import MyPTV2DTracker, Tracking as MyPTV2DTrackingPlugin
from openptv2.plugins.loader import resolve_plugin_module, BUILTIN_TRACKING_PLUGINS


def test_myptv_3d_tracker_synthetic_linear_motion():
    tracker = MyPTV3DTracker(v_max=5.0, a_max=10.0, max_gap=1, dt=0.1)

    # 10 frames of linear particle motion: p(t) = p0 + v * t
    v1 = np.array([2.0, 1.0, 0.0])
    v2 = np.array([-1.0, 3.0, 0.5])

    frames = []
    for f in range(10):
        t = f * 0.1
        p1 = t * v1
        p2 = np.array([10.0, 0.0, 0.0]) + t * v2
        frames.append(np.array([p1, p2]))

    results = tracker.track_frames(frames)

    # Should find 2 continuous trajectories of length 10
    assert len(results) == 2
    lens = sorted([len(tr["pos"]) for tr in results])
    assert lens == [10, 10]


def test_myptv_2d_tracker_synthetic_pixel_motion():
    tracker = MyPTV2DTracker(max_pixel_disp=15.0, max_gap=1)

    # 2D pixel trajectory
    frames_2d = []
    for f in range(8):
        pt1 = np.array([100.0 + f * 5.0, 200.0 + f * 2.0])
        pt2 = np.array([300.0 - f * 3.0, 400.0 + f * 4.0])
        frames_2d.append(np.array([pt1, pt2]))

    results = tracker.track_2d_blobs(frames_2d)

    assert len(results) == 2
    lens = sorted([len(tr["pos_2d"]) for tr in results])
    assert lens == [8, 8]


def test_myptv_plugins_plugin_loader_resolution():
    mod_3d = resolve_plugin_module("myptv_3d_tracking", BUILTIN_TRACKING_PLUGINS)
    assert hasattr(mod_3d, "Tracking")

    mod_2d = resolve_plugin_module("myptv_2d_tracking", BUILTIN_TRACKING_PLUGINS)
    assert hasattr(mod_2d, "Tracking")
