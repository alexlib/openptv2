"""Unit tests for Quality3DTracker plugin (src/openptv2/plugins/quality_3d_tracking.py)."""

import numpy as np
import pytest

from openptv2.plugins.quality_3d_tracking import Quality3DTracker


def test_quality_3d_tracker_synthetic_flow():
    tracker = Quality3DTracker(v_max=10.0, a_max=5.0)

    # 3 linear trajectories across 5 frames
    frame0 = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    v1 = np.array([1.0, 0.5, 0.0])
    v2 = np.array([0.0, 1.0, 0.5])
    v3 = np.array([-0.5, 0.0, 1.0])

    frames = [
        frame0,
        frame0 + np.array([v1, v2, v3]),
        frame0 + 2 * np.array([v1, v2, v3]),
        frame0 + 3 * np.array([v1, v2, v3]),
        frame0 + 4 * np.array([v1, v2, v3]),
    ]

    trajectories = tracker.track_frames(frames)

    # Should reconstruct 3 unbroken 5-step trajectories
    long_tracks = [tr for tr in trajectories if len(tr["time"]) == 5]
    assert len(long_tracks) == 3


def test_quality_3d_tracker_empty_frames():
    tracker = Quality3DTracker()
    assert tracker.track_frames([]) == []
    assert len(tracker.track_frames([np.empty((0, 3)), np.empty((0, 3))])) == 0


def test_quality_3d_track_directory_integration(tmp_path):
    import shutil
    from pathlib import Path
    from openptv2.benchmarking.runner import read_trajectories

    src = Path("test_data/synthetic_turbulent")
    for sub in ("cal", "res", "img"):
        shutil.copytree(src / sub, tmp_path / sub)
    shutil.copy(src / "parameters_Run1.yaml", tmp_path / "parameters_Run1.yaml")

    tracker = Quality3DTracker(v_max=6.0, a_max=6.0)
    tracker.track_directory(tmp_path)

    trajectories = read_trajectories(tmp_path / "res", 10001, 10030)
    assert len(trajectories) > 0

