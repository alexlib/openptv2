import pytest
import numpy as np
import os
import shutil
from pathlib import Path

from openptv2.algorithms.compat.calibration import Calibration
from openptv2.algorithms.compat.parameters import (
    ControlParams, VolumeParams, TrackingParams, SequenceParams
)
from openptv2.algorithms.compat.tracker import Tracker


def read_all_calibration(num_cams, base_path="test_data/track"):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration()
        cal.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


def test_tracker_3d_compat_loop():
    original = os.getcwd()
    try:
        test_dir = os.path.join(os.path.dirname(__file__), '../../test_data/track')
        os.chdir(test_dir)
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        # Read control parameters to find correct number of cameras
        cpar = ControlParams(num_cams=2)
        cpar.read_control_par("parameters/ptv.par")
        num_cams = cpar.get_num_cams()

        vpar = VolumeParams()
        vpar.read_volume_par("parameters/criteria.par")

        tpar = TrackingParams()
        tpar.read_track_par("parameters/track.par")
        # Ensure track_mode is 1 (3D Segment Tracking) and add is 0
        tpar.set_track_mode(1)
        tpar.set_add(0)

        spar = SequenceParams(num_cams=num_cams)
        spar.read_sequence_par("parameters/sequence.par", num_cams)

        cals = read_all_calibration(num_cams, base_path=".")

        # Initialize the tracker
        tracker = Tracker(cpar, vpar, tpar, spar, cals)
        
        # Test full forward 3D tracking
        tracker.full_forward_3d()

        # Verify that tracker._run accumulated statistics
        assert tracker._run is not None
        assert tracker._run.npart > 0
        assert tracker._run.nlinks > 0

    finally:
        os.chdir(original)


def test_tracker_3d_compat_step_forward():
    original = os.getcwd()
    try:
        test_dir = os.path.join(os.path.dirname(__file__), '../../test_data/track')
        os.chdir(test_dir)
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = ControlParams(num_cams=2)
        cpar.read_control_par("parameters/ptv.par")
        num_cams = cpar.get_num_cams()

        vpar = VolumeParams()
        vpar.read_volume_par("parameters/criteria.par")

        tpar = TrackingParams()
        tpar.read_track_par("parameters/track.par")
        tpar.set_track_mode(1)
        tpar.set_add(0)

        spar = SequenceParams(num_cams=num_cams)
        spar.read_sequence_par("parameters/sequence.par", num_cams)

        cals = read_all_calibration(num_cams, base_path=".")

        tracker = Tracker(cpar, vpar, tpar, spar, cals)
        tracker.restart()

        # Run step_forward_3d in a loop
        has_more = True
        steps = 0
        while has_more:
            has_more = tracker.step_forward_3d()
            steps += 1

        assert steps > 0
        assert tracker._run.npart > 0

    finally:
        os.chdir(original)
