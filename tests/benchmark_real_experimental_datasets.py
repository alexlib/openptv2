"""Real Experimental Dataset Benchmark Script for OpenPTV2 Trackers.

Evaluates tracker performance on real experimental flow datasets:
- test_cavity (3D cavity fluid flow)
- burgers (3D Burgers vortex flow)
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np

from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.calibration import Calibration
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.tracker import Tracker


def read_all_calibration(num_cams, base_path="."):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration()
        cal.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


def benchmark_cavity_dataset():
    repo_root = Path(__file__).parent.parent
    cavity_dir = repo_root / "test_data" / "test_cavity"

    if not cavity_dir.exists():
        print(f"Skipping cavity dataset: {cavity_dir} not found")
        return

    orig_cwd = os.getcwd()
    try:
        os.chdir(cavity_dir)

        # Reset res directory
        res_dir = cavity_dir / "res"
        res_orig = cavity_dir / "res_orig"
        if res_dir.exists():
            shutil.rmtree(res_dir)
        shutil.copytree(res_orig, res_dir)

        cpar = ControlPar.from_yaml("parameters.yaml")
        vpar = VolumePar.from_yaml("parameters.yaml")
        tpar = TrackPar.from_yaml("parameters.yaml")
        spar = SequencePar.from_yaml("parameters.yaml")
        cals = read_all_calibration(cpar.num_cams, base_path=".")

        print("\n" + "=" * 90)
        print(
            f"--- Real Experimental Dataset Benchmark: TEST_CAVITY ({spar.first} .. {spar.last}) ---"
        )
        print("=" * 90)

        # 1. OpenPTV2 Classic Tracker
        tracker = Tracker(cpar, vpar, tpar, spar, cals)
        t0 = time.perf_counter()
        tracker.full_forward_3d()
        t_openptv = max(time.perf_counter() - t0, 1e-6)

        links_openptv = tracker.nlinks
        parts_openptv = tracker.npart
        fps_openptv = (spar.last - spar.first) / t_openptv

        print(
            f"{'OpenPTV2 Classic Tracker':<30} | Links: {links_openptv:5d} | Parts: {parts_openptv:5d} | Speed: {fps_openptv:7.1f} FPS"
        )

        # 2. MyPTV 3D Tracker on same cavity data
        from openptv2.algorithms.tracking_frame_buf import Frame

        frames = []
        frame_particles = []
        for fn in range(spar.first, spar.last + 1):
            fr = Frame(cpar.num_cams, 20000)
            fr.read(
                "res/rt_is",
                "",
                prio_file_base="res/added",
                target_file_base="",
                frame_num=fn,
            )
            frames.append(fr)
            frame_particles.append(fr.positions())

        myptv = MyPTV3DTracker(
            v_max=float(tpar.dvxmax), a_max=float(tpar.dacc), max_gap=1, dt=1.0
        )
        t0 = time.perf_counter()
        trajectories = myptv.track_frames(frame_particles)
        t_myptv = max(time.perf_counter() - t0, 1e-6)

        myptv_links = sum(len(tr["pos"]) - 1 for tr in trajectories)
        fps_myptv = (spar.last - spar.first) / t_myptv

        print(
            f"{'MyPTV 3D Tracker Baseline':<30} | Links: {myptv_links:5d} | Parts: {parts_openptv:5d} | Speed: {fps_myptv:7.1f} FPS"
        )
        print("=" * 90)

    finally:
        os.chdir(orig_cwd)


def benchmark_burgers_dataset():
    repo_root = Path(__file__).parent.parent
    burgers_dir = repo_root / "test_data" / "burgers"

    if not burgers_dir.exists():
        print(f"Skipping burgers dataset: {burgers_dir} not found")
        return

    orig_cwd = os.getcwd()
    try:
        os.chdir(burgers_dir)

        # Reset res directory
        res_dir = burgers_dir / "res"
        res_orig = burgers_dir / "res_orig"
        if res_dir.exists():
            shutil.rmtree(res_dir)
        shutil.copytree(res_orig, res_dir)

        cpar = ControlPar.from_yaml("parameters.yaml")
        vpar = VolumePar.from_yaml("parameters.yaml")
        tpar = TrackPar.from_yaml("parameters.yaml")
        spar = SequencePar.from_yaml("parameters.yaml")
        cals = read_all_calibration(cpar.num_cams, base_path=".")

        print("\n" + "=" * 90)
        print(
            f"--- Real Experimental Dataset Benchmark: BURGERS ({spar.first} .. {spar.last}) ---"
        )
        print("=" * 90)

        # 1. OpenPTV2 Classic Tracker
        tracker = Tracker(cpar, vpar, tpar, spar, cals)
        t0 = time.perf_counter()
        tracker.full_forward_3d()
        t_openptv = max(time.perf_counter() - t0, 1e-6)

        links_openptv = tracker.nlinks
        parts_openptv = tracker.npart
        fps_openptv = (spar.last - spar.first) / t_openptv

        print(
            f"{'OpenPTV2 Classic Tracker':<30} | Links: {links_openptv:5d} | Parts: {parts_openptv:5d} | Speed: {fps_openptv:7.1f} FPS"
        )

        # 2. MyPTV 3D Tracker
        from openptv2.algorithms.tracking_frame_buf import Frame

        frames = []
        frame_particles = []
        for fn in range(spar.first, spar.last + 1):
            fr = Frame(cpar.num_cams, 20000)
            fr.read(
                "res/rt_is",
                "",
                prio_file_base="res/added",
                target_file_base="",
                frame_num=fn,
            )
            frames.append(fr)
            frame_particles.append(fr.positions())

        myptv = MyPTV3DTracker(
            v_max=float(tpar.dvxmax), a_max=float(tpar.dacc), max_gap=1, dt=1.0
        )
        t0 = time.perf_counter()
        trajectories = myptv.track_frames(frame_particles)
        t_myptv = max(time.perf_counter() - t0, 1e-6)

        myptv_links = sum(len(tr["pos"]) - 1 for tr in trajectories)
        fps_myptv = (spar.last - spar.first) / t_myptv

        print(
            f"{'MyPTV 3D Tracker Baseline':<30} | Links: {myptv_links:5d} | Parts: {parts_openptv:5d} | Speed: {fps_myptv:7.1f} FPS"
        )
        print("=" * 90)

    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    benchmark_cavity_dataset()
    benchmark_burgers_dataset()
