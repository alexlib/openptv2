import os
import time
from pathlib import Path

from openptv2.batch.pyptv_batch import (
    build_processing_experiment,
    validate_experiment_setup,
)
from openptv2.plugins import run_tracking_plugin
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker

yaml_path = Path(
    r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\wp1\parameters_wp1_sample.yaml"
)

print("=== Fair Side-by-Side Timing Benchmark of Tracking Engines ===")

exp_path = validate_experiment_setup(yaml_path)
os.chdir(exp_path)
exp = build_processing_experiment(yaml_path, 1, 10)

# 1. Benchmark fast_3d (track_mode = 1)
exp.track_par.track_mode = 1
t0 = time.perf_counter()
run_tracking_plugin("fast_3d", exp)
t1 = time.perf_counter()
time_fast3d = t1 - t0

# 2. Benchmark MyPTV 3D
frames = []
for f in range(exp.spar.first, exp.spar.last + 1):
    exp.tracker._run.fb.read_frame_at_end(f)
    pts = (
        exp.tracker._run.fb.buf[3].path_x[: exp.tracker._run.fb.buf[3].num_parts].copy()
    )
    frames.append(pts)

tracker_myptv = MyPTV3DTracker(v_max=20.0, a_max=30.0, max_gap=1, dt=1.0)
t0 = time.perf_counter()
trajs_myptv = tracker_myptv.track_frames(frames)
t1 = time.perf_counter()
time_myptv = t1 - t0

links_myptv = sum(len(t["pos"]) - 1 for t in trajs_myptv) / float(len(frames) - 1)

print("\n=== Exact Benchmark Summary ===")
print(f"fast_3d (track_3d): {time_fast3d:.3f} s")
print(f"myptv_3d:          {time_myptv:.3f} s ({links_myptv:.1f} links/step)")
