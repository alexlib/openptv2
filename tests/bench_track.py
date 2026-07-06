"""Benchmark tracking pipeline."""
import os, shutil, time, sys
sys.path.insert(0, 'src')

os.chdir('test_data/track')
for d in ['res', 'img']:
    if os.path.exists(d): shutil.rmtree(d)
shutil.copytree('res_orig', 'res')
shutil.copytree('img_orig', 'img')

from openptv2.calibration import Calibration
from openptv2.parameters import get_control_par
from openptv2.tracking_run import tr_new
from openptv2.track import track_forward_start, trackcorr_c_loop

cpar = get_control_par("parameters/ptv.par")
calib = [Calibration.from_file(f"cal/cam{c+1}.tif.ori", f"cal/cam{c+1}.tif.addpar") 
         for c in range(cpar.num_cams)]

run = tr_new("parameters/sequence.par", "parameters/track.par", "parameters/criteria.par",
             "parameters/ptv.par", 4, 20000, "res/rt_is", "res/ptv_is", "res/added", calib, 0.0001)
run.tpar = run.tpar._replace(add=0)

print("Warming...")
track_forward_start(run)
for step in range(run.seq_par.first, run.seq_par.last):
    trackcorr_c_loop(run, step)

# Time one complete tracking sequence
times = []
for _ in range(5):
    shutil.rmtree('res', ignore_errors=True)
    shutil.copytree('res_orig', 'res')
    run = tr_new("parameters/sequence.par", "parameters/track.par", "parameters/criteria.par",
                 "parameters/ptv.par", 4, 20000, "res/rt_is", "res/ptv_is", "res/added", calib, 0.0001)
    run.tpar = run.tpar._replace(add=0)
    track_forward_start(run)
    t0 = time.perf_counter()
    for step in range(run.seq_par.first, run.seq_par.last):
        trackcorr_c_loop(run, step)
    t1 = time.perf_counter()
    times.append(t1 - t0)

times.sort()
print(f"Total (3 frames): median={times[len(times)//2]*1000:.1f}ms  "
      f"min={min(times)*1000:.1f}ms  max={max(times)*1000:.1f}ms")
