import numpy as np
from algorithms.calibration import read_calibration
from algorithms.track import point_to_pixel
from algorithms.parameters import ControlPar, MultimediaPar
from pathlib import Path

CAVITY_DATA = Path("test_data/test_cavity")
ori = CAVITY_DATA / "cal/cam1.tif.ori"
add = CAVITY_DATA / "cal/cam1.tif.addpar"
cal = read_calibration(ori, add)

mm = MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1)
cpar = ControlPar(imx=1280, imy=1024, pix_x=0.008, pix_y=0.008, mm=mm)

# Particle 1 in frame 10001
p1_10001 = np.array([22.142, 41.030, 9.046])
px_10001 = point_to_pixel(p1_10001, cal, cpar)
print(f"P1 10001 proj: {px_10001}")

# Particle 1 in frame 10002
p1_10002 = np.array([21.955, 41.452, 8.638])
px_10002 = point_to_pixel(p1_10002, cal, cpar)
print(f"P1 10002 proj: {px_10002}")
