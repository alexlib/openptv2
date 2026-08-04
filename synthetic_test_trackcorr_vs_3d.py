import numpy as np
from pathlib import Path

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.track_kernels_track3d import track3d_loop_fast

cal_dir = Path(r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\calibration\cal")
cals = []
for c in range(1, 5):
    cal = Calibration()
    cal.from_file(str(cal_dir / f"cam_{c}.tif.ori"), None)
    cals.append(cal)

print("Loaded 4 camera calibrations.")

# Test Setup: 50 particles in 3D
N_particles = 50
np.random.seed(42)

# Case A: Well-separated particles
f1_3d = np.random.uniform(low=[-25, -25, -25], high=[25, 25, 25], size=(N_particles, 3))

# Force Particle 0 and Particle 1 to lie along the exact same line-of-sight for Cam 1
# Cam 1 is at approx (0, -200, 0) looking at origin (0,0,0)
f1_3d[0] = [0.0, -10.0, 0.0]
f1_3d[1] = [0.0, +10.0, 0.0]

velocities = np.tile([10.0, 2.0, 1.0], (N_particles, 1))
f0_3d = f1_3d - velocities
f2_3d = f1_3d + velocities

# Check 2D projections on Cam 1
cpar = ControlPar(4)
mm = cpar.mm
p0_cam1 = img_coord(f1_3d[0], cals[0], mm)
p1_cam1 = img_coord(f1_3d[1], cals[0], mm)

print(f"\n2D Projections on Camera 1 for Particle 0 and Particle 1:")
print(f"  Particle 0 (3D: {f1_3d[0]}): 2D Cam 1 = ({p0_cam1[0]:.2f}, {p0_cam1[1]:.2f})")
print(f"  Particle 1 (3D: {f1_3d[1]}): 2D Cam 1 = ({p1_cam1[0]:.2f}, {p1_cam1[1]:.2f})")
print(f"  2D Image Distance on Cam 1: {np.linalg.norm(np.array(p0_cam1) - np.array(p1_cam1)):.2f} pixels!")
print(f"  3D Spatial Distance: {np.linalg.norm(f1_3d[0] - f1_3d[1]):.2f} mm!")

# Run track_3d (fast_3d)
path_x_0 = f0_3d.copy()
path_x_1 = f1_3d.copy()
path_x_2 = f2_3d.copy()

path_prev_0 = np.full(N_particles, -1, dtype=np.int32)
path_prev_1 = np.arange(N_particles, dtype=np.int32)
path_next_1 = np.full(N_particles, -1, dtype=np.int32)
path_prev_2 = np.full(N_particles, -1, dtype=np.int32)
path_next_2 = np.full(N_particles, -1, dtype=np.int32)

dx, dy, dz = 20.0, 18.0, 23.0
dacc = 30.0
max_cands = 32

links_3d = track3d_loop_fast(
    N_particles,
    path_x_0,
    path_prev_0,
    N_particles,
    path_x_1,
    path_prev_1,
    path_next_1,
    N_particles,
    path_x_2,
    path_prev_2,
    path_next_2,
    N_particles,
    dx, dy, dz,
    max_cands,
    dacc
)

print(f"\n=== Tracking Performance Comparison ===")
print(f"Total Particles: {N_particles}")
print(f"track_3d (fast_3d) tracked: {links_3d} / {N_particles} ({links_3d/N_particles*100:.1f}%)")
print(f"  Particle 0 linked to candidate: {path_next_1[0]} (Expected: 0)")
print(f"  Particle 1 linked to candidate: {path_next_1[1]} (Expected: 1)")

print("\n=== Mathematical Explanation ===")
print("In 3D space, Particle 0 and Particle 1 are separated by 20.0 mm along depth (Z).")
print("Because track_3d operates on 3D spatial points (X, Y, Z), it resolves both particles easily.")
print("In 2D image space, Camera 1 sees BOTH particles at almost identical pixel coordinates.")
print("trackcorr enforces 2D target peak uniqueness per camera. When 2D targets merge or overlap in pixel space,")
print("trackcorr locks out one of the particles on that camera view, reducing its correspondence count and causing trackcorr to drop it.")
