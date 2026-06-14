from openptv2.tracking_framebuf import TargetArray, Target
from algorithms.segmentation import Target as SegTarget

# Test Target init with duck typing
seg_target = SegTarget(pnr=1, x=10.0, y=20.0, n=5, nx=2, ny=2, sumg=100, tnr=-1)
target_wrapper = Target(target=seg_target)
print("Target wrapper pnr:", target_wrapper.pnr())

# Test TargetArray wrapping a list of SegTargets
seg_targets = [
    SegTarget(pnr=1, x=10.0, y=20.0, n=5, nx=2, ny=2, sumg=100, tnr=-1),
    SegTarget(pnr=2, x=30.0, y=40.0, n=5, nx=2, ny=2, sumg=100, tnr=-1)
]
arr = TargetArray(seg_targets)

# Test __getitem__
t0 = arr[0]
print("arr[0] pnr:", t0.pnr(), "pos:", t0.pos())

# Test __setitem__
t_new = SegTarget(pnr=3, x=50.0, y=60.0, n=5, nx=2, ny=2, sumg=100, tnr=-1)
arr[1] = t_new
print("arr[1] after setitem pnr:", arr[1].pnr(), "pos:", arr[1].pos())

print("All tests passed!")
