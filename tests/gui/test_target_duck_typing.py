from openptv2.tracking_framebuf import TargetArray, Target
from openptv2.algorithms.segmentation import Target as SegTarget

def test_target_duck_typing():
    # Test Target init with duck typing
    seg_target = SegTarget(pnr=1, x=10.0, y=20.0, n=5, nx=2, ny=2, sumg=100, tnr=-1)
    target_wrapper = Target(target=seg_target)
    assert target_wrapper.pnr() == 1

    # Test TargetArray wrapping a list of SegTargets
    seg_targets = [
        SegTarget(pnr=1, x=10.0, y=20.0, n=5, nx=2, ny=2, sumg=100, tnr=-1),
        SegTarget(pnr=2, x=30.0, y=40.0, n=5, nx=2, ny=2, sumg=100, tnr=-1)
    ]
    arr = TargetArray(seg_targets)

    # Test __getitem__
    t0 = arr[0]
    assert t0.pnr() == 1
    assert t0.x() == 10.0
    assert t0.y() == 20.0

    # Test __setitem__
    t_new = SegTarget(pnr=3, x=50.0, y=60.0, n=5, nx=2, ny=2, sumg=100, tnr=-1)
    arr[1] = t_new
    assert arr[1].pnr() == 3
    assert arr[1].x() == 50.0
    assert arr[1].y() == 60.0
