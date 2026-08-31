"""Which grid node does the coded L corner occupy?  Read it off the data.

The datum -- the coded L-corner dot -- is what pins the world to a physical piece
of the plate, and its grid index `(ix, iy)` is an input to every later step
(`label_plate(corner_index=...)`, the object-point shift, calibration_block.txt).
Guessing it silently offsets the whole world by a multiple of the pitch, so the
tutorial says to verify it from the data.  This is that check.

How it works: on a view that sees the COMPLETE nx x ny lattice, labelling with
`corner_index=None` anchors the grid on the smallest detected index, which is
then genuinely 0 -- so the coded corner's index comes out absolute and correct.
That is only true when nothing is missing, which is exactly why the anchoring is
unsafe in general (see plate_labeler.label_coded_6x7) and why this script uses
complete views only.

Run it before anything else on a new camera group, and put the answer in
plate.yaml:datum.

    ILLMENAU_DIR=<folder> ILLMENAU_CAMS=5,6,7,8 python find_datum.py [--frames N]
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from openptv2.detect_plate import (  # noqa: E402
    detect_plate_targets,
    plate_tpar_from_yaml,
)
from openptv2.plate_labeler import label_plate  # noqa: E402

MAX_FRAMES = 8          # complete views to collect per camera
MAX_SCAN = 12           # images to open per camera before giving up
for i, a in enumerate(sys.argv):
    if a == "--frames":
        MAX_FRAMES = int(sys.argv[i + 1])
    elif a == "--scan":
        MAX_SCAN = int(sys.argv[i + 1])

CODED_THR = (30, 25, 20, 15, 10)
tpar = plate_tpar_from_yaml(CFG.DIR / "parameters_Run1.yaml")
cpar = CFG.control_par()
full = CFG.NX * CFG.NY

print(CFG.banner())
print(f"\nlooking for views that see all {full} dots, so the grid anchors at a true 0\n")
print("  cam  frame       dots   coded L corner at (ix, iy)")

votes = Counter()
for ci in range(CFG.NCAM):
    seen = scanned = 0
    for f in sorted(CFG.image_dir(ci).glob("*.tif*")):
        # Bounded on BOTH counts.  A camera group whose views never show the
        # complete lattice -- the far wall of this rig is one -- would otherwise
        # open every image at five detector thresholds and look like a hang.
        if seen >= MAX_FRAMES or scanned >= MAX_SCAN:
            break
        scanned += 1
        fr = f.name.split("_")[0]
        try:
            img = np.array(Image.open(f))
            res = None
            for thr in CODED_THR:
                r = detect_plate_targets(img, tpar, cpar, cam=ci, coded_thr=float(thr))
                if int(r.coded_mask.sum()) == 3:
                    res = r
                    break
            if res is None or len(res.centroids) < full:
                continue
            # deliberately NO corner_index here: that is the whole point
            ip, _, idx = label_plate(res.centroids, res.coded_mask,
                                     pitch_x=CFG.PITCH, pitch_y=CFG.PITCH,
                                     nx=CFG.NX, ny=CFG.NY, y_sign=1)
        except Exception as e:                       # noqa: BLE001 - report and move on
            print(f"  {CFG.cam_number(ci)}    {fr}   {type(e).__name__}: {e}")
            continue
        if len(ip) < full:
            continue
        # the corner is the coded dot whose two partners sit at 1 and 2 pitch
        coded = res.centroids[res.coded_mask]
        d = np.linalg.norm(coded[:, None, :] - coded[None, :, :], axis=2)
        corner_xy = coded[int(np.argmin(np.sort(d, axis=1)[:, 1:].sum(1)))]
        k = int(np.argmin(np.linalg.norm(ip - corner_xy, axis=1)))
        ixiy = (int(idx[k, 0]), int(idx[k, 1]))
        votes[ixiy] += 1
        seen += 1
        print(f"  {CFG.cam_number(ci)}    {fr}   {len(ip):4d}   {ixiy}", flush=True)

print()
if not votes:
    raise SystemExit(
        "no view saw the complete lattice, so the datum cannot be read off this way.\n"
        "Either capture a frame where the whole plate is visible in one camera, or "
        "identify the corner dot by eye on one image and enter its (ix, iy) manually.")
(best, n), = votes.most_common(1)
total = sum(votes.values())
print(f"datum grid index = {best}   ({n}/{total} complete views agree)")
if len(votes) > 1:
    print(f"  DISAGREEMENT: {dict(votes)}")
    print("  Do not proceed until this is one value -- a split vote means the L code "
          "is resolving to different dots in different views.")
else:
    ix, iy = best
    print(f"  put this in {CFG.DIR / 'plate.yaml'} under plate.datum:")
    print(f"      ix: {ix}\n      iy: {iy}")
    print(f"  point id of the datum dot = iy*nx + ix + 1 = {iy * CFG.NX + ix + 1}")
