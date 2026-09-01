"""Write cal/calibration_block.txt for the configured plate and datum.

The block is openptv2's table of known 3D points: one line `id X Y Z` per plate
dot, in the same world the `.ori` use.  It has to agree exactly with the object
points every other step uses, or the GUI and the batch pipeline will be
measuring against a different plate than the one that was calibrated -- so it is
generated from the same `_config.obj_of` rather than written by hand.

Point id convention is row-major and 1-based from the bottom-left corner of the
lattice, `id = iy*nx + ix + 1`, and the datum dot (the coded L corner, whose
grid index comes from plate.yaml) sits at the origin.

    ILLMENAU_DIR=<folder> ILLMENAU_CAMS=5,6,7,8 python make_calibration_block.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np  # noqa: E402

dst = CFG.CAL / "calibration_block.txt"
CFG.CAL.mkdir(parents=True, exist_ok=True)

ids = np.arange(1, CFG.NX * CFG.NY + 1)
xyz = CFG.obj_of(ids)

lines = [f"{i} {p[0]:.1f} {p[1]:.1f} {p[2]:.1f}" for i, p in zip(ids, xyz)]
if dst.exists() and dst.read_text(encoding="utf-8").strip() == "\n".join(lines):
    print(f"{dst} already matches the configured plate; unchanged")
else:
    if dst.exists():
        dst.replace(dst.with_suffix(".txt.bak"))
        print(f"existing block backed up to {dst.name}.bak")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dst}")

datum_id = CFG.DATUM_IY * CFG.NX + CFG.DATUM_IX + 1
print(f"{len(ids)} points, {CFG.NX}x{CFG.NY} at {CFG.PITCH} mm pitch")
print(
    f"datum grid ({CFG.DATUM_IX},{CFG.DATUM_IY}) -> point id {datum_id}: "
    f"{lines[datum_id - 1]}"
)
assert np.allclose(xyz[datum_id - 1], 0.0), "datum dot is not at the origin"
print(f"first: {lines[0]}\nlast:  {lines[-1]}")
