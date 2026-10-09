"""Determine the ONE shared cc from multi-plane consistency.

For a trial cc: fit each camera's pose on the reference frame, then for every
other frame ask each camera separately where the plate is.  If cc is right the
four answers coincide
if cc is wrong each camera's world frame sits at the
wrong distance and the answers spread apart, the more so the further the plate
is from the reference plane.  Minimise that spread.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np  # noqa: E402

from openptv2.plate_multiplane import fit_shared_cc, plate_spread  # noqa: E402

views = CFG.load_views()
geom = dict(imx=CFG.IMX, imy=CFG.IMY, pix_mm=CFG.PIX)
fit = fit_shared_cc(
    views,
    CFG.GRID,
    CFG.REF,
    CFG.NCAM,
    np.arange(7.6, 11.61, 0.20),
    bracket=0.20,
    **geom,
)

print("  cc [mm]   frames   median cross-camera spread of the plate centre [mm]")
for cc, n, s in fit.scan:
    print(f"   {cc:5.2f}     {n:3d}      {s:10.2f}")
was = plate_spread(9.44, views, CFG.GRID, CFG.REF, fit.frames, CFG.NCAM, **geom)[0]
print(
    f"\nbest shared cc = {fit.cc:.4f} mm   median spread {fit.spread_mm:.2f} mm "
    f"(was {was:.2f} mm at cc=9.44)"
)
np.save(CFG.DIR / "cal" / "fitted_cc.npy", fit.cc)
print(
    "\nframe      plate distance from reference plane [mm]   cross-camera spread [mm]"
)
for fr, dist, sp in sorted(fit.rows, key=lambda r: r[1])[:14]:
    print(f"{fr}          {dist:10.1f}                        {sp:8.2f}")
