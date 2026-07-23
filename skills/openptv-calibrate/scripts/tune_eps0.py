#!/usr/bin/env python
"""Visual assistance for picking criteria.par's eps0 (epipolar-band half-width).

eps0 trades off two failure modes when finding correspondences across
cameras: too tight and real matches fall outside the band (quadruplets
degrade to triplets/pairs, or vanish); too loose and spurious matches get
accepted (inflated pair/triplet counts, garbage-in to point_positions()).
There is no formula for the right value -- it depends on the calibration's
actual RMS and the scene's particle density -- so this sweeps eps0 against
real detected targets from one frame and plots quad/triplet/pair counts,
so the "knee" of the quad curve (where it flattens out) is visible directly
instead of guessed at.

Run with:
  uv run python skills/openptv-calibrate/scripts/tune_eps0.py <dataset> [frame]

Requires: <dataset>/img/camN.<frame>_targets already exist (run detection on
a real sequence frame first) and a calibration already written to cal/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, VolumePar
from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.autocalibration import cam_files, _find_yaml
from openptv2.gui.ptv import read_targets

# Reasonable default sweep in mm; override by editing if your pix size is
# very different from ~0.02mm (each dataset's own pixel size is printed
# alongside so you can read the eps0(px) column instead of picking blind).
DEFAULT_SWEEP_MM = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]


def sweep(base: Path, frame: int, eps0_values=None):
    yaml_path = _find_yaml(base)
    if yaml_path is None:
        raise FileNotFoundError(f"no parameters_*.yaml found in {base}")
    cpar = ControlPar.from_yaml(str(yaml_path))
    vpar = VolumePar.from_yaml(str(yaml_path))
    nc = cpar.num_cams

    cals = []
    for i in range(nc):
        _, ori, addpar = cam_files(base, i)
        c = Calibration()
        c.from_file(str(ori), str(addpar))
        cals.append(c)

    detections, corrected = [], []
    for i in range(nc):
        t = read_targets(str(base / "img" / f"cam{i + 1}"), frame)
        if not t:
            raise RuntimeError(
                f"no targets for cam{i + 1} frame {frame} -- run detection first"
            )
        t.sort_y()
        detections.append(t)
        corrected.append(MatchedCoords(t, cpar, cals[i]))

    values = eps0_values or DEFAULT_SWEEP_MM
    rows = []  # (eps0_mm, eps0_px, quad, trip, pair)
    for eps0 in values:
        vpar.eps0 = eps0
        sorted_pos, _, _ = correspondences(detections, corrected, cals, vpar, cpar)
        quad, trip, pair = (sorted_pos[0].shape[1], sorted_pos[1].shape[1],
                            sorted_pos[2].shape[1])
        rows.append((eps0, eps0 / cpar.pix_x, quad, trip, pair))

    return rows, [len(d) for d in detections], cpar.pix_x


def main():
    if len(sys.argv) < 2:
        print("Usage: tune_eps0.py <dataset> [frame]", file=sys.stderr)
        return 1

    base = Path(sys.argv[1]).resolve()
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    rows, counts, pix_x = sweep(base, frame)

    print(f"frame {frame}: detections per cam = {counts}  (pixel size = {pix_x} mm)")
    print(f"{'eps0(mm)':<10}{'eps0(px)':<10}{'quad':<8}{'trip':<8}{'pair':<8}")
    for eps0_mm, eps0_px, quad, trip, pair in rows:
        print(f"{eps0_mm:<10.3f}{eps0_px:<10.2f}{quad:<8}{trip:<8}{pair:<8}")

    # Visual: where does the quad curve flatten out? That's the "as small as
    # possible but not too small" point -- tighter loses real quads, looser
    # only adds risk (more pairs/triplets, no meaningful quad gain).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps0_mm = [r[0] for r in rows]
    quad = [r[2] for r in rows]
    trip = [r[3] for r in rows]
    pair = [r[4] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(eps0_mm, quad, "o-", color="red", label="quadruplets (4 cams)", linewidth=2)
    ax1.plot(eps0_mm, trip, "o-", color="green", label="triplets (3 cams)")
    ax1.plot(eps0_mm, pair, "o-", color="gold", label="pairs (2 cams)")
    ax1.set_xlabel("eps0 [mm]")
    ax1.set_ylabel("count")
    ax1.set_title(f"Correspondence counts vs. eps0 -- frame {frame}\n"
                  "(colors match the GUI's own pair/triplet/quad overlay colors)")
    ax1.legend(loc="center right")
    ax1.grid(alpha=0.3)

    ax2 = ax1.secondary_xaxis("top", functions=(lambda x: x / pix_x, lambda x: x * pix_x))
    ax2.set_xlabel("eps0 [pixels]")

    fig.tight_layout()
    dest = base / "eps0_tuning.png"
    fig.savefig(dest, dpi=120)
    print(f"\nSaved {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
