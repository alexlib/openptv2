"""End-to-end headless calibration on the real test_cavity dataset.

Exercises openptv2.autocalibration.calibrate_dataset without touching the
committed .ori/.addpar files (write=False). Asserts the reprojection RMS and
matched-point counts stay within the quality envelope established when the
turnkey calibrator was built.
"""
from pathlib import Path

import numpy as np
import pytest

from openptv2.autocalibration import calibrate_dataset

DATASET = Path("test_data/test_cavity")

# Physical ground truth for this dataset: cam1/cam2 have the lower target rows
# occluded by the black step, so their matched count is < 73 by design.
EXPECTED_MIN_MATCHED = {0: 40, 1: 40, 2: 70, 3: 70}


@pytest.mark.integration
@pytest.mark.skipif(not DATASET.exists(), reason="test_cavity dataset not present")
def test_autocalibrate_cavity():
    results = calibrate_dataset(DATASET, write=False, overlays=False)

    assert len(results) == 4
    for r in results:
        assert r.rms < 5.0, f"cam{r.cam + 1} RMS {r.rms:.3f}px too high"
        assert r.matched >= EXPECTED_MIN_MATCHED[r.cam], (
            f"cam{r.cam + 1} matched only {r.matched}/{r.nfix}"
        )
        # Every flag-set includes the exterior + principal-point terms.
        assert {"cc", "xh", "yh"}.issubset(set(r.flags))
        # Calibration must be finite.
        assert np.all(np.isfinite(r.cal.get_pos()))
        assert np.all(np.isfinite(r.cal.get_angles()))

    mean_rms = float(np.mean([r.rms for r in results]))
    assert mean_rms < 3.5, f"mean RMS {mean_rms:.3f}px regressed"
