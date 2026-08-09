"""Ground-truth quality regression floor for the default tracker.

Fixture: ``test_data/synthetic_turbulent_1k`` (~1000 particles/frame, 30
frames; see scripts/create_synthetic_turbulent.py --density 1000 and
docs/plans/master-plan.md Stage 0). Unlike
tests/unit/test_track3d.py, which pins a raw *link count* on test_cavity,
this pins link-level *correctness* against exact ground truth (see
src/openptv2/tracking_metrics.py's yield/precision, which requires both
endpoints of a predicted link to match the same true link).

Floors are set from the measured baseline (fast_3d, default BASE_OVERRIDES),
with margin for incidental rebuild/tie-break drift -- not aspirational
targets. Raised once already after Stage 1b (global cost-ordered claiming
within a level instead of particle-by-particle in index order) measurably
improved precision 0.718 -> 0.871 and recall 0.648 -> 0.812 at this density.
A future quality improvement is expected to raise these floors further; a
drop below them without an intentional algorithm change is a regression.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import benchmark_utils as bu  # noqa: E402
from create_synthetic_turbulent import make_dataset  # noqa: E402

DATASET = Path(__file__).resolve().parents[2] / "test_data" / "synthetic_turbulent_1k"


@pytest.mark.slow
def test_fast_3d_quality_floor_at_1k_density():
    # res/img are gitignored repo-wide (test_data/**/res/, **/*_targets) --
    # only cal/ and parameters_Run1.yaml are checked in. Regenerate the rest
    # deterministically (fixed seed) on a fresh checkout.
    if not (DATASET / "res").exists():
        make_dataset(DATASET, num_particles=1000, num_frames=bu.N_FRAMES, seed=2026)

    results = bu.run_all_trackers(
        ["fast_3d"], silent=True, src=DATASET, first=bu.FIRST, n_frames=bu.N_FRAMES,
    )
    row = results["fast_3d"]["row"]
    assert row is not None, results["fast_3d"].get("error")

    # Measured baseline (post Stage 1b): precision 0.871, yield_recall
    # 0.812, ghost_capture 0.038 (see docstring). Floors keep ~8-9% margin
    # below the measurement.
    assert row["precision"] >= 0.80, row
    assert row["yield_recall"] >= 0.75, row
    assert row["ghost_capture_rate"] <= 0.10, row
