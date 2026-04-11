
from __future__ import annotations
import pytest
import yaml
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

@pytest.mark.slow
@pytest.mark.parity
def test_tracking_parameters_from_data_statistics():
    """
    Tutorial: Choosing tracking parameters from basic data statistics.

    This test demonstrates how to set velocity, acceleration, and angle limits
    based on simple statistics of the dataset (as computed by a quick probe script):
      - max observed displacement per frame (velocity)
      - max observed acceleration
      - typical interparticle distance
    The chosen parameters are set just above the real motion, but below the ambiguity threshold.
    """
    # Example values from quick probe (replace with actual script output if available)
    max_disp = 0.08  # mm/frame (max observed displacement)
    max_acc = 0.09   # mm/frame^2 (max observed acceleration)
    interparticle_dist = 1.53  # mm (typical)

    # Set velocity window just above max displacement, but below interparticle distance
    vlim = round(max_disp * 1.1, 3)  # 10% margin
    vlim = min(vlim, interparticle_dist * 0.9)  # don't exceed 90% of spacing
    velocity_lims = [[-vlim, vlim], [-vlim, vlim], [-vlim, vlim]]

    # Set acceleration limit just above max observed
    accel_lim = round(max_acc * 1.1, 3)

    # Set angle limit to 20 gon (18 degrees), typical for smooth motion
    angle_lim = 20  # gon

    conf = yaml.safe_load((Path("test_data/burgers/conf.yaml")).read_text())
    baseline, _ = _run_forward_metrics(conf)

    stat_m, stat_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update({
            "velocity_lims": velocity_lims,
            "accel_lim": accel_lim,
            "angle_lim": angle_lim,
        }),
    )
    stat_report = ScenarioReport(
        scenario="parameters from data statistics",
        flag=FailureFlag.RECOVERED,
        tracking=stat_t,
        metrics=stat_m,
        detail=(
            f"velocity_lims set to {velocity_lims} (from max_disp={max_disp}), "
            f"accel_lim={accel_lim} (from max_acc={max_acc}), "
            f"angle_lim={angle_lim} gon (typical for smooth motion). "
            f"All values chosen just above real motion, below ambiguity threshold (interparticle_dist={interparticle_dist})."
        ),
    )
    # Allow for a small margin of missed links if statistics are tight
    assert stat_m.linked_real >= baseline.linked_real - 1, stat_report.as_text()


"""Burgers parameter-sensitivity test suite for tracking failure diagnostics.

This suite intentionally drives tracking into known failure modes and then
validates recovery by changing only the target parameter.

It provides explicit failure flags and detailed assertion reports so users can
understand *why* tracking failed:
- strict velocity windows (isotropic / non-isotropic)
- strict acceleration windows
- angle threshold effects and angle/acceleration coupling
- overly broad search regions causing ambiguity under cluttered candidates
"""


from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pytest
import yaml

from .test_burgers_tracking_value_parity import (
    _copy_burgers_workspace,
    _localize_conf,
    _parse_ptv,
    _parse_rt,
    _run_tracker,
)


FRAMES = [10001, 10002, 10003, 10004, 10005]
REAL_CORRS = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3), (4, 4, 4, 4)]


class FailureFlag(str, Enum):
    STRICT_VELOCITY_LIMIT = "STRICT_VELOCITY_LIMIT"
    NONISOTROPIC_COMPONENT_CLIP = "NONISOTROPIC_COMPONENT_CLIP"
    STRICT_ACCELERATION_LIMIT = "STRICT_ACCELERATION_LIMIT"
    ANGLE_MASKED_BY_ACCEL_SHORTCIRCUIT = "ANGLE_MASKED_BY_ACCEL_SHORTCIRCUIT"
    STRICT_ANGLE_LIMIT = "STRICT_ANGLE_LIMIT"
    TOO_MANY_OPTIONS_AMBIGUOUS = "TOO_MANY_OPTIONS_AMBIGUOUS"
    RECOVERED = "RECOVERED"


@dataclass
class TrackingMetrics:
    linked_real: int
    possible_real: int
    wrong_real_links: int
    p2_10004_next: int | None


@dataclass
class ScenarioReport:
    scenario: str
    flag: FailureFlag
    tracking: dict[str, Any]
    metrics: TrackingMetrics
    detail: str

    def as_text(self) -> str:
        return (
            f"scenario={self.scenario}\n"
            f"flag={self.flag.value}\n"
            f"tracking={self.tracking}\n"
            f"metrics="
            f"linked_real={self.metrics.linked_real}/{self.metrics.possible_real}, "
            f"wrong_real_links={self.metrics.wrong_real_links}, "
            f"p2_10004_next={self.metrics.p2_10004_next}\n"
            f"detail={self.detail}"
        )


def _clone_conf(conf: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(conf))


def _xyz_to_row_index(rows: list[dict[str, Any]], xyz, tol: float = 1e-9) -> int | None:
    for i, row in enumerate(rows):
        if (
            abs(row["xyz"][0] - xyz[0]) < tol
            and abs(row["xyz"][1] - xyz[1]) < tol
            and abs(row["xyz"][2] - xyz[2]) < tol
        ):
            return i
    return None


def _evaluate_real_tracking(out_dir: Path) -> TrackingMetrics:
    rt_by_frame: dict[int, dict[tuple[int, int, int, int], dict[str, Any]]] = {}
    ptv_by_frame: dict[int, list[dict[str, Any]]] = {}

    for f in FRAMES:
        _, rt_rows = _parse_rt(out_dir / f"rt_is.{f}")
        _, ptv_rows = _parse_ptv(out_dir / f"ptv_is.{f}")
        rt_by_frame[f] = {row["p"]: row for row in rt_rows}
        ptv_by_frame[f] = ptv_rows

    linked_real = 0
    possible_real = 0
    wrong_real_links = 0

    for i in range(len(FRAMES) - 1):
        f0 = FRAMES[i]
        f1 = FRAMES[i + 1]

        for corr in REAL_CORRS:
            if corr not in rt_by_frame[f0] or corr not in rt_by_frame[f1]:
                continue

            possible_real += 1
            xyz0 = rt_by_frame[f0][corr]["xyz"]
            row_idx = _xyz_to_row_index(ptv_by_frame[f0], xyz0)
            if row_idx is None:
                continue

            row = ptv_by_frame[f0][row_idx]
            if row["next"] < 0:
                continue

            linked_real += 1

            next_idx = row["next"]
            if not (0 <= next_idx < len(ptv_by_frame[f1])):
                wrong_real_links += 1
                continue

            xyz_expected = rt_by_frame[f1][corr]["xyz"]
            xyz_actual = ptv_by_frame[f1][next_idx]["xyz"]
            if (
                abs(xyz_expected[0] - xyz_actual[0]) > 1e-9
                or abs(xyz_expected[1] - xyz_actual[1]) > 1e-9
                or abs(xyz_expected[2] - xyz_actual[2]) > 1e-9
            ):
                wrong_real_links += 1

    p2_10004_next = None
    p2_corr = (2, 2, 2, 2)
    if p2_corr in rt_by_frame[10004]:
        p2_xyz = rt_by_frame[10004][p2_corr]["xyz"]
        p2_idx = _xyz_to_row_index(ptv_by_frame[10004], p2_xyz)
        if p2_idx is not None:
            p2_10004_next = ptv_by_frame[10004][p2_idx]["next"]

    return TrackingMetrics(
        linked_real=linked_real,
        possible_real=possible_real,
        wrong_real_links=wrong_real_links,
        p2_10004_next=p2_10004_next,
    )


def _run_forward_metrics(
    conf: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None] | None = None,
    augment: Callable[[Path], None] | None = None,
) -> tuple[TrackingMetrics, dict[str, Any]]:
    with TemporaryDirectory() as td:
        work = _copy_burgers_workspace(Path(td))
        if augment is not None:
            augment(work)

        local_conf = _localize_conf(_clone_conf(conf), work)
        if mutate is not None:
            mutate(local_conf)

        out = _run_tracker("python", "forward", work, local_conf)
        metrics = _evaluate_real_tracking(out)
        return metrics, local_conf["tracking"]


def _augment_with_distractors(work: Path, offset: float = 0.02) -> None:
    """Add one synthetic distractor near each particle in every frame.

    This preserves Burgers dynamics while making candidate density higher, which
    helps expose failures under overly broad search windows.
    """
    for f in FRAMES:
        rt_path = work / "res_orig" / f"rt_is.{f}"
        with open(rt_path) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

        n = int(lines[0])
        rows = lines[1 : 1 + n]

        added_rows = []
        for i, ln in enumerate(rows):
            t = ln.split()
            x, y, z = float(t[1]), float(t[2]), float(t[3])
            dx, dy, dz = x + offset, y - offset, z + offset
            added_rows.append(
                f"{n + i:4d} {dx:9.4f} {dy:9.4f} {dz:9.4f} 9999 9999 9999 9999"
            )

        with open(rt_path, "w") as fh:
            fh.write(str(n + len(added_rows)) + "\n")
            for row in rows:
                fh.write(row + "\n")
            for row in added_rows:
                fh.write(row + "\n")


@pytest.mark.slow
@pytest.mark.parity
def test_velocity_isotropic_intentional_failure_then_recovery():
    conf = yaml.safe_load((Path("test_data/burgers/conf.yaml")).read_text())

    baseline, _ = _run_forward_metrics(conf)

    fail_m, fail_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update(
            {"velocity_lims": [[-0.03, 0.03], [-0.03, 0.03], [-0.03, 0.03]]}
        ),
    )
    fail_report = ScenarioReport(
        scenario="isotropic velocity too strict",
        flag=FailureFlag.STRICT_VELOCITY_LIMIT,
        tracking=fail_t,
        metrics=fail_m,
        detail="Search windows are tighter than true per-frame displacement.",
    )

    rec_m, rec_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update(
            {"velocity_lims": [[-0.09, 0.09], [-0.09, 0.09], [-0.09, 0.09]]}
        ),
    )
    rec_report = ScenarioReport(
        scenario="isotropic velocity recovered",
        flag=FailureFlag.RECOVERED,
        tracking=rec_t,
        metrics=rec_m,
        detail="Wider windows include true motion and restore linking.",
    )

    assert fail_m.linked_real <= 1, fail_report.as_text()
    assert rec_m.linked_real == baseline.linked_real, rec_report.as_text()
    assert rec_m.p2_10004_next is not None and rec_m.p2_10004_next >= 0, rec_report.as_text()


@pytest.mark.slow
@pytest.mark.parity
def test_velocity_nonisotropic_component_clip_failure_then_recovery():
    conf = yaml.safe_load((Path("test_data/burgers/conf.yaml")).read_text())

    baseline, _ = _run_forward_metrics(conf)

    fail_m, fail_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update(
            {"velocity_lims": [[-0.04, 0.04], [-0.2, 0.2], [-0.2, 0.2]]}
        ),
    )
    fail_report = ScenarioReport(
        scenario="non-isotropic x component clipped",
        flag=FailureFlag.NONISOTROPIC_COMPONENT_CLIP,
        tracking=fail_t,
        metrics=fail_m,
        detail="x-bound clips true dx while y/z bounds remain permissive.",
    )

    rec_m, rec_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update(
            {"velocity_lims": [[-0.12, 0.12], [-0.2, 0.2], [-0.2, 0.2]]}
        ),
    )
    rec_report = ScenarioReport(
        scenario="non-isotropic x component recovered",
        flag=FailureFlag.RECOVERED,
        tracking=rec_t,
        metrics=rec_m,
        detail="Relaxing x-bound alone recovers expected links.",
    )

    assert fail_m.linked_real < baseline.linked_real - 10, fail_report.as_text()
    assert rec_m.linked_real == baseline.linked_real, rec_report.as_text()


@pytest.mark.slow
@pytest.mark.parity
def test_acceleration_strict_failure_then_recovery():
    conf = yaml.safe_load((Path("test_data/burgers/conf.yaml")).read_text())

    baseline, _ = _run_forward_metrics(conf)

    fail_m, fail_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update({"accel_lim": 0.001}),
    )
    fail_report = ScenarioReport(
        scenario="acceleration too strict",
        flag=FailureFlag.STRICT_ACCELERATION_LIMIT,
        tracking=fail_t,
        metrics=fail_m,
        detail="accel_lim is below observed acceleration envelope.",
    )

    rec_m, rec_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update({"accel_lim": 0.1}),
    )
    rec_report = ScenarioReport(
        scenario="acceleration recovered",
        flag=FailureFlag.RECOVERED,
        tracking=rec_t,
        metrics=rec_m,
        detail="Restoring accel_lim to baseline recovers link quality.",
    )

    assert fail_m.linked_real <= 1, fail_report.as_text()
    assert rec_m.linked_real == baseline.linked_real, rec_report.as_text()


@pytest.mark.slow
@pytest.mark.parity
def test_angle_non_intuitive_masking_and_effective_recovery():
    conf = yaml.safe_load((Path("test_data/burgers/conf.yaml")).read_text())

    baseline, _ = _run_forward_metrics(conf)

    masked_m, masked_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update({"angle_lim": 2, "accel_lim": 0.1}),
    )
    masked_report = ScenarioReport(
        scenario="angle tightened but masked by acceleration shortcut",
        flag=FailureFlag.ANGLE_MASKED_BY_ACCEL_SHORTCIRCUIT,
        tracking=masked_t,
        metrics=masked_m,
        detail=(
            "With high accel_lim, the `(acc < dacc/10)` branch can bypass angle gating, "
            "so very small angle_lim may not reduce links."
        ),
    )

    fail_m, fail_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update({"angle_lim": 1, "accel_lim": 0.03}),
    )
    fail_report = ScenarioReport(
        scenario="angle genuinely too strict (accel coupled)",
        flag=FailureFlag.STRICT_ANGLE_LIMIT,
        tracking=fail_t,
        metrics=fail_m,
        detail="Reducing accel shortcut exposure makes angle threshold active and restrictive.",
    )

    rec_m, rec_t = _run_forward_metrics(
        conf,
        mutate=lambda c: c["tracking"].update({"angle_lim": 10, "accel_lim": 0.03}),
    )
    rec_report = ScenarioReport(
        scenario="angle recovery under same acceleration",
        flag=FailureFlag.RECOVERED,
        tracking=rec_t,
        metrics=rec_m,
        detail="Relaxing only angle_lim substantially restores links under fixed accel_lim.",
    )

    assert masked_m.linked_real == baseline.linked_real, masked_report.as_text()
    assert fail_m.linked_real <= 10, fail_report.as_text()
    assert rec_m.linked_real >= fail_m.linked_real + 6, rec_report.as_text()


@pytest.mark.slow
@pytest.mark.parity
def test_overly_broad_search_window_ambiguity_failure_then_recovery():
    conf = yaml.safe_load((Path("test_data/burgers/conf.yaml")).read_text())

    base_m, base_t = _run_forward_metrics(conf, augment=_augment_with_distractors)

    fail_m, fail_t = _run_forward_metrics(
        conf,
        augment=_augment_with_distractors,
        mutate=lambda c: c["tracking"].update(
            {
                "velocity_lims": [[-20, 20], [-20, 20], [-20, 20]],
                "accel_lim": 10.0,
                "angle_lim": 200,
            }
        ),
    )
    fail_report = ScenarioReport(
        scenario="too-broad search window under candidate clutter",
        flag=FailureFlag.TOO_MANY_OPTIONS_AMBIGUOUS,
        tracking=fail_t,
        metrics=fail_m,
        detail=(
            "Very broad windows and permissive scoring increase ambiguity; "
            "a real-particle link regresses (P2 10004->10005 breaks)."
        ),
    )

    rec_m, rec_t = _run_forward_metrics(
        conf,
        augment=_augment_with_distractors,
        mutate=lambda c: c["tracking"].update(
            {
                "velocity_lims": [[-0.12, 0.12], [-0.12, 0.12], [-0.12, 0.12]],
                "accel_lim": 0.1,
                "angle_lim": 100,
            }
        ),
    )
    rec_report = ScenarioReport(
        scenario="ambiguity reduced by tighter windows",
        flag=FailureFlag.RECOVERED,
        tracking=rec_t,
        metrics=rec_m,
        detail="Tightening search bounds restores real-link quality under same clutter.",
    )

    assert (
        fail_m.linked_real < base_m.linked_real
        or fail_m.wrong_real_links > base_m.wrong_real_links
        or (fail_m.p2_10004_next is not None and fail_m.p2_10004_next < 0)
    ), fail_report.as_text()

    assert rec_m.linked_real >= base_m.linked_real, rec_report.as_text()
    assert rec_m.wrong_real_links <= base_m.wrong_real_links, rec_report.as_text()
