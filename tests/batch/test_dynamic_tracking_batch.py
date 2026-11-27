"""End-to-end coverage for opt-in dynamic (per-step) tracking parameters,
running through the real batch pipeline (pyptv_batch.main -> tracking
plugin -> py_trackcorr_init -> Tracker.step_forward), not mocked.

Complements tests/unit/test_dynamic_tracking.py, which covers
DynamicTrackParams/flag_low_quality_steps/etc. in isolation.
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

from openptv2.batch import pyptv_batch
from openptv2.dynamic_tracking import (
    flag_low_quality_steps,
    per_frame_link_rate,
    resolve_dynamic_params_path,
    suggest_step_overrides,
    write_dynamic_params_yaml,
)


def _env_with_pythonpath() -> dict:
    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent.parent / "src")
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{env['PYTHONPATH']}" if "PYTHONPATH" in env else src_dir
    )
    return env


def _update_track_cfg(yaml_path, **updates) -> None:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    data.setdefault("track", {}).update(updates)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)


def test_dynamic_tracking_absent_key_runs_static_unchanged(cavity_workdir, capsys):
    """No `track.dynamic_tracking` key -> no dynamic-tracking log line and
    the batch run behaves exactly as before (the backward-compat contract)."""
    yaml_file = cavity_workdir / "parameters_Run1.yaml"
    start_frame, end_frame = 10000, 10001

    pyptv_batch.main(yaml_file, start_frame, end_frame)

    out = capsys.readouterr().out
    assert "[dynamic tracking]" not in out

    res_dir = cavity_workdir / "res"
    for frame in range(start_frame, end_frame + 1):
        assert (res_dir / f"ptv_is.{frame}").exists()


def test_dynamic_tracking_enabled_missing_file_falls_back(cavity_workdir, capsys):
    """`dynamic_tracking: true` with no sidecar file present must warn and
    fall back to static parameters rather than crash the batch run."""
    yaml_file = cavity_workdir / "parameters_Run1.yaml"
    _update_track_cfg(yaml_file, dynamic_tracking=True)
    start_frame, end_frame = 10000, 10001

    pyptv_batch.main(yaml_file, start_frame, end_frame)

    out = capsys.readouterr().out
    assert "enabled but" in out and "not found" in out

    res_dir = cavity_workdir / "res"
    for frame in range(start_frame, end_frame + 1):
        assert (res_dir / f"ptv_is.{frame}").exists()


def test_dynamic_tracking_enabled_with_overrides_runs_end_to_end(cavity_workdir, capsys):
    """`dynamic_tracking: true` plus a sidecar overrides file is picked up by
    the real batch pipeline and still produces valid per-frame tracking
    output (the per-step swap doesn't corrupt the run)."""
    yaml_file = cavity_workdir / "parameters_Run1.yaml"
    start_frame, end_frame = 10000, 10001

    with open(yaml_file) as f:
        base_dvxmax = float(yaml.safe_load(f)["track"]["dvxmax"])

    dyn_path = cavity_workdir / "dynamic_track.yaml"
    write_dynamic_params_yaml(
        dyn_path,
        {start_frame: {"dvxmax": base_dvxmax * 2, "dvxmin": -base_dvxmax * 2}},
    )
    _update_track_cfg(yaml_file, dynamic_tracking=True)

    pyptv_batch.main(yaml_file, start_frame, end_frame)

    out = capsys.readouterr().out
    assert f"[dynamic tracking] per-step overrides loaded from {dyn_path}" in out

    res_dir = cavity_workdir / "res"
    for frame in range(start_frame, end_frame + 1):
        ptv_is_file = res_dir / f"ptv_is.{frame}"
        assert ptv_is_file.exists()
        lines = ptv_is_file.read_text().strip().splitlines()
        num_tracks = int(lines[0]) if lines else 0
        assert num_tracks > 0, f"No tracks found in {ptv_is_file}."


def test_dynamic_tracking_custom_params_file(cavity_workdir):
    """`track.dynamic_params_file` overrides the default sidecar filename."""
    yaml_file = cavity_workdir / "parameters_Run1.yaml"
    start_frame, end_frame = 10000, 10001

    custom_path = cavity_workdir / "custom_overrides.yaml"
    write_dynamic_params_yaml(custom_path, {start_frame: {"dacc": 5.0}})
    _update_track_cfg(
        yaml_file,
        dynamic_tracking=True,
        dynamic_params_file="custom_overrides.yaml",
    )

    # No exception raised -> the custom sidecar file was found and parsed.
    pyptv_batch.main(yaml_file, start_frame, end_frame)

    res_dir = cavity_workdir / "res"
    assert (res_dir / f"ptv_is.{start_frame}").exists()


def test_tune_dynamic_pipeline_end_to_end(cavity_workdir):
    """The `openptv tune-dynamic` mechanics, run directly against real batch
    output: static tracking pass -> per-frame link rate -> flag steps ->
    derive & write per-step overrides."""
    yaml_file = cavity_workdir / "parameters_Run1.yaml"
    start_frame, end_frame = 10000, 10004

    # Step 1: static tracking pass produces rt_is (positions) and ptv_is
    # (linkage) on disk.
    pyptv_batch.main(yaml_file, start_frame, end_frame)

    res_dir = cavity_workdir / "res"
    rates = per_frame_link_rate(res_dir / "ptv_is", start_frame, end_frame)
    assert rates, "static pass should have produced linkage data"
    assert set(rates) == set(range(start_frame, end_frame))

    # threshold=2.0 forces every step to flag, deterministically exercising
    # the "some steps flagged" path regardless of this dataset's own link
    # quality (which is normally high enough that nothing gets flagged).
    flagged = flag_low_quality_steps(rates, threshold=2.0)
    assert flagged == sorted(rates)

    with open(yaml_file) as f:
        num_cams = yaml.safe_load(f)["num_cams"]

    overrides = suggest_step_overrides(
        res_dir, flagged, start_frame, end_frame, num_cams=num_cams, window=2
    )
    assert set(overrides) == set(flagged)
    for step, step_overrides in overrides.items():
        assert step_overrides.get("dvxmax", 0) > 0, (
            f"step {step}: expected a positive suggested dvxmax, got {step_overrides}"
        )

    out_path = resolve_dynamic_params_path({}, cavity_workdir)
    write_dynamic_params_yaml(out_path, overrides)
    assert out_path.exists()

    written = yaml.safe_load(out_path.read_text())
    assert set(written["steps"]) == set(flagged)

    # Step 2: feed the generated file straight back into a real batch run.
    _update_track_cfg(yaml_file, dynamic_tracking=True)
    pyptv_batch.main(yaml_file, start_frame, end_frame)
    for frame in range(start_frame, end_frame + 1):
        assert (res_dir / f"ptv_is.{frame}").exists()


def test_tune_dynamic_cli_writes_overrides_file(cavity_workdir):
    """`openptv tune-dynamic` invoked as a real CLI subprocess: static pass,
    force-flag every step (--threshold 2.0, same determinism trick as
    test_tune_dynamic_pipeline_end_to_end), write the sidecar file.

    tune-dynamic has no --first/--last of its own -- it always operates over
    the YAML's own `sequence.first`/`sequence.last`, so the prep run below
    must cover that same range (not an arbitrary sub-range) for its static
    pass to find rt_is/ptv_is on disk for every step it will measure."""
    yaml_file = cavity_workdir / "parameters_Run1.yaml"
    with open(yaml_file) as f:
        seq_cfg = yaml.safe_load(f)["sequence"]
    start_frame, end_frame = int(seq_cfg["first"]), int(seq_cfg["last"])

    # Sequence + tracking must exist on disk before tune-dynamic can measure
    # link rate; pyptv_batch.main mode="both" (default) is a separate,
    # earlier stage from the CLI subprocess below.
    pyptv_batch.main(yaml_file, start_frame, end_frame)

    cmd = [
        sys.executable,
        "-m",
        "openptv2.cli",
        "tune-dynamic",
        str(yaml_file),
        "--threshold",
        "2.0",
        "--window",
        "2",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cavity_workdir,
        env=_env_with_pythonpath(),
    )
    assert result.returncode == 0, (
        f"tune-dynamic failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Wrote per-step overrides" in result.stdout
    assert "dynamic_tracking: true" in result.stdout

    out_path = cavity_workdir / "dynamic_track.yaml"
    assert out_path.exists()
    written = yaml.safe_load(out_path.read_text())
    assert set(written["steps"]) == set(range(start_frame, end_frame))


def test_tune_dynamic_cli_default_threshold_is_consistent_with_output_file(cavity_workdir):
    """Default (auto mean - 1 std) threshold, run through the real CLI: the
    reported outcome and the presence/absence of the sidecar file must agree
    with each other, whichever way this dataset's own link-rate distribution
    happens to fall (not asserting a specific frame count -- that would be
    testing this dataset's kinematics, not the tune-dynamic mechanism)."""
    yaml_file = cavity_workdir / "parameters_Run1.yaml"
    with open(yaml_file) as f:
        seq_cfg = yaml.safe_load(f)["sequence"]
    start_frame, end_frame = int(seq_cfg["first"]), int(seq_cfg["last"])

    pyptv_batch.main(yaml_file, start_frame, end_frame)

    cmd = [
        sys.executable,
        "-m",
        "openptv2.cli",
        "tune-dynamic",
        str(yaml_file),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cavity_workdir,
        env=_env_with_pythonpath(),
    )
    assert result.returncode == 0, (
        f"tune-dynamic failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    out_path = cavity_workdir / "dynamic_track.yaml"
    if "sufficient across the whole run" in result.stdout:
        assert not out_path.exists()
    else:
        assert "Wrote per-step overrides" in result.stdout
        assert out_path.exists()


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
