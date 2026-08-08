import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from openptv2.batch.pyptv_batch import run_batch

TEST_CAVITY_YAML = (
    Path(__file__).parent.parent.parent
    / "test_data"
    / "test_cavity"
    / "parameters_Run1.yaml"
)


def analyze_trajectories(res_dir: Path, first_frame: int, last_frame: int) -> dict:
    """Parse ptv_is linkage files to extract trajectory metrics."""
    frames = {}
    for f in range(first_frame, last_frame + 1):
        file_path = res_dir / f"ptv_is.{f}"
        if not file_path.exists():
            continue
        lines = file_path.read_text().strip().splitlines()
        if not lines:
            continue
        if len(lines[0].split()) == 1:
            lines = lines[1:]
        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                rows.append((int(parts[0]), int(parts[1])))
        frames[f] = rows

    total_links = 0
    traj_lengths = []

    for f in range(first_frame, last_frame + 1):
        rows = frames.get(f, [])
        for idx, (prev, next_idx) in enumerate(rows):
            if next_idx >= 0:
                total_links += 1
            if prev == -1:
                length = 1
                curr_f = f
                curr_next = next_idx
                while curr_next >= 0 and (curr_f + 1) in frames:
                    curr_f += 1
                    next_rows = frames[curr_f]
                    if curr_next < len(next_rows):
                        length += 1
                        curr_next = next_rows[curr_next][1]
                    else:
                        break
                traj_lengths.append(length)

    traj_count = len(traj_lengths)
    mean_len = sum(traj_lengths) / traj_count if traj_count > 0 else 0.0
    max_len = max(traj_lengths) if traj_count > 0 else 0
    return {
        "total_links": total_links,
        "trajectories_count": traj_count,
        "mean_length": round(mean_len, 2),
        "max_length": max_len,
    }


def get_benchmark_dataset(use_aorta: bool = False):
    """Returns (yaml_path, first_frame, last_frame).
    Uses test_cavity (5 frames, ~200 particles) for quick pytest execution.
    The optional aorta path was removed as unsupported/dead in this checkout.
    """
    return TEST_CAVITY_YAML, 10000, 10004


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("preset", ["fast_3d", "standard_forward", "full_multipass"])
def test_tracking_preset_execution_and_benchmark(preset, tmp_path):
    """Run batch mode for each preset, verify outputs, and print trajectory metrics."""
    yaml_src, first_frame, last_frame = get_benchmark_dataset()
    data_dir = yaml_src.parent

    # Copy dataset into tmp_path sandbox
    for item in data_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, tmp_path / item.name)
        else:
            shutil.copy2(item, tmp_path / item.name)

    sandbox_yaml = tmp_path / yaml_src.name

    # Override preset and selected_tracking in sandboxed YAML
    with open(sandbox_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("track", {})["preset"] = preset
    cfg.setdefault("plugins", {})["selected_tracking"] = preset
    with open(sandbox_yaml, "w") as f:
        yaml.safe_dump(cfg, f)

    t0 = time.perf_counter()
    run_batch(sandbox_yaml, first_frame, last_frame, mode="both")
    elapsed = time.perf_counter() - t0

    stats = analyze_trajectories(tmp_path / "res", first_frame, last_frame)
    stats["time_sec"] = round(elapsed, 3)

    # Outcome quality checks: verify that tracking produced healthy link yield and non-trivial trajectories
    MIN_EXPECTED_LINKS = 500
    assert stats["total_links"] >= MIN_EXPECTED_LINKS, (
        f"Preset {preset} lost too many links! Yielded only {stats['total_links']} links (expected >= {MIN_EXPECTED_LINKS})"
    )
    assert stats["trajectories_count"] > 0, f"Preset {preset} yielded 0 trajectories"
    assert stats["mean_length"] >= 1.2, (
        f"Preset {preset} trajectory mean length too short: {stats['mean_length']} (expected >= 1.2)"
    )

    print(
        f"\n[BENCHMARK] Preset: {preset:18s} | Dataset: 'Cavity' | "
        f"Time: {stats['time_sec']}s | Links: {stats['total_links']} | "
        f"Trajectories: {stats['trajectories_count']} | Mean Len: {stats['mean_length']} | "
        f"Max Len: {stats['max_length']}"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_preset_comparison_summary_table(tmp_path, capsys):
    """Runs all 3 presets and prints a Markdown comparison table suitable for documentation."""
    yaml_src, first_frame, last_frame = get_benchmark_dataset()
    data_dir = yaml_src.parent

    results = {}
    presets = ["fast_3d", "standard_forward", "full_multipass"]

    for preset in presets:
        preset_dir = tmp_path / preset
        preset_dir.mkdir()
        for item in data_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, preset_dir / item.name)
            else:
                shutil.copy2(item, preset_dir / item.name)

        sandbox_yaml = preset_dir / yaml_src.name
        with open(sandbox_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("track", {})["preset"] = preset
        cfg.setdefault("plugins", {})["selected_tracking"] = preset
        with open(sandbox_yaml, "w") as f:
            yaml.safe_dump(cfg, f)

        t0 = time.perf_counter()
        run_batch(sandbox_yaml, first_frame, last_frame, mode="both")
        elapsed = time.perf_counter() - t0

        stats = analyze_trajectories(preset_dir / "res", first_frame, last_frame)
        stats["time_sec"] = round(elapsed, 2)
        results[preset] = stats

    dataset_name = "test_cavity (5 frames)"
    table_md = f"""
### Tracking Pipeline Benchmark Comparison ({dataset_name})

| Tracking Preset | Pipeline Description | Time (s) | Total Links | Trajectories Count | Mean Length | Max Length |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`fast_3d`** | Single-pass 3D Segment (`track_mode=1`) | {results["fast_3d"]["time_sec"]}s | {results["fast_3d"]["total_links"]:,} | {results["fast_3d"]["trajectories_count"]:,} | {results["fast_3d"]["mean_length"]} | {results["fast_3d"]["max_length"]} |
| **`standard_forward`** | Single-pass Forward (`track_mode=0`) | {results["standard_forward"]["time_sec"]}s | {results["standard_forward"]["total_links"]:,} | {results["standard_forward"]["trajectories_count"]:,} | {results["standard_forward"]["mean_length"]} | {results["standard_forward"]["max_length"]} |
| **`full_multipass`** | 3-Pass (Forward + Backward + Postprocess) | {results["full_multipass"]["time_sec"]}s | {results["full_multipass"]["total_links"]:,} | {results["full_multipass"]["trajectories_count"]:,} | {results["full_multipass"]["mean_length"]} | {results["full_multipass"]["max_length"]} |
"""
    print(table_md)
    assert len(results) == 3
    for p_name, p_stats in results.items():
        assert p_stats["total_links"] >= 500, (
            f"Preset {p_name} in summary comparison failed link retention: {p_stats['total_links']} links"
        )
