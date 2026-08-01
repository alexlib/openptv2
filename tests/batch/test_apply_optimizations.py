"""Apply optimized tracking parameters to improve linking performance"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def apply_optimized_parameters(yaml_file: Path) -> bool:
    """Apply the optimized tracking parameters found through testing"""
    if not yaml_file.exists():
        print(f"❌ YAML file not found: {yaml_file}")
        return False

    print("🔧 Applying optimized tracking parameters...")
    content = yaml_file.read_text()
    lines = content.split("\n")
    changes_made = []

    for i, line in enumerate(lines):
        if "track:" in content[: content.find(line)] or "track:" in line:
            if "angle:" in line:
                old_value = line.split(":")[1].strip()
                lines[i] = "  angle: 0.5"
                changes_made.append(f"angle: {old_value} → 0.5")
            elif "dacc:" in line:
                old_value = line.split(":")[1].strip()
                lines[i] = "  dacc: 10.0"
                changes_made.append(f"dacc: {old_value} → 10.0")

    modified_content = "\n".join(lines)
    yaml_file.write_text(modified_content)

    print("✅ Applied optimizations:")
    for change in changes_made:
        print(f"   {change}")

    return True


def test_optimized_performance(tmp_path):
    """Test tracking performance with optimized parameters on a temporary copy."""
    src_path = Path(__file__).parent.parent.parent / "test_data" / "test_splitter"
    if not src_path.exists():
        pytest.skip("test_splitter fixture not available")

    # Copy test_splitter fixture to tmp_path to avoid mutating shared test_data
    dest_path = tmp_path / "test_splitter"
    shutil.copytree(src_path, dest_path)

    yaml_file = dest_path / "parameters_Run1.yaml"
    apply_optimized_parameters(yaml_file)

    script_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "openptv2"
        / "batch"
        / "pyptv_batch_plugins.py"
    )
    cmd = [
        sys.executable,
        str(script_path),
        str(yaml_file),
        "1000001",
        "1000003",
        "--mode",
        "both",
        "--sequence",
        "ext_sequence_splitter",
        "--tracking",
        "ext_tracker_splitter",
    ]

    print("🚀 Testing performance with optimized parameters...")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60, cwd=dest_path
    )

    assert result.returncode == 0, f"Test failed: {result.stderr}"

    lines = result.stdout.split("\n")
    tracking_lines = [line for line in lines if "step:" in line and "links:" in line]

    total_particles = 0
    total_links = 0
    frames_count = 0

    for line in tracking_lines:
        print(f"📊 {line}")
        try:
            parts = line.split(",")
            curr_part = [p for p in parts if "curr:" in p][0]
            curr_count = int(curr_part.split(":")[1].strip())

            links_part = [p for p in parts if "links:" in p][0]
            links_count = int(links_part.split(":")[1].strip())

            total_particles += curr_count
            total_links += links_count
            frames_count += 1
        except (ValueError, IndexError):
            continue

    assert frames_count > 0 and total_particles > 0, "No tracking data found"

    avg_particles = total_particles / frames_count
    avg_links = total_links / frames_count
    link_ratio = avg_links / avg_particles * 100

    print("\n📈 Performance Results:")
    print(f"Average particles per frame: {avg_particles:.1f}")
    print(f"Average links per frame: {avg_links:.1f}")
    print(f"Link ratio: {link_ratio:.1f}%")

    assert link_ratio > 5.0, f"Link ratio too low: {link_ratio:.1f}%"
