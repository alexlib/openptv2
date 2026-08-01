"""PyPTV_BATCH: Batch processing script with plugin support

Script for PyPTV experiments that have been set up using the GUI.
Supports custom tracking and sequence plugins.

Example:
    python pyptv_batch_plugins.py tests/test_splitter 10000 10004 --tracking splitter --sequence splitter
"""

import importlib
import json
import sys
from pathlib import Path

# Register optv package and its submodules as aliases in sys.modules for legacy compatibility
try:
    import openptv2

    sys.modules["optv"] = openptv2
    for sub in [
        "correspondences",
        "tracker",
        "orientation",
        "calibration",
        "parameters",
        "imgcoord",
    ]:
        try:
            mod = importlib.import_module(f"openptv2.{sub}")
            sys.modules[f"optv.{sub}"] = mod
        except ImportError:
            pass
except ImportError:
    pass

# Register pyptv package and its submodules as aliases in sys.modules
try:
    import openptv2.gui as _gui
    import openptv2.gui.pyptv as _pyptv_base

    sys.modules["pyptv"] = _pyptv_base

    for sub, target in _gui.submodule_mapping.items():
        try:
            # Map the alias directly using the registered lazy shim
            shim = sys.modules[f"openptv2.gui.pyptv.{sub}"]
            sys.modules[f"pyptv.{sub}"] = shim
        except KeyError:
            try:
                mod = importlib.import_module(target)
                sys.modules[f"pyptv.{sub}"] = mod
                sys.modules[f"openptv2.gui.pyptv.{sub}"] = mod
            except ImportError:
                pass
except ImportError:
    pass

from openptv2.gui.experiment import Experiment


def load_plugins_config(exp_path: Path):
    """Load available plugins from experiment parameters (YAML) with fallback to plugins.json"""
    try:
        experiment = Experiment()
        experiment.pm.from_yaml(exp_path)  # Corrected to use exp_path
        plugins_params = experiment.pm.parameters.get("plugins", None)
        if plugins_params is not None:
            return {
                "tracking": plugins_params.get("available_tracking", ["default"]),
                "sequence": plugins_params.get("available_sequence", ["default"]),
                "selected_tracking": plugins_params.get("selected_tracking", "default"),
                "selected_sequence": plugins_params.get("selected_sequence", "default"),
            }
    except Exception as e:
        print(f"Error loading plugins from YAML: {e}")
    # Fallback to plugins.json for backward compatibility (deprecated)
    plugins_file = exp_path.parent / "plugins.json"  # Corrected to use exp_path
    if plugins_file.exists():
        print(
            "WARNING: Using deprecated plugins.json - please migrate to YAML parameters"
        )
        with open(plugins_file, "r") as f:
            return json.load(f)
    return {"tracking": ["default"], "sequence": ["default"]}


def run_batch(
    yaml_file: Path,
    seq_first: int,
    seq_last: int,
    tracking_plugin: str = "default",
    sequence_plugin: str = "default",
    mode: str = "both",
):
    """Deprecated: use openptv2.batch.pyptv_batch.run_batch instead.

    Kept for backward compatibility with direct callers/tests using this
    module's legacy signature and keyword order (tracking_plugin before
    sequence_plugin). "default" now resolves to the core pipeline through
    the same plugin loader as any other plugin, so the two batch runners are
    behaviorally identical — this module no longer maintains its own copy
    of the sequence/tracking dispatch logic.
    """
    from openptv2.batch.pyptv_batch import run_batch as _run_batch

    _run_batch(
        Path(yaml_file),
        seq_first,
        seq_last,
        mode=mode,
        sequence_plugin=sequence_plugin,
        tracking_plugin=tracking_plugin,
    )


def main():
    """Main entry point with argparse and --mode support"""
    import argparse

    parser = argparse.ArgumentParser(
        description="PyPTV batch processing with plugins. Supports running only sequence, only tracking, or both."
    )
    parser.add_argument("yaml_file", type=str, help="Path to YAML parameter file.")
    parser.add_argument("first_frame", type=int, help="First frame number.")
    parser.add_argument("last_frame", type=int, help="Last frame number.")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "sequence", "tracking"],
        help="Which steps to run: both (default), sequence, or tracking.",
    )
    parser.add_argument("--sequence", type=str, help="Sequence plugin to use.")
    parser.add_argument("--tracking", type=str, help="Tracking plugin to use.")
    args = parser.parse_args()
    yaml_file = Path(args.yaml_file).resolve()
    first_frame = args.first_frame
    last_frame = args.last_frame
    mode = args.mode
    # Show available plugins
    plugins_config = load_plugins_config(yaml_file)
    print(f"Available tracking plugins: {plugins_config.get('tracking', ['default'])}")
    print(f"Available sequence plugins: {plugins_config.get('sequence', ['default'])}")
    # Prefer the selection saved in the YAML over the first available name.
    tracking_plugin = args.tracking or plugins_config.get(
        "selected_tracking", plugins_config.get("tracking", ["default"])[0]
    )
    sequence_plugin = args.sequence or plugins_config.get(
        "selected_sequence", plugins_config.get("sequence", ["default"])[0]
    )
    run_batch(
        yaml_file, first_frame, last_frame, tracking_plugin, sequence_plugin, mode
    )


if __name__ == "__main__":
    main()
