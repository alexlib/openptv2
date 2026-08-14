"""PyPTV_BATCH: Batch processing script for 3D-PTV (http://ptv.origo.ethz.ch)

This module provides batch processing capabilities for PyPTV, allowing users to
process sequences of images without the GUI interface.

The script expects:
- A YAML parameter file (e.g., parameters_Run1.yaml)
- img/ directory with image sequences (relative to YAML file location)
- cal/ directory with calibration files (relative to YAML file location)
- res/ directory (created automatically if missing)

To convert legacy parameters to YAML format:
    python -m pyptv.parameter_util legacy-to-yaml /path/to/parameters/

Example:
    Command line usage:
    >>> python pyptv_batch.py tests/test_cavity/parameters_Run1.yaml 10000 10004

    Python API usage:
    >>> from .pyptv_batch import main
    >>> main("tests/test_cavity/parameters_Run1.yaml", 10000, 10004)
"""

import os
import sys
import time
from pathlib import Path
from typing import Union


class ProcessingError(Exception):
    """Custom exception for PyPTV batch processing errors."""

    pass


# AttrDict removed - using direct dictionary access with Experiment object


def validate_experiment_setup(yaml_file: Path) -> Path:
    """Validate that the YAML file exists and required directories are available.

    Args:
        yaml_file: Path to the YAML parameter file

    Returns:
        Path to the experiment directory (parent of YAML file)

    Raises:
        ProcessingError: If required files or directories are missing
    """
    if not yaml_file.exists():
        raise ProcessingError(f"YAML parameter file does not exist: {yaml_file}")

    if not yaml_file.is_file():
        raise ProcessingError(f"Path is not a file: {yaml_file}")

    if yaml_file.suffix.lower() not in [".yaml", ".yml"]:
        raise ProcessingError(f"File must have .yaml or .yml extension: {yaml_file}")

    # Get experiment directory (parent of YAML file)
    exp_path = yaml_file.parent

    # Check for required subdirectories relative to YAML file location
    # Note: 'res' directory is created automatically if missing
    # required_dirs = ["img", "cal"]
    # missing_dirs = []

    # for dir_name in required_dirs:
    #     dir_path = exp_path / dir_name
    #     if not dir_path.exists():
    #         missing_dirs.append(dir_name)

    # if missing_dirs:
    #     raise ProcessingError(
    #         f"Missing required directories relative to {yaml_file}: {', '.join(missing_dirs)}"
    #     )

    return exp_path


def resolve_selected_plugins(pm, sequence_plugin=None, tracking_plugin=None):
    """Resolve plugin names: explicit argument wins, else the YAML
    ``plugins.selected_*`` selection saved by the GUI, else "default".

    This is what makes a GUI-tuned YAML self-contained: running the batch
    with no plugin flags reproduces the GUI's plugin selection.
    """
    plugins_params = pm.parameters.get("plugins") or {}
    if sequence_plugin is None:
        sequence_plugin = plugins_params.get("selected_sequence") or "default"
    if tracking_plugin is None:
        tracking_plugin = plugins_params.get("selected_tracking") or "default"
    return sequence_plugin, tracking_plugin


class ProcessingExperiment:
    """Minimal experiment-shaped object consumed by openptv2.gui.ptv
    functions and by sequence/tracking plugins (which expect .pm, the
    parameter objects, .cals and .target_filenames)."""

    def __init__(self, pm, cpar, spar, vpar, track_par, tpar, cals, epar):
        self.pm = pm
        self.cpar = cpar
        self.spar = spar
        self.vpar = vpar
        self.track_par = track_par
        self.tpar = tpar
        self.cals = cals
        self.epar = epar
        self.num_cams = pm.num_cams
        # Attributes that may be set during processing
        self.detections = []
        self.corrected = []
        self.target_filenames = pm.get_target_filenames()


def build_processing_experiment(
    yaml_file: Path, seq_first: int, seq_last: int
) -> ProcessingExperiment:
    """Load the YAML and construct the processing experiment for one run
    (or one frame chunk of a parallel run — each worker builds its own)."""
    from openptv2.gui.experiment import Experiment
    from openptv2.gui.ptv import py_start_proc_c

    experiment = Experiment()
    print(f"Loading parameters from: {yaml_file}")
    experiment.pm.from_yaml(yaml_file)

    print(f"Initializing processing with num_cams = {experiment.pm.num_cams}")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(experiment.pm)

    spar.set_first(seq_first)
    spar.set_last(seq_last)

    return ProcessingExperiment(
        experiment.pm, cpar, spar, vpar, track_par, tpar, cals, epar
    )


def _warn_if_tracking_poorly_conditioned(proc_exp, seq_first: int, seq_last: int) -> None:
    """Best-effort advisory check, run once before tracking starts: is the
    true flow fast enough, relative to this calibration's z-reconstruction
    noise floor, for individual trajectories to be trustworthy? See
    openptv2.tracking_feasibility for the reasoning and
    docs/plans/two-subrig-calibration.md for where this was first measured.

    Purely informational -- never blocks or alters tracking. Any failure to
    read the data (sequence didn't produce rt_is yet, too few points, etc.)
    is swallowed silently rather than interrupting the actual pipeline.
    """
    if seq_last <= seq_first:
        return
    try:
        import numpy as np

        from openptv2.tracking_feasibility import assess_tracking_conditioning

        pos_a = np.loadtxt(f"res/rt_is.{seq_first}", skiprows=1, ndmin=2)[:, 1:4]
        pos_b = np.loadtxt(f"res/rt_is.{seq_first + 1}", skiprows=1, ndmin=2)[:, 1:4]
        report = assess_tracking_conditioning(pos_a, pos_b, proc_exp.cals, proc_exp.cpar)
        if report is not None and report.verdict != "well-conditioned":
            print(f"[WARNING] {report.message}")
    except Exception:
        pass


def run_batch(
    yaml_file: Path,
    seq_first: int,
    seq_last: int,
    mode: str = "both",
    track3d: bool = False,
    sequence_plugin: str | None = None,
    tracking_plugin: str | None = None,
) -> None:
    """Run batch processing for a sequence of frames.

    Args:
        seq_first: First frame number in the sequence
        seq_last: Last frame number in the sequence
        yaml_file: Path to the YAML parameter file
        track3d: Whether to use 3D segment tracking (only affects the
            "default" tracking plugin; see openptv2.plugins.default_tracking)
        sequence_plugin: Sequence plugin name; None (default) uses the
            YAML ``plugins.selected_sequence`` selection
        tracking_plugin: Tracking plugin name; None (default) uses the
            YAML ``plugins.selected_tracking`` selection

    Raises:
        ProcessingError: If processing fails
    """
    print(f"Starting batch processing: frames {seq_first} to {seq_last}")
    print(f"Using parameter file: {yaml_file}")

    # Validate experiment setup and get experiment directory
    exp_path = validate_experiment_setup(yaml_file)

    # Store original working directory
    original_cwd = Path.cwd()

    try:
        # Change to experiment directory
        os.chdir(exp_path)

        proc_exp = build_processing_experiment(yaml_file, seq_first, seq_last)

        sequence_plugin, tracking_plugin = resolve_selected_plugins(
            proc_exp.pm, sequence_plugin, tracking_plugin
        )

        if track3d:
            proc_exp.track3d = True

        # default_naming (res/rt_is, res/ptv_is, res/added) is relative to
        # cwd and used by every sequence/tracking plugin, so the output
        # directory must exist before any of them run.
        Path("res").mkdir(exist_ok=True)

        from openptv2.plugins import run_sequence_plugin, run_tracking_plugin

        plugins_dir = exp_path / "plugins"
        if mode not in ("both", "sequence", "tracking"):
            raise ProcessingError(
                f"Unknown mode: {mode}. Use 'both', 'sequence', or 'tracking'."
            )
        if mode in ("both", "sequence"):
            print(f"Running sequence plugin: {sequence_plugin}")
            run_sequence_plugin(sequence_plugin, proc_exp, plugins_dir)
        if mode in ("both", "tracking"):
            _warn_if_tracking_poorly_conditioned(proc_exp, seq_first, seq_last)
            print(f"Running tracking plugin: {tracking_plugin}")
            run_tracking_plugin(tracking_plugin, proc_exp, plugins_dir)

        print("Batch processing completed successfully")

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise ProcessingError(f"Batch processing failed: {e}")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main(
    yaml_file: Union[str, Path],
    first: Union[str, int],
    last: Union[str, int],
    repetitions: int = 1,
    mode: str = "both",
    track3d: bool = False,
    sequence_plugin: str | None = None,
    tracking_plugin: str | None = None,
) -> None:
    """Run PyPTV batch processing.

    Args:
        yaml_file: Path to the YAML parameter file (e.g., parameters_Run1.yaml)
        first: First frame number in the sequence
        last: Last frame number in the sequence
        repetitions: Number of times to repeat the processing (default: 1)
        mode: Which steps to run: both (default), sequence, or tracking
        track3d: Whether to use 3D segment tracking
        sequence_plugin: Sequence plugin name; None uses the YAML
            ``plugins.selected_sequence`` selection
        tracking_plugin: Tracking plugin name; None uses the YAML
            ``plugins.selected_tracking`` selection

    Raises:
        ProcessingError: If processing fails
        ValueError: If parameters are invalid

    Note:
        If you have legacy .par files, convert them first using:
        python -m pyptv.parameter_util legacy-to-yaml /path/to/parameters/
    """
    start_time = time.time()

    try:
        # Validate and convert parameters
        yaml_file = Path(yaml_file).resolve()
        seq_first = int(first)
        seq_last = int(last)

        exp_path = yaml_file.parent

        if seq_first > seq_last:
            raise ValueError(
                f"First frame ({seq_first}) must be <= last frame ({seq_last})"
            )

        if repetitions < 1:
            raise ValueError(f"Repetitions must be >= 1, got {repetitions}")

        print(f"Starting batch processing with YAML file: {yaml_file}")
        print(f"Frame range: {seq_first} to {seq_last}")
        print(f"Repetitions: {repetitions}")
        # Validate YAML file and experiment setup
        # exp_path = validate_experiment_setup(yaml_file)
        print(f"Experiment directory: {exp_path}")
        # Create results directory if it doesn't exist
        res_path = exp_path / "res"
        if not res_path.exists():
            print("Creating 'res' directory")
            res_path.mkdir(parents=True, exist_ok=True)

        # Run processing for specified repetitions
        for i in range(repetitions):
            if repetitions > 1:
                print(f"Starting repetition {i + 1} of {repetitions}")
            run_batch(
                yaml_file,
                seq_first,
                seq_last,
                mode=mode,
                track3d=track3d,
                sequence_plugin=sequence_plugin,
                tracking_plugin=tracking_plugin,
            )
        elapsed_time = time.time() - start_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")

    except (ValueError, ProcessingError) as e:
        print(f"Processing failed: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during processing: {e}")
        raise ProcessingError(f"Unexpected error: {e}")


def parse_command_line_args(
    args_list=None,
) -> tuple[Path, int, int, str, bool, str, str]:
    """Parse and validate command line arguments.

    Returns:
        Tuple of (yaml_file_path, first_frame, last_frame, mode, track3d,
        sequence_plugin, tracking_plugin)

    Raises:
        ValueError: If arguments are invalid
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyPTV batch processing")
    parser.add_argument("yaml_file", type=str, nargs="?", help="YAML parameter file")
    parser.add_argument("first_frame", type=int, nargs="?", help="First frame number")
    parser.add_argument("last_frame", type=int, nargs="?", help="Last frame number")
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        help="YAML parameter file or experiment directory",
    )
    parser.add_argument(
        "--first",
        "-f",
        type=int,
        help="First frame number",
    )
    parser.add_argument(
        "--last",
        "-l",
        type=int,
        help="Last frame number",
    )
    parser.add_argument(
        "--engine",
        "-e",
        help="Deprecated compatibility flag. Ignored in the single-engine runtime.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "sequence", "tracking"],
        default="both",
        help="Which steps to run: both (default), sequence, or tracking",
    )
    parser.add_argument(
        "--track3d",
        action="store_true",
        help="Use 3D segment tracking instead of standard tracking",
    )
    parser.add_argument(
        "--sequence-plugin",
        default=None,
        help=(
            "Sequence plugin name. Default: the plugins.selected_sequence "
            "saved in the YAML (falling back to the core pipeline)."
        ),
    )
    parser.add_argument(
        "--tracking-plugin",
        default=None,
        help=(
            "Tracking plugin name. Default: the plugins.selected_tracking "
            "saved in the YAML (falling back to the core pipeline)."
        ),
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Deprecated compatibility flag. Ignored in the single-engine runtime.",
    )
    args = parser.parse_args(args_list)

    if args.engine or args.debug_mode:
        print(
            "Ignoring legacy engine-selection flags; openptv2 now uses a single "
            "Cython 3 runtime."
        )

    print("Using tracking engine: cython3-pure-python")

    yaml_arg = args.workdir or args.yaml_file
    if not yaml_arg:
        parser.print_help()
        raise ValueError(
            "Please provide a YAML parameter file or experiment directory "
            "via --workdir/-w or as a positional argument."
        )

    yaml_path = Path(yaml_arg).resolve()
    if yaml_path.is_dir():
        yaml_files = list(yaml_path.glob("*parameters_*.yaml"))
        if not yaml_files:
            yaml_files = list(yaml_path.glob("*.yaml")) + list(yaml_path.glob("*.yml"))

        if not yaml_files:
            raise ValueError(f"No YAML parameter files found in directory {yaml_path}")

        yaml_file = sorted(set(yaml_files))[0]
        print(f"Directory provided. Selected parameter file: {yaml_file}")
    else:
        yaml_file = yaml_path

    from openptv2.gui.parameter_manager import ParameterManager

    pm = ParameterManager()
    pm.from_yaml(yaml_file)

    first_frame = args.first if args.first is not None else args.first_frame
    if first_frame is None:
        seq = pm.parameters.get("sequence")
        if seq:
            first_frame = seq.get("first")

    last_frame = args.last if args.last is not None else args.last_frame
    if last_frame is None:
        seq = pm.parameters.get("sequence")
        if seq:
            last_frame = seq.get("last")

    mode = args.mode
    track3d = args.track3d
    sequence_plugin = args.sequence_plugin
    tracking_plugin = args.tracking_plugin

    return (
        yaml_file,
        first_frame,
        last_frame,
        mode,
        track3d,
        sequence_plugin,
        tracking_plugin,
    )


def main_cli() -> None:
    """Entry point for command line execution."""
    try:
        print("Starting batch processing")
        print(f"Command line arguments: {sys.argv}")

        (
            yaml_file,
            first_frame,
            last_frame,
            mode,
            track3d,
            sequence_plugin,
            tracking_plugin,
        ) = parse_command_line_args()
        main(
            yaml_file,
            first_frame,
            last_frame,
            mode=mode,
            track3d=track3d,
            sequence_plugin=sequence_plugin,
            tracking_plugin=tracking_plugin,
        )

        print("Batch processing completed successfully")

    except (ValueError, ProcessingError) as e:
        print(f"Batch processing failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """Entry point for command line execution.

    Command line usage:
        python pyptv_batch.py <yaml_file> <first_frame> <last_frame>

    Example:
        python pyptv_batch.py tests/test_cavity/parameters_Run1.yaml 10000 10004

    Python API usage:
        from .pyptv_batch import main
        main("tests/test_cavity/parameters_Run1.yaml", 10000, 10004)
    """
    main_cli()
