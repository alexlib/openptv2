#!/usr/bin/env python
"""
Unified command-line interface (CLI) for openptv2.

Exposes subcommands for headless tracking, runtime validation, and launching the GUI.
"""

import sys


def print_help():
    """Print the unified help message."""
    print("=" * 60)
    print("openptv2 Unified Command-Line Interface")
    print("=" * 60)
    print("Usage: openptv <command> [options]")
    print()
    print("Available Commands:")
    print("  track               Run headless batch sequence and tracking processing")
    print("  benchmark-tracking  Run quantitative tracking benchmark & metrics evaluation")
    print("  inspect             Inspect Zarr store data across all pipeline stages")
    print("  validate            Validate the single Cython runtime on bundled test data")
    print("  gui                 Launch the interactive 3D-PTV GUI")
    print()
    print("For help on any specific command, run:")
    print("  openptv <command> --help")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "track":
        # Headless batch sequence / tracking
        try:
            from openptv2.batch.pyptv_batch import main as batch_main
            from openptv2.batch.pyptv_batch import parse_command_line_args

            # Re-parse sys.argv[2:] using pyptv_batch's parser
            (
                yaml_file,
                first_frame,
                last_frame,
                mode,
                track3d,
                sequence_plugin,
                tracking_plugin,
            ) = parse_command_line_args(sys.argv[2:])
            batch_main(
                yaml_file,
                first_frame,
                last_frame,
                mode=mode,
                track3d=track3d,
                sequence_plugin=sequence_plugin,
                tracking_plugin=tracking_plugin,
            )

        except Exception as e:
            print(f"Tracking command failed: {e}")
            sys.exit(1)

    elif command in ("benchmark-tracking", "benchmark"):
        try:
            import argparse
            import numpy as np
            from openptv2.tracking_metrics import (
                generate_synthetic_benchmark_dataset,
                calculate_tracking_metrics,
            )
            from openptv2.tracking_cost import CostWeights
            from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker

            parser = argparse.ArgumentParser(prog="openptv benchmark-tracking")
            parser.add_argument("--flow", choices=["vortex", "linear", "burgers"], default="vortex", help="Synthetic flow field type")
            parser.add_argument("--particles", type=int, default=30, help="Number of particles")
            parser.add_argument("--frames", type=int, default=15, help="Number of frames")
            parser.add_argument("--noise", type=float, default=0.15, help="Spatial noise std dev")
            parser.add_argument("--gaps", type=float, default=0.10, help="Probability of detection dropout / gap per frame")
            parser.add_argument("--spurious", type=float, default=0.15, help="Ratio of false positive ghost noise particles")
            parser.add_argument("--w-vel", type=float, default=0.0, help="Velocity continuity cost weight")
            parser.add_argument("--w-acc", type=float, default=0.0, help="Acceleration cost weight")
            args, _ = parser.parse_known_args(sys.argv[2:])

            from openptv2.tracking_metrics import (
                generate_synthetic_benchmark_dataset,
                calculate_tracking_metrics,
                run_multi_tracker_benchmark,
            )

            print(f"--- Running Tracking Benchmark ({args.flow.upper()} flow, {args.particles} particles, {args.frames} frames, noise={args.noise}, gaps={args.gaps}, spurious={args.spurious}) ---")
            true_tracks, frame_blobs = generate_synthetic_benchmark_dataset(
                num_particles=args.particles,
                num_frames=args.frames,
                noise_std=args.noise,
                gap_probability=args.gaps,
                false_positive_ratio=args.spurious,
                flow_type=args.flow,
            )

            results = run_multi_tracker_benchmark(true_tracks, frame_blobs)

            print("=" * 105)
            print(f"{'Tracker Engine':<28} | {'Yield':<7} | {'Precision':<9} | {'Mean Length':<11} | {'RMS Error':<9} | {'FPS':<8} | {'Throughput':<12}")
            print("-" * 105)
            for engine_name, m in results.items():
                print(
                    f"{engine_name:<28} | {m.yield_recall*100:5.1f}% | {m.precision*100:7.1f}% | {m.mean_track_length:9.2f} fr | {m.rms_position_error:8.4f} | {m.fps:7.1f} | {m.particles_per_sec:10.0f} p/s"
                )
            print("=" * 105)

        except Exception as e:
            print(f"Benchmark failed: {e}")
            sys.exit(1)

    elif command in ("inspect", "peek"):
        try:
            if len(sys.argv) < 3:
                print("Usage: openptv inspect <zarr_path> [--frame FRAME] [--cam CAM]")
                sys.exit(1)
            from openptv2.storage.zarr_store import inspect_zarr_store, ZarrFrameStore
            zarr_path = sys.argv[2]
            if "--frame" in sys.argv or "-f" in sys.argv:
                sys.argv = [sys.argv[0]] + sys.argv[2:]
                from openptv2.storage.zarr_store import main_cli
                main_cli()
            else:
                print(inspect_zarr_store(zarr_path))
        except Exception as e:
            print(f"Inspection failed: {e}")
            sys.exit(1)

    elif command == "validate":
        # Engine consistency validation tool
        try:
            # We must trick validate's argparser into parsing sys.argv[2:] instead of sys.argv
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            from openptv2.validate import main as validate_main

            sys.exit(validate_main())

        except Exception as e:
            print(f"Validation command failed: {e}")
            sys.exit(1)

    elif command == "gui":
        # Interactive 3D-PTV GUI
        try:
            print("Launching interactive OpenPTV GUI...")
            from openptv2.gui.pyptv_gui import main as gui_main

            gui_main()

        except Exception as e:
            print(f"GUI launch failed: {e}")
            sys.exit(1)

    else:
        print(f"Unknown command: '{command}'")
        print()
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
