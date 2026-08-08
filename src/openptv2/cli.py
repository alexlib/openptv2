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
    print(
        "  benchmark-tracking  Run quantitative tracking benchmark & metrics evaluation"
    )
    print("  inspect             Inspect Zarr store data across all pipeline stages")
    print(
        "  validate            Validate the single Cython runtime on bundled test data"
    )
    print("  gui                 Launch the interactive 3D-PTV GUI")
    print("  list-trackers       List all available trackers with capabilities")
    print("  recommend           Analyse a dataset and recommend a tracker & parameters")
    print("  benchmark           Generate datasets, sweep params, compare trackers")
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

    elif command == "benchmark-tracking":
        try:
            import argparse

            from openptv2.tracking_metrics import (
                generate_synthetic_benchmark_dataset,
            )

            parser = argparse.ArgumentParser(prog="openptv benchmark-tracking")
            parser.add_argument(
                "--flow",
                choices=["vortex", "linear", "burgers"],
                default="vortex",
                help="Synthetic flow field type",
            )
            parser.add_argument(
                "--particles", type=int, default=30, help="Number of particles"
            )
            parser.add_argument(
                "--frames", type=int, default=15, help="Number of frames"
            )
            parser.add_argument(
                "--noise", type=float, default=0.15, help="Spatial noise std dev"
            )
            parser.add_argument(
                "--gaps",
                type=float,
                default=0.10,
                help="Probability of detection dropout / gap per frame",
            )
            parser.add_argument(
                "--spurious",
                type=float,
                default=0.15,
                help="Ratio of false positive ghost noise particles",
            )
            parser.add_argument(
                "--w-vel",
                type=float,
                default=0.0,
                help="Velocity continuity cost weight",
            )
            parser.add_argument(
                "--w-acc", type=float, default=0.0, help="Acceleration cost weight"
            )
            args, _ = parser.parse_known_args(sys.argv[2:])

            from openptv2.tracking_metrics import (
                run_multi_tracker_benchmark,
            )

            print(
                f"--- Running Tracking Benchmark ({args.flow.upper()} flow, {args.particles} particles, {args.frames} frames, noise={args.noise}, gaps={args.gaps}, spurious={args.spurious}) ---"
            )
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
            print(
                f"{'Tracker Engine':<28} | {'Yield':<7} | {'Precision':<9} | {'Mean Length':<11} | {'RMS Error':<9} | {'FPS':<8} | {'Throughput':<12}"
            )
            print("-" * 105)
            for engine_name, m in results.items():
                print(
                    f"{engine_name:<28} | {m.yield_recall * 100:5.1f}% | {m.precision * 100:7.1f}% | {m.mean_track_length:9.2f} fr | {m.rms_position_error:8.4f} | {m.fps:7.1f} | {m.particles_per_sec:10.0f} p/s"
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
            from openptv2.storage.zarr_store import inspect_zarr_store

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

    elif command in ("list-trackers", "list"):
        from openptv2.tracking_registry import (
            TRACKER_REGISTRY,
            print_tracker_detail,
            print_tracker_table,
        )

        if len(sys.argv) >= 3 and sys.argv[2] in ("--show", "-s"):
            name = sys.argv[3] if len(sys.argv) > 3 else ""
            if name:
                info = TRACKER_REGISTRY.get(name)
                if info:
                    print(print_tracker_detail(name))
                else:
                    print(f"Unknown tracker: {name!r}")
                    print(f"Available: {', '.join(sorted(TRACKER_REGISTRY))}")
            else:
                print("Usage: openptv list-trackers --show <tracker_name>")
        else:
            print("\nAvailable Trackers:\n")
            print(print_tracker_table())
            print()
            print("For full details: openptv list-trackers --show <name>")

    elif command == "benchmark":
        import argparse

        parser = argparse.ArgumentParser(prog="openptv benchmark")
        sub = parser.add_subparsers(dest="action", required=True)

        p_ds = sub.add_parser("dataset", help="Generate a ground-truth dataset")
        p_ds.add_argument("out_dir")
        p_ds.add_argument("--particles", type=int, default=60)
        p_ds.add_argument("--frames", type=int, default=40)
        p_ds.add_argument("--velocity", type=float, default=1.0)
        p_ds.add_argument("--crossings", type=int, default=0)
        p_ds.add_argument("--entering", type=int, default=0)
        p_ds.add_argument("--leaving", type=int, default=0)
        p_ds.add_argument("--gap", type=float, default=0.05)
        p_ds.add_argument("--noise", type=float, default=0.02)
        p_ds.add_argument("--ghost", type=float, default=0.02)
        p_ds.add_argument("--refract", action="store_true")
        p_ds.add_argument("--seed", type=int, default=42)

        p_sw = sub.add_parser("sweep", help="Sweep a tracking parameter for a tracker")
        p_sw.add_argument("out_dir")
        p_sw.add_argument("--tracker", default="fast_3d")
        p_sw.add_argument("--param", default="dvxmax",
                          choices=["dvxmax", "dvy", "dvz", "dacc", "angle"])
        p_sw.add_argument("--values", nargs="+", type=float,
                          default=[1.0, 2.0, 4.0, 8.0])
        p_sw.add_argument("--particles", type=int, default=60)
        p_sw.add_argument("--frames", type=int, default=30)
        p_sw.add_argument("--refract", action="store_true")
        p_sw.add_argument("--seed", type=int, default=42)

        p_cp = sub.add_parser("compare", help="Compare trackers on the same data")
        p_cp.add_argument("out_dir")
        p_cp.add_argument("--trackers", nargs="+", default=None)
        p_cp.add_argument("--particles", type=int, default=60)
        p_cp.add_argument("--frames", type=int, default=30)
        p_cp.add_argument("--refract", action="store_true")
        p_cp.add_argument("--seed", type=int, default=42)

        args, _ = parser.parse_known_args(sys.argv[2:])

        try:
            from openptv2.benchmarking.cli_benchmark import (
                cmd_dataset,
                cmd_sweep,
                cmd_compare,
            )

            if args.action == "dataset":
                cmd_dataset(
                    args.out_dir, args.particles, args.frames, args.velocity,
                    args.crossings, args.entering, args.leaving, args.gap,
                    args.noise, args.ghost, args.refract, args.seed,
                )
            elif args.action == "sweep":
                cmd_sweep(
                    args.out_dir, args.tracker, args.param, args.values,
                    args.particles, args.frames, args.refract, args.seed,
                )
            elif args.action == "compare":
                cmd_compare(
                    args.out_dir, args.trackers, args.particles, args.frames,
                    args.refract, args.seed,
                )
        except Exception as e:
            print(f"Benchmark failed: {e}")
            sys.exit(1)

    elif command == "recommend":
        try:
            import argparse

            parser = argparse.ArgumentParser(prog="openptv recommend")
            parser.add_argument(
                "rt_is_dir",
                nargs="?",
                default="res",
                help="Directory containing rt_is.# files (default: res)",
            )
            parser.add_argument(
                "--first", type=int, default=None, help="First frame"
            )
            parser.add_argument(
                "--last", type=int, default=None, help="Last frame"
            )
            parser.add_argument(
                "--priority",
                choices=["speed", "accuracy", "default"],
                default="default",
                help="Optimisation priority",
            )
            parser.add_argument(
                "--require-backward",
                action="store_true",
                help="Require backward tracking support",
            )
            parser.add_argument(
                "--force",
                action="store_true",
                help="Show recommendation even if stats are incomplete",
            )
            args, _ = parser.parse_known_args(sys.argv[2:])

            from openptv2.tracking_recommender import (
                recommend_from_files,
                print_recommendation,
            )

            rec = recommend_from_files(
                args.rt_is_dir,
                args.first or 0,
                args.last or 0,
                user_preferences={
                    "priority": args.priority,
                    "require_backward": args.require_backward,
                },
            )
            print(print_recommendation(rec))

        except Exception as e:
            print(f"Recommendation failed: {e}")
            sys.exit(1)

    else:
        print(f"Unknown command: '{command}'")
        print()
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
