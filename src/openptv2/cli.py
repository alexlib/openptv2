#!/usr/bin/env python
"""
Unified command-line interface (CLI) for openptv2.

Exposes subcommands for headless tracking, runtime validation, and launching the GUI.
"""

import sys
from pathlib import Path


def print_help():
    """Print the unified help message."""
    print("=" * 60)
    print("openptv2 Unified Command-Line Interface")
    print("=" * 60)
    print("Usage: openptv <command> [options]")
    print()
    print("Available Commands:")
    print("  track       Run headless batch sequence and tracking processing")
    print("  validate    Validate the single Cython runtime on bundled test data")
    print("  gui         Launch the interactive 3D-PTV GUI")
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
            from openptv2.gui.pyptv.pyptv_batch import run_batch, parse_command_line_args, main as batch_main
            
            # Re-parse sys.argv[2:] using pyptv_batch's parser
            yaml_file, first_frame, last_frame, mode = parse_command_line_args(sys.argv[2:])
            batch_main(yaml_file, first_frame, last_frame, mode=mode)
            
        except Exception as e:
            print(f"Tracking command failed: {e}")
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
            from openptv2.gui.pyptv.pyptv_gui import main as gui_main
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
