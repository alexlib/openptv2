#!/usr/bin/env python3
"""Convert legacy OpenPTV ASCII run files to a unified Zarr store.

Usage:
    uv run python scripts/convert_legacy_to_zarr.py <experiment_folder> [--store <path>] [--remove-ascii]

Examples:
    # Convert test_cavity and create res/run.zarr
    uv run python scripts/convert_legacy_to_zarr.py test_data/test_cavity

    # Convert and delete legacy ASCII target and result files
    uv run python scripts/convert_legacy_to_zarr.py test_data/test_cavity --remove-ascii
"""

import sys

from openptv2.storage.legacy import main

if __name__ == "__main__":
    sys.exit(main())
