# openptv2 Documentation

Particle Tracking Velocimetry: track particles in 3D from 4 camera views.

## Start here (10 minutes)

New to openptv2? Do this first:

1. **Install** — [Installation](installation.md):
   ```bash
   pip install "openptv2[gui]"
   ```
2. **Run your first tracking** — [GUI First Tracking tutorial](tutorials/getting_started_tutorial.md):
   open the bundled `test_data/test_cavity` example in the desktop GUI and
   track particles in 5 clicks. No data to download, no parameters to tune.
3. **Understand what happened** — [Tracking Pipeline & Results](tracking_guide.md):
   what detection, correspondences, and tracking do, and where results land.

```bash
openptv2-gui -w test_data/test_cavity/parameters_Run1.yaml
```

If you prefer the command line, the same example runs headless:

```bash
pyptv_batch --workdir=test_data/test_cavity --first=10000 --last=10001
```

## Documentation map

| I want to...                | Go to                                                        |
| --------------------------- | ------------------------------------------------------------ |
| Track my first particles    | [GUI First Tracking](tutorials/getting_started_tutorial.md)  |
| Install on my platform      | [Installation](installation.md)                              |
| Run without the GUI         | [Command-Line Batch Processing](tutorials/batch_processing.md) |
| Calibrate my own cameras    | [Plate Calibration How-To](plate-calibration-howto.md)       |
| Pick a tracker              | [Tracker Engines & Parameters](tracker-tutorials.md)         |
| Store large runs efficiently| [Zarr & HDF5 Storage](zarr-hdf5-storage.md)                  |
| Contribute code             | [Building from Source](developer_guide/building.md)          |

Advanced case studies ([cavity flow](tutorials/cavity_flow_tutorial.md),
[aorta flow](aorta_tutorial.md)) and calibration references live in the
navigation sidebar. Start with the tutorial above first.
