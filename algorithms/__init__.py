"""Pure-Python PTV algorithms using NumPy vectorized operations.

This package is a clean-room translation of the C library in lib/src/
into Python using:
- NumPy vectorized operations for all numerical computations
- Structure-of-Arrays (SoA) layout for batch data
- dataclasses for parameter/configuration objects
- No adapter layers, no dual storage, no getter/setter boilerplate

Each module corresponds directly to a C source file:
  vec_utils.py       <- lib/src/vec_utils.c
  lsqadj.py          <- lib/src/lsqadj.c
  calibration.py     <- lib/src/calibration.c
  parameters.py      <- lib/src/parameters.c
  trafo.py           <- lib/src/trafo.c
  multimed.py        <- lib/src/multimed.c
  ray_tracing.py     <- lib/src/ray_tracing.c
  imgcoord.py        <- lib/src/imgcoord.c
  image_processing.py <- lib/src/image_processing.c
  segmentation.py    <- lib/src/segmentation.c
  epi.py             <- lib/src/epi.c
  correspondences.py <- lib/src/correspondences.c
  orientation.py     <- lib/src/orientation.c
  sortgrid.py        <- lib/src/sortgrid.c
  tracking_frame_buf.py <- lib/src/tracking_frame_buf.c
  tracking_run.py    <- lib/src/tracking_run.c
  track.py           <- lib/src/track.c
  track3d.py         <- lib/src/track3d.c

Design principles:
1. Single data model - no adapter layers
2. SoA-only storage - no dual object/array storage
3. Data model separated from serialization
4. No getter/setter methods - use dataclass fields directly
5. Small, testable function kernels
6. Named constants for all magic numbers
7. Clear module boundaries with minimal imports
8. Consistent error handling with specific exceptions
9. No dead code
10. Consistent use of dataclasses/ndarrays
"""
