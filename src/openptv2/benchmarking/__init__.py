"""Ground-truth tracking benchmarking for openptv2.

Provides the components to build synthetic-but-physics-grounded benchmark
cases that exercise the real tracking pipeline through a calibration and
multimedia model, and to compare trackers / tune parameters against them.

Modules
-------
camera_rig
    Build a simple 4-camera rig (calibration + multimedia) in code.
scenario
    Define configurable ground-truth trajectories (velocities, crossings,
    entering/leaving particles, density, noise, gaps).
datawriter
    Project trajectories to per-camera pixels and write rt_is / targets /
    origin ground-truth files.
metrics
    Compute proPTV-style identity metrics (F, C, Cr, pmt) plus the standard
    yield/precision/RMS set.
runner
    Run a chosen tracker on a generated dataset and read trajectories back.
jhtdb_client
    Fetch (or synthesize) ground-truth Lagrangian HIT trajectories from the
    Johns Hopkins Turbulence Database.
synthetic_optical_projector
    Render synthetic multi-camera PTV images (PSF, laser sheet, noise,
    ghosts) from 3D particle positions through the real camera model.
"""

from .camera_rig import (
    N_AIR,
    N_GLASS,
    N_WATER,
    CameraRig,
    make_standard_rig,
    project_to_pixels,
)
from .datawriter import DatasetSpec, write_dataset
from .experiment import write_experiment
from .jhtdb_client import (
    fetch_hit_trajectories,
    get_hit_velocity,
    synthetic_hit_trajectories,
)
from .metrics import (
    IdentityMetrics,
    compute_identity_metrics,
    ghost_positions_from_frame_gt,
)
from .runner import read_trajectories, run_tracker
from .scenario import CrossingSpec, ScenarioSpec, generate_scenario
from .synthetic_optical_projector import RenderConfig, render_frame

__all__ = [
    "CameraRig",
    "make_standard_rig",
    "project_to_pixels",
    "N_AIR",
    "N_GLASS",
    "N_WATER",
    "ScenarioSpec",
    "CrossingSpec",
    "generate_scenario",
    "DatasetSpec",
    "write_dataset",
    "write_experiment",
    "run_tracker",
    "read_trajectories",
    "IdentityMetrics",
    "compute_identity_metrics",
    "ghost_positions_from_frame_gt",
    "fetch_hit_trajectories",
    "get_hit_velocity",
    "synthetic_hit_trajectories",
    "RenderConfig",
    "render_frame",
]
