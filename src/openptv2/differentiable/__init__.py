"""OpenPTV3 differentiable pipeline core (Phase 2 of
docs/plans/differentiable_ptv_nextgen_plan.md).

PyTorch soft-differentiable operators replacing the discrete legacy
algorithms, enabling gradient flow from Stage-1 micro-parameters (2D
intensity thresholds) through to Stage-5 Lagrangian turbulence physics.

Modules
-------
centroiding
    Subpixel 2D target centroiding: ``SoftArgmax2D``, ``DifferentiableGaussianFit``.
geometry
    Differentiable pinhole camera model (Brown-Conrady distortion) and
    epipolar ray intersection.
matching
    ``SoftSinkhornMatcher`` -- entropy-regularized optimal transport for
    soft stereo correspondence.
tracking
    ``DifferentiableSegmentTracker`` (soft frame-to-frame linkage) and
    ``DifferentiableSavitzkyGolay`` (smooth velocity/acceleration).
physics_loss
    The end-to-end Lagrangian turbulence physics loss (kurtosis, spectral,
    track-length, reprojection, ghost-penalty terms) that gradients flow
    back through to Stage-1 micro-parameters.

Requires the ``differentiable`` extra (``uv sync --extra differentiable``);
``torch`` is not a core dependency.
"""

from openptv2.differentiable.centroiding import (
    DifferentiableGaussianFit,
    SoftArgmax2D,
    gaussian_moment_fit,
    soft_argmax_2d,
    soft_threshold,
)
from openptv2.differentiable.geometry import closest_point_between_rays, project_pinhole
from openptv2.differentiable.matching import SoftSinkhornMatcher, sinkhorn_soft_assign
from openptv2.differentiable.physics_loss import (
    PhysicsLossWeights,
    delta_kurtosis_loss,
    ghost_penalty,
    kurtosis,
    reprojection_error,
    spectral_loss,
    total_physics_loss,
    track_length_term,
    velocity_power_spectrum,
)
from openptv2.differentiable.tracking import (
    DifferentiableSavitzkyGolay,
    DifferentiableSegmentTracker,
    differentiable_savitzky_golay,
    savitzky_golay_kernels,
)

__all__ = [
    "soft_threshold",
    "soft_argmax_2d",
    "SoftArgmax2D",
    "gaussian_moment_fit",
    "DifferentiableGaussianFit",
    "project_pinhole",
    "closest_point_between_rays",
    "sinkhorn_soft_assign",
    "SoftSinkhornMatcher",
    "DifferentiableSegmentTracker",
    "savitzky_golay_kernels",
    "differentiable_savitzky_golay",
    "DifferentiableSavitzkyGolay",
    "kurtosis",
    "delta_kurtosis_loss",
    "velocity_power_spectrum",
    "spectral_loss",
    "track_length_term",
    "reprojection_error",
    "ghost_penalty",
    "PhysicsLossWeights",
    "total_physics_loss",
]
