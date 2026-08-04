"""Unit tests for openptv2.plugins.stb_4d_refinement."""

import numpy as np
import pytest
from openptv2.plugins.stb_4d_refinement import shake_particle_position_3d


def test_stb_shake_particle_position_smoke():
    """Smoke test for 4D STB particle position shaking refinement."""
    pos_3d = np.array([0.0, 0.0, 0.0])
    cals = []  # Empty calibration list -> returns unchanged position
    cpar = None
    image_crops = []

    refined = shake_particle_position_3d(pos_3d, cals, cpar, image_crops, max_iterations=2)
    assert np.allclose(refined, pos_3d)
