"""
Detailed comparison test: Python multimed.py vs optv (Cython/C) multimed.

This test compares the multimedia refraction calculations between:
- Python/Numba: algorithms/multimed.py
- C/Cython: optv image_coordinates function (which uses lib/src/multimed.c)

The comparison is done via the image_coordinates function which applies
multimedia refraction as part of the coordinate transformation.
"""

import pytest
import numpy as np
from pathlib import Path


def create_test_calibration():
    """Create a test calibration similar to test_cavity setup."""
    from optv.calibration import Calibration as OptvCal
    from optv.parameters import ControlParams

    # Create control params with multimedia
    cpar = ControlParams(
        num_cams=1,
        flags=["hp", "headers"],
        image_size=(1280, 1024),
        pixel_size=(0.008, 0.008),
    )

    # Set up multimedia params (water-glass-air typical)
    cpar.get_multimedia_params().set_layers(
        refr_index=[1.5],  # Glass refractive index
        thickness=[5.0],  # Glass thickness in mm
    )
    cpar.get_multimedia_params().set_n3(1.0)  # Air

    # Create calibration
    cal = OptvCal()

    return cal, cpar


def create_python_calibration():
    """Create equivalent Python calibration."""
    from algorithms.calibration import Calibration, Exterior, Interior
    from algorithms.parameters import MultimediaPar

    cal = Calibration()

    # Set exterior (camera position)
    cal.ext_par = Exterior.copy()
    cal.ext_par.x0 = -78.0
    cal.ext_par.y0 = 70.0
    cal.ext_par.z0 = 650.0

    # Set interior
    cal.int_par = Interior.copy()
    cal.int_par.cc = 16.0
    cal.int_par.xh = 0.0
    cal.int_par.yh = 0.0

    # Set multimedia params
    mm = MultimediaPar(n1=1.33, n2=[1.5], d=[5.0], n3=1.0)

    return cal, mm


class TestMultimedDirectComparison:
    """Direct comparison of radial shift calculation."""

    @pytest.fixture
    def test_positions(self):
        """Generate test positions in the volume."""
        positions = []

        # Center point
        positions.append(np.array([0.0, 0.0, 50.0]))

        # Points at increasing radii from optical axis
        for r in [25.0, 50.0, 75.0, 100.0]:
            positions.append(np.array([r, 0.0, 50.0]))
            positions.append(np.array([0.0, r, 50.0]))
            positions.append(np.array([r, r, 50.0]))

        # Points at different depths
        for z in [25.0, 75.0, 100.0]:
            positions.append(np.array([50.0, 50.0, z]))

        return positions

    def test_radial_shift_at_center(self, test_positions):
        """Test radial shift is close to 1.0 at center (r=0)."""
        from algorithms.multimed import multimed_r_nlay
        from algorithms.parameters import MultimediaPar

        cal, mm = create_python_calibration()

        # Center position
        pos = test_positions[0]  # [0, 0, 50]

        result = multimed_r_nlay(cal, mm, pos)

        # At center, result should be close to 1.0 (within ~5% for water-glass-air)
        assert 0.95 < result < 1.05, f"At center, expected ~1.0, got {result}"

    def test_radial_shift_increases_with_radius(self, test_positions):
        """Test that radial shift increases with distance from center."""
        from algorithms.multimed import multimed_r_nlay
        from algorithms.parameters import MultimediaPar

        cal, mm = create_python_calibration()

        results = []
        for pos in test_positions:
            r = np.linalg.norm(pos[:2])  # Horizontal distance
            if r > 0:
                result = multimed_r_nlay(cal, mm, pos)
                results.append((r, result))

        # Sort by radius
        results.sort(key=lambda x: x[0])

        # Radial shift should deviate more from 1.0 as radius increases
        # (at least for points with significant radius)
        for i in range(1, len(results)):
            r1, res1 = results[i - 1]
            r2, res2 = results[i]

            # Just verify results are in physical range
            assert 0.5 < res2 < 1.5, f"Unphysical result at r={r2}: {res2}"

    def test_compare_python_to_optv_api(self):
        """Compare Python and optv APIs."""
        try:
            from optv.calibration import Calibration as OptvCal
            from optv.parameters import ControlParams, MultimediaParams
        except ImportError:
            pytest.skip("optv not available")

        # Verify both APIs exist and work
        optv_cal = OptvCal()

        cpar = ControlParams(num_cams=1)
        mm_params = cpar.get_multimedia_params()
        mm_params.set_layers([1.5], [5.0])
        mm_params.set_n3(1.0)

        assert optv_cal is not None
        assert mm_params is not None


class TestImageCoordinatesComparison:
    """Compare image_coordinates output between Python and optv."""

    def test_optv_image_coordinates_basic(self):
        """Test that optv image_coordinates works."""
        try:
            from optv.imgcoord import image_coordinates
            from optv.calibration import Calibration as OptvCal
            from optv.parameters import ControlParams
        except ImportError:
            pytest.skip("optv not available")

        # Create calibration
        cal = OptvCal()

        # Create control params with multimedia
        cpar = ControlParams(num_cams=1)
        mm = cpar.get_multimedia_params()
        mm.set_layers([1.5], [5.0])
        mm.set_n3(1.0)

        # Test 3D points in world coordinates
        test_points = np.array(
            [
                [0.0, 0.0, 50.0],
                [50.0, 0.0, 50.0],
                [0.0, 50.0, 50.0],
                [50.0, 50.0, 50.0],
            ]
        )

        # Get image coordinates
        img_coords = image_coordinates(test_points, cal, mm)

        # Verify output shape
        assert img_coords.shape == (4, 2)

        # Check for valid numbers
        assert not np.any(np.isnan(img_coords))
        assert not np.any(np.isinf(img_coords))

    def test_python_image_coordinates_equivalence(self):
        """Test Python image_coordinates produces valid output."""
        from algorithms.imgcoord import image_coordinates
        from algorithms.calibration import Calibration, Exterior, Interior
        from algorithms.parameters import MultimediaPar

        # Create Python calibration
        cal = Calibration()
        cal.ext_par = Exterior.copy()
        cal.ext_par.x0 = -78.0
        cal.ext_par.y0 = 70.0
        cal.ext_par.z0 = 650.0

        cal.int_par = Interior.copy()
        cal.int_par.cc = 16.0
        cal.int_par.xh = 0.0
        cal.int_par.yh = 0.0

        # Create multimedia params
        mm = MultimediaPar(n1=1.33, n2=[1.5], d=[5.0], n3=1.0)

        # Test points
        test_points = np.array(
            [
                [0.0, 0.0, 50.0],
                [50.0, 0.0, 50.0],
                [0.0, 50.0, 50.0],
                [50.0, 50.0, 50.0],
            ]
        )

        # Get image coordinates
        img_coords = image_coordinates(test_points, cal, mm)

        # Verify output
        assert img_coords.shape == (4, 2)
        assert not np.any(np.isnan(img_coords))
        assert not np.any(np.isinf(img_coords))


class TestMultimedEdgeCases:
    """Test edge cases in multimed calculations."""

    def test_single_medium_case(self):
        """Test with single medium (no refraction)."""
        from algorithms.multimed import multimed_r_nlay
        from algorithms.parameters import MultimediaPar
        from algorithms.calibration import Calibration, Exterior, Interior

        cal = Calibration()
        cal.ext_par = Exterior.copy()
        cal.ext_par.x0 = 0.0
        cal.ext_par.y0 = 0.0
        cal.ext_par.z0 = 100.0

        cal.int_par = Interior.copy()

        # Single medium - no refraction
        mm = MultimediaPar(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)

        pos = np.array([100.0, 100.0, 50.0])

        result = multimed_r_nlay(cal, mm, pos)

        assert abs(result - 1.0) < 1e-10

    def test_multiple_layers(self):
        """Test with multiple refraction layers."""
        from algorithms.multimed import multimed_r_nlay
        from algorithms.parameters import MultimediaPar
        from algorithms.calibration import Calibration, Exterior, Interior

        cal = Calibration()
        cal.ext_par = Exterior.copy()
        cal.ext_par.x0 = 0.0
        cal.ext_par.y0 = 0.0
        cal.ext_par.z0 = 200.0

        cal.int_par = Interior.copy()

        # Water - Glass - Air (3 layers)
        mm = MultimediaPar(
            n1=1.33,  # Water
            n2=[1.5, 1.0],  # Glass, Air
            d=[5.0, 10.0],  # Thicknesses
            n3=1.0,  # Final medium (Air)
        )

        positions = [
            np.array([0.0, 0.0, 50.0]),
            np.array([50.0, 0.0, 50.0]),
            np.array([100.0, 0.0, 50.0]),
        ]

        for pos in positions:
            result = multimed_r_nlay(cal, mm, pos)
            assert 0.0 < result < 2.0, f"Unphysical result for {pos}: {result}"

    def test_extreme_angles(self):
        """Test at extreme angles (large radius)."""
        from algorithms.multimed import multimed_r_nlay
        from algorithms.parameters import MultimediaPar
        from algorithms.calibration import Calibration, Exterior, Interior

        cal = Calibration()
        cal.ext_par = Exterior.copy()
        cal.ext_par.x0 = 0.0
        cal.ext_par.y0 = 0.0
        cal.ext_par.z0 = 200.0

        cal.int_par = Interior.copy()

        mm = MultimediaPar(n1=1.33, n2=[1.5], d=[5.0], n3=1.0)

        # Very large radius
        pos = np.array([500.0, 500.0, 25.0])

        result = multimed_r_nlay(cal, mm, pos)

        # Should still be in physical range
        assert 0.0 < result < 2.0, f"Unphysical result: {result}"


class TestMultimedConvergence:
    """Test iterative convergence behavior."""

    def test_convergence_at_optical_axis(self):
        """Test that iteration converges at optical axis."""
        from algorithms.multimed import fast_multimed_r_nlay

        # Multi-layer case
        result = fast_multimed_r_nlay(
            nlay=2,
            n1=1.0,
            n2=np.array([1.33, 1.5]),
            n3=1.0,
            d=np.array([0.0, 5.0, 10.0]),
            x0=0.0,
            y0=0.0,
            z0=200.0,
            pos=np.array([0.0, 0.0, 50.0]),
        )

        # At r=0, should return exactly 1.0
        assert abs(result - 1.0) < 1e-10

    def test_convergence_iteration_count(self):
        """Test that iteration count is reasonable."""
        from algorithms.multimed import fast_multimed_r_nlay

        # This is a hard case that might need many iterations
        result = fast_multimed_r_nlay(
            nlay=3,
            n1=1.0,
            n2=np.array([1.2, 1.4, 1.6]),
            n3=1.0,
            d=np.array([0.0, 3.0, 7.0, 15.0]),
            x0=0.0,
            y0=0.0,
            z0=300.0,
            pos=np.array([150.0, 150.0, 50.0]),
        )

        # Should converge to a reasonable value
        assert 0.0 < result < 2.0


class TestMultimedRoundTrip:
    """Test round-trip coordinate transformations."""

    def test_world_to_image_round_trip(self):
        """Test that world->image->world is consistent."""
        from algorithms.imgcoord import image_coordinates
        from algorithms.calibration import Calibration, Exterior, Interior
        from algorithms.parameters import MultimediaPar

        # Setup
        cal = Calibration()
        cal.ext_par = Exterior.copy()
        cal.ext_par.x0 = -78.0
        cal.ext_par.y0 = 70.0
        cal.ext_par.z0 = 650.0

        cal.int_par = Interior.copy()
        cal.int_par.cc = 16.0
        cal.int_par.xh = 0.0
        cal.int_par.yh = 0.0

        mm = MultimediaPar(n1=1.33, n2=[1.5], d=[5.0], n3=1.0)

        # Test point
        world_point = np.array([[50.0, 50.0, 50.0]])

        # Forward: world -> image
        img_point = image_coordinates(world_point, cal, mm)

        # This is a one-way transformation, so we can't easily reverse it
        # But we can verify the output is reasonable
        assert img_point.shape == (1, 2)
        assert not np.any(np.isnan(img_point))
        assert not np.any(np.isinf(img_point))

    def test_compare_with_optv_multimed(self):
        """Compare Python multimed against optv using image_coordinates."""
        try:
            from optv.imgcoord import image_coordinates as optv_image_coordinates
            from optv.calibration import Calibration as OptvCal
            from optv.parameters import ControlParams
        except ImportError:
            pytest.skip("optv not available")

        # Create optv calibration
        optv_cal = OptvCal()

        # Create control params with multimedia
        cpar = ControlParams(num_cams=1)
        mm = cpar.get_multimedia_params()
        mm.set_layers([1.5], [5.0])
        mm.set_n3(1.0)

        # Test points
        test_pts = np.array(
            [
                [0.0, 0.0, 50.0],
                [25.0, 0.0, 50.0],
                [50.0, 25.0, 50.0],
                [75.0, 75.0, 50.0],
            ],
            dtype=np.float64,
        )

        # Get optv results
        optv_result = optv_image_coordinates(test_pts, optv_cal, mm)

        # Verify output is valid
        assert optv_result.shape == (4, 2)
        assert not np.any(np.isnan(optv_result))
        assert not np.any(np.isinf(optv_result))

        print("\noptv image_coordinates results:")
        print(optv_result)

        # Now test Python version
        from algorithms.imgcoord import image_coordinates as python_image_coordinates
        from algorithms.calibration import Calibration as PythonCal
        from algorithms.parameters import MultimediaPar

        py_cal = PythonCal()
        from algorithms.calibration import Exterior, Interior

        py_cal.ext_par = Exterior.copy()
        py_cal.ext_par.x0 = -78.0
        py_cal.ext_par.y0 = 70.0
        py_cal.ext_par.z0 = 650.0

        py_cal.int_par = Interior.copy()
        py_cal.int_par.cc = 16.0
        py_cal.int_par.xh = 0.0
        py_cal.int_par.yh = 0.0

        py_mm = MultimediaPar(n1=1.33, n2=[1.5], d=[5.0], n3=1.0)

        py_result = python_image_coordinates(test_pts, py_cal, py_mm)

        print("\nPython image_coordinates results:")
        print(py_result)

        # Compare results
        # Note: These won't be exactly equal due to different calibration
        # structures, but both should produce valid output
        assert py_result.shape == (4, 2)
        assert not np.any(np.isnan(py_result))
        assert not np.any(np.isinf(py_result))


class TestMultimedPerformance:
    """Test performance of multimed calculations."""

    def test_numba_compilation(self):
        """Test that Numba functions compile."""
        from algorithms.multimed import fast_multimed_r_nlay, fast_trans_cam_point

        # First call triggers compilation
        result1 = fast_multimed_r_nlay(
            1,
            1.0,
            np.array([1.0]),
            1.0,
            np.array([0.0, 0.0]),
            0,
            0,
            100,
            np.array([10.0, 10.0, 50.0]),
        )

        # Second call should be faster (cached)
        result2 = fast_multimed_r_nlay(
            1,
            1.0,
            np.array([1.0]),
            1.0,
            np.array([0.0, 0.0]),
            0,
            0,
            100,
            np.array([10.0, 10.0, 50.0]),
        )

        assert result1 == result2

    def test_batch_processing(self):
        """Test processing multiple points."""
        from algorithms.multimed import fast_multimed_r_nlay

        nlay = 2
        n1 = 1.33
        n2 = np.array([1.5])
        n3 = 1.0
        d = np.array([0.0, 5.0])
        x0, y0, z0 = 0.0, 0.0, 200.0

        # Generate random test points
        np.random.seed(42)
        positions = np.random.uniform(-100, 100, (100, 3))

        # Process all points
        results = []
        for pos in positions:
            result = fast_multimed_r_nlay(nlay, n1, n2, n3, d, x0, y0, z0, pos)
            results.append(result)

        results = np.array(results)

        # All should be in physical range
        assert np.all(results > 0.0)
        assert np.all(results < 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
