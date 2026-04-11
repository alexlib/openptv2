"""
Debug correspondences to isolate SoA refactoring bugs.

Small unit tests comparing against liboptv references.
"""
import numpy as np
import pytest
from pathlib import Path
from ..conftest import (
    calibration_files,
    control_params_file,
    volume_params_file,
    FIXTURES,
)


class TestCorrespondencesBasic:
    """Minimal correspondences tests to verify SoA correctness."""

    def test_read_targets_preserves_data(self):
        """Test that read_targets correctly preserves target data."""
        from algorithms.tracking_frame_buf import read_targets, Target

        targets = read_targets(str(FIXTURES / "frame" / "cam1."), 333)
        
        # Verify targets read correctly
        assert isinstance(targets, list)
        assert len(targets) > 0
        for t in targets:
            assert isinstance(t, Target)
            print(f"Target: pnr={t.pnr}, x={t.x:.3f}, y={t.y:.3f}, n={t.n}")

    def test_match_coords_basic(self, file_control_params, file_calibration_cam1):
        """Test match_coords SoA conversion."""
        from algorithms.tracking_frame_buf import read_targets, match_coords

        optv_cpar, python_cpar = file_control_params
        optv_cal, python_cal = file_calibration_cam1
        
        targets1 = read_targets(str(FIXTURES / "frame" / "cam1."), 333)
        targets2 = read_targets(str(FIXTURES / "frame" / "cam2."), 333)
        
        print(f"\nTargets cam1: {len(targets1)}")
        print(f"Targets cam2: {len(targets2)}")
        
        # match_coords should return sorted x, y, pnr arrays
        x_sorted, y_sorted, pnr_sorted = match_coords(
            targets1[:min(10, len(targets1))], python_cpar, python_cal
        )
        
        print(f"\nMatched coords: {len(x_sorted)}")
        print(f"X sorted: {x_sorted}")
        print(f"Y sorted: {y_sorted}")
        print(f"PNR sorted: {pnr_sorted}")
        
        # Verify x is sorted
        assert np.all(x_sorted[1:] >= x_sorted[:-1]), "X not sorted!"
        
    def test_correspondences_simple_data(self, file_control_params, file_volume_params, file_calibration_cam1, file_calibration_4cam):
        """Test correspondences function with small data."""
        from algorithms.tracking_frame_buf import Frame, read_targets, Target, match_coords
        from algorithms import correspondences as corr_module

        optv_cpar, python_cpar = file_control_params
        optv_vpar, python_vpar = file_volume_params
        optv_cal1, python_cal1 = file_calibration_cam1
        optv_cals4, python_cals4 = file_calibration_4cam
       
        # Check cpar num_cams
        num_cams = python_cpar.num_cams if hasattr(python_cpar, 'num_cams') else 4
        print(f"num_cams from cpar: {python_cpar.num_cams}")
        
        # Create frame with correct number of cameras
        frm = Frame(num_cams=num_cams)
        
        cam1_targets = read_targets(str(FIXTURES / "frame" / "cam1."), 333)[:5]
        cam2_targets = read_targets(str(FIXTURES / "frame" / "cam2."), 333)[:5]
        
        for i, t in enumerate(cam1_targets):
            frm.targets[0][i] = t
            frm.num_targets[0] += 1
            
        for i, t in enumerate(cam2_targets):
            frm.targets[1][i] = t
            frm.num_targets[1] += 1
        
        # Zero out remaining cameras
        for cam_idx in range(2, num_cams):
            frm.num_targets[cam_idx] = 0
        
        print(f"\nFrame cam0: {frm.num_targets[0]} targets")
        print(f"Frame cam1: {frm.num_targets[1]} targets")
        
        # Create matched coords
        corrected = []
        cals = python_cals4
        for i in range(num_cams):
            if i < 2:
                # Use actual targets for first 2 cameras
                x, y, pnr = match_coords(
                    [frm.targets[i][j] for j in range(frm.num_targets[i])],
                    python_cpar, cals[i]
                )
            else:
                # Empty data for remaining cameras
                x = np.array([], dtype=np.float64)
                y = np.array([], dtype=np.float64)
                pnr = np.array([], dtype=np.int64)
            
            # Convert to recarray with x, y, pnr fields for compatibility
            crd = np.recarray((len(pnr),), dtype=[('x', np.float64), ('y', np.float64), ('pnr', np.int64)])
            crd.x = x
            crd.y = y
            crd.pnr = pnr
            corrected.append(crd)
            print(f"  Camera {i}: matched={len(pnr)}")
        
        # Now call correspondences  
        match_counts = [0, 0, 0, 0]
        try:
            con = corr_module.correspondences(frm, corrected, python_vpar, python_cpar, cals, match_counts)
            print(f"\nCorrespondences result:")
            print(f"  con dtype: {con.dtype}")
            print(f"  con shape: {con.shape}")
            print(f"  match_counts: {match_counts}")
            if len(con) > 0:
                print(f"  First correspondence: p={con[0].p}, corr={con[0].corr}")
        except Exception as e:
            print(f"\nERROR in correspondences: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
