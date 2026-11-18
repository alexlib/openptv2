import os
import shutil
from pathlib import Path

import pytest

from openptv2.gui.parameter_manager import ParameterManager
from openptv2.tracker import Tracker, default_naming


@pytest.mark.usefixtures("tmp_path")
def test_tracker_minimal(tmp_path):
    # Use the real test data from test_data/test_cavity
    test_data_dir = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    # Copy all necessary files and folders to tmp_path for isolation
    # Copy 'cal' folder
    shutil.copytree(test_data_dir / "cal", tmp_path / "cal")
    # Copy 'img_orig' to 'img'
    shutil.copytree(test_data_dir / "img_orig", tmp_path / "img")
    # Copy 'res_orig' to 'res'
    shutil.copytree(test_data_dir / "res_orig", tmp_path / "res")
    # Copy parameters file
    shutil.copy(
        test_data_dir / "parameters_Run1.yaml", tmp_path / "parameters_Run1.yaml"
    )

    # Change working directory to tmp_path
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Load parameters using ParameterManager
        param_path = tmp_path / "parameters_Run1.yaml"
        pm = ParameterManager()
        pm.from_yaml(param_path)

        from openptv2.gui.ptv import py_start_proc_c

        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)

        for cam_id, short_name in enumerate(pm.get_target_filenames()):
            # print(f"Setting tracker image base name for cam {cam_id+1}: {Path(short_name).resolve()}")
            spar.set_img_base_name(cam_id, str(Path(short_name).resolve()) + ".")

        import numpy as np

        from openptv2.storage import RunStore

        res_dir = tmp_path / "res"
        img_dir = tmp_path / "img"
        store = RunStore(res_dir / "run.zarr", mode="w")
        for rt_file in sorted(res_dir.glob("rt_is.*")):
            frame = int(rt_file.suffix.lstrip("."))
            lines = [ln.strip() for ln in rt_file.read_text().splitlines() if ln.strip()]
            n = int(lines[0]) if lines else 0
            if n > 0:
                rows = np.loadtxt(lines[1 : n + 1], dtype=np.float64).reshape(n, 8)
                pos_3d = rows[:, 1:4]
                cam_ids = rows[:, 4:8].astype(np.int32)
            else:
                pos_3d = np.empty((0, 3), dtype=np.float64)
                cam_ids = np.empty((0, 4), dtype=np.int32)
            store.write_correspondences(frame=frame, pos_3d=pos_3d, cam_target_ids=cam_ids)

        for cam in range(cpar.get_num_cams()):
            for tfile in sorted(img_dir.glob(f"cam{cam + 1}.*_targets")):
                frame = int(tfile.name.split(".")[1].split("_")[0])
                lines = [ln.strip() for ln in tfile.read_text().splitlines() if ln.strip()]
                n = int(lines[0]) if lines else 0
                arr = (
                    np.loadtxt(lines[1 : n + 1], dtype=np.float64).reshape(n, 8)
                    if n > 0
                    else np.empty((0, 8), dtype=np.float64)
                )
                store.write_targets(cam, frame, arr)

        # Set up tracker using loaded parameters
        tracker = Tracker(cpar, vpar, track_par, spar, cals, default_naming, store=store)
        tracker.full_forward()

        # Check that output linkage is created in RunStore and contains tracks
        first = spar.get_first()
        last = spar.get_last()
        for frame in range(first, last + 1):
            assert store.has_linkage(frame, "ptv_is"), (
                f"Output linkage for frame {frame} not created."
            )
            prev_ids, _, _ = store.read_linkage(frame, "ptv_is")
            num_tracks = len(prev_ids)

            # Every frame in [first, last] must be populated. The last frame
            # (written by trackcorr_c_finish) carries each particle's prev-link
            # with next = -2; it must NOT be empty. An empty last frame was the
            # symptom of the forward off-by-one (an extra step that ran one past
            # `last` and clobbered the final write).
            assert num_tracks > 0, f"No tracks found in RunStore frame {frame}."
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
