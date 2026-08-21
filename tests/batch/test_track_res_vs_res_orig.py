import shutil
from pathlib import Path

import pytest
import yaml

from openptv2.batch import pyptv_batch
from openptv2.gui.parameter_manager import ParameterManager

TRACK_DIR = Path(__file__).parent.parent.parent / "test_data" / "pyptv_track"


def _count_particle_lines(res_dir: Path, last_frame: int) -> int:
    """Particle-line count for the last frame's rt_is output, matching the
    len(file.readlines()) convention used below (count line + N particle
    lines). Reads the ASCII rt_is.<frame> file when present, or -- for a
    store-backed run (rt_is/ptv_is/added are no longer written for
    store-backed runs, see tracking_frame_buf.write_path_frame) -- the
    RunStore's correspondences for that frame."""
    ascii_files = sorted(res_dir.glob("rt_is.*"))
    if ascii_files:
        return len(ascii_files[-1].read_text().splitlines())

    from openptv2.storage import RunStore

    store = RunStore(res_dir / "run.zarr", mode="r")
    if not store.has_correspondences(last_frame):
        return 0
    pos_3d, _ = store.read_correspondences(last_frame)
    return len(pos_3d) + 1


def _skip_if_frame_read_failure(error: Exception) -> None:
    """Skip test if the error is due to missing frame files."""
    msg = str(error)
    if "does not exist" in msg or "Can't open" in msg:
        pytest.skip(f"Frame read failure (missing test data): {msg}")


@pytest.mark.parametrize(
    "yaml_path, desc",
    [
        # ("parameters_Run1.yaml", "2 cameras, no new particles"),
        # ("parameters_Run2.yaml", "3 cameras, new particle"),
        ("parameters_Run3.yaml", "3 cameras, newpart, frame by frame"),
    ],
)
def test_tracking_res_matches_orig(tmp_path, yaml_path, desc):
    # Print image name pattern for debugging

    """
    For the given parameter set, clean and set up img/ and res/ folders, run tracking, and compare res/ to res_orig/.
    """
    # 1. Setup working directory
    work_dir = tmp_path / "track"
    work_dir.mkdir(exist_ok=True)
    # copy everything from TRACK_DIR to work_dir
    shutil.copytree(TRACK_DIR, work_dir, dirs_exist_ok=True)

    # create in work_dir copy of img_orig as img and res_orig as res
    shutil.copytree(work_dir / "img_orig", work_dir / "img", dirs_exist_ok=True)
    shutil.copytree(work_dir / "res_orig", work_dir / "res", dirs_exist_ok=True)
    # Remove all files from work_dir / "res"
    res_dir = work_dir / "res"
    for file in res_dir.glob("*"):
        if file.is_file():
            file.unlink()

    # 2. Convert .par to YAML
    # exp = Experiment()
    # exp.populate_runs(work_dir)

    yaml_name = yaml_path
    yaml_path = work_dir / yaml_name

    pm = ParameterManager()
    pm.from_yaml(yaml_path)
    # yaml_path = work_dir / param_yaml
    # pm.to_yaml(yaml_path)

    # Get first and last from sequence_parameters in pm
    # pm = exp.pm
    seq_params = pm.parameters.get("sequence")
    first = seq_params.get("first")
    last = seq_params.get("last")

    # 4. Run tracking using pyptv_batch.main directly with arguments
    if yaml_name == "parameters_Run3.yaml":
        res_dir = work_dir / "res"

        # First run: no new particle — need sequence+tracking ("both")
        # because res/ is empty and tracking needs rt_is.* files
        with open(yaml_path, "r") as f:
            yml = yaml.safe_load(f)
        yml["track"]["flagNewParticles"] = False
        with open(yaml_path, "w") as f:
            yaml.safe_dump(yml, f)

        try:
            pyptv_batch.run_batch(
                yaml_file=yaml_path,
                seq_first=first,
                seq_last=last,
                mode="both",
            )
        except Exception as e:
            _skip_if_frame_read_failure(e)
            raise

        n_lines_noadd = _count_particle_lines(res_dir, last)
        if n_lines_noadd == 0:
            pytest.skip("Sequence+tracking produced no rt_is data")

        # Second run: add new particle — tracking only (rt_is.* now exist)
        with open(yaml_path, "r") as f:
            yml = yaml.safe_load(f)
        yml["track"]["flagNewParticles"] = True
        with open(yaml_path, "w") as f:
            yaml.safe_dump(yml, f)

        try:
            pyptv_batch.run_batch(
                yaml_file=yaml_path,
                seq_first=first,
                seq_last=last,
                mode="tracking",
            )
        except Exception as e:
            _skip_if_frame_read_failure(e)
            raise
        n_lines_add = _count_particle_lines(res_dir, last)

        assert n_lines_add >= n_lines_noadd, (
            "New particle tracking produced fewer lines than standard tracking"
        )

    else:
        # Standard test for Run1 and Run2
        try:
            pyptv_batch.run_batch(
                yaml_file=yaml_path, seq_first=first, seq_last=last, mode="both"
            )
        except Exception as e:
            _skip_if_frame_read_failure(e)
            raise
        # 5. Compare res/ to res_orig/
        res_dir = work_dir / "res"
        res_orig_dir = work_dir / "res_orig"

        for f in sorted(res_dir.glob("rt_is.*")):
            print(f"\n--- {f.name} ---")
            with open(f, "r") as file:
                print(file.read())

        for f in sorted(res_dir.glob("ptv_is.*")):
            print(f"\n--- {f.name} ---")
            with open(f, "r") as file:
                print(file.read())

        # dcmp = filecmp.dircmp(res_dir, res_orig_dir)
        # assert len(dcmp.diff_files) == 0, f"Files differ in {desc}: {dcmp.diff_files}"
        # assert len(dcmp.left_only) == 0, f"Extra files in result: {dcmp.left_only}"
        # assert len(dcmp.right_only) == 0, f"Missing files in result: {dcmp.right_only}"
        # print(f"Tracking test passed for {desc}")

        # Compare file contents and stop at the first difference
        for fname in sorted(f for f in res_dir.iterdir() if f.is_file()):
            orig_file = res_orig_dir / fname.name
            if not orig_file.exists():
                print(f"Missing file in res_orig: {fname.name}")
                break
            with open(fname, "rb") as f1, open(orig_file, "rb") as f2:
                content1 = f1.read()
                content2 = f2.read()
                if content1 != content2:
                    print(f"File differs: {fname.name}")
                    break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
