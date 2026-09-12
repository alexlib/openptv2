import numpy as np
import pandas as pd
import pytest

from openptv2.gui.flowtracks_utils import export_ptv_is_to_paraview


def make_fake_ptv_is_files(tmpdir, n_frames=3, n_particles=2):
    """Create fake ptv_is.# files for testing export_ptv_is_to_paraview."""
    # We'll create a minimalistic trajectories_ptvis-compatible loader
    # But for this test, we patch the function in the module
    # Instead, we create a fake loader below
    # This is a placeholder for a real test with actual files
    pass  # The real test will patch trajectories_ptvis


def test_export_ptv_is_to_paraview(monkeypatch, tmp_path):
    # Prepare fake data to be returned by trajectories_ptvis
    fake_trajs = [
        [
            (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0, 42),
            (1.1, 2.1, 3.1, 0.1, 0.2, 0.3, 0, 42),
        ],
        [
            (4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 1, 43),
            (4.1, 5.1, 6.1, 0.4, 0.5, 0.6, 1, 43),
        ],
    ]

    def fake_trajectories_ptvis(pattern, xuap=False):
        return fake_trajs

    monkeypatch.setattr(
        "openptv2.gui.flowtracks_utils.trajectories_ptvis", fake_trajectories_ptvis
    )
    outdir = tmp_path
    export_ptv_is_to_paraview(
        ptv_is_pattern="doesnotmatter", output_dir=str(outdir), xuap=False
    )
    # Check that output files exist and have correct content
    files = list(outdir.glob("ptv_*.txt"))
    assert files, "No output files created"
    # Read and check content
    for f in files:
        df = pd.read_csv(f, sep=",|	", engine="python")
        assert set(["particle", "x", "y", "z", "dx", "dy", "dz"]).issubset(df.columns)
        assert not df.empty


def test_export_ptv_is_to_paraview_writes_vtp_when_zarr_store_exists(tmp_path):
    """When a run.zarr store exists, export writes a .vtp instead of CSVs."""
    pytest.importorskip("pyvista")
    import zarr

    n_traj, n_pts = 3, 4
    pos = np.random.default_rng(0).uniform(0, 1, (n_traj * n_pts, 3))
    vel = np.random.default_rng(1).normal(0, 1, (n_traj * n_pts, 3))
    time = np.tile(np.arange(n_pts), n_traj)
    trajid = np.repeat(np.arange(n_traj), n_pts)

    group = zarr.open_group(str(tmp_path / "run.zarr"), mode="w")
    traj_group = group.create_group("trajectories")
    traj_group["pos"] = pos
    traj_group["vel"] = vel
    traj_group["time"] = time
    traj_group["trajid"] = trajid

    export_ptv_is_to_paraview(output_dir=str(tmp_path))

    vtp_path = tmp_path / "trajectories.vtp"
    assert vtp_path.exists()
    assert not list(tmp_path.glob("ptv_*.txt"))

    import pyvista as pv

    poly = pv.read(vtp_path)
    assert poly.n_points == n_traj * n_pts
    assert poly.n_lines == n_traj
