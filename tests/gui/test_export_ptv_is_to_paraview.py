import numpy as np
import pandas as pd

from openptv2.gui.flowtracks_utils import export_ptv_is_to_paraview
from openptv2.storage import RunStore


def test_export_ptv_is_to_paraview(tmp_path):
    store = RunStore(tmp_path / "run.zarr", mode="w")
    store.write_linkage(
        0,
        np.array([-1, -1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
    )
    store.write_linkage(
        1,
        np.array([0, 1], dtype=np.int32),
        np.array([-2, -2], dtype=np.int32),
        np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]], dtype=np.float64),
    )

    outdir = tmp_path
    export_ptv_is_to_paraview(output_dir=str(outdir), xuap=False)
    # Check that output files exist and have correct content
    files = list(outdir.glob("ptv_*.txt"))
    assert files, "No output files created"
    # Read and check content
    for f in files:
        df = pd.read_csv(f, sep=",|	", engine="python")
        assert set(["particle", "x", "y", "z", "dx", "dy", "dz"]).issubset(df.columns)
        assert not df.empty
