
import numpy as np
from flowtracks.io import trajectories_ptvis  # Expose for testing/monkeypatching

from openptv2.imgcoord import image_coordinates
from openptv2.transforms import convert_arr_metric_to_pixel


def compute_flowtracks_trajectories_from_guiobj(guiobj):
    """
    Compute 2D projected trajectories for each camera from a flowtracks dataset, using info.object from GUI.
    Returns dict with keys: heads_x, heads_y, tails_x, tails_y, ends_x, ends_y (each is a list of lists per camera)
    """
    seq_params = guiobj.get_parameter("sequence")
    seq_first = seq_params["first"]
    seq_last = seq_params["last"]
    seq_params["base_name"]

    dataset = []
    from openptv2.storage import RunStore

    store = RunStore.open(getattr(guiobj, "exp_path", "."), mode="a")
    dataset = store.to_flowtracks_trajectories(
        first=seq_first, last=seq_last, traj_min_len=3
    )
    cals = guiobj.cals
    cpar = guiobj.cpar
    num_cams = guiobj.num_cams

    heads_x, heads_y = [], []
    tails_x, tails_y = [], []
    ends_x, ends_y = [], []
    for i_cam in range(num_cams):
        head_x, head_y = [], []
        tail_x, tail_y = [], []
        end_x, end_y = [], []
        for traj in dataset:
            pos_arr = np.array(traj.pos()) * 1000
            projected = image_coordinates(
                np.atleast_2d(pos_arr),
                cals[i_cam],
                cpar.get_multimedia_params(),
            )
            pos = convert_arr_metric_to_pixel(projected, cpar)
            head_x.append(pos[0, 0])
            head_y.append(pos[0, 1])
            tail_x.extend(list(pos[1:-1, 0]))
            tail_y.extend(list(pos[1:-1, 1]))
            end_x.append(pos[-1, 0])
            end_y.append(pos[-1, 1])
        heads_x.append(head_x)
        heads_y.append(head_y)
        tails_x.append(tail_x)
        tails_y.append(tail_y)
        ends_x.append(end_x)
        ends_y.append(end_y)
    return dict(
        heads_x=heads_x,
        heads_y=heads_y,
        tails_x=tails_x,
        tails_y=tails_y,
        ends_x=ends_x,
        ends_y=ends_y,
    )


def export_ptv_is_to_paraview(
    ptv_is_pattern="res/ptv_is.%d", output_dir="./res", xuap=False
):
    """
    Reads ptv_is.# files or Zarr store and exports per-frame CSVs for Paraview visualization.
    Each output file is named ptv_<frame>.txt and contains columns:
    particle, x, y, z, dx, dy, dz
    """
    import pandas as pd

    from openptv2.storage import find_existing_store

    dataset = []
    zarr_store = find_existing_store(output_dir)
    if zarr_store is not None:
        try:
            from openptv2.storage import RunStore

            store = RunStore(zarr_store, mode="a")
            dataset = store.to_flowtracks_trajectories()
        except Exception:
            dataset = []

    if not dataset:
        dataset = trajectories_ptvis(ptv_is_pattern, xuap=xuap)


    dataframes = []
    for traj in dataset:
        dataframes.append(
            pd.DataFrame.from_records(
                traj, columns=["x", "y", "z", "dx", "dy", "dz", "frame", "particle"]
            )
        )
    if not dataframes:
        print("No trajectories found.")
        return
    df = pd.concat(dataframes, ignore_index=True)
    df["particle"] = df["particle"].astype(np.int32)
    df["frame"] = df["frame"].astype(np.int32)
    df.reset_index(inplace=True, drop=True)
    df_grouped = df.groupby("frame")
    for index, group in df_grouped:
        group.to_csv(
            f"{output_dir}/ptv_{index:05d}.txt",
            mode="w",
            columns=["particle", "x", "y", "z", "dx", "dy", "dz"],
            index=False,
        )
    print(
        f"Saving trajectories to Paraview finished. {len(df_grouped)} frames exported."
    )

