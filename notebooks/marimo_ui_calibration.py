# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.9",
#     "wigglystuff>=0.5.22",
#     "matplotlib",
#     "numpy",
#     "scikit-image==0.26.0",
#     "imageio",
#     "pyyaml",
#     "openptv2==0.2.1",
#     "scipy==1.18.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from imageio.v3 import imread
    from skimage.color import rgb2gray
    from skimage.util import img_as_ubyte
    from wigglystuff import ChartPuck

    from openptv2.gui import ptv


    return ChartPuck, img_as_ubyte, imread, mo, np, ptv, rgb2gray


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Interactive Manual Orientation with `pyptv`

    This notebook demonstrates how to load parameters from a YAML file, display calibration images, and allow manual adjustment of orientation points (pucks).

    ### Workflow:
    1.  **Load Parameters**: Using `pyptv.parameter_manager.ParameterManager` to read from `tests/test_cavity/parameters_Run1.yaml`.
    2.  **Display Images**: Loading calibration images specified in `cal_ori` parameters.
    3.  **Interactive Adjustment**: Using `wigglystuff.ChartPuck` to create draggable points for the 4 manual orientation markers on each image.
    4.  **Save Changes**: Updating the in-memory parameters and saving back to the YAML file.
    """)
    return


@app.cell
def _(test_yaml):
    from pathlib import Path

    from openptv2.gui.pyptv.parameter_manager import ParameterManager

    # Path to the YAML file - check LV calibration first, then fallback
    lv_yaml = Path(r"C:\Users\alex\Downloads\hidimaging_test\LV\calibration\parameters_Run_Cal.yaml")

    yaml_path = lv_yaml if lv_yaml.exists() else test_yaml

    # Check if file exists
    if yaml_path.exists():
        pm = ParameterManager()
        pm.from_yaml(yaml_path)
        print(f"YAML loaded successfully from {yaml_path}.")

        # Check keys
        print("Keys in parameters:", pm.parameters.keys())

        # Look for manual orientation parameters
        if "man_ori" in pm.parameters:
            print("\nManual Orientation Parameters (man_ori):")
            print(pm.parameters["man_ori"])

        if "man_ori_coordinates" in pm.parameters:
            print("\nManual Orientation Coordinates (man_ori_coordinates):")
            print(pm.parameters["man_ori_coordinates"])
    else:
        print(f"File not found: {yaml_path}")
    return pm, yaml_path


@app.cell
def _(ChartPuck, img_as_ubyte, imread, mo, pm, ptv, rgb2gray, yaml_path):
    import imageio.v3 as iio

    # Assuming pm is already initialized and populated
    cal_images = pm.parameters["cal_ori"]["img_cal_name"]
    coords = pm.parameters.get("man_ori_coordinates", {})
    # Get the manual orientation IDs (nr)
    man_ori_nr = pm.parameters.get("man_ori", {}).get("nr", [])
    num_cams = pm.parameters.get("num_cams", len(cal_images))

    calibration_widgets = {}

    # Base directory derived from YAML path
    base_dir = yaml_path.parent

    # Check if images need splitting (e.g. 4-view single image)
    ptv_params = pm.parameters.get("ptv", {})
    split_order = ptv_params.get("splitter_order", [0, 1, 3, 2])
    is_splitter = ptv_params.get("splitter", False)

    # Load raw image for camera 0 to test
    first_img_path = (base_dir / cal_images[0]).resolve() if cal_images else None

    H = 0
    W = 0

    # Colors for the 4 pucks (vibrant, distinct, high contrast)
    puck_colors = ["#ff4d6d", "#00b4d8", "#38b000", "#ff9f1c"]

    if is_splitter and first_img_path and first_img_path.exists():
        temp_img = imread(first_img_path)
        if temp_img.ndim > 2:
            temp_img = rgb2gray(temp_img[:, :, :3])

        H, W = temp_img.shape

        x_init = []
        y_init = []

        for _i in range(num_cams):
            _cam_key = f"camera_{_i}"
            _cam_coords = coords.get(_cam_key, {})

            _quadrant = split_order[_i]
            _dx = (_quadrant % 2) * (W // 2)
            _dy = (_quadrant // 2) * (H // 2)

            _start_idx = _i * 4
            _end_idx = _start_idx + 4
            if _end_idx <= len(man_ori_nr):
                _cam_point_ids = man_ori_nr[_start_idx:_end_idx]
            else:
                _cam_point_ids = list(range(1, 5))  # Fallback

            for _pt_idx in range(1, 5):
                _pt_key = f"point_{_pt_idx}"
                _pt = _cam_coords.get(_pt_key, {"x": 100, "y": 100})

                # Map from split to unsplit
                _x_val = _pt["x"] + _dx
                _y_val = _pt["y"] + _dy

                x_init.append(_x_val)
                y_init.append(_y_val)

        def draw_single_view(ax, widget):
            ax.imshow(temp_img, cmap="gray")
            ax.axis("off")

            _global_idx = 0
            for _i in range(num_cams):
                _quadrant = split_order[_i]
                _dx = (_quadrant % 2) * (W // 2)
                _dy = (_quadrant // 2) * (H // 2)

                _start_idx = _i * 4
                _end_idx = _start_idx + 4
                if _end_idx <= len(man_ori_nr):
                    _cam_point_ids = man_ori_nr[_start_idx:_end_idx]
                else:
                    _cam_point_ids = list(range(1, 5))

                for _pt_idx in range(1, 5):
                    if _global_idx < len(widget.x) and _global_idx < len(widget.y):
                        _x_val = widget.x[_global_idx]
                        _y_val = widget.y[_global_idx]

                        if _pt_idx - 1 < len(_cam_point_ids):
                            _pid = _cam_point_ids[_pt_idx - 1]
                            ax.text(
                                _x_val + 20,
                                _y_val + 20,
                                f"C{_i+1}: {_pid}",
                                color="yellow",
                                fontsize=14,
                                fontweight="bold",
                            )
                    _global_idx += 1

        puck = ChartPuck.from_callback(
            draw_fn=draw_single_view,
            x_bounds=(-0.5, W - 0.5),
            y_bounds=(H - 0.5, -0.5),
            figsize=(20, 20),
            x=x_init,
            y=y_init,
            puck_color=puck_colors,
            puck_radius=15,
        )

        widget = mo.ui.anywidget(puck)
        calibration_widgets["SingleView"] = widget

        # Also save the split images so that calibration has them
        split_images = [img_as_ubyte(img) for img in ptv.image_split(temp_img, order=split_order)]
        img_cal_paths = ptv_params.get("img_cal", [f"cal/cam_{k+1}.tif" for k in range(num_cams)])
        for _i, _img_data in enumerate(split_images):
            if _i < len(img_cal_paths):
                _out_path = (base_dir / img_cal_paths[_i]).resolve()
                _out_path.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(_out_path, _img_data)

    else:
        # Standard 1 tab per camera mode
        for _i in range(num_cams):
            _cam_key = f"camera_{_i}"
            _img_name = cal_images[_i] if _i < len(cal_images) else ""
            _img_path = (base_dir / _img_name).resolve()

            if _img_path.exists():
                _image = imread(_img_path)
            else:
                print(f"Warning: Image not found: {_img_path}")
                continue

            _cam_coords = coords.get(_cam_key, {})
            _x_init = []
            _y_init = []

            _start_idx = _i * 4
            _end_idx = _start_idx + 4
            if _end_idx <= len(man_ori_nr):
                _cam_point_ids = man_ori_nr[_start_idx:_end_idx]
            else:
                _cam_point_ids = list(range(1, 5))

            for _pt_idx in range(1, 5):
                _pt_key = f"point_{_pt_idx}"
                _pt = _cam_coords.get(_pt_key, {"x": 100, "y": 100})
                _x_init.append(_pt["x"])
                _y_init.append(_pt["y"])

            _h, _w = _image.shape[:2]

            def _make_draw_cam(_img, _ids):
                def _draw(ax, widget):
                    ax.imshow(_img, cmap="gray")
                    ax.axis("off")
                    for _pt_idx in range(1, 5):
                        if (_pt_idx - 1) < len(widget.x) and (_pt_idx - 1) < len(widget.y):
                            _x_val = widget.x[_pt_idx - 1]
                            _y_val = widget.y[_pt_idx - 1]
                            if (_pt_idx - 1) < len(_ids):
                                _pid = _ids[_pt_idx - 1]
                                ax.text(
                                    _x_val + 15,
                                    _y_val + 15,
                                    str(_pid),
                                    color="yellow",
                                    fontsize=12,
                                    fontweight="bold",
                                )
                return _draw

            puck = ChartPuck.from_callback(
                draw_fn=_make_draw_cam(_image, _cam_point_ids),
                x_bounds=(-0.5, _w - 0.5),
                y_bounds=(_h - 0.5, -0.5),
                figsize=(10, 10),
                x=_x_init,
                y=_y_init,
                puck_color=puck_colors,
                puck_radius=10,
            )

            widget = mo.ui.anywidget(puck)
            calibration_widgets[f"Camera {_i + 1}"] = widget

    tabs = mo.ui.tabs(calibration_widgets)
    save_btn = mo.ui.run_button(label="Save Parameters to YAML")
    mo.vstack([tabs, save_btn])
    return (
        H,
        W,
        base_dir,
        cal_images,
        calibration_widgets,
        first_img_path,
        is_splitter,
        num_cams,
        save_btn,
        split_order,
    )


@app.cell
def _(
    H,
    W,
    calibration_widgets,
    is_splitter,
    num_cams,
    pm,
    save_btn,
    split_order,
):

    # This cell reacts to the save button click
    if save_btn.value:
        _updated_coords = {}

        if is_splitter and "SingleView" in calibration_widgets:
            _w = calibration_widgets["SingleView"]
            _x_vals = _w.x
            _y_vals = _w.y

            _global_idx = 0
            for _i in range(num_cams):
                _c_key = f"camera_{_i}"
                _updated_coords[_c_key] = {}

                _quadrant = split_order[_i]
                _dx = (_quadrant % 2) * (W // 2)
                _dy = (_quadrant // 2) * (H // 2)

                for _p_idx in range(4):
                    if _global_idx < len(_x_vals) and _global_idx < len(_y_vals):
                        _updated_coords[_c_key][f"point_{_p_idx + 1}"] = {
                            "x": float(_x_vals[_global_idx]) - _dx,
                            "y": float(_y_vals[_global_idx]) - _dy,
                        }
                    _global_idx += 1
        else:
            for _idx in range(num_cams):
                _c_key = f"camera_{_idx}"
                _widget_key = f"Camera {_idx + 1}"
                if _widget_key in calibration_widgets:
                    _w = calibration_widgets[_widget_key]
                    _x_vals = _w.x
                    _y_vals = _w.y

                    _updated_coords[_c_key] = {}
                    for _p_idx in range(4):
                        if _p_idx < len(_x_vals) and _p_idx < len(_y_vals):
                            _updated_coords[_c_key][f"point_{_p_idx + 1}"] = {
                                "x": float(_x_vals[_p_idx]),
                                "y": float(_y_vals[_p_idx]),
                            }

        # Update parameter manager
        if _updated_coords:
            pm.parameters["man_ori_coordinates"] = _updated_coords

            # Save to YAML
            pm.to_yaml(pm.yaml_path)
            print(
                f"? Successfully saved manual orientation coordinates to {pm.yaml_path}"
            )
        else:
            print("?? No coordinates found to save.")
    return


@app.cell
def _(pm):

    print("cal_images inside notebook:", pm.parameters["cal_ori"]["img_cal_name"])
    print("ptv splitter flag:", pm.parameters.get("ptv", {}).get("splitter"))
    print("ptv img_cal:", pm.parameters.get("ptv", {}).get("img_cal"))
    return


@app.cell
def _(
    base_dir,
    cal_images,
    first_img_path,
    img_as_ubyte,
    imread,
    is_splitter,
    mo,
    np,
    num_cams,
    ptv,
    rgb2gray,
    split_order,
):
    # Define state and load/split images for centroid detection
    images_to_use = []
    if is_splitter and first_img_path and first_img_path.exists():
        _temp_img = imread(first_img_path)
        if _temp_img.ndim > 2:
            _temp_img = rgb2gray(_temp_img[:, :, :3])
        images_to_use = [img_as_ubyte(img) for img in ptv.image_split(_temp_img, order=split_order)]
    else:
        for _i in range(num_cams):
            _img_name = cal_images[_i] if _i < len(cal_images) else ""
            _img_path = (base_dir / _img_name).resolve()
            if _img_path.exists():
                _image = imread(_img_path)
                if _image.ndim > 2:
                    _image = rgb2gray(_image[:, :, :3])
                images_to_use.append(img_as_ubyte(_image))
            else:
                images_to_use.append(None)

    # 2. Compute initial automatic centroids
    from scipy.ndimage import center_of_mass, label
    from skimage.filters import threshold_otsu

    _initial_detected = {}
    for _i, _img in enumerate(images_to_use):
        if _img is not None:
            _norm_img = (_img - _img.min()) / max(1.0, _img.max() - _img.min())
            try:
                _thresh = threshold_otsu(_norm_img)
            except Exception:
                _thresh = 0.2
            _binary = _norm_img > _thresh
            _labeled, _num = label(_binary)
            _centroids = []
            for _j in range(1, _num + 1):
                _mask = (_labeled == _j)
                _size = np.sum(_mask)
                if 4 <= _size <= 1000:
                    _cy, _cx = center_of_mass(_norm_img, _labeled, _j)
                    _centroids.append([_cx, _cy])
            _initial_detected[f"camera_{_i}"] = np.array(_centroids)
        else:
            _initial_detected[f"camera_{_i}"] = np.array([])

    # Define state and UI controls
    centroids_state, set_centroids_state = mo.state(_initial_detected)

    threshold_slider = mo.ui.slider(
        start=0.01,
        stop=1.0,
        step=0.01,
        value=0.2,
        label="Manual Detection Threshold"
    )
    return (
        center_of_mass,
        centroids_state,
        images_to_use,
        label,
        set_centroids_state,
        threshold_slider,
    )


@app.cell
def _(
    base_dir,
    center_of_mass,
    centroids_state,
    images_to_use,
    label,
    mo,
    np,
    num_cams,
    pm,
    set_centroids_state,
    threshold_slider,
):
    # Construct and display the interactive centroid filtering dashboard
    from wigglystuff import ChartMultiSelect

    select_widgets = {}

    for _i, _img in enumerate(images_to_use):
        if _img is not None:
            _h, _w = _img.shape[:2]
            _cam_key = f"camera_{_i}"

            def _make_draw_centroids(_image_data, _ckey):
                def _draw(ax, widget):
                    ax.imshow(_image_data, cmap="gray")
                    ax.axis("off")
                    _pts = centroids_state().get(_ckey, np.array([]))
                    if len(_pts) > 0:
                        ax.scatter(_pts[:, 0], _pts[:, 1], color="red", s=15, alpha=0.8)
                return _draw

            _puck_select = ChartMultiSelect.from_callback(
                draw_fn=_make_draw_centroids(_img, _cam_key),
                x_bounds=(-0.5, _w - 0.5),
                y_bounds=(_h - 0.5, -0.5),
                figsize=(10, 10),
                n_classes=1,
                mode="lasso",
            )

            select_widgets[f"Camera {_i + 1}"] = _puck_select

    tabs_widget = mo.ui.tabs(select_widgets)

    # Callbacks for buttons
    def detect_centroids_callback(_):
        _new_detected = {}
        for _i, _img in enumerate(images_to_use):
            if _img is not None:
                _norm_img = (_img - _img.min()) / max(1.0, _img.max() - _img.min())
                _binary = _norm_img > threshold_slider.value
                _labeled, _num = label(_binary)
                _centroids = []
                for _j in range(1, _num + 1):
                    _mask = (_labeled == _j)
                    _size = np.sum(_mask)
                    if 4 <= _size <= 1000:
                        _cy, _cx = center_of_mass(_norm_img, _labeled, _j)
                        _centroids.append([_cx, _cy])
                _new_detected[f"camera_{_i}"] = np.array(_centroids)
            else:
                _new_detected[f"camera_{_i}"] = np.array([])
        set_centroids_state(_new_detected)

    def delete_selected_centroids(_):
        _selected_tab_label = tabs_widget.value
        if _selected_tab_label:
            _cam_idx = int(_selected_tab_label.split()[-1]) - 1
            _cam_key = f"camera_{_cam_idx}"

            _widget_key = f"Camera {_cam_idx + 1}"
            if _widget_key in select_widgets:
                _w = select_widgets[_widget_key]
                _pts = centroids_state().get(_cam_key, np.array([]))
                if len(_pts) > 0:
                    _indices_to_delete = _w.get_indices(_pts[:, 0], _pts[:, 1])
                    if len(_indices_to_delete) > 0:
                        _new_pts = np.delete(_pts, _indices_to_delete, axis=0)

                        _updated_dict = dict(centroids_state())
                        _updated_dict[_cam_key] = _new_pts
                        set_centroids_state(_updated_dict)

                        _w.clear()

    def save_targets_callback(_):
        _ptv_params = pm.parameters.get("ptv", {})
        _img_cal_paths = _ptv_params.get("img_cal", [f"cal/cam_{k+1}.tif" for k in range(num_cams)])

        for _i, _path in enumerate(_img_cal_paths):
            _cam_key = f"camera_{_i}"
            _pts = centroids_state().get(_cam_key, np.array([]))

            _target_path = (base_dir / f"{_path}_targets").resolve()
            _target_path.parent.mkdir(parents=True, exist_ok=True)

            with open(_target_path, "w") as f:
                f.write(f"{len(_pts)}\n")
                for _idx, _pt in enumerate(_pts):
                    _cx, _cy = _pt
                    f.write(f"   {_idx:d}  {_cx:8.4f}  {_cy:8.4f}    50     8     8  5000    -1\n")
        print("Successfully saved target files for all cameras!")

    # Define buttons with callbacks
    detect_btn = mo.ui.button(
        label="Detect Centroids on All Camera Views",
        kind="success",
        on_click=detect_centroids_callback
    )

    delete_btn = mo.ui.button(
        label="Delete Selected Centroids in Current Tab",
        kind="danger",
        on_click=delete_selected_centroids
    )

    save_targets_btn = mo.ui.button(
        label="Save Filtered Centroids as OpenPTV Targets",
        kind="neutral",
        on_click=save_targets_callback
    )

    counts_md = []
    for _i in range(len(images_to_use)):
        _pts = centroids_state().get(f"camera_{_i}", [])
        counts_md.append(f"**Camera {_i+1}**: {len(_pts)} dots")

    info_md = mo.md(f"""
    ### Centroid Detection and Spurious Points Filtering
    Use this section to automatically detect calibration dots (centroids) on each camera view and filter out spurious noise.

    -   **Select a tab** below to view a camera's split view.
    -   **Draw lasso selections** (click and drag) around spurious dots you want to delete.
    -   Click **"Delete Selected Centroids in Current Tab"** to remove them.
    -   Adjust the manual slider and click **"Detect Centroids on All Camera Views"** to rerun detection if needed.
    -   Click **"Save Filtered Centroids as OpenPTV Targets"** to save to `cal/cam_*.tif_targets`.

    **Current Active Counts:**
    {" | ".join(counts_md)}
    """)

    mo.vstack([
        info_md,
        mo.hstack([threshold_slider, detect_btn]),
        tabs_widget,
        mo.hstack([delete_btn, save_targets_btn])
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
