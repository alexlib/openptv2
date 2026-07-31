import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml


@app.cell
def _():
    base_path = Path("/home/user/Documents/GitHub/openptv2/test_data/test_cavity")
    res_dir = base_path / "res"
    res_dir.mkdir(exist_ok=True)
    img_dir = base_path / "img"
    return base_path, img_dir, res_dir


@app.cell
def _(base_path, res_dir):
    # Run batch sequence with the single runtime
    from openptv2.gui.pyptv import pyptv_batch

    yaml_path = base_path / "parameters_Run1.yaml"
    with open(yaml_path) as f:
        params = yaml.safe_load(f)

    # res_dir = base_path / "res"


    params["sequence"]["output"] = str(res_dir)
    params["sequence"]["first"] = 10001
    params["sequence"]["last"] = 10004

    temp_yaml = base_path / "temp_run.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(params, f)

    print("Running batch: frames 10001-10004")
    pyptv_batch.main(temp_yaml, 10001, 10004)
    print("Batch complete!")
    return


@app.cell
def _():
    from openptv2.gui.pyptv.ptv import read_rt_is_file
    # def read_rt_is_file(filename) -> List[List[float]]:
    #     """Read data from an rt_is file and return the parsed values."""
    #     try:
    #         with open(filename, "r", encoding="utf-8") as file:
    #             num_rows = int(file.readline().strip())
    #             if num_rows == 0:
    #                 raise ValueError("Failed to read the number of rows")

    #             data = []
    #             for _ in range(num_rows):
    #                 line = file.readline().strip()
    #                 if not line:
    #                     break

    #                 values = line.split()
    #                 if len(values) != 8:
    #                     raise ValueError("Incorrect number of values in line")

    #                 x = float(values[1])
    #                 y = float(values[2])
    #                 z = float(values[3])
    #                 p1 = int(values[4])
    #                 p2 = int(values[5])
    #                 p3 = int(values[6])
    #                 p4 = int(values[7])

    #                 data.append([x, y, z, p1, p2, p3, p4])

    #             return data

    #     except IOError as e:
    #         print(f"Can't open ascii file: {filename}")
    #         raise e
    return (read_rt_is_file,)


@app.cell
def _(read_rt_is_file, res_dir):
    frame_num = 10001
    rt_file = res_dir / f"rt_is.{frame_num}"
    pts_3d, img_xs, img_ys = read_rt_is_file(rt_file)
    print(f"Loaded {len(pts_3d)} correspondences")

    pts_3d, img_xs, img_ys
    return img_xs, img_ys


@app.cell
def _(img_dir):
    img_names = ["cam1.10001", "cam2.10001", "cam3.10001", "cam4.10001"]
    images = []
    for name in img_names:
        img_path = img_dir / name
        if img_path.exists():
            images.append(plt.imread(str(img_path)))
        else:
            images.append(np.zeros((1024, 1024)))
    print(f"Loaded {len(images)} images")
    return (images,)


@app.cell
def _(images, img_xs, img_ys):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    colors = {4: "red", 3: "green", 2: "yellow"}
    labels = {4: "Quadruplets", 3: "Triplets", 2: "Pairs"}

    for cam_idx in range(4):
        ax = axes_flat[cam_idx]
        ax.imshow(images[cam_idx], cmap="gray")
        ax.set_title(f"Camera {cam_idx + 1}")

        for n_cams in [4, 3, 2]:
            xs_list, ys_list = [], []
            for i in range(len(img_xs[0])):
                valid = sum(1 for c in range(4) if img_xs[c][i] >= 0)
                if valid == n_cams:
                    x, y = img_xs[cam_idx][i], img_ys[cam_idx][i]
                    if x >= 0:
                        xs_list.append(x)
                        ys_list.append(y)
            if xs_list:
                ax.scatter(
                    xs_list,
                    ys_list,
                    c=colors[n_cams],
                    s=20,
                    alpha=0.7,
                    label=labels[n_cams] if cam_idx == 0 else "",
                )

    axes_flat[0].legend(loc="upper right")

    click_data = {
        "img_xs": img_xs,
        "img_ys": img_ys,
        "axes": axes_flat,
        "fig": fig,
    }


    def onclick(event):
        if not event.inaxes:
            return
        click_cam = None
        for i, ax in enumerate(click_data["axes"]):
            if ax == event.inaxes:
                click_cam = i
                break
        if click_cam is None:
            return

        x, y = event.xdata, event.ydata
        print(f"\n=== Click on Camera {click_cam + 1} at ({x:.1f}, {y:.1f}) ===")

        img_xs = click_data["img_xs"]
        img_ys = click_data["img_ys"]

        point = np.array([x, y])
        min_dist = float("inf")
        found_idx = None

        for i in range(len(img_xs[0])):
            if img_xs[click_cam][i] < 0:
                continue
            px, py = img_xs[click_cam][i], img_ys[click_cam][i]
            dist = np.sqrt((px - x) ** 2 + (py - y) ** 2)
            if dist < 20 and dist < min_dist:
                min_dist = dist
                found_idx = i

        if found_idx is None:
            print("  No correspondence found")
            return

        valid_cams = 0
        for cam_idx in range(4):
            if img_xs[cam_idx][found_idx] >= 0:
                valid_cams += 1
                px, py = img_xs[cam_idx][found_idx], img_ys[cam_idx][found_idx]
                print(f"  Cam {cam_idx + 1}: ({px:.1f}, {py:.1f})")
                click_data["axes"][cam_idx].plot(
                    px, py, "o", color="cyan", markersize=10
                )

        print(f"  {valid_cams} cameras, dist {min_dist:.1f}px")
        click_data["fig"].canvas.draw_idle()


    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _():
    print(
        "Click on points to see correspondences. Red=4 cams, Green=3 cams, Yellow=2 cams"
    )
    return


if __name__ == "__main__":
    app.run()
