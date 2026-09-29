"""Record the GUI-first tutorial by driving the real PyPTV GUI.

Launches MainGUI on test_data/test_cavity, invokes the same handlers the
menu clicks trigger (Start -> Init / Reload, Preprocess -> Image coord,
...), and captures the window with Qt's grab() after each step.

Output (docs/tutorials/images/):
    gui_step1_open.png ... gui_step7_trajectories.png  (downscaled, captioned)
    gui_tutorial.gif  (animated walkthrough for the tutorial page)

Run from the repository root:
    uv run python docs/tutorials/record_gui_tutorial.py

No accessibility permissions needed (everything runs in-process).
The GUI window appears on screen briefly while recording.
"""

import os
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

from openptv2.gui.experiment import Experiment  # noqa: E402
from openptv2.gui.parameter_manager import ParameterManager  # noqa: E402
from openptv2.gui.pyptv_gui import MainGUI, TreeMenuHandler  # noqa: E402

YAML = REPO_ROOT / "test_data/test_cavity/parameters_Run1.yaml"
IMG_DIR = REPO_ROOT / "docs/tutorials/images"
TMP_DIR = Path("/tmp/gui_rec")
CAPTURE_WIDTH = 1400
GIF_WIDTH = 1000

# (method on TreeMenuHandler or None, caption, output name, gif duration ms)
STEPS = [
    (None, "Step 1: open the example", "gui_step1_open.png", 1500),
    ("init_action", "Step 2: Start -> Init / Reload", "gui_step2_init.png", 1200),
    (
        "img_coord_action",
        "Step 3: Preprocess -> Image coord (blue = detected)",
        "gui_step3_detect.png",
        1200,
    ),
    (
        "corresp_action",
        "Step 4: Preprocess -> Correspondences",
        "gui_step4_corresp.png",
        1200,
    ),
    (
        "sequence_action",
        "Step 5: Sequence -> Sequence without display",
        "gui_step5_sequence.png",
        1500,
    ),
    (
        "track_no_disp_action",
        "Step 6: Tracking -> Tracking without display",
        "gui_step6_tracking.png",
        1500,
    ),
    (
        "traject_action_flowtracks",
        "Step 7: Tracking -> Show trajectories",
        "gui_step7_trajectories.png",
        2500,
    ),
]


def settle(app, seconds=2.0):
    """Let Qt/Chaco repaint before grabbing."""
    app.processEvents()
    time.sleep(seconds)
    app.processEvents()


def grab(app, ui, path):
    from pyface.qt.QtWidgets import QApplication

    widget = ui.control
    if widget is None:  # some actions rebuild editors; fall back to window
        visible = [w for w in QApplication.topLevelWidgets() if w.isVisible()]
        widget = max(visible, key=lambda w: w.width() * w.height())
    app.processEvents()
    pixmap = widget.grab()
    pixmap.save(str(path))
    print(f"captured {path} ({pixmap.width()}x{pixmap.height()})")


def caption_and_resize(src, dst, caption, width):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(src).convert("RGB")
    w, h = img.size
    img = img.resize((width, int(h * width / w)), Image.LANCZOS)
    try:
        import matplotlib

        font = ImageFont.truetype(
            str(
                Path(matplotlib.__file__).parent
                / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"
            ),
            30,
        )
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    banner_h = 56
    banner = Image.new("RGB", (img.width, banner_h), (20, 20, 20))
    img.paste(banner, (0, 0))
    draw.text((16, 10), caption, fill=(255, 255, 255), font=font)
    img.save(dst)
    print(f"wrote {dst}")


def main():
    from pyface.qt.QtWidgets import QApplication

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    pm = ParameterManager()
    pm.from_yaml(str(YAML))
    exp = Experiment(pm=pm)
    exp.populate_runs(YAML.parent, active_yaml=YAML)

    app = QApplication.instance() or QApplication([])
    os.chdir(YAML.parent)
    try:
        main_gui = MainGUI(YAML, exp)
        ui = main_gui.edit_traits(kind="live")
        # Full-screen recording: the TraitsUI dialog pins its height, so
        # setFixedSize (not resize/showMaximized) to fill the display.
        # 2880x1700 fits a 3024x1964 Retina screen minus menu bar and Dock.
        ui.control.setFixedSize(2880, 1700)
        settle(app, 2.0)
        # Keep the parameter tree readable: give it ~450 px, rest to cameras.
        from pyface.qt.QtWidgets import QSplitter

        for splitter in ui.control.findChildren(QSplitter):
            sizes = splitter.sizes()
            if len(sizes) == 2:
                splitter.setSizes([450, sum(sizes) - 450])
        settle(app, 2.0)

        handler = TreeMenuHandler()
        info = types.SimpleNamespace(object=main_gui, ui=ui)

        raws = []
        for method_name, caption, fname, _duration in STEPS:
            if method_name is not None:
                print(f"running {method_name} ...")
                getattr(handler, method_name)(info)
                settle(app, 2.0)
            raw = TMP_DIR / fname
            grab(app, ui, raw)
            raws.append((raw, caption, fname))

        # Post-process: captioned PNGs + GIF
        from PIL import Image

        gif_frames = []
        for raw, caption, fname in raws:
            dst = IMG_DIR / fname
            caption_and_resize(raw, dst, caption, CAPTURE_WIDTH)
            gif_frames.append(Image.open(dst).resize(
                (GIF_WIDTH, int(Image.open(dst).height * GIF_WIDTH / Image.open(dst).width)),
                Image.LANCZOS,
            ))

        gif_path = IMG_DIR / "gui_tutorial.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=[d for _, _, _, d in STEPS],
            loop=0,
        )
        print(f"wrote {gif_path}")
    finally:
        os.chdir(REPO_ROOT)

    print("done - closing GUI")
    ui.dispose()


if __name__ == "__main__":
    main()
