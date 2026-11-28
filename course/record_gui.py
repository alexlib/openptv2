# ruff: noqa
"""Snapshots of the real GUI for the course (docs/course/figures/gui_*.png).

    uv run python docs/course/record_gui.py

Drives the GUI in-process on a scratch COPY of test_data/test_cavity_small (the bundled data is never written to) and saves
only the application window (Qt grab), so nothing else on the screen can end up in a picture. The window appears briefly.
The macOS menu bar is not part of the window, so the captions name the menu entry instead.
"""
import os, shutil, sys, tempfile, time, types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO); sys.path.insert(0, str(REPO / "src"))
from PIL import Image, ImageDraw, ImageFont

from openptv2.gui.experiment import Experiment
from openptv2.gui.parameter_manager import ParameterManager
from openptv2.gui.pyptv_gui import MainGUI, TreeMenuHandler

OUT = REPO / "docs/course/figures"
work = Path(tempfile.mkdtemp(prefix="course_gui_")) / "my_run"
shutil.copytree(REPO / "test_data/test_cavity_small", work, ignore=shutil.ignore_patterns("tmp*.txt", "res"))
YAML = work / "parameters_Run1.yaml"
STEPS = [(None, "Start > Init / Reload: the four camera tabs show frame 10000", "gui_1_init.png"),
         ("img_coord_action", "Preprocess > Image coord: blue + marks the detected targets", "gui_2_detect.png"),
         ("corresp_action", "Preprocess > Correspondences: matched particles get colour (yellow 2, green 3, red 4 cameras)", "gui_3_match.png")]


def caption(path, text, width=1100):
    im = Image.open(path).convert("RGB"); im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    bar = Image.new("RGB", (width, 40), (24, 24, 24)); d = ImageDraw.Draw(bar)
    try:
        import matplotlib
        f = ImageFont.truetype(str(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"), 17)
    except Exception:
        f = ImageFont.load_default()
    d.text((12, 10), text, fill=(255, 255, 255), font=f)
    out = Image.new("RGB", (width, im.height + 40)); out.paste(bar, (0, 0)); out.paste(im, (0, 40)); out.save(path)


from pyface.qt.QtWidgets import QApplication
pm = ParameterManager(); pm.from_yaml(str(YAML)); exp = Experiment(pm=pm); exp.populate_runs(YAML.parent, active_yaml=YAML)
app = QApplication.instance() or QApplication([]); os.chdir(YAML.parent)
gui = MainGUI(YAML, exp); ui = gui.edit_traits(kind="live"); ui.control.setFixedSize(1500, 900)
settle = lambda s=2.5: (app.processEvents(), time.sleep(s), app.processEvents())
settle(3)
h = TreeMenuHandler(); info = types.SimpleNamespace(object=gui, ui=ui)
getattr(h, "init_action")(info); settle(3)
for fn, text, name in STEPS:
    if fn:
        getattr(h, fn)(info); settle(3)
    app.processEvents(); p = OUT / name; ui.control.grab().save(str(p)); caption(p, text); print("saved", p)
os.chdir(REPO); ui.dispose()
