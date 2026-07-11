"""OpenPTV2 Tk main application (migration target).

Layout: menubar + parameter tree (left) + N-camera grid (center) + status bar.
Cross-view click propagation goes through a single EventBus. Business logic is
untouched — the app calls the same ParameterManager / ptv APIs as the Traits GUI.

Run:  uv run python -m openptv2.gui_tk.app <dataset-dir-or-yaml>
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from . import theme
from .events import CameraClick, EventBus, ExperimentLoaded, ParamsChanged, Status
from .paramform import ParamForm
from .widgets import MplImageView


def _load_pm(dataset: str):
    """Load a ParameterManager from a dataset dir or a YAML file."""
    from openptv2.gui.parameter_manager import ParameterManager

    p = Path(dataset)
    yaml = p if p.suffix in (".yaml", ".yml") else (p / "parameters_Run1.yaml")
    pm = ParameterManager()
    pm.from_yaml(str(yaml))
    return pm, yaml


def _camera_image(dataset_dir: Path, cam: int) -> np.ndarray:
    """Best-effort image for a camera panel (cal image if present, else blank)."""
    for cand in (dataset_dir / f"cal/cam{cam+1}.tif",):
        if cand.exists():
            try:
                import imageio.v3 as iio
                img = np.asarray(iio.imread(cand))
                return img[..., 0] if img.ndim > 2 else img
            except Exception:
                break
    return np.zeros((64, 64), dtype=np.uint8)


class MainWindow:
    def __init__(self, root: tk.Tk, dataset: str):
        self.root = root
        self.bus = EventBus()
        self.pm, self.yaml = _load_pm(dataset)
        self.dataset_dir = Path(self.yaml).parent
        self.num_cams = int(self.pm.parameters.get("num_cams", 4) or 4)
        self.views: dict[int, MplImageView] = {}

        self.palette = theme.apply(root, "dark")
        root.title(f"OpenPTV2 (Tk) — {self.dataset_dir.name}")
        root.geometry("1200x800")
        self._build_menu()

        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        paned.add(self._build_param_tree(paned), weight=1)
        paned.add(self._build_camera_grid(paned), weight=4)

        self.status = ttk.Label(root, text="ready", anchor="w", relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # cross-view wiring
        self.bus.subscribe(CameraClick, self._on_camera_click)
        self.bus.subscribe(Status, lambda e: self.status.config(text=e.text))
        self.bus.subscribe(ParamsChanged,
                           lambda e: self._set_status(f"saved: {e.section}"))
        self.bus.publish(ExperimentLoaded(str(self.yaml), self.num_cams))
        self._set_status(f"loaded {self.yaml.name}  ({self.num_cams} cameras)")

    # --- menus ------------------------------------------------------------- #

    def _build_menu(self) -> None:
        m = tk.Menu(self.root)
        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="Save parameters", command=self.save_params)
        filem.add_separator()
        filem.add_command(label="Quit", command=self.root.destroy)
        m.add_cascade(label="File", menu=filem)

        runm = tk.Menu(m, tearoff=0)
        runm.add_command(label="Detection", command=lambda: self._run("detection"))
        runm.add_command(label="Sequence", command=lambda: self._run("sequence"))
        runm.add_command(label="Tracking", command=lambda: self._run("tracking"))
        m.add_cascade(label="Run", menu=runm)

        calm = tk.Menu(m, tearoff=0)
        calm.add_command(label="Open calibration…", command=self.open_calibration)
        m.add_cascade(label="Calibration", menu=calm)

        viewm = tk.Menu(m, tearoff=0)
        viewm.add_command(label="Toggle dark / light", command=self.toggle_theme)
        m.add_cascade(label="View", menu=viewm)

        winm = tk.Menu(m, tearoff=0)
        winm.add_command(label="3D positions view", command=self.open_view3d)
        winm.add_command(label="Detach all cameras", command=self._detach_cameras)
        m.add_cascade(label="Windows", menu=winm)

        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(label="About", command=lambda: self._set_status(
            "OpenPTV2 Tk GUI (migration in progress)"))
        m.add_cascade(label="Help", menu=helpm)
        self.root.config(menu=m)
        self._menus = {"File": filem, "Run": runm, "View": viewm,
                       "Calibration": calm, "Windows": winm, "Help": helpm}

    # --- parameter tree ---------------------------------------------------- #

    def _build_param_tree(self, master) -> ttk.Frame:
        frame = ttk.Frame(master)
        ttk.Label(frame, text="Parameters").pack(side=tk.TOP, anchor="w", padx=4)
        tree = ttk.Treeview(frame, show="tree")
        tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tree = tree
        self._tree_section: dict[str, str] = {}

        for section, val in self.pm.parameters.items():
            node = tree.insert("", "end", text=section, open=False)
            self._tree_section[node] = section
            if isinstance(val, dict):
                for k, v in val.items():
                    if not isinstance(v, dict):
                        tree.insert(node, "end", text=f"{k} = {v}")

        menu = tk.Menu(tree, tearoff=0)
        menu.add_command(label="Edit…", command=self._edit_selected_section)
        self._tree_menu = menu

        def popup(event):
            iid = tree.identify_row(event.y)
            if iid:
                tree.selection_set(iid)
                menu.tk_popup(event.x_root, event.y_root)
        tree.bind("<Button-3>", popup)
        tree.bind("<Double-1>", lambda e: self._edit_selected_section())
        return frame

    def _selected_section(self) -> str | None:
        sel = self.tree.selection()
        if not sel:
            return None
        node = sel[0]
        # walk up to the top-level section node
        while self.tree.parent(node):
            node = self.tree.parent(node)
        return self._tree_section.get(node)

    def _edit_selected_section(self) -> None:
        section = self._selected_section()
        if section is None:
            return
        data = self.pm.parameters.get(section, {})
        if not isinstance(data, dict):
            self._set_status(f"'{section}' is not editable")
            return
        top = tk.Toplevel(self.root)
        top.title(f"Edit: {section}")
        form = ParamForm(top, section, data, on_save=self._save_section)
        form.pack(fill=tk.BOTH, expand=True)
        self._last_form = form  # for tests

    def _save_section(self, section: str, values: dict) -> None:
        self.pm.parameters.setdefault(section, {}).update(values)
        self.bus.publish(ParamsChanged(section))
        self._refresh_tree_section(section)

    def _refresh_tree_section(self, section: str) -> None:
        for node, sec in self._tree_section.items():
            if sec == section:
                for child in self.tree.get_children(node):
                    self.tree.delete(child)
                for k, v in self.pm.parameters[section].items():
                    if not isinstance(v, dict):
                        self.tree.insert(node, "end", text=f"{k} = {v}")
                break

    # --- camera grid + click propagation ----------------------------------- #

    def _build_camera_grid(self, master) -> ttk.Frame:
        frame = ttk.Frame(master)
        cols = 2
        for cam in range(self.num_cams):
            view = MplImageView(frame, cam=cam, bus=self.bus, palette=self.palette)
            view.set_image(_camera_image(self.dataset_dir, cam))
            r, c = divmod(cam, cols)
            view.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            self.views[cam] = view
        for i in range((self.num_cams + cols - 1) // cols):
            frame.rowconfigure(i, weight=1)
        for i in range(cols):
            frame.columnconfigure(i, weight=1)
        return frame

    def _on_camera_click(self, ev: CameraClick) -> None:
        """A click in one camera marks the point and echoes a marker in the
        others — the hook where epipolar-line propagation will plug in."""
        self._set_status(f"cam{ev.cam} click at ({ev.x:.1f}, {ev.y:.1f})")
        for cam, view in self.views.items():
            if cam == ev.cam:
                view.clear_overlays()
                view.add_points([[ev.x, ev.y]], color="yellow", marker="+")
            else:
                view.clear_overlays()
                view.add_points([[ev.x, ev.y]], color="magenta", marker="x")

    def _detach_cameras(self) -> None:
        self._set_status("detach cameras — placeholder (per-panel detach next)")

    def toggle_theme(self) -> None:
        mode = theme.toggle(self.root)
        self.palette = theme.palette(mode)
        for view in self.views.values():
            view.set_palette(self.palette)
        self._set_status(f"theme: {mode}")

    # --- run actions (delegate to existing ptv/tracker) -------------------- #

    def _run(self, what: str) -> None:
        self._set_status(f"Run '{what}' — wiring to ptv.* in the run slice")

    def open_calibration(self) -> None:
        try:
            from .calibration import CalibrationWindow
            CalibrationWindow(self.root, self.pm, self.dataset_dir, self.bus,
                              palette=self.palette)
            self._set_status("calibration window opened")
        except Exception as exc:  # scaffold may be incomplete
            self._set_status(f"calibration window: {exc}")

    def open_view3d(self) -> None:
        try:
            from .view3d import View3DWindow
            View3DWindow(self.root, self.dataset_dir, self.bus, palette=self.palette)
            self._set_status("3D view opened")
        except Exception as exc:
            self._set_status(f"3D view: {exc}")

    # --- helpers ----------------------------------------------------------- #

    def save_params(self) -> None:
        self.pm.to_yaml(str(self.yaml))
        self._set_status(f"parameters saved to {self.yaml.name}")

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    root = tk.Tk()
    MainWindow(root, argv[0])
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
