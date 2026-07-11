"""OpenPTV2 CustomTkinter application (chosen migration target).

Modern CustomTkinter chrome (sidebar + rounded controls + appearance switch)
hosting the two components ctk lacks natively — the parameter ``ttk.Treeview``
(dark-styled to match) and the embedded matplotlib camera grid. A classic
``tk.Menu`` menubar provides File/Run/Calibration/View/Windows/Help.

Business logic is untouched: the same ParameterManager / ptv APIs as the Traits
GUI. Run:  uv run python -m openptv2.gui_tk.app_ctk <dataset-dir-or-yaml>
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .app import _camera_image, _load_pm
from .events import CameraClick, EventBus, ParamsChanged
from .paramform import ParamForm
from .widgets import MplImageView

CTK_DARK = {
    "bg": "#2b2b2b", "panel": "#2b2b2b", "fg": "#dce4ee",
    "axes": "#2b2b2b", "grid": "#4a4a4a", "accent": "#1f6aa5",
}
CTK_LIGHT = {
    "bg": "#ebebeb", "panel": "#ebebeb", "fg": "#1a1a1a",
    "axes": "#ffffff", "grid": "#c0c0c0", "accent": "#1f6aa5",
}
SIDEBAR_ACTIONS = [
    ("Detection", "detection"), ("Sequence", "sequence"),
    ("Tracking", "tracking"), ("Calibration…", "calibration"),
    ("3D positions", "view3d"), ("Save parameters", "save"),
]


class CtkApp:
    def __init__(self, dataset: str):
        import customtkinter as ctk
        self.ctk = ctk
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.pm, self.yaml = _load_pm(dataset)
        self.dataset_dir = Path(self.yaml).parent
        self.num_cams = int(self.pm.parameters.get("num_cams", 4) or 4)
        self.bus = EventBus()
        self.views: dict[int, MplImageView] = {}
        self.palette = CTK_DARK

        self.root = ctk.CTk()
        self.root.title(f"OpenPTV2 — {self.dataset_dir.name}")
        self.root.geometry("1240x820")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self._build_menu()
        self._build_sidebar()
        self._build_camera_grid()

        self.status = ctk.CTkLabel(self.root, text=f"loaded {self.yaml.name}",
                                   anchor="w")
        self.status.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 4))

        self.bus.subscribe(CameraClick, self._on_click)
        self.bus.subscribe(ParamsChanged,
                           lambda e: self._set_status(f"saved: {e.section}"))

    # --- menubar ----------------------------------------------------------- #

    def _build_menu(self) -> None:
        m = tk.Menu(self.root)
        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="Save parameters", command=self.save_params)
        filem.add_separator()
        filem.add_command(label="Quit", command=self.root.destroy)
        m.add_cascade(label="File", menu=filem)

        runm = tk.Menu(m, tearoff=0)
        for label, key in (("Detection", "detection"), ("Sequence", "sequence"),
                           ("Tracking", "tracking")):
            runm.add_command(label=label, command=lambda k=key: self._run(k))
        m.add_cascade(label="Run", menu=runm)

        calm = tk.Menu(m, tearoff=0)
        calm.add_command(label="Open calibration…", command=self.open_calibration)
        m.add_cascade(label="Calibration", menu=calm)

        viewm = tk.Menu(m, tearoff=0)
        viewm.add_command(label="Dark", command=lambda: self._set_appearance("Dark"))
        viewm.add_command(label="Light", command=lambda: self._set_appearance("Light"))
        m.add_cascade(label="View", menu=viewm)

        winm = tk.Menu(m, tearoff=0)
        winm.add_command(label="3D positions view", command=self.open_view3d)
        m.add_cascade(label="Windows", menu=winm)

        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(label="About",
                          command=lambda: self._set_status("OpenPTV2 CustomTkinter GUI"))
        m.add_cascade(label="Help", menu=helpm)
        self.root.config(menu=m)
        self._menus = {"File": filem, "Run": runm, "Calibration": calm,
                       "View": viewm, "Windows": winm, "Help": helpm}

    # --- sidebar (ctk) + parameter tree ------------------------------------ #

    def _build_sidebar(self) -> None:
        ctk = self.ctk
        bar = ctk.CTkFrame(self.root, width=220, corner_radius=0)
        bar.grid(row=0, column=0, rowspan=2, sticky="nsw")
        ctk.CTkLabel(bar, text="OpenPTV2",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(padx=20, pady=(18, 12))
        for label, key in SIDEBAR_ACTIONS:
            ctk.CTkButton(bar, text=label,
                          command=lambda k=key: self._sidebar(k)).pack(
                              padx=16, pady=5, fill="x")
        ctk.CTkLabel(bar, text="Appearance").pack(padx=16, pady=(16, 0), anchor="w")
        ctk.CTkOptionMenu(bar, values=["Dark", "Light", "System"],
                          command=self._set_appearance).pack(padx=16, pady=4, fill="x")
        ctk.CTkLabel(bar, text="Parameters").pack(padx=16, pady=(14, 0), anchor="w")
        self._build_tree(bar).pack(padx=12, pady=6, fill="both", expand=True)

    def _build_tree(self, master):
        self._style_tree()
        tree = ttk.Treeview(master, show="tree", style="Ctk.Treeview")
        self.tree = tree
        self._tree_section: dict[str, str] = {}
        for section, val in self.pm.parameters.items():
            node = tree.insert("", "end", text=section)
            self._tree_section[node] = section
            if isinstance(val, dict):
                for k, v in val.items():
                    if not isinstance(v, dict):
                        tree.insert(node, "end", text=f"{k} = {v}")
        menu = tk.Menu(tree, tearoff=0)
        menu.add_command(label="Edit…", command=self._edit_selected_section)

        def popup(event):
            iid = tree.identify_row(event.y)
            if iid:
                tree.selection_set(iid)
                menu.tk_popup(event.x_root, event.y_root)
        tree.bind("<Button-3>", popup)
        tree.bind("<Double-1>", lambda e: self._edit_selected_section())
        return tree

    def _style_tree(self) -> None:
        dark = self.ctk.get_appearance_mode() == "Dark"
        bg = "#2b2b2b" if dark else "#f7f7f7"
        fg = "#dce4ee" if dark else "#1a1a1a"
        style = ttk.Style()
        style.configure("Ctk.Treeview", background=bg, fieldbackground=bg,
                        foreground=fg, borderwidth=0, rowheight=22)
        style.map("Ctk.Treeview", background=[("selected", "#1f6aa5")],
                  foreground=[("selected", "#ffffff")])

    def _selected_section(self):
        sel = self.tree.selection()
        if not sel:
            return None
        node = sel[0]
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
        top = self.ctk.CTkToplevel(self.root)
        top.title(f"Edit: {section}")
        top.after(50, top.lift)
        form = ParamForm(top, section, data, on_save=self._save_section)
        form.pack(fill="both", expand=True)
        self._last_form = form

    def _save_section(self, section: str, values: dict) -> None:
        self.pm.parameters.setdefault(section, {}).update(values)
        self.bus.publish(ParamsChanged(section))
        for node, sec in self._tree_section.items():
            if sec == section:
                for child in self.tree.get_children(node):
                    self.tree.delete(child)
                for k, v in self.pm.parameters[section].items():
                    if not isinstance(v, dict):
                        self.tree.insert(node, "end", text=f"{k} = {v}")
                break

    # --- camera grid ------------------------------------------------------- #

    def _build_camera_grid(self) -> None:
        grid = self.ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        grid.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        for cam in range(self.num_cams):
            v = MplImageView(grid, cam=cam, bus=self.bus, palette=self.palette)
            v.set_image(_camera_image(self.dataset_dir, cam))
            r, c = divmod(cam, 2)
            v.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
            self.views[cam] = v
        for i in range((self.num_cams + 1) // 2):
            grid.rowconfigure(i, weight=1)
        for i in range(2):
            grid.columnconfigure(i, weight=1)

    def _on_click(self, ev: CameraClick) -> None:
        for cam, view in self.views.items():
            view.clear_overlays()
            view.add_points([[ev.x, ev.y]],
                            color=("yellow" if cam == ev.cam else "magenta"),
                            marker=("+" if cam == ev.cam else "x"))
        self._set_status(f"cam{ev.cam} click ({ev.x:.1f}, {ev.y:.1f})")

    # --- actions ----------------------------------------------------------- #

    def _sidebar(self, key: str) -> None:
        if key == "calibration":
            self.open_calibration()
        elif key == "view3d":
            self.open_view3d()
        elif key == "save":
            self.save_params()
        else:
            self._run(key)

    def _run(self, what: str) -> None:
        self._set_status(f"Run '{what}' — wiring to ptv.* in the run slice")

    def open_calibration(self) -> None:
        try:
            from .calibration import CalibrationWindow
            CalibrationWindow(self.root, self.pm, self.dataset_dir, self.bus,
                              palette=self.palette)
            self._set_status("calibration window opened")
        except Exception as exc:
            self._set_status(f"calibration window: {exc}")

    def open_view3d(self) -> None:
        try:
            from .view3d import View3DWindow
            View3DWindow(self.root, self.dataset_dir, self.bus, palette=self.palette)
            self._set_status("3D view opened")
        except Exception as exc:
            self._set_status(f"3D view: {exc}")

    def _set_appearance(self, mode: str) -> None:
        self.ctk.set_appearance_mode(mode)
        self.palette = CTK_LIGHT if self.ctk.get_appearance_mode() == "Light" \
            else CTK_DARK
        self._style_tree()
        for view in self.views.values():
            view.set_palette(self.palette)
        self._set_status(f"appearance: {self.ctk.get_appearance_mode()}")

    def save_params(self) -> None:
        self.pm.to_yaml(str(self.yaml))
        self._set_status(f"parameters saved to {self.yaml.name}")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def run(self) -> None:
        self.root.mainloop()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    CtkApp(argv[0]).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
