"""CustomTkinter look-spike for the GUI migration (comparison vs the ttkbootstrap
``app.py``). Not wired to actions — it exists so the modern CustomTkinter chrome
can be judged side-by-side against ttkbootstrap, using the two hard components
(the parameter ``ttk.Treeview`` and the embedded matplotlib camera grid) that
CustomTkinter cannot provide natively.

Run:  uv run python -m openptv2.gui_tk.app_ctk <dataset-dir-or-yaml>
"""
from __future__ import annotations

import sys
from pathlib import Path
from tkinter import ttk

from .app import _camera_image, _load_pm
from .events import CameraClick, EventBus
from .widgets import MplImageView

# palette matching CustomTkinter's dark theme, fed to the matplotlib panels
CTK_DARK = {
    "bg": "#2b2b2b", "panel": "#2b2b2b", "fg": "#dce4ee",
    "axes": "#2b2b2b", "grid": "#4a4a4a", "accent": "#1f6aa5",
}
SIDEBAR_ACTIONS = ["Detection", "Sequence", "Tracking",
                   "Calibration…", "3D positions", "Save parameters"]


class CtkApp:
    def __init__(self, dataset: str):
        import customtkinter as ctk
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.pm, self.yaml = _load_pm(dataset)
        self.dataset_dir = Path(self.yaml).parent
        self.num_cams = int(self.pm.parameters.get("num_cams", 4) or 4)
        self.bus = EventBus()
        self.views: dict[int, MplImageView] = {}

        self.root = ctk.CTk()
        self.root.title(f"OpenPTV2 (CustomTkinter spike) — {self.dataset_dir.name}")
        self.root.geometry("1200x800")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # --- sidebar (CustomTkinter-native modern chrome) ------------------ #
        sidebar = ctk.CTkFrame(self.root, width=210, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsw")
        ctk.CTkLabel(
            sidebar, text="OpenPTV2",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(padx=20, pady=(18, 12))
        for label in SIDEBAR_ACTIONS:
            ctk.CTkButton(sidebar, text=label,
                          command=lambda t=label: self._action(t)).pack(
                              padx=16, pady=5, fill="x")

        ctk.CTkLabel(sidebar, text="Appearance").pack(
            padx=16, pady=(18, 0), anchor="w")
        ctk.CTkOptionMenu(
            sidebar, values=["Dark", "Light", "System"],
            command=ctk.set_appearance_mode,
        ).pack(padx=16, pady=4, fill="x")

        ctk.CTkLabel(sidebar, text="Parameters").pack(padx=16, pady=(14, 0), anchor="w")
        self._make_tree(sidebar).pack(padx=12, pady=6, fill="both", expand=True)

        # --- camera grid (matplotlib panels, dark to match ctk) ------------ #
        grid = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        grid.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        for cam in range(self.num_cams):
            v = MplImageView(grid, cam=cam, bus=self.bus, palette=CTK_DARK)
            v.set_image(_camera_image(self.dataset_dir, cam))
            r, c = divmod(cam, 2)
            v.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
            self.views[cam] = v
        for i in range((self.num_cams + 1) // 2):
            grid.rowconfigure(i, weight=1)
        for i in range(2):
            grid.columnconfigure(i, weight=1)

        self.status = ctk.CTkLabel(self.root, text=f"loaded {self.yaml.name}",
                                   anchor="w")
        self.status.grid(row=1, column=1, sticky="ew", padx=8)

        self.bus.subscribe(CameraClick, self._on_click)

    def _make_tree(self, master):
        # ttk.Treeview styled dark so it blends into the CustomTkinter chrome
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Ctk.Treeview", background="#2b2b2b",
                        fieldbackground="#2b2b2b", foreground="#dce4ee",
                        borderwidth=0, rowheight=22)
        style.map("Ctk.Treeview", background=[("selected", "#1f6aa5")])
        tree = ttk.Treeview(master, show="tree", style="Ctk.Treeview")
        for section, val in self.pm.parameters.items():
            node = tree.insert("", "end", text=section)
            if isinstance(val, dict):
                for k, v in val.items():
                    if not isinstance(v, dict):
                        tree.insert(node, "end", text=f"{k} = {v}")
        self.tree = tree
        return tree

    def _on_click(self, ev: CameraClick):
        for cam, view in self.views.items():
            view.clear_overlays()
            view.add_points([[ev.x, ev.y]],
                            color=("yellow" if cam == ev.cam else "magenta"),
                            marker=("+" if cam == ev.cam else "x"))
        self.status.configure(text=f"cam{ev.cam} click ({ev.x:.1f}, {ev.y:.1f})")

    def _action(self, label: str):
        self.status.configure(text=f"[spike] '{label}' — look comparison only")

    def run(self):
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
