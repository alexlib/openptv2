"""Generic ttk parameter editor built from a YAML/dict parameter section.

Widgets are chosen by value type (bool→Checkbutton, int/float→Entry, list→CSV
Entry, str→Entry). ``get_values()`` parses the widgets back to a dict with the
original types preserved, so a load→edit→save round-trip is exact when nothing
is changed. Used both inline and inside a right-click Toplevel from the tree.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class ParamForm(ttk.Frame):
    def __init__(self, master, section: str, data: dict[str, Any],
                 on_save: Callable[[str, dict], None] | None = None):
        super().__init__(master)
        self.section = section
        self._orig = dict(data)
        self._on_save = on_save
        self._vars: dict[str, tuple[str, tk.Variable]] = {}

        grid = ttk.Frame(self)
        grid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        row = 0
        for key, val in data.items():
            if isinstance(val, dict):
                continue  # nested sections get their own tree node/form
            ttk.Label(grid, text=key).grid(row=row, column=0, sticky="w", pady=1)
            kind, var, widget = self._make_widget(grid, val)
            widget.grid(row=row, column=1, sticky="ew", pady=1)
            self._vars[key] = (kind, var)
            row += 1
        grid.columnconfigure(1, weight=1)

        btns = ttk.Frame(self)
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)
        ttk.Button(btns, text="Save", command=self.save).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Reset", command=self.reset).pack(side=tk.RIGHT, padx=4)

    def _make_widget(self, master, val):
        if isinstance(val, bool):
            var = tk.BooleanVar(value=val)
            return "bool", var, ttk.Checkbutton(master, variable=var)
        if isinstance(val, (list, tuple)):
            var = tk.StringVar(value=", ".join(str(x) for x in val))
            return "list", var, ttk.Entry(master, textvariable=var)
        if _is_number(val):
            var = tk.StringVar(value=repr(val))
            return ("int" if isinstance(val, int) else "float"), var, \
                ttk.Entry(master, textvariable=var)
        var = tk.StringVar(value="" if val is None else str(val))
        return "str", var, ttk.Entry(master, textvariable=var)

    def get_values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, (kind, var) in self._vars.items():
            raw = var.get()
            if kind == "bool":
                out[key] = bool(raw)
            elif kind == "int":
                out[key] = int(float(raw))
            elif kind == "float":
                out[key] = float(raw)
            elif kind == "list":
                items = [s.strip() for s in str(raw).split(",") if s.strip() != ""]
                orig = self._orig.get(key) or []
                cast = float
                if orig and all(isinstance(x, int) and not isinstance(x, bool)
                                for x in orig):
                    cast = int
                elif orig and all(isinstance(x, str) for x in orig):
                    cast = str
                out[key] = [cast(s) for s in items]
            else:
                out[key] = raw
        return out

    def reset(self) -> None:
        for key, (kind, var) in self._vars.items():
            val = self._orig[key]
            if kind == "bool":
                var.set(bool(val))
            elif kind == "list":
                var.set(", ".join(str(x) for x in val))
            else:
                var.set("" if val is None else (repr(val) if _is_number(val)
                                                else str(val)))

    def save(self) -> dict[str, Any]:
        values = self.get_values()
        if self._on_save is not None:
            self._on_save(self.section, values)
        return values
