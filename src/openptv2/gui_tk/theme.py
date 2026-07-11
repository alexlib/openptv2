"""Modern look for the Tk GUI.

Uses the Sun Valley (sv-ttk) theme for all ttk widgets (buttons, tree, panes)
and provides a matching matplotlib palette so the embedded image/plot panels
blend into the dark/light chrome instead of glaring white. Pure Python/Tcl —
free-threaded-safe, no Qt/C-extension theming stack.
"""
from __future__ import annotations

DARK = {
    "bg": "#1c1c1c", "panel": "#1c1c1c", "fg": "#fafafa",
    "axes": "#242424", "grid": "#3a3a3a", "accent": "#3b8ed0",
}
LIGHT = {
    "bg": "#fafafa", "panel": "#fafafa", "fg": "#1c1c1c",
    "axes": "#ffffff", "grid": "#c8c8c8", "accent": "#1f6aa5",
}


def palette(mode: str) -> dict:
    return DARK if mode == "dark" else LIGHT


def apply(root, mode: str = "dark") -> dict:
    """Apply the Sun Valley theme to the whole app; return the mpl palette."""
    import sv_ttk
    sv_ttk.set_theme(mode)
    return palette(mode)


def current_mode() -> str:
    try:
        import sv_ttk
        return sv_ttk.get_theme()
    except Exception:
        return "dark"


def toggle(root) -> str:
    import sv_ttk
    sv_ttk.toggle_theme()
    return current_mode()


def style_figure(fig, pal: dict) -> None:
    """Recolor a matplotlib Figure to match the chrome."""
    fig.set_facecolor(pal["panel"])
    for ax in fig.get_axes():
        ax.set_facecolor(pal["axes"])
        ax.tick_params(colors=pal["fg"], which="both")
        for spine in ax.spines.values():
            spine.set_color(pal["grid"])
        for lbl in (ax.xaxis.label, ax.yaxis.label, ax.title):
            lbl.set_color(pal["fg"])
        if hasattr(ax, "zaxis"):  # Axes3D
            ax.zaxis.label.set_color(pal["fg"])
            ax.tick_params(axis="z", colors=pal["fg"])


def style_toolbar(toolbar, pal: dict) -> None:
    """Recolor the matplotlib Tk navigation toolbar.

    matplotlib's toolbar icons are dark glyphs with no theme awareness, so a
    dark strip makes them invisible. Keep the strip LIGHT in both modes so the
    icons always read; the surrounding chrome is themed by sv-ttk."""
    strip = "#f0f0f0"  # always light so the dark toolbar icons stay visible
    try:
        toolbar.config(background=strip)
        for child in toolbar.winfo_children():
            try:
                child.config(background=strip)
            except Exception:
                pass
    except Exception:
        pass
