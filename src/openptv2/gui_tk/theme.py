"""Modern look for the Tk GUI, via ttkbootstrap.

ttkbootstrap restyles *all* standard ttk widgets — including the parameter
``ttk.Treeview`` — with Bootstrap-style themes (accent colours, flat modern
controls, ~18 light/dark themes). It is a drop-in over ttk (no widget rewrite),
pure Python (free-threaded-safe), and unlike CustomTkinter it themes the tree
and works with our embedded matplotlib panels.

We derive a matplotlib palette from the active theme's colours so the image/plot
panels blend into the chrome. The matplotlib navigation-toolbar icons are dark
glyphs with no theme awareness, so that one strip is kept light in all themes.
"""
from __future__ import annotations

# Curated shortlist shown in the View → Theme menu (all 18 are available).
LIGHT_THEMES = ["litera", "cosmo", "flatly", "minty", "yeti", "sandstone"]
DARK_THEMES = ["darkly", "superhero", "cyborg", "solar"]
DEFAULT_THEME = "litera"


def _colors(theme_name: str) -> dict:
    from ttkbootstrap.themes.standard import STANDARD_THEMES
    return STANDARD_THEMES[theme_name]["colors"]


def palette(theme_name: str) -> dict:
    c = _colors(theme_name)
    return {
        "bg": c["bg"], "panel": c["bg"], "fg": c["fg"],
        "axes": c["bg"], "grid": c["border"], "accent": c["primary"],
    }


def theme_names() -> list[str]:
    return LIGHT_THEMES + DARK_THEMES


def apply(root, theme_name: str = DEFAULT_THEME) -> dict:
    """Apply a ttkbootstrap theme to the app; return the matplotlib palette."""
    import ttkbootstrap as tb
    from ttkbootstrap import Style
    style = getattr(root, "_tb_style", None)
    if style is None:
        # ttkbootstrap's Style is a process singleton bound to one interpreter;
        # drop any stale instance (e.g. a previous root/app) so it binds to THIS
        # root instead of a destroyed one.
        try:
            tb.style.Style.instance = None
        except Exception:
            pass
        style = Style(theme=theme_name)
        root._tb_style = style
    else:
        style.theme_use(theme_name)
    root._tb_theme = theme_name
    return palette(theme_name)


def set_theme(root, theme_name: str) -> dict:
    return apply(root, theme_name)


def current_theme(root) -> str:
    return getattr(root, "_tb_theme", DEFAULT_THEME)


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
    """Keep the matplotlib toolbar strip light so its dark icons stay visible in
    every theme (the icons are not theme-aware)."""
    strip = "#f0f0f0"
    try:
        toolbar.config(background=strip)
        for child in toolbar.winfo_children():
            try:
                child.config(background=strip)
            except Exception:
                pass
    except Exception:
        pass
