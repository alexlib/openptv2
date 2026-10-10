"""Movie of the merged Ilmenau trajectories in the look of the delivered
TimeStep_10000_to_12000.mp4: same viewing direction, same colour bar
(vertical velocity v, -0.5..0.5 m/s), trailing beads, grey back walls, 500 mm grid,
plus the barrel's bottom (heating) and top (cooling) rims as circles.

Coordinates are OURS, not the movie's: the barrel frame of merged_fulldiam.zarr
(mm, origin on the barrel axis at mid-height, +Y up); v is the Y velocity.
Rendering is postptv's flowtracks.graphics.animate_trajectories_3d (PyVista).

    uv run --project /Users/alex/Documents/Github/postptv python \
        scripts/illmenau/make_barrel_movie.py [--first 10250 --last 10250] [--out DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from flowtracks.graphics import animate_trajectories_3d
from matplotlib.colors import LinearSegmentedColormap

R = Path("/Users/alex/Downloads/Ilmenau")

# colour bar of the reference movie, sampled from its frames at -0.5, -0.4 ... +0.5
REF_COLOURS = np.array([
    [0, 0, 81], [0, 0, 236], [49, 51, 251], [118, 119, 254], [185, 186, 251],
    [250, 250, 250],
    [251, 172, 173], [253, 90, 88], [251, 8, 8], [212, 0, 1], [173, 0, 4],
]) / 255.0
REF_CMAP = LinearSegmentedColormap.from_list("ilmenau_reference", REF_COLOURS, N=256)

# viewing direction of the reference movie, fitted to the six outer corners of
# its box (residual 6-15 px of 2920x1840): elevation 31 deg, from the -x/+z
# corner, practically parallel projection
VIEW_DIRECTION = (-0.668, 0.510, 0.542)  # from the scene towards the camera
VIEW_UP = (0.343, 0.857, -0.383)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"  # VTK's own font has no [ ]


def barrel_rims(n=360):
    """Bottom and top rim of the barrel in the barrel frame [mm]: radius and
    height from plate.yaml's test_section, origin at mid-height."""
    ts = yaml.safe_load((R / "openptv_illmenau_4cam" / "plate.yaml").read_text())
    ts = ts["plate"]["test_section"] if "plate" in ts else ts["test_section"]
    r, h = float(ts["radius"]), float(ts["height"])
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([r * np.cos(a), np.zeros(n), r * np.sin(a)])
    return [ring + [0, -h / 2, 0], ring + [0, h / 2, 0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(R / "merged_fulldiam.zarr"))
    ap.add_argument("--first", type=int, default=10001)
    ap.add_argument("--last", type=int, default=10500)
    ap.add_argument("--tail", type=int, default=20, help="trail length [frames]")
    ap.add_argument("--out", default=str(R / "movie_merged_fulldiam"))
    ap.add_argument("--fps", type=int, default=10, help="movie frame rate (real: 10)")
    a = ap.parse_args()

    out = Path(a.out)
    frames = np.arange(a.first, a.last + 1)
    movie = out.with_suffix(".mp4") if len(frames) > 1 else None
    paths = animate_trajectories_3d(
        a.store, out, frames, component=1, tail=a.tail, pos_scale=1000.0,
        bounds=(-4000, 4000, -2000, 2000, -4000, 4000), outlines=barrel_rims(),
        cmap=REF_CMAP, clim=(-0.5, 0.5), bar_title="Velocity v [m/s]",
        view_direction=VIEW_DIRECTION, view_up=VIEW_UP, zoom=1.1,
        window_center=(0.09, 0.0), window_size=(2920, 1840), point_size=5.0,
        grid_step=500.0, axis_titles=("x [mm]", "y [mm]", "z [mm]"),
        font_size=20, font_file=FONT if Path(FONT).exists() else None,
        movie=movie, fps=a.fps, label="frame {frame}",
    )
    print(f"{len(paths)} frames -> {out}" + (f", movie {movie}" if movie else ""))


if __name__ == "__main__":
    main()
