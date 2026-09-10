"""3D trajectory ROI defined as inclusion polygons on three projections.

Drawn interactively in ``openptv2.gui.trajectory_roi_gui`` on the XY, XZ, and
YZ projections of a run's 3D trajectory positions. A position is inside the
ROI when it falls inside every polygon that is set -- a plane left undrawn
imposes no constraint, and no polygons at all means everything passes.

Same ``point_rule``/``combine_rule`` JSON convention as
aortic-particle-pipeline's ``roi.json`` (``all_points_inside`` / ``and``), so
the two are interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PLANES = {"xy_polygon": (0, 1), "xz_polygon": (0, 2), "yz_polygon": (1, 2)}


@dataclass(frozen=True)
class TrajectoryROI:
    xy_polygon: np.ndarray | None = None
    xz_polygon: np.ndarray | None = None
    yz_polygon: np.ndarray | None = None

    @classmethod
    def from_json(cls, path: Path | str) -> "TrajectoryROI":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kwargs = {
            name: np.asarray(data[name], dtype=float) if data.get(name) else None
            for name in PLANES
        }
        return cls(**kwargs)

    def to_json(self, path: Path | str) -> None:
        payload = {"point_rule": "all_points_inside", "combine_rule": "and"}
        for name in PLANES:
            polygon = getattr(self, name)
            payload[name] = polygon.tolist() if polygon is not None else None
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def is_empty(self) -> bool:
        return all(getattr(self, name) is None for name in PLANES)

    def contains(self, positions: np.ndarray) -> np.ndarray:
        """True for each row of ``positions`` (N, 3) inside every set polygon."""
        from matplotlib.path import Path as MplPath

        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        inside = np.ones(len(positions), dtype=bool)
        for name, (a, b) in PLANES.items():
            polygon = getattr(self, name)
            if polygon is not None:
                inside &= MplPath(polygon).contains_points(positions[:, [a, b]])
        return inside
