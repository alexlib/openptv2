"""Load one or more calibration models and check them for physical plausibility.

Shared by the interactive viewer (`gui/visualize_calibration_nb.py`, a marimo
notebook) and the headless CLI (`scripts/calibration_diagnostics.py`) so the
loading/diagnostic logic lives exactly once.

A passing reprojection RMS at the matched calibration points only proves
local self-consistency -- it does not prove the pose is physically sane. The
checks here catch what RMS misses: is each camera's optical axis actually
pointed at the calibration body (`angle`), and are the cameras' distances to
the body roughly symmetric (`spread`) the way a real rig's cameras usually
are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from openptv2.algorithms.sortgrid import read_calblock
from openptv2.calibration import Calibration


@dataclass
class CameraDiagnostic:
    name: str
    pos: np.ndarray
    rot: np.ndarray
    ori_path: Path
    dist: float = 0.0
    angle: float = 0.0
    flag: bool = False
    rms: float | None = None
    matched: int | None = None


@dataclass
class ModelDiagnostic:
    label: str
    cameras: list[CameraDiagnostic] = field(default_factory=list)
    spread: float = 0.0
    flag: bool = False


def viewing_dir(rot: np.ndarray) -> np.ndarray:
    """Optical axis in world coords. Column 2 of the .ori rotation matrix
    points backward out of the lens for this convention -- verified against
    openptv2.algorithms.imgcoord.img_coord."""
    return -rot[:, 2]


def angle_to_target(pos: np.ndarray, rot: np.ndarray, target: np.ndarray) -> float:
    v = target - pos
    v = v / np.linalg.norm(v)
    cos_a = np.clip(np.dot(viewing_dir(rot), v), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _addpar_for(ori_path: Path) -> Path:
    return Path(str(ori_path).replace(".ori", ".addpar"))


def load_model(
    path_str: str,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray, Path]], Path | None]:
    """Load one model's cameras from a YAML (`cal_ori` section) or a
    directory of `cam*.tif.ori`/`.addpar` files directly.

    Returns ([(cam_name, pos, rot, ori_path), ...], calblock_path_or_None).
    """
    path = Path(path_str).expanduser().resolve()
    if path.suffix.lower() in (".yaml", ".yml"):
        from openptv2.gui.parameter_manager import ParameterManager

        pm = ParameterManager()
        pm.from_yaml(path)
        cal_ori = pm.get_parameter("cal_ori") or {}
        img_ori = cal_ori.get("img_ori") or []
        if not img_ori:
            raise ValueError(f"{path}: cal_ori.img_ori is empty")
        base = path.parent
        ori_paths = [
            (base / o) if not Path(o).is_absolute() else Path(o) for o in img_ori
        ]
        fixp = cal_ori.get("fixp_name")
        calblock = (base / fixp) if fixp else None
    else:
        if not path.is_dir():
            raise ValueError(f"{path}: not a YAML file or a directory")
        ori_paths = sorted(path.glob("cam_*.tif.ori")) or sorted(
            path.glob("cam*.tif.ori")
        )
        if not ori_paths:
            raise ValueError(f"{path}: no cam*.tif.ori files found")
        calblock = None
        for candidate_dir in (path, path.parent):
            hits = sorted(candidate_dir.glob("*calblock*.txt")) + sorted(
                candidate_dir.glob("*calib*block*.txt")
            )
            if hits:
                calblock = hits[0]
                break

    cams = []
    for i, ori_path in enumerate(ori_paths, start=1):
        cal = Calibration()
        cal.from_file(str(ori_path), str(_addpar_for(ori_path)))
        pos = np.asarray(cal.get_pos(), dtype=float)
        rot = np.asarray(cal.get_rotation_matrix(), dtype=float)
        cams.append((f"cam{i}", pos, rot, ori_path))
    return cams, calblock


def find_rms(ori_path: Path) -> tuple[float | None, int | None]:
    """Reprojection RMS from a sibling calib_matches/cam{N}_matches.txt, if present."""
    m = re.search(r"cam_?(\d+)", ori_path.stem)
    if not m:
        return None, None
    n = m.group(1)
    for base in (ori_path.parent, ori_path.parent.parent):
        candidate = base / "calib_matches" / f"cam{n}_matches.txt"
        if candidate.exists():
            ids, det, rep = [], [], []
            for line in candidate.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                ids.append(int(parts[0]))
                det.append((float(parts[1]), float(parts[2])))
                rep.append((float(parts[3]), float(parts[4])))
            if not ids:
                return None, None
            det, rep = np.array(det), np.array(rep)
            rms = float(np.sqrt(np.mean(np.sum((det - rep) ** 2, axis=1))))
            return rms, len(ids)
    return None, None


def compute_diagnostics(
    models: dict[str, list[tuple[str, np.ndarray, np.ndarray, Path]]],
    centroid: np.ndarray,
    angle_flag_deg: float = 15.0,
    spread_flag_ratio: float = 0.3,
) -> dict[str, ModelDiagnostic]:
    """Per-camera sight-line angle + reprojection RMS, per-model centroid-distance
    spread. `angle_flag_deg`/`spread_flag_ratio` are the thresholds used to set
    each CameraDiagnostic/ModelDiagnostic's `flag`."""
    out: dict[str, ModelDiagnostic] = {}
    for label, cams in models.items():
        cam_diags = []
        for name, pos, rot, ori_path in cams:
            ang = angle_to_target(pos, rot, centroid)
            dist = float(np.linalg.norm(pos - centroid))
            rms, matched = find_rms(ori_path)
            cam_diags.append(
                CameraDiagnostic(
                    name=name,
                    pos=pos,
                    rot=rot,
                    ori_path=ori_path,
                    dist=dist,
                    angle=ang,
                    flag=ang > angle_flag_deg,
                    rms=rms,
                    matched=matched,
                )
            )
        dists = [c.dist for c in cam_diags] or [1.0]
        spread = float(np.ptp(dists))
        out[label] = ModelDiagnostic(
            label=label,
            cameras=cam_diags,
            spread=spread,
            flag=spread > spread_flag_ratio * float(np.mean(dists)),
        )
    return out


def resolve_centroid(
    models: dict[str, list[tuple[str, np.ndarray, np.ndarray, Path]]],
    calblock_path: Path | None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Calibration-body point cloud (or None) + its centroid, falling back to
    the camera-cluster centroid when no calblock is available."""
    body = None
    if calblock_path is not None and calblock_path.exists():
        body, _ = read_calblock(str(calblock_path))
    if body is not None and len(body):
        return body, body.mean(axis=0)
    all_pos = [pos for cams in models.values() for _, pos, _, _ in cams]
    centroid = np.mean(all_pos, axis=0) if all_pos else np.zeros(3)
    return body, centroid


def parse_models_arg(models_spec: str) -> list[tuple[str, str]]:
    """'label1=path1,label2=path2' -> [(label1, path1), (label2, path2)]."""
    parsed = []
    for entry in models_spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, _, path = entry.partition("=")
        parsed.append((label.strip() or path.strip(), path.strip()))
    return parsed
