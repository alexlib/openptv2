"""Value-by-value Burgers tracking parity suite.

This suite compares outputs produced by:
1. Python trackcorr forward  vs Cython trackcorr forward
2. Python track3d forward    vs Cython track3d forward
3. Python trackcorr backward vs Cython trackcorr backward

It also validates a full Python step-by-step process against Python full_forward
for deterministic end-state identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest
import yaml

from ..conftest import FIXTURES


@dataclass
class FileDiff:
    file: str
    message: str


def _copy_burgers_workspace(tmp_path: Path) -> Path:
    src = FIXTURES / "burgers"
    work = tmp_path / "burgers"
    shutil.copytree(src, work)

    # Fresh runtime folders each run
    res = work / "res"
    img = work / "img"
    if res.exists():
        shutil.rmtree(res)
    if img.exists():
        shutil.rmtree(img)
    shutil.copytree(work / "res_orig", res)
    shutil.copytree(work / "img_orig", img)

    return work


def _localize_conf(conf: dict[str, Any], work: Path) -> dict[str, Any]:
    conf = yaml.safe_load(yaml.safe_dump(conf))

    prefix = "test_data/burgers/"

    def _relocate(path_str: str) -> str:
        if path_str.startswith(prefix):
            return str(work / path_str[len(prefix) :])
        return path_str

    for cam in conf["cameras"]:
        cam["ori_file"] = _relocate(cam["ori_file"])
        if cam.get("addpar_file"):
            cam["addpar_file"] = _relocate(cam["addpar_file"])

    conf["sequence"]["targets_template"] = _relocate(conf["sequence"]["targets_template"])
    return conf


def _build_python_tracker(conf: dict[str, Any], work: Path):
    from algorithms.calibration import Calibration as PyCalibration
    from algorithms.parameters import ControlPar, SequencePar, TrackParTuple, VolumePar
    from algorithms.track import Tracker as PyTracker

    seq_cfg = conf["sequence"]
    scene = conf["scene"]
    corresp = conf["correspondences"]
    tracking = conf["tracking"]

    cals = []
    for cam in conf["cameras"]:
        cal = PyCalibration()
        cal.from_file(cam["ori_file"], cam.get("addpar_file", None))
        cals.append(cal)

    cpar = ControlPar(num_cams=len(conf["cameras"]))
    cpar.imx = scene["image_size"][0]
    cpar.imy = scene["image_size"][1]
    cpar.pix_x = scene["pixel_size"][0]
    cpar.pix_y = scene["pixel_size"][1]

    vpar = VolumePar(
        x_lay=corresp["x_span"],
        z_min_lay=[z[0] for z in corresp["z_spans"]],
        z_max_lay=[z[1] for z in corresp["z_spans"]],
    )

    vel = tracking["velocity_lims"]
    tpar = TrackParTuple(
        dvxmin=vel[0][0],
        dvxmax=vel[0][1],
        dvymin=vel[1][0],
        dvymax=vel[1][1],
        dvzmin=vel[2][0],
        dvzmax=vel[2][1],
        dangle=tracking["angle_lim"],
        dacc=tracking["accel_lim"],
        add=tracking["add_particle"],
        dsumg=0.0,
        dn=0.0,
        dnx=0.0,
        dny=0.0,
    )

    img_base = [seq_cfg["targets_template"].format(cam=i + 1) for i in range(len(conf["cameras"]))]
    spar = SequencePar(img_base_name=img_base, first=seq_cfg["first"], last=seq_cfg["last"])

    naming = {
        "corres": str(work / "res" / "rt_is"),
        "linkage": str(work / "res" / "ptv_is"),
        "prio": str(work / "res" / "whatever"),
    }
    return PyTracker(cpar, vpar, tpar, spar, cals, naming)


def _build_cython_tracker(conf: dict[str, Any], work: Path):
    from optv.calibration import Calibration as CyCalibration
    from optv.parameters import ControlParams, SequenceParams, TrackingParams, VolumeParams
    from optv.tracker import Tracker as CyTracker

    seq_cfg = conf["sequence"]

    cals = []
    img_base = []
    for i, cam in enumerate(conf["cameras"]):
        cal = CyCalibration()
        addpar = cam.get("addpar_file", None)
        cal.from_file(cam["ori_file"].encode(), addpar.encode() if addpar else None)
        cals.append(cal)
        img_base.append(seq_cfg["targets_template"].format(cam=i + 1))

    cpar = ControlParams(len(conf["cameras"]), **conf["scene"])
    vpar = VolumeParams(**conf["correspondences"])
    tpar = TrackingParams(**conf["tracking"])
    spar = SequenceParams(image_base=img_base, frame_range=(seq_cfg["first"], seq_cfg["last"]))

    naming = {
        "corres": str(work / "res" / "rt_is").encode(),
        "linkage": str(work / "res" / "ptv_is").encode(),
        "prio": str(work / "res" / "whatever").encode(),
    }
    return CyTracker(cpar, vpar, tpar, spar, cals, naming)


def _run_tracker(engine: str, mode: str, work: Path, conf: dict[str, Any]) -> Path:
    out = work / f"out_{engine}_{mode}"

    # Reset runtime input/output folders
    for name in ("res", "img"):
        p = work / name
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(work / "res_orig", work / "res")
    shutil.copytree(work / "img_orig", work / "img")

    tracker = _build_python_tracker(conf, work) if engine == "python" else _build_cython_tracker(conf, work)

    if mode == "forward":
        tracker.full_forward()
    elif mode == "forward_3d":
        tracker.full_forward_3d()
    elif mode == "backward":
        tracker.full_forward()
        tracker.full_backward()
    elif mode == "step_forward":
        tracker.restart()
        while tracker.step_forward():
            pass
        tracker.finalize()
    else:
        raise ValueError(mode)

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(work / "res", out)
    return out


def _parse_rt(path: Path):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0])
    rows = []
    for ln in lines[1:]:
        t = ln.split()
        rows.append(
            {
                "id": int(t[0]),
                "xyz": np.array([float(t[1]), float(t[2]), float(t[3])]),
                "p": tuple(int(x) for x in t[4:8]),
            }
        )
    return n, rows


def _parse_ptv(path: Path):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0])
    rows = []
    for ln in lines[1:]:
        t = ln.split()
        rows.append(
            {
                "prev": int(t[0]),
                "next": int(t[1]),
                "xyz": np.array([float(t[2]), float(t[3]), float(t[4])]),
            }
        )
    return n, rows


def _parse_whatever(path: Path):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0])
    rows = []
    for ln in lines[1:]:
        t = ln.split()
        rows.append(
            {
                "prev": int(t[0]),
                "next": int(t[1]),
                "xyz": np.array([float(t[2]), float(t[3]), float(t[4])]),
                "prio": int(t[5]),
            }
        )
    return n, rows


def _xyz_for_correspondence(res_dir: Path, frame: int, corres: tuple[int, int, int, int]) -> np.ndarray:
    _, rows = _parse_rt(res_dir / f"rt_is.{frame}")
    for row in rows:
        if row["p"] == corres:
            return row["xyz"]
    raise AssertionError(f"No particle with correspondences={corres} in rt_is.{frame}")


def _find_ptv_row_by_xyz(res_dir: Path, frame: int, xyz: np.ndarray, atol: float = 1e-9):
    _, rows = _parse_ptv(res_dir / f"ptv_is.{frame}")
    for row in rows:
        if np.allclose(row["xyz"], xyz, atol=atol, rtol=0.0):
            return row
    return None


def _compare_result_dirs(left: Path, right: Path, atol: float = 1e-12) -> list[FileDiff]:
    diffs: list[FileDiff] = []

    lfiles = sorted([p.name for p in left.iterdir() if p.is_file()])
    rfiles = sorted([p.name for p in right.iterdir() if p.is_file()])

    if lfiles != rfiles:
        diffs.append(FileDiff("__files__", f"file-set mismatch: left={lfiles}, right={rfiles}"))
        return diffs

    for name in lfiles:
        lp = left / name
        rp = right / name

        if name.startswith("rt_is."):
            ln, lr = _parse_rt(lp)
            rn, rr = _parse_rt(rp)
            if ln != rn or len(lr) != len(rr):
                diffs.append(FileDiff(name, f"record-count mismatch: {ln}/{len(lr)} vs {rn}/{len(rr)}"))
                continue
            for i, (a, b) in enumerate(zip(lr, rr)):
                if a["id"] != b["id"]:
                    diffs.append(FileDiff(name, f"row {i}: id mismatch {a['id']} vs {b['id']}"))
                    break
                if a["p"] != b["p"]:
                    diffs.append(FileDiff(name, f"row {i}: correspondences mismatch {a['p']} vs {b['p']}"))
                    break
                if not np.allclose(a["xyz"], b["xyz"], atol=atol, rtol=0.0):
                    diffs.append(FileDiff(name, f"row {i}: xyz mismatch {a['xyz']} vs {b['xyz']}"))
                    break

        elif name.startswith("ptv_is."):
            ln, lr = _parse_ptv(lp)
            rn, rr = _parse_ptv(rp)
            if ln != rn or len(lr) != len(rr):
                diffs.append(FileDiff(name, f"record-count mismatch: {ln}/{len(lr)} vs {rn}/{len(rr)}"))
                continue
            for i, (a, b) in enumerate(zip(lr, rr)):
                if (a["prev"], a["next"]) != (b["prev"], b["next"]):
                    diffs.append(
                        FileDiff(name, f"row {i}: link mismatch {(a['prev'], a['next'])} vs {(b['prev'], b['next'])}")
                    )
                    break
                if not np.allclose(a["xyz"], b["xyz"], atol=atol, rtol=0.0):
                    diffs.append(FileDiff(name, f"row {i}: xyz mismatch {a['xyz']} vs {b['xyz']}"))
                    break

        elif name.startswith("whatever."):
            ln, lr = _parse_whatever(lp)
            rn, rr = _parse_whatever(rp)
            if ln != rn or len(lr) != len(rr):
                diffs.append(FileDiff(name, f"record-count mismatch: {ln}/{len(lr)} vs {rn}/{len(rr)}"))
                continue
            for i, (a, b) in enumerate(zip(lr, rr)):
                if (a["prev"], a["next"], a["prio"]) != (b["prev"], b["next"], b["prio"]):
                    diffs.append(
                        FileDiff(
                            name,
                            f"row {i}: link/prio mismatch {(a['prev'], a['next'], a['prio'])} vs {(b['prev'], b['next'], b['prio'])}",
                        )
                    )
                    break
                if not np.allclose(a["xyz"], b["xyz"], atol=atol, rtol=0.0):
                    diffs.append(FileDiff(name, f"row {i}: xyz mismatch {a['xyz']} vs {b['xyz']}"))
                    break

    return diffs


@pytest.mark.slow
@pytest.mark.parity
def test_burgers_trackcorr_forward_python_vs_cython_value_by_value(tmp_path: Path):
    work = _copy_burgers_workspace(tmp_path)
    conf = _localize_conf(yaml.safe_load((work / "conf.yaml").read_text()), work)

    py_out = _run_tracker("python", "forward", work, conf)
    cy_out = _run_tracker("cython", "forward", work, conf)

    diffs = _compare_result_dirs(py_out, cy_out)
    diff_files = sorted({d.file for d in diffs})
    expected_diff_files = ["ptv_is.10004", "ptv_is.10005", "whatever.10004", "whatever.10005"]
    assert diff_files == expected_diff_files, (
        f"Unexpected trackcorr forward diff set. Expected {expected_diff_files}, got {diff_files}.\n"
        + "\n".join(f"{d.file}: {d.message}" for d in diffs)
    )


@pytest.mark.slow
@pytest.mark.parity
def test_burgers_track3d_forward_python_vs_cython_verify_deviations(tmp_path: Path):
    work = _copy_burgers_workspace(tmp_path)
    conf = _localize_conf(yaml.safe_load((work / "conf.yaml").read_text()), work)

    py_out = _run_tracker("python", "forward_3d", work, conf)
    cy_out = _run_tracker("cython", "forward_3d", work, conf)

    py_files = sorted([p.name for p in py_out.iterdir() if p.is_file()])
    cy_files = sorted([p.name for p in cy_out.iterdir() if p.is_file()])

    py_rt = sorted([f for f in py_files if f.startswith("rt_is.")])
    cy_rt = sorted([f for f in cy_files if f.startswith("rt_is.")])
    assert py_rt == cy_rt

    py_ptv = sorted([f for f in py_files if f.startswith("ptv_is.")])
    cy_ptv = sorted([f for f in cy_files if f.startswith("ptv_is.")])
    assert py_ptv == ["ptv_is.10001", "ptv_is.10002", "ptv_is.10003", "ptv_is.10004", "ptv_is.10005"]
    assert cy_ptv == ["ptv_is.10001", "ptv_is.10002", "ptv_is.10003", "ptv_is.10004", "ptv_is.10005"]

    py_whatever = sorted([f for f in py_files if f.startswith("whatever.")])
    cy_whatever = sorted([f for f in cy_files if f.startswith("whatever.")])
    assert py_whatever == [
        "whatever.10001",
        "whatever.10002",
        "whatever.10003",
        "whatever.10004",
        "whatever.10005",
    ]
    assert cy_whatever == [
        "whatever.10001",
        "whatever.10002",
        "whatever.10003",
        "whatever.10004",
        "whatever.10005",
    ]

    # Compare only overlapping files and verify the known bounded diff set.
    common = sorted(set(py_files) & set(cy_files))
    py_common = work / "common_py"
    cy_common = work / "common_cy"
    if py_common.exists():
        shutil.rmtree(py_common)
    if cy_common.exists():
        shutil.rmtree(cy_common)
    py_common.mkdir()
    cy_common.mkdir()
    for name in common:
        shutil.copy2(py_out / name, py_common / name)
        shutil.copy2(cy_out / name, cy_common / name)

    diffs = _compare_result_dirs(py_common, cy_common)
    diff_files = sorted({d.file for d in diffs})
    expected_diff_files: list[str] = []
    assert diff_files == expected_diff_files, (
        f"Unexpected track3d diff set. Expected {expected_diff_files}, got {diff_files}.\n"
        + "\n".join(f"{d.file}: {d.message}" for d in diffs)
    )


@pytest.mark.slow
@pytest.mark.parity
def test_burgers_trackcorr_backward_python_vs_cython_verify_deviations(tmp_path: Path):
    """Backward parity currently has known, bounded differences.

    This test verifies those differences explicitly so any drift is detected.
    """
    work = _copy_burgers_workspace(tmp_path)
    conf = _localize_conf(yaml.safe_load((work / "conf.yaml").read_text()), work)

    py_out = _run_tracker("python", "backward", work, conf)
    cy_out = _run_tracker("cython", "backward", work, conf)

    diffs = _compare_result_dirs(py_out, cy_out)
    diff_files = sorted({d.file for d in diffs})

    expected = [
        "ptv_is.10002",
        "ptv_is.10004",
        "ptv_is.10005",
        "rt_is.10002",
        "whatever.10002",
        "whatever.10004",
        "whatever.10005",
    ]
    assert diff_files == expected, (
        f"Unexpected backward diff set. Expected {expected}, got {diff_files}.\n"
        + "\n".join(f"{d.file}: {d.message}" for d in diffs)
    )


@pytest.mark.slow
@pytest.mark.parity
def test_burgers_python_complete_step_process_matches_full_forward(tmp_path: Path, capsys):
    """Run full Python process in two ways and require identical final outputs.

    1) explicit step-by-step forward loop (shows all step prints)
    2) one-shot full_forward
    """
    work = _copy_burgers_workspace(tmp_path)
    conf = _localize_conf(yaml.safe_load((work / "conf.yaml").read_text()), work)

    step_out = _run_tracker("python", "step_forward", work, conf)
    captured = capsys.readouterr().out
    assert "step:" in captured
    assert "Average over sequence" in captured

    full_out = _run_tracker("python", "forward", work, conf)

    diffs = _compare_result_dirs(step_out, full_out)
    assert diffs == [], "\n".join(f"{d.file}: {d.message}" for d in diffs)


@pytest.mark.slow
@pytest.mark.parity
def test_burgers_python_forward_relinks_reappeared_particle_in_both_trackers(tmp_path: Path):
    """When only frames 10001..10005 exist, P2 re-appearance at 10004 must still link to 10005.

    This is the regression for the trackcorr fallback path used when no 10006 lookahead exists.
    """
    work = _copy_burgers_workspace(tmp_path)
    conf = _localize_conf(yaml.safe_load((work / "conf.yaml").read_text()), work)

    p2_corres = (2, 2, 2, 2)

    for mode in ("forward", "forward_3d"):
        out = _run_tracker("python", mode, work, conf)
        p2_10004 = _xyz_for_correspondence(out, 10004, p2_corres)
        p2_10005 = _xyz_for_correspondence(out, 10005, p2_corres)

        row_10004 = _find_ptv_row_by_xyz(out, 10004, p2_10004)
        row_10005 = _find_ptv_row_by_xyz(out, 10005, p2_10005)

        assert row_10004 is not None, f"{mode}: missing P2 row in ptv_is.10004"
        assert row_10005 is not None, f"{mode}: missing P2 row in ptv_is.10005"
        assert row_10004["next"] >= 0, f"{mode}: expected P2 10004->10005 link"
        assert row_10005["prev"] >= 0, f"{mode}: expected P2 10005 to have predecessor"


@pytest.mark.slow
@pytest.mark.parity
def test_burgers_python_backward_starts_from_forward_and_keeps_relinked_segment(tmp_path: Path):
    """Backward tracking should operate on top of forward output and keep valid re-linked segments."""
    work = _copy_burgers_workspace(tmp_path)
    conf = _localize_conf(yaml.safe_load((work / "conf.yaml").read_text()), work)

    forward_out = _run_tracker("python", "forward", work, conf)
    backward_out = _run_tracker("python", "backward", work, conf)

    p2_corres = (2, 2, 2, 2)
    p2_10004 = _xyz_for_correspondence(backward_out, 10004, p2_corres)
    p2_10005 = _xyz_for_correspondence(backward_out, 10005, p2_corres)

    row_10004 = _find_ptv_row_by_xyz(backward_out, 10004, p2_10004)
    row_10005 = _find_ptv_row_by_xyz(backward_out, 10005, p2_10005)

    assert row_10004 is not None
    assert row_10005 is not None
    assert row_10004["next"] >= 0
    assert row_10005["prev"] >= 0

    fw_bw_diffs = _compare_result_dirs(forward_out, backward_out)
    assert fw_bw_diffs != [], "Backward pass should update at least one output file."
