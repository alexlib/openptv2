"""A synthetic dumbbell calibration with a known answer.

1. Copy ``test_data/test_cavity`` (4 cameras, its cameras are the TRUTH).
2. Move a rigid bar of known length through the volume and render its two end points
   into every camera (``openptv2.gui.dumbbell_ground_truth``), optionally with pixel noise.
3. Spoil the cameras of the copy (position and angles), keep camera 1 as the reference.
4. Run the dumbbell calibration (``openptv2.gui.standalone_dumbbell_calibration``).
5. Compare with the truth: camera position and angle errors before and after, and the bar length
   measured with the spoiled and with the fitted cameras.

    uv run python scripts/synthetic_dumbbell.py WORKDIR [--frames 40] [--noise 0.1] [--length 46]
        [--claimed-length L]   # the length you TELL the program (default: the true one); try a wrong one
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _read_cals(work: Path):
    from openptv2.calibration import Calibration

    out = []
    for i in range(1, 5):
        c = Calibration()
        c.from_file(str(work / "cal" / f"cam{i}.tif.ori"), str(work / "cal" / f"cam{i}.tif.addpar"))
        out.append(c)
    return out


def _bar_length(work: Path, first: int, last: int, cals) -> float:
    """Median distance between the two end points, triangulated with ``cals`` from the stored targets."""
    from openptv2.gui import ptv
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.standalone_dumbbell_calibration import (
        _read_dumbbell_targets_metric,
        _to_flat,
    )
    from openptv2.orientation import multi_cam_point_positions
    from openptv2.storage import RunStore, find_existing_store

    pm = ParameterManager()
    pm.from_yaml(work / "parameters_Run1.yaml")
    cpar, *_ = ptv.py_start_proc_c(pm)
    bases = [str((work / b).resolve()) for b in pm.get_target_filenames()]
    store = RunStore(find_existing_store(work), mode="r")
    metric, n, _ = _read_dumbbell_targets_metric(
        target_bases=bases, first_frame=first, last_frame=last, step=1, num_cams=4, cpar=cpar, store=store
    )
    per = metric.reshape(4, n, 2, 2)
    lengths = []
    for f in range(n):
        flat = _to_flat(per[:, f], cals)
        a, _ = multi_cam_point_positions(flat[:, 0, :][np.newaxis], cpar, cals)
        b, _ = multi_cam_point_positions(flat[:, 1, :][np.newaxis], cpar, cals)
        lengths.append(float(np.linalg.norm(a[0] - b[0])))
    return float(np.median(lengths))


def run(workdir: Path, frames: int = 40, noise: float = 0.1, length: float = 46.0,
        claimed_length: float | None = None, seed: int = 1, quiet: bool = False,
        maxiter: int = 600) -> dict:
    import yaml

    from openptv2.gui.dumbbell_ground_truth import (
        DumbbellGTSpec,
        generate_dumbbell_target_files,
    )
    from openptv2.gui.standalone_dumbbell_calibration import run_dumbbell_calibration

    work = Path(workdir)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(ROOT / "test_data" / "test_cavity", work, ignore=shutil.ignore_patterns("res*", "tmp*.txt"))
    first, last = 10001, 10000 + frames
    yp = work / "parameters_Run1.yaml"
    d = yaml.safe_load(yp.read_text())
    d["sequence"]["first"], d["sequence"]["last"] = first, last
    d["dumbbell"].update(dumbbell_scale=float(claimed_length or length), dumbbell_eps=0.0, dumbbell_step=1)
    yp.write_text(yaml.safe_dump(d, sort_keys=False))

    truth = _read_cals(work)
    with contextlib.redirect_stdout(io.StringIO()):
        generate_dumbbell_target_files(
            yp, spec=DumbbellGTSpec(first=first, last=last, length=length, noise_sigma_px=noise, seed=seed,
                                    max_tries_per_frame=5000)
        )

    # spoil cameras 2-4 (camera 1 is the reference and stays)
    rng = np.random.default_rng(seed + 100)
    for i in (1, 2, 3):
        c = truth[i]
        new = _read_cals(work)[i]
        new.set_pos(c.get_pos() + rng.normal(0, 1.5, 3))
        new.set_angles(c.get_angles() + rng.normal(0, 0.004, 3))
        p = work / "cal" / f"cam{i + 1}.tif.ori"
        new.write(str(p).encode(), str(p).replace(".ori", ".addpar").encode())

    def errors(cals):
        pos = [float(np.linalg.norm(cals[i].get_pos() - truth[i].get_pos())) for i in (1, 2, 3)]
        ang = [float(np.degrees(np.linalg.norm(cals[i].get_angles() - truth[i].get_angles()))) for i in (1, 2, 3)]
        return pos, ang

    before = _read_cals(work)
    pos0, ang0 = errors(before)
    len0 = _bar_length(work, first, last, before)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = run_dumbbell_calibration(yp, fixed_cams=[0], maxiter=maxiter, write=True)
    after = _read_cals(work)
    pos1, ang1 = errors(after)
    len1 = _bar_length(work, first, last, after)
    out = dict(frames_used=res.n_frames_used, frames_total=res.n_frames_total, pos_before=pos0, pos_after=pos1,
               ang_before=ang0, ang_after=ang1, len_before=len0, len_after=len1, true_length=length,
               claimed_length=float(claimed_length or length), fun_initial=res.fun_initial, fun_final=res.fun_final)
    if not quiet:
        print(f"frames used {res.n_frames_used} of {res.n_frames_total}; true bar {length} mm, told to the program {out['claimed_length']} mm")
        print("camera   position error (mm)   angle error (deg)")
        for k, i in enumerate((2, 3, 4)):
            print(f"  {i}      {pos0[k]:5.2f} -> {pos1[k]:5.2f}        {ang0[k]:5.3f} -> {ang1[k]:5.3f}")
        print(f"bar length measured with the cameras: {len0:.2f} -> {len1:.2f} mm")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--length", type=float, default=46.0)
    ap.add_argument("--claimed-length", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run(Path(a.workdir), a.frames, a.noise, a.length, a.claimed_length, a.seed)
