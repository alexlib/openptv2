"""Hybrid tracker: 3MA/4BE's cheap 3D-only linker, plus a back-projection
consistency filter as a cheap secondary pass -- not a new core algorithm,
a post-processing filter over an existing tracker's output.

Motivation (docs/plans/2026-08-17-lagrangian-accuracy-program.md, Phase 3):
3MA/4BE never touch 2D image space, so a ghost 3D point (a spurious
epipolar-tolerance match with weak real multi-camera support) is invisible
to them -- it looks like any other 3D point once triangulated. trackcorr
avoids this by checking image-space consistency during the two-hop search
itself, at the cost of running that search on every candidate.

This filter is cheaper: run the primary linker once (its full cost, paid
once), then for every point ALONG each resulting track, reproject it into
every camera using the SAME calibration the tracker used and check it
actually lands near a real detected 2D target within `tol_px`. A point
that fails in enough cameras had weak real support to begin with (the
ghost signature) -- split the track there rather than trust it.

Important scope note (found empirically, 2026-08-18's 4BE investigation):
this catches GHOST points (spurious 3D correspondences with no real
particle behind them) -- it does NOT catch identity swaps between two
REAL points (4BE's confirmed failure mode: frames 13-14 correctly follow
true track 42303, frame 15 jumps to true track 46936 -- both are real,
valid, well-supported 3D points, so back-projection passes for both).
Identity swaps need a different mechanism (tighter kinematic gating,
Hungarian conflict resolution, or trackcorr's own compound two-hop check).
This filter and that problem are complementary, not the same fix.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import benchmark_utils as bu  # noqa: E402


def apply_backprojection_filter(
    pred_tracks: dict,
    src: Path,
    first: int,
    tol_px: float = 3.0,
    min_cams: int = 3,
) -> dict:
    """Split every track at any point whose back-projection lands near a
    real 2D target in fewer than `min_cams` cameras. Returns a new
    {track_id: [(frame, x, y, z), ...]} dict -- fragments, not merges, so
    downstream kinematics/yield metrics from benchmark_utils work unchanged."""
    import yaml

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.parameters import ControlPar, MmNp
    from openptv2.algorithms.trafo import metric_to_pixel
    from openptv2.storage import RunStore

    yaml_data = yaml.safe_load((src / "parameters_Run1.yaml").read_text())
    ptv = yaml_data["ptv"]
    cpar = ControlPar(
        num_cams=4,
        imx=ptv["imx"],
        imy=ptv["imy"],
        pix_x=ptv["pix_x"],
        pix_y=ptv["pix_y"],
        mm=MmNp(
            n1=ptv["mmp_n1"], n2=[ptv["mmp_n2"]], n3=ptv["mmp_n3"], d=[ptv["mmp_d"]]
        ),
    )
    cals = [
        Calibration.from_file(
            str(src / "cal" / f"cam{c + 1}.tif.ori"),
            str(src / "cal" / f"cam{c + 1}.tif.addpar"),
        )
        for c in range(4)
    ]
    store = RunStore.open(src, mode="r")

    target_cache: dict[tuple[int, int], np.ndarray] = {}

    def targets_xy(cam: int, frame: int) -> np.ndarray:
        key = (cam, frame)
        if key not in target_cache:
            if store.has_targets(cam, frame):
                t = store.read_targets(cam, frame)
                target_cache[key] = (
                    np.array([[tt.x, tt.y] for tt in t]) if len(t) else np.zeros((0, 2))
                )
            else:
                target_cache[key] = np.zeros((0, 2))
        return target_cache[key]

    def supported(f_offset: int, x: float, y: float, z: float) -> bool:
        frame = first + f_offset
        ok = 0
        for c in range(4):
            txy = targets_xy(c, frame)
            if len(txy) == 0:
                continue
            mx, my = img_coord((x, y, z), cals[c], cpar.mm)
            px, py = metric_to_pixel(mx, my, cpar)
            if np.min(np.linalg.norm(txy - [px, py], axis=1)) < tol_px:
                ok += 1
        return ok >= min_cams

    out: dict = {}
    next_tid = 0
    for pts in pred_tracks.values():
        pts = sorted(pts)
        run: list = []
        for f, x, y, z in pts:
            if supported(f, x, y, z):
                run.append((f, x, y, z))
            else:
                if len(run) >= 1:
                    out[next_tid] = run
                    next_tid += 1
                run = []
        if run:
            out[next_tid] = run
            next_tid += 1
    return out


def main():
    SRC = Path("test_data/proptv_500_30")
    FIRST, N = 10001, 30
    from bench_proptv_kinematics import kinematics, stats

    frames = bu.read_gt_frames(SRC, FIRST, N)
    tt = bu.build_true_tracks(frames, FIRST)
    _v_t, a_t = kinematics(tt)
    a_rms_t, a_k_t = stats(a_t)
    print(f"truth: a_rms {a_rms_t:.5f}  K_a {a_k_t:.2f}\n")

    overrides = bu.per_tracker_overrides(
        ["priority_segment_3d"], src=SRC, first=FIRST, n_frames=N
    )
    pred0, dt = bu.run_single_tracker(
        "priority_segment_3d",
        track_overrides=overrides["priority_segment_3d"],
        src=SRC,
        first=FIRST,
    )

    for label, tracks in [
        ("primary (3MA, unfiltered)", pred0),
        (
            "hybrid (3MA + back-projection filter)",
            apply_backprojection_filter(pred0, SRC, FIRST),
        ),
    ]:
        m = bu.combined_metrics(tt, tracks, eps=1.0)
        v_p, a_p = kinematics(tracks)
        a_rms, a_k = stats(a_p)
        lens = np.array([len(v) for v in tracks.values()]) if tracks else np.zeros(1)
        outl = (
            100 * np.mean(np.abs(a_p - a_t.mean()) > 5 * a_rms_t)
            if a_p.size
            else float("nan")
        )
        print(f"{label}:")
        print(
            f"  n_tracks={len(tracks)}  meanlen={lens.mean():.2f}  "
            f"a_err={100 * (a_rms / a_rms_t - 1):+.1f}%  K_a={a_k:.2f}  >5sig={outl:.3f}%  "
            f"yield={m['yield_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
