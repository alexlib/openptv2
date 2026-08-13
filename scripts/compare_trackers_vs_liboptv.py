"""Comprehensive forward-mode benchmark: openptv2's 5 canonical trackers vs
ground truth, and vs the original liboptv C tracker (the ``optv`` package --
https://github.com/alexlib/openptv, the C library this project was
translated from).

Two independent questions, both scored per-tracker:

  1. Correctness vs ground truth -- identity metrics (purity/pmt/
     fragmentation) and link-level metrics (precision/yield_recall), using
     EACH TRACKER'S OWN recommended kinematic-bound parameters (see
     benchmark_utils.per_tracker_overrides) rather than one shared override
     dict applied to every engine -- a tracker whose parameters are
     borrowed from a different algorithm is not a fair test of that
     tracker.
  2. Trajectory-by-trajectory agreement vs liboptv -- the same link-level
     metrics, but with liboptv's own forward-mode output as the reference
     trajectory set instead of ground truth. priority_segment_3d and
     trackcorr are direct translations of liboptv algorithms, so this is a
     near-parity check for them; kalman_hungarian_3d/nearest_hungarian_3d/
     predictive_gmm_3d are intentionally different algorithms, so disagreement
     there is expected and diagnostic, not a bug.

Forward mode only (no backward pass / postprocessing on either side), per
the redesigned Tracking Parameters GUI's Tracker x Direction split.

Usage:
    uv run python scripts/compare_trackers_vs_liboptv.py
    uv run python scripts/compare_trackers_vs_liboptv.py --src test_data/burgers --first 10001 --n-frames 4
    uv run python scripts/compare_trackers_vs_liboptv.py --report scratch/tracker_report.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu  # noqa: E402

from openptv2.tracking_metrics import calculate_tracking_metrics  # noqa: E402

_WORKER = Path(__file__).parent / "_tracker_run_worker.py"


def _run_via_subprocess(
    tracker: str, src: Path, first: int, n_frames: int, overrides: dict | None,
) -> tuple[dict, float]:
    """Run one tracker (openptv2 name, or "liboptv:fast3d"/"liboptv:trackcorr")
    in its own process -- see _tracker_run_worker.py's docstring for why:
    openptv2's own Cython extensions and the optv C bindings corrupt each
    other's memory when run back-to-back in one process."""
    spec = {
        "tracker": tracker, "src": str(src), "first": first,
        "n_frames": n_frames, "overrides": overrides,
    }
    work = Path(tempfile.mkdtemp())
    spec_path, out_path = work / "spec.json", work / "result.json"
    spec_path.write_text(json.dumps(spec))
    proc = subprocess.run(
        [sys.executable, str(_WORKER), str(spec_path), str(out_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"{tracker} worker failed (exit {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    payload = json.loads(out_path.read_text())
    tracks = {int(k): [tuple(pt) for pt in v] for k, v in payload["tracks"].items()}
    return tracks, payload["time_s"]

# Which liboptv forward-mode engine each openptv2 tracker is checked
# against. priority_segment_3d/kalman/myptv/proptv are all 3D-only linkers
# over already-triangulated rt_is.# points (liboptv's fast3d counterpart);
# trackcorr is the multi-camera 2D+3D epipolar engine (liboptv's own
# trackcorr, same input target files).
LIBOPTV_REFERENCE = {
    "priority_segment_3d": "fast3d",
    "trackcorr": "trackcorr",
    "kalman_hungarian_3d": "fast3d",
    "nearest_hungarian_3d": "fast3d",
    "predictive_gmm_3d": "fast3d",
}

TRACKERS = list(LIBOPTV_REFERENCE)


def run_comparison(src: Path, first: int, n_frames: int) -> list[dict]:
    if not bu.has_liboptv():
        raise SystemExit(
            "optv (liboptv C/Cython bindings, https://github.com/alexlib/openptv) "
            "is not installed -- cannot compare against the reference tracker. "
            "uv sync should have pulled it in (pyproject.toml: optv>=0.3.2)."
        )

    overrides = bu.per_tracker_overrides(TRACKERS, src=src, first=first, n_frames=n_frames)

    frames = bu.read_gt_frames(src, first, n_frames)
    tt = bu.build_true_tracks(frames, first)
    ghosts = bu.build_ghost_frames(frames, first)

    liboptv_cache: dict[str, dict] = {}
    rows = []
    for tr in TRACKERS:
        ref_mode = LIBOPTV_REFERENCE[tr]
        if ref_mode not in liboptv_cache:
            try:
                ref_tracks, ref_dt = _run_via_subprocess(
                    f"liboptv:{ref_mode}", src, first, n_frames, overrides[tr],
                )
                liboptv_cache[ref_mode] = {
                    "tracks": ref_tracks, "time_s": ref_dt, "src": src, "n_frames": n_frames,
                }
            except Exception as e:
                if ref_mode == "trackcorr":
                    # liboptv's compiled full_forward() crashes on this
                    # dataset above trivial density/frame-count (see
                    # benchmark_utils.LIBOPTV_TRACKCORR_MAX_PARTICLES's
                    # docstring for the verified, non-formulaic root cause)
                    # -- retry at an empirically-safe operating point rather
                    # than give up on the comparison entirely.
                    capped_frames = min(n_frames, bu.LIBOPTV_TRACKCORR_MAX_FRAMES)
                    try:
                        capped_src, _ = bu.make_density_capped_copy(
                            src, bu.LIBOPTV_TRACKCORR_MAX_PARTICLES,
                            first, first + capped_frames - 1,
                        )
                        ref_tracks, ref_dt = _run_via_subprocess(
                            f"liboptv:{ref_mode}", capped_src, first, capped_frames, overrides[tr],
                        )
                        liboptv_cache[ref_mode] = {
                            "tracks": ref_tracks, "time_s": ref_dt, "src": capped_src,
                            "n_frames": capped_frames,
                            "capped_at": bu.LIBOPTV_TRACKCORR_MAX_PARTICLES,
                        }
                    except Exception as e2:
                        liboptv_cache[ref_mode] = {"error": str(e2)}
                else:
                    liboptv_cache[ref_mode] = {"error": str(e)}
        ref = liboptv_cache[ref_mode]

        row = {"tracker": tr, "liboptv_ref": ref_mode, "params": overrides[tr]}
        if "error" in ref:
            row["liboptv_error"] = ref["error"]
        if "capped_at" in ref:
            row["liboptv_capped_at"] = ref["capped_at"]
        try:
            pred, dt = _run_via_subprocess(tr, src, first, n_frames, overrides[tr])
            row["time_s"] = dt
            row["n_tracks"] = len(pred)

            gt_row = bu.combined_metrics(tt, pred, eps=1.0, ghosts=ghosts)
            for k, v in gt_row.items():
                row[f"gt_{k}"] = v

            if "error" not in ref:
                # ref may be on a density-capped copy of src (see above) --
                # run this tracker there too, so the link-agreement
                # comparison is apples-to-apples on identical input.
                cmp_pred = pred if ref["src"] == src else _run_via_subprocess(
                    tr, ref["src"], first, ref["n_frames"], overrides[tr],
                )[0]
                vs_ref_link = calculate_tracking_metrics(ref["tracks"], cmp_pred, distance_tolerance=1.0)
                row["vs_liboptv_precision"] = vs_ref_link.precision
                row["vs_liboptv_yield_recall"] = vs_ref_link.yield_recall
                row["vs_liboptv_false_connection_rate"] = vs_ref_link.false_connection_rate
        except Exception as e:  # keep going for the other trackers
            row["error"] = str(e)
        rows.append(row)

    # liboptv itself as a row, run with the SAME parameters as our own
    # trackcorr (its direct counterpart) -- reuses the already-cached run,
    # no extra liboptv invocation. GT metrics are computed on whatever
    # src/n_frames that run actually used (the full dataset, or the
    # density-capped fallback -- see the "trackcorr" row's liboptv_ref
    # entry above for which applied).
    trackcorr_ref = liboptv_cache.get("trackcorr")
    if trackcorr_ref is not None and "error" not in trackcorr_ref:
        optv_row = {
            "tracker": "optv (liboptv, trackcorr params)",
            "liboptv_ref": "trackcorr",
            "params": overrides["trackcorr"],
            "time_s": trackcorr_ref["time_s"],
            "n_tracks": len(trackcorr_ref["tracks"]),
        }
        if trackcorr_ref["src"] == src:
            gt_tt, gt_ghosts = tt, ghosts
        else:
            ref_frames = bu.read_gt_frames(trackcorr_ref["src"], first, trackcorr_ref["n_frames"])
            gt_tt = bu.build_true_tracks(ref_frames, first)
            gt_ghosts = bu.build_ghost_frames(ref_frames, first)
            cap = trackcorr_ref.get("capped_at")
            optv_row["tracker"] += (
                f" (capped <= {cap}/frame, {trackcorr_ref['n_frames']} frames)"
            )
        gt_row = bu.combined_metrics(gt_tt, trackcorr_ref["tracks"], eps=1.0, ghosts=gt_ghosts)
        for k, v in gt_row.items():
            optv_row[f"gt_{k}"] = v
        rows.append(optv_row)

    return rows


def format_report(rows: list[dict], src: Path, first: int, n_frames: int) -> str:
    lines = [
        "# Tracker correctness report: openptv2 vs liboptv (forward mode)",
        "",
        f"Dataset: `{src}`  frames {first}..{first + n_frames - 1}",
        "",
        "liboptv = the original C tracker (https://github.com/alexlib/openptv) "
        "this project was translated from, run via the `optv` Cython bindings.",
        "",
        "## vs ground truth (each tracker's own recommended parameters)",
        "",
        "| Tracker | liboptv ref | pmt% | purity | fragmentation | precision | yield_recall | ghost_capture | time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['tracker']} | {row['liboptv_ref']} | ERROR: {row['error']} | | | | | | |")
            continue
        lines.append(
            f"| {row['tracker']} | {row['liboptv_ref']} | "
            f"{row['gt_pmt']:.1f} | {row['gt_purity']:.3f} | {row['gt_fragmentation']:.2f} | "
            f"{row['gt_precision']:.3f} | {row['gt_yield_recall']:.3f} | "
            f"{row['gt_ghost_capture_rate']:.3f} | {row['time_s']:.2f} |"
        )

    lines += [
        "",
        "## vs liboptv (trajectory-by-trajectory link agreement, same input rt_is.#)",
        "",
        "precision = fraction of this tracker's links that liboptv also made; "
        "yield_recall = fraction of liboptv's links this tracker reproduced. "
        "Near 1.0 is expected for priority_segment_3d/trackcorr (same "
        "algorithm); lower values for kalman/myptv/proptv reflect a genuinely "
        "different algorithm, not a bug. trackcorr's row runs on a "
        f"density-capped subset (<= {bu.LIBOPTV_TRACKCORR_MAX_PARTICLES} "
        "particles/frame) when the full dataset exceeds it -- see "
        "benchmark_utils.LIBOPTV_TRACKCORR_MAX_PARTICLES's docstring for why. "
        "That subset still uses the full dataset's recommended search-cone "
        "parameters, which are oversized for it (few particles left, still "
        "the same absolute mm bounds) -- expect near-zero agreement there, "
        "not because the engines disagree, but because the search cone no "
        "longer discriminates between candidates on such a sparse subset.",
        "",
        "| Tracker | liboptv ref | agree-precision | agree-yield_recall | false_connection_rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if "error" in row or row["tracker"].startswith("optv "):
            continue
        if "vs_liboptv_precision" not in row:
            err = " ".join(row.get("liboptv_error", "unknown error").split())
            lines.append(
                f"| {row['tracker']} | {row['liboptv_ref']} | "
                f"liboptv reference unavailable: {err[:120]} | | |"
            )
            continue
        tracker_label = row["tracker"]
        if "liboptv_capped_at" in row:
            tracker_label += f" (capped <= {row['liboptv_capped_at']}/frame)"
        lines.append(
            f"| {tracker_label} | {row['liboptv_ref']} | "
            f"{row['vs_liboptv_precision']:.3f} | {row['vs_liboptv_yield_recall']:.3f} | "
            f"{row['vs_liboptv_false_connection_rate']:.3f} |"
        )

    lines += ["", "## Per-tracker parameters used", ""]
    for row in rows:
        lines.append(f"- **{row['tracker']}**: `{row['params']}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=bu.SRC)
    ap.add_argument("--first", type=int, default=bu.FIRST)
    ap.add_argument("--n-frames", type=int, default=bu.N_FRAMES)
    ap.add_argument("--report", type=Path, default=None, help="write the markdown report to this path")
    args = ap.parse_args()

    rows = run_comparison(args.src, args.first, args.n_frames)
    report = format_report(rows, args.src, args.first, args.n_frames)
    print(report)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
