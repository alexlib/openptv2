"""Post-PTV clean chain for one Illmenau rig.

reconnect (openptv2.reconnect) -> the flowtracks cleaning chain
(flowtracks.clean: repair -> smooth -> kink split -> smooth -> stitch ->
smooth -> kink split -> smooth -> tight stitch -> smooth), on the tracker
linkage of RIGDIR/res/<store>, saved as a flowtracks trajectory store.

Every setting comes from the `postptv_clean:` section of the rig's parameter
file (RIGDIR/parameters_Run1.yaml, or --params); a command-line flag overrides
it; a setting missing from both takes the default (reconnect: below; the
rest: flowtracks.clean.CLEAN_DEFAULTS). fps has no default. The effective
values and their source are printed at the start.

    postptv_clean:
      fps: 10.0                 # frame rate [Hz]; velocities in m/s
      reconnect_max_gap: 10     # reconnect: max frame gap
      reconnect_tol_sigmas: 4.0 # reconnect: tolerance in sigmas
      repair_max_gap: 1         # repair: attach / join across this many frames,
      repair_max_distance_mm: 0.3  #   within this distance
      repair_link_percentile: 99.9
      repair_cut_factor: 2.0
      repair_cut: true
      repair_attach: true
      repair_join: true
      sg_window: 7              # Savitzky-Golay window / order
      sg_order: 2
      kink_deg: 90.0            # kink split: turn angle ...
      kink_step_mm: 25.0        # ... on steps longer than this
      gap1: 12                  # first stitch: max gap [frames],
      dist1: 0.05               #   max distance [m],
      vd1: 0.5                  #   max velocity difference [m/s]
      gap2: 8                   # second (tight) stitch on clean ends
      dist2: 0.025
      vd2: 0.35
      win_long: 0               # >0: final smoothing (win_long, order_long)
      order_long: 3             #   for tracks with >= len_long points
      len_long: 25

Usage (flowtracks >= 1.7.0, postptv):
    uv run python \
        scripts/illmenau/build_postptv_clean.py RIGDIR [--first F] [--last L] \
        [--store run_fulldiam.zarr] [--out run_fulldiam_clean.zarr] [--fps 10 ...]
"""

import argparse
import time
from pathlib import Path

import yaml
from flowtracks.clean import CLEAN_DEFAULTS, clean_trajectories, resolve_settings
from flowtracks.io import save_zarr_trajectories

from openptv2.reconnect import reconnect_pieces
from openptv2.storage.linkage_arrays import read_linkage_arrays

RECONNECT_DEFAULTS = dict(reconnect_max_gap=10, reconnect_tol_sigmas=4.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rigdir")
    ap.add_argument("--params", default=None,
                    help="parameter file (default RIGDIR/parameters_Run1.yaml)")
    ap.add_argument("--first", type=int, default=10001)
    ap.add_argument("--last", type=int, default=10500)
    ap.add_argument("--store", default="run_fulldiam.zarr",
                    help="run store filename inside rigdir/res")
    ap.add_argument("--out", default="run_fulldiam_clean.zarr")
    keys = {**RECONNECT_DEFAULTS, **CLEAN_DEFAULTS}
    for k, v in keys.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k,
                        type=float if v is None else type(v), default=None,
                        help=f"override the parameter file (default {v})")
    args = ap.parse_args()
    cli = vars(args)

    path = Path(args.params) if args.params else Path(args.rigdir) / "parameters_Run1.yaml"
    section = dict((yaml.safe_load(path.read_text()) or {}).get("postptv_clean") or {})
    rec = {}
    rec_src = {}
    for k, v in RECONNECT_DEFAULTS.items():
        if cli.get(k) is not None:
            rec[k], rec_src[k] = cli[k], "override"
        elif k in section:
            rec[k], rec_src[k] = type(v)(section.pop(k)), path.name
        else:
            rec[k], rec_src[k] = v, "default"
    try:
        c, src = resolve_settings(section, {k: cli.get(k) for k in CLEAN_DEFAULTS},
                                  source_name=path.name)
    except ValueError as exc:
        raise SystemExit(f"{path}: postptv_clean: {exc}") from None
    print("postptv_clean settings:", flush=True)
    for k in keys:
        v, s = (rec[k], rec_src[k]) if k in rec else (c[k], src[k])
        print(f"  {k:24s} {v!s:>8s}   ({s})", flush=True)

    t0 = time.perf_counter()
    res = Path(args.rigdir) / "res"
    trajid0, time0, pos0 = read_linkage_arrays(res / args.store, first=args.first,
                                               last=args.last)
    print(f"linkage rows: {len(trajid0)}", flush=True)
    tj = reconnect_pieces(trajid0, time0, pos0, max_gap=rec["reconnect_max_gap"],
                          tol_sigmas=rec["reconnect_tol_sigmas"])
    trajs = clean_trajectories(tj, time0, pos0, c,
                               log=lambda s: print(s, flush=True))
    save_zarr_trajectories(trajs, str(res / args.out), overwrite=True)
    print(f"saved {args.out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
