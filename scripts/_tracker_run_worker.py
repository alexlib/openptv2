"""Run exactly one tracker (an openptv2 tracker, or liboptv fast3d/trackcorr)
in its own process and print the resulting trajectories as JSON on stdout.

Isolation reason: openptv2's own compiled Cython extensions and the `optv`
package (liboptv's separately-compiled C/Cython bindings) corrupt each
other's memory when run back-to-back in one process (observed: a segfault,
and in milder cases a tracker's C-level state re-firing after it already
reported completion). One tracker per process sidesteps this entirely --
see scripts/compare_trackers_vs_liboptv.py, which is the only caller.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu  # noqa: E402


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    tracker = spec["tracker"]
    src = Path(spec["src"])
    first = spec["first"]
    n_frames = spec["n_frames"]
    overrides = spec.get("overrides") or None

    if tracker.startswith("liboptv:"):
        mode = tracker.split(":", 1)[1]
        pred, dt = bu.run_liboptv_tracker(
            mode=mode,
            track_overrides=overrides,
            src=src,
            first=first,
            n_frames=n_frames,
        )
    else:
        pred, dt = bu.run_single_tracker(
            tracker,
            track_overrides=overrides,
            src=src,
            first=first,
        )

    result = {
        "time_s": dt,
        "tracks": {str(k): v for k, v in pred.items()},
    }
    # File-based handoff, not stdout: some tracker backends print more
    # diagnostic output after their own "done" point (observed with the
    # optv/openptv2 in-process interaction this worker exists to isolate),
    # which would otherwise corrupt a stdout-marker-based parse.
    out_path.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
