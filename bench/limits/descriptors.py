"""Truth-free descriptors and the advice for run stores (the same code as ``openptv recommend``).

    uv run python bench/limits/descriptors.py <label>=<run.zarr> [...]
"""

from __future__ import annotations

import sys

from openptv2.storage import RunStore
from openptv2.tracking_advice import advise, describe_store


def main() -> None:
    for a in sys.argv[1:]:
        label, _, path = a.partition("=")
        d = describe_store(RunStore(path, mode="r"))
        adv = advise(d)
        print(f"{label[-55:]:55s} ppf {d.ppf:5.0f}  sp/st {d.spacing_over_step:5.1f}  mutual {d.mutual_rate:.2f}  "
              f"noise/st {d.kink_over_step:.2f}  partial {d.frac_partial:.2f}  flag {d.merge_flag_rate:.2f}  ->  "
              f"{adv.tracker}{'+corr' if adv.shared_correction else ''}{'+pass' if adv.shared_pass else ''}")


if __name__ == "__main__":
    main()
