#!/usr/bin/env python3
"""Compare timing of a pure-Python function against its compiled Cython
counterpart, on identical inputs.

This assumes you have:
  - the original .py module, importable under its normal name (or a copy
    renamed so it doesn't collide with the compiled extension, e.g.
    mymodule_py.py vs the compiled mymodule.so)
  - the compiled extension already built (see build_template.py)

Usage:
    python benchmark.py \\
        --py-module mymodule_py --compiled-module mymodule \\
        --func process --setup "import numpy as np; arg = np.random.rand(1000, 1000)" \\
        --call "process(arg)" \\
        --number 20 --repeat 5

--setup runs once per timeit repeat, before the timed calls, and should
define any variables your --call expression references (it does NOT need to
import the module under test — that's done automatically).

If --compiled-module is omitted, only the plain-Python timing is reported
(useful before you've built the extension yet, as a baseline).
"""

import argparse
import os
import sys
import timeit

# Running this script as `python /path/to/benchmark.py` puts the script's own
# directory on sys.path[0], not the current directory — so without this, the
# modules being benchmarked (which live in the caller's project directory)
# would not be importable.
sys.path.insert(0, os.getcwd())


def time_module(module_name, call_expr, setup_code, number, repeat):
    setup = f"import {module_name}\n{setup_code}\n"
    # Make the module's names available unqualified too, mirroring
    # `from module import *`-ish convenience for the --call expression,
    # while still keeping the qualified name available.
    globs = {}
    exec(setup, globs)
    mod = globs[module_name]
    for name in dir(mod):
        if not name.startswith("_"):
            globs.setdefault(name, getattr(mod, name))

    timer = timeit.Timer(call_expr, globals=globs)
    samples = timer.repeat(repeat=repeat, number=number)
    best = min(samples) / number
    return best, samples


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--py-module", required=True, help="Importable name of the plain-Python module"
    )
    parser.add_argument(
        "--compiled-module",
        default=None,
        help="Importable name of the compiled extension (optional)",
    )
    parser.add_argument(
        "--setup", default="", help="Setup code defining variables used in --call"
    )
    parser.add_argument(
        "--call", required=True, help="Expression to time, e.g. 'process(arg)'"
    )
    parser.add_argument(
        "--number", type=int, default=10, help="Calls per timing sample"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of timing samples (best is reported)",
    )
    args = parser.parse_args()

    print(f"Timing {args.py_module}.{args.call!r} ...")
    py_best, py_samples = time_module(
        args.py_module, args.call, args.setup, args.number, args.repeat
    )
    print(
        f"  plain Python: {py_best * 1e3:.4f} ms/call (best of {args.repeat} runs of {args.number} calls)"
    )

    if args.compiled_module:
        print(f"Timing {args.compiled_module}.{args.call!r} ...")
        c_best, c_samples = time_module(
            args.compiled_module, args.call, args.setup, args.number, args.repeat
        )
        print(
            f"  compiled:     {c_best * 1e3:.4f} ms/call (best of {args.repeat} runs of {args.number} calls)"
        )
        if c_best > 0:
            print(f"\nSpeedup: {py_best / c_best:.2f}x")
    else:
        print(
            "\n(no --compiled-module given — build the extension, then rerun with it to see the speedup)"
        )


if __name__ == "__main__":
    main()
