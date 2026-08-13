"""``openptv2-autotune`` CLI (Phase 4): Sobol sensitivity, optimization, and
runtime-vs-accuracy Pareto plots over pipeline hyperparameters.

Subcommands
-----------
sensitivity
    Sobol first/total-order indices -- which parameters actually move the
    objective.
optimize
    Tune parameters via ``--method blackbox`` (scipy differential evolution,
    for the legacy runtime) or ``--method gradient`` (Adam, requires a
    ``--objective`` that returns a differentiable ``torch.Tensor``).
pareto
    Sample the parameter space, time each objective evaluation, and plot
    execution runtime vs. objective value (accuracy proxy), marking the
    non-dominated (Pareto-optimal) points.

Objectives are loaded via ``--objective module:function`` (a callable taking
a ``(k,)`` or ``(N, k)`` ndarray). ``--demo`` runs a built-in placeholder
bowl function so the CLI is runnable end-to-end before a real pipeline
objective is wired in.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time

import numpy as np

from openptv2.autoresearch.agent import AutoResearchAgent

DEFAULT_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]


def demo_objective(params: np.ndarray) -> np.ndarray:
    """Placeholder bowl-shaped objective (stand-in for the real physics loss).

    Accepts either a single ``(k,)`` point or a batch ``(N, k)``.
    """
    p = np.atleast_2d(params)
    x, y = p[:, 0], p[:, 1]
    val = (x - 3.0) ** 2 + (y + 1.0) ** 2 + 0.5 * np.sin(3 * x) * np.cos(3 * y)
    return val if params.ndim > 1 else val[0]


def _load_objective(spec: str):
    module_name, func_name = spec.split(":")
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def _resolve_objective(args) -> tuple:
    if args.demo or args.objective is None:
        return demo_objective, DEFAULT_BOUNDS
    bounds = json.loads(args.bounds) if args.bounds else DEFAULT_BOUNDS
    return _load_objective(args.objective), [tuple(b) for b in bounds]


def cmd_sensitivity(args) -> None:
    objective, bounds = _resolve_objective(args)
    agent = AutoResearchAgent()
    result = agent.sensitivity(objective, bounds, n_base=args.n_base, seed=args.seed)
    print(f"{'param':<10} | {'S1':>8} | {'ST':>8}")
    print("-" * 32)
    for i in range(len(bounds)):
        print(f"p{i:<9} | {result['S1'][i]:>8.4f} | {result['ST'][i]:>8.4f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({k: v.tolist() for k, v in result.items()}, fh, indent=2)
        print(f"wrote {args.json}")


def cmd_optimize(args) -> None:
    objective, bounds = _resolve_objective(args)
    agent = AutoResearchAgent()
    if args.method == "blackbox":
        result = agent.optimize_blackbox(objective, bounds, seed=args.seed, maxiter=args.n_iters)
    else:
        init = np.array([0.5 * (lo + hi) for lo, hi in bounds])
        result = agent.optimize_gradient(objective, init, n_steps=args.n_iters, lr=args.lr)
    print(f"best_params = {result.best_params.tolist()}")
    print(f"best_loss   = {result.best_loss}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {"best_params": result.best_params.tolist(), "best_loss": result.best_loss},
                fh,
                indent=2,
            )
        print(f"wrote {args.json}")


def cmd_pareto(args) -> None:
    objective, bounds = _resolve_objective(args)
    rng = np.random.default_rng(args.seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    samples = lo + rng.random((args.n_samples, len(bounds))) * (hi - lo)

    runtimes = np.empty(args.n_samples)
    losses = np.empty(args.n_samples)
    for i, p in enumerate(samples):
        t0 = time.perf_counter()
        losses[i] = float(objective(p))
        runtimes[i] = time.perf_counter() - t0

    # Non-dominated: no other point has runtime<= and loss<= with a strict
    # improvement in at least one.
    dominated = np.zeros(args.n_samples, dtype=bool)
    for i in range(args.n_samples):
        better_or_equal = (runtimes <= runtimes[i]) & (losses <= losses[i])
        strictly_better = (runtimes < runtimes[i]) | (losses < losses[i])
        dominated[i] = np.any(better_or_equal & strictly_better)
    pareto_mask = ~dominated

    print(f"{args.n_samples} samples, {pareto_mask.sum()} on the Pareto front")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(runtimes[~pareto_mask], losses[~pareto_mask], c="lightgray", label="dominated")
        order = np.argsort(runtimes[pareto_mask])
        ax.plot(
            runtimes[pareto_mask][order],
            losses[pareto_mask][order],
            "o-",
            c="crimson",
            label="Pareto front",
        )
        ax.set_xlabel("execution runtime [s]")
        ax.set_ylabel("objective (physics accuracy proxy)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openptv2-autotune")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--objective", help="module:function returning a scalar (or batched) loss")
    common.add_argument("--bounds", help='JSON [[lo, hi], ...], one pair per parameter')
    common.add_argument("--demo", action="store_true", help="use the built-in placeholder objective")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--json", help="write results to this JSON file")

    sub = parser.add_subparsers(dest="command", required=True)

    p_sens = sub.add_parser("sensitivity", parents=[common], help="Sobol sensitivity indices")
    p_sens.add_argument("--n-base", type=int, default=256)
    p_sens.set_defaults(func=cmd_sensitivity)

    p_opt = sub.add_parser("optimize", parents=[common], help="tune parameters")
    p_opt.add_argument("--method", choices=["blackbox", "gradient"], default="blackbox")
    p_opt.add_argument("--n-iters", type=int, default=100)
    p_opt.add_argument("--lr", type=float, default=0.05, help="gradient method learning rate")
    p_opt.set_defaults(func=cmd_optimize)

    p_pareto = sub.add_parser("pareto", parents=[common], help="runtime vs. accuracy Pareto plot")
    p_pareto.add_argument("--n-samples", type=int, default=50)
    p_pareto.add_argument("--plot", help="output image path (requires matplotlib)")
    p_pareto.set_defaults(func=cmd_pareto)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
