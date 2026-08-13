"""Auto-Research optimization engine (Phase 4 of
docs/plans/differentiable_ptv_nextgen_plan.md).

Sobol global sensitivity analysis for causal attribution between Stage-1..4
micro-parameters and the Stage-5 Lagrangian physics loss, plus an agent that
tunes those parameters via gradient descent (the differentiable PyTorch
runtime) or black-box optimization (the legacy C/Cython runtime).

Modules
-------
sensitivity
    Sobol first/total-order variance decomposition (Saltelli sampling).
agent
    ``AutoResearchAgent`` -- sensitivity, gradient (Adam), and black-box
    (differential evolution) optimization.

Backs the ``openptv2-autotune`` CLI (:mod:`openptv2.autoresearch.cli_autotune`).
"""

from openptv2.autoresearch.agent import AutoResearchAgent, OptimizationResult
from openptv2.autoresearch.sensitivity import saltelli_sample, sobol_indices

__all__ = [
    "AutoResearchAgent",
    "OptimizationResult",
    "saltelli_sample",
    "sobol_indices",
]
