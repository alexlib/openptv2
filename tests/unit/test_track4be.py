"""Kernel-level tests for the 4BE (four-frame best estimate) tracker.

Ouellette, Xu & Bodenschatz, Exp. Fluids 40:301-313 (2006), eqs. 10/12/14.
These pin the two things the tracker is: the eq. 12 estimate ``x~ = 2q - x1``
that defines its cost, and the give-up-on-conflict claiming rule.

The kernel is called directly with four hand-built frames, so a failure
points at the kernel rather than at the frame buffer or the plugin stack.
"""

import numpy as np
import pytest

from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE
from openptv2.algorithms.track_kernels import track4be_loop_fast

MAX_CANDS = 64
BIG = 100.0  # velocity window; wide enough not to gate these tiny scenes


def _frame(points, prev=None, nxt=None):
    """(path_x, path_prev, path_next, num_parts) for a list of xyz points."""
    x = np.ascontiguousarray(np.array(points, dtype=np.float64).reshape(-1, 3))
    n = len(x)
    p = (
        np.full(n, PREV_NONE, dtype=np.int32)
        if prev is None
        else np.array(prev, np.int32)
    )
    q = (
        np.full(n, NEXT_NONE, dtype=np.int32)
        if nxt is None
        else np.array(nxt, np.int32)
    )
    return x, p, q, n


def _run(f0, f1, f2, f3, dv=BIG, **kw):
    x0, p0, _n0, c0 = f0
    x1, p1, n1, c1 = f1
    x2, p2, n2, c2 = f2
    x3, _p3, _n3, c3 = f3
    links = track4be_loop_fast(
        c1,
        x0,
        p0,
        c0,
        x1,
        p1,
        n1,
        c1,
        x2,
        p2,
        n2,
        c2,
        x3,
        c3,
        dv,
        dv,
        dv,
        MAX_CANDS,
        **kw,
    )
    return links, n1, p2


def test_eq12_estimate_picks_the_candidate_that_predicts_frame_n2():
    """Eq. 12: a candidate q is scored by ||x^{n+2} - (2q - x1)||.

    Two candidates sit equally far from the eq. 10 search centre, so nearest
    neighbour cannot separate them. Only one of them extrapolates onto a real
    particle in frame n+2 -- 4BE must take that one. If the kernel scored on
    distance to the search centre instead, this test could not distinguish
    them and would fail roughly half the time.
    """
    # Track: x0=(0,0,0) -> x1=(1,0,0). Eq. 10 centre in n+1 is 2*x1-x0=(2,0,0).
    f0 = _frame([[0.0, 0.0, 0.0]])
    f1 = _frame([[1.0, 0.0, 0.0]], prev=[0])
    # Both candidates are exactly 0.5 from the centre (2,0,0).
    cand_a = [2.0, 0.5, 0.0]
    cand_b = [2.0, -0.5, 0.0]
    f2 = _frame([cand_a, cand_b])
    # Eq. 12 estimate for cand_b: 2*(2,-0.5,0) - (1,0,0) = (3,-1,0). Put a
    # real particle exactly there, and nothing near cand_a's estimate (3,1,0).
    f3 = _frame([[3.0, -1.0, 0.0]])

    links, next1, prev2 = _run(f0, f1, f2, f3)

    assert links == 1
    assert next1[0] == 1, "must link to cand_b, whose eq. 12 estimate is supported"
    assert prev2[1] == 0
    assert prev2[0] == PREV_NONE


def test_eq12_estimate_is_2q_minus_x1_not_a_velocity_extrapolation():
    """Pin the exact estimate formula: ``x~ = 2q - x1``.

    Eq. 12 extrapolates from the *candidate's own* implied velocity
    (``q + (q - x1)``), NOT from the velocity already established in frames
    n-1..n (``q + (x1 - x0)``). The two agree whenever motion is uniform, so
    this scene gives the candidates an implied velocity different from the
    incoming one, and places one real n+2 particle on each formula's
    prediction. Whichever candidate is chosen names the formula that ran.

    A one-candidate scene cannot do this: with a wide velocity window the
    wrong estimate still finds support and links anyway. This was verified by
    mutating the kernel to ``q + (x1 - x0)`` -- the earlier version of this
    test passed under the mutant; this one fails.
    """
    #   x0=(0,0,0) -> x1=(1,0,0), so the eq. 10 centre in n+1 is (2,0,0).
    f0 = _frame([[0.0, 0.0, 0.0]])
    f1 = _frame([[1.0, 0.0, 0.0]], prev=[0])
    #   Two candidates, equidistant (0.4) from that centre.
    #     A=(2, 0.4,0): 2q-x1 = (3, 0.8,0)   q+(x1-x0) = (3, 0.4,0)
    #     B=(2,-0.4,0): 2q-x1 = (3,-0.8,0)   q+(x1-x0) = (3,-0.4,0)
    f2 = _frame([[2.0, 0.4, 0.0], [2.0, -0.4, 0.0]])
    #   One n+2 particle on eq. 12's estimate for A, one on the WRONG
    #   formula's estimate for B:
    #     eq. 12   -> cost(A)=0.0, cost(B)=0.4  -> picks A
    #     q+(x1-x0)-> cost(A)=0.4, cost(B)=0.0  -> picks B
    f3 = _frame([[3.0, 0.8, 0.0], [3.0, -0.4, 0.0]])

    links, next1, prev2 = _run(f0, f1, f2, f3)

    assert links == 1
    assert next1[0] == 0, "eq. 12 is 2q - x1, not q + (x1 - x0)"
    assert prev2[0] == 0


def test_strict_support_rejects_an_unsupported_candidate():
    """With strict_support=1 a candidate whose eq. 12 estimate has no real
    particle near it is discarded outright (the paper's literal rule)."""
    f0 = _frame([[0.0, 0.0, 0.0]])
    f1 = _frame([[1.0, 0.0, 0.0]], prev=[0])
    f2 = _frame([[2.0, 0.0, 0.0]])
    f3 = _frame([[50.0, 50.0, 50.0]])  # nowhere near the estimate (3,0,0)

    strict, next1, _ = _run(f0, f1, f2, f3, dv=10.0, strict_support=1)
    assert strict == 0
    assert next1[0] == NEXT_NONE

    # The default (0) keeps it as a penalised 3MA fallback, so the link is made.
    loose, next1b, _ = _run(f0, f1, f2, f3, dv=10.0, strict_support=0)
    assert loose == 1
    assert next1b[0] == 0


def test_give_up_on_conflict_drops_both_claimants():
    """The paper's conflict rule: a frame n+1 particle claimed by more than
    one frame-n particle links to NONE of them. Both tracks end."""
    # Two tracks converging on the same single candidate.
    f0 = _frame([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    f1 = _frame([[1.0, 0.5, 0.0], [1.0, 1.5, 0.0]], prev=[0, 1])
    f2 = _frame([[2.0, 1.0, 0.0]])  # the single contested candidate
    f3 = _frame([[3.0, 1.0, 0.0]])

    links, next1, prev2 = _run(f0, f1, f2, f3)

    assert links == 0, "contested candidate must go to neither claimant"
    assert next1[0] == NEXT_NONE
    assert next1[1] == NEXT_NONE
    assert prev2[0] == PREV_NONE


def test_greedy_conflicts_awards_the_contested_candidate_to_lower_cost():
    """greedy_conflicts=1 switches to cost-ordered claiming instead: the
    better-scoring claimant wins rather than both losing."""
    f0 = _frame([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    f1 = _frame([[1.0, 0.5, 0.0], [1.0, 1.5, 0.0]], prev=[0, 1])
    f2 = _frame([[2.0, 1.0, 0.0]])
    # Eq. 12 estimate from the contested candidate is 2*(2,1,0) - x1:
    #   for particle 0 (x1=(1,0.5,0)) -> (3, 1.5, 0)
    #   for particle 1 (x1=(1,1.5,0)) -> (3, 0.5, 0)
    # Put the real n+2 particle exactly on particle 0's estimate, so particle
    # 0 has cost 0 and particle 1 has cost 1.
    f3 = _frame([[3.0, 1.5, 0.0]])

    links, next1, prev2 = _run(f0, f1, f2, f3, greedy_conflicts=1)

    assert links == 1
    assert next1[0] == 0, "lower-cost claimant wins the contested candidate"
    assert next1[1] == NEXT_NONE
    assert prev2[0] == 0


def test_greedy_conflicts_lets_the_loser_take_its_second_choice():
    """Unlike give-up-on-conflict, cost-ordered claiming is a full pass over
    every (particle, candidate) edge, so a particle that loses its first
    choice can still claim another candidate."""
    f0 = _frame([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    f1 = _frame([[1.0, 0.5, 0.0], [1.0, 1.5, 0.0]], prev=[0, 1])
    # Two candidates; both particles prefer index 0, but index 1 is reachable.
    f2 = _frame([[2.0, 1.0, 0.0], [2.0, 1.9, 0.0]])
    f3 = _frame([[3.0, 1.5, 0.0]])  # supports particle 0 on candidate 0

    links, next1, _prev2 = _run(f0, f1, f2, f3, greedy_conflicts=1)

    assert links == 2, "both particles link: one first choice, one second"
    assert next1[0] == 0
    assert next1[1] == 1

    # The default rule cannot do this -- particle 1 simply loses.
    links_paper, next1p, _ = _run(f0, f1, f2, f3)
    assert links_paper < 2
    assert next1p[1] == NEXT_NONE


def test_unseeded_particle_falls_back_to_nearest_neighbour():
    """The first two points of a track have no velocity, so the paper joins
    them by nearest neighbour."""
    f0 = _frame([])
    f1 = _frame([[0.0, 0.0, 0.0]])  # no prev link -> unseeded
    f2 = _frame([[9.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    f3 = _frame([])

    links, next1, _prev2 = _run(f0, f1, f2, f3)

    assert links == 1
    assert next1[0] == 1, "nearest candidate, not the first one in index order"


def test_missing_frame_n2_degrades_to_the_3ma_residual():
    """At the tail of a sequence frame n+2 is empty; seeded scoring falls
    back to the 3MA acceleration residual so the last steps still link."""
    f0 = _frame([[0.0, 0.0, 0.0]])
    f1 = _frame([[1.0, 0.0, 0.0]], prev=[0])
    # Constant velocity continues to (2,0,0); the decoy implies acceleration.
    f2 = _frame([[2.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    f3 = _frame([])  # no n+2 at all

    links, next1, _prev2 = _run(f0, f1, f2, f3)

    assert links == 1
    assert next1[0] == 0, "zero-acceleration candidate wins the 3MA residual"


def test_already_claimed_candidate_is_not_relinked():
    """A frame n+1 particle that already has a prev link (claimed by an
    earlier pass) must not be stolen."""
    f0 = _frame([[0.0, 0.0, 0.0]])
    f1 = _frame([[1.0, 0.0, 0.0]], prev=[0])
    f2 = _frame([[2.0, 0.0, 0.0]], prev=[7])  # already claimed
    f3 = _frame([[3.0, 0.0, 0.0]])

    links, next1, prev2 = _run(f0, f1, f2, f3)

    assert links == 0
    assert next1[0] == NEXT_NONE
    assert prev2[0] == 7, "existing claim untouched"


@pytest.mark.parametrize("greedy", [0, 1])
def test_empty_frames_are_safe(greedy):
    """Zero-particle frames must not index out of bounds in either mode."""
    empty = _frame([])
    links, _n, _p = _run(empty, empty, empty, empty, greedy_conflicts=greedy)
    assert links == 0
