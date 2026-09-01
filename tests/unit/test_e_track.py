"""Tests for Ouellette's track-level error metric.

The point of most of these is the coverage condition: the form this metric
was first written in (docs/plans/2026-08-16-tracking-next-steps.md §5)
checked only "all points map to one true particle" + "same start frame", so
a short fragment scored as a perfect reproduction. Several tests below fail
under that form and pass under the corrected one.
"""

import pytest

from openptv2.benchmarking.metrics import e_track


def _line(tid_frames, x0=0.0, step=1.0, y=0.0, z=0.0):
    """A straight track: [(frame, x, y, z)] over the given frames."""
    return [(f, x0 + step * f, y, z) for f in tid_frames]


EPS = 0.25


def test_exact_reproduction_is_perfect():
    true = {0: _line(range(10))}
    pred = {100: _line(range(10))}

    m = e_track(true, pred, eps=EPS)

    assert m.n_perfect == 1
    assert m.e_track == 0.0
    assert (m.n_fragmented, m.n_contaminated, m.n_incomplete, m.n_missed) == (
        0,
        0,
        0,
        0,
    )


def test_short_fragment_is_not_a_perfect_reproduction():
    """THE regression: a 2-point fragment starting on the right frame used to
    score as a perfectly reproduced 10-point trajectory."""
    true = {0: _line(range(10))}
    pred = {100: _line(range(2))}  # same start frame, same positions, 2 of 10 points

    m = e_track(true, pred, eps=EPS)

    assert m.n_perfect == 0
    assert m.e_track == 1.0
    assert m.n_incomplete == 1


def test_fragmenting_a_track_scores_worse_not_better():
    """A tracker that splits one true track into two must not be rewarded.
    Under the loose form both halves 'match only one true id', and the half
    starting at the true start frame made the track count as perfect."""
    true = {0: _line(range(10))}
    whole = {100: _line(range(10))}
    split = {100: _line(range(5)), 101: _line(range(5, 10))}

    m_whole = e_track(true, whole, eps=EPS)
    m_split = e_track(true, split, eps=EPS)

    assert m_whole.e_track == 0.0
    assert m_split.e_track == 1.0
    assert m_split.n_fragmented == 1


def test_missing_the_last_point_is_incomplete():
    true = {0: _line(range(10))}
    pred = {100: _line(range(9))}  # right start, one point short at the end

    m = e_track(true, pred, eps=EPS)

    assert m.n_perfect == 0
    assert m.n_incomplete == 1


def test_a_foreign_point_contaminates():
    """A predicted track holding a point from another particle is not a
    perfect reproduction of either."""
    true = {0: _line(range(10)), 1: _line(range(10), y=50.0)}
    # Takes particle 0 for 9 frames then jumps onto particle 1 at frame 9.
    pred = {100: _line(range(9)) + [(9, 9.0, 50.0, 0.0)]}

    m = e_track(true, pred, eps=EPS)

    assert m.n_perfect == 0
    assert m.n_contaminated + m.n_incomplete + m.n_missed == 2


def test_an_unmatched_point_contaminates():
    """A point matching no true particle at all (a ghost pickup) also breaks
    the reproduction, even though every *matched* point is consistent."""
    true = {0: _line(range(10))}
    pred = {100: _line(range(10)) + [(10, 999.0, 999.0, 999.0)]}

    m = e_track(true, pred, eps=EPS)

    assert m.n_perfect == 0
    assert m.n_contaminated == 1


def test_untracked_true_track_is_missed():
    true = {0: _line(range(10)), 1: _line(range(10), y=50.0)}
    pred = {100: _line(range(10))}

    m = e_track(true, pred, eps=EPS)

    assert m.n_perfect == 1
    assert m.n_missed == 1
    assert m.e_track == pytest.approx(0.5)


def test_a_gap_in_ground_truth_can_be_reproduced_perfectly():
    """Ground truth here contains detection gaps (91% of trajectories in
    synthetic_turbulent do). A true track is its OBSERVED frames, so a
    tracker that bridges the gap reproduces it exactly -- which is why
    e_track must be evaluated with gap bridging on."""
    frames = [0, 1, 2, 4, 5]  # frame 3 missing
    true = {0: _line(frames)}

    bridged = {100: _line(frames)}
    assert e_track(true, bridged, eps=EPS).e_track == 0.0

    # A tracker that cannot bridge splits at the gap and can never be perfect.
    unbridged = {100: _line([0, 1, 2]), 101: _line([4, 5])}
    m = e_track(true, unbridged, eps=EPS)
    assert m.e_track == 1.0
    assert m.n_fragmented == 1


def test_breakdown_partitions_the_failures():
    true = {
        0: _line(range(10)),  # perfect
        1: _line(range(10), y=50.0),  # missed
        2: _line(range(10), y=100.0),  # fragmented
        3: _line(range(10), y=150.0),  # incomplete
    }
    pred = {
        100: _line(range(10)),
        102: _line(range(5), y=100.0),
        103: _line(range(5, 10), y=100.0),
        104: _line(range(4), y=150.0),
    }

    m = e_track(true, pred, eps=EPS)

    assert m.n_true_tracks == 4
    assert m.n_perfect == 1
    assert (
        m.n_fragmented + m.n_contaminated + m.n_incomplete + m.n_missed
        == m.n_true_tracks - m.n_perfect
    )
    assert m.e_track == pytest.approx(0.75)


def test_empty_inputs():
    assert e_track({}, {}, eps=EPS).e_track == 1.0
    assert e_track({}, {}, eps=EPS).n_true_tracks == 0
    m = e_track({0: _line(range(5))}, {}, eps=EPS)
    assert m.e_track == 1.0 and m.n_missed == 1
