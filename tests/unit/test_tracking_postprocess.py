"""Tests for disk-level tracking post-processes (FB-consistency + cold-start)."""

import numpy as np
import pytest

from openptv2.tracking_postprocess import (
    count_links,
    enforce_reciprocity,
    link_step,
    read_linkage,
    relink_trajectory_gaps,
    seed_cold_start,
    write_linkage,
)


def _write(base, frame, prev, nxt, xyz):
    write_linkage(base, frame, np.array(prev), np.array(nxt), np.array(xyz, float))


def test_write_linkage_with_store_skips_ascii(tmp_path):
    """Regression test: when a store is given, write_linkage must not also
    try to write the ASCII file -- linkage_base can be a store-only scratch
    namespace (e.g. warmup's "warmup/cycle1") with no real on-disk directory,
    so an unconditional ASCII write raises FileNotFoundError. See
    tracking_warmup._forward_backward_agreement, which hit exactly this."""
    from openptv2.storage import RunStore

    store = RunStore(tmp_path / "run.zarr", mode="w")
    base = "warmup/cycle1"  # deliberately not a real directory on disk

    write_linkage(base, 1, np.array([-1]), np.array([-2]), np.array([[0.0, 0.0, 0.0]]), store=store)

    prev, nxt, xyz = read_linkage(base, 1, store=store)
    assert list(prev) == [-1]
    assert list(nxt) == [-2]


def test_enforce_reciprocity_noop_on_symmetric_links(tmp_path):
    base = str(tmp_path / "ptv_is")
    # frame0: i0->j0, i1->j1 ; frame1: j0.prev=i0, j1.prev=i1  (fully reciprocal)
    _write(base, 0, [-1, -1], [0, 1], [[0, 0, 0], [1, 0, 0]])
    _write(base, 1, [0, 1], [-2, -2], [[0, 1, 0], [1, 1, 0]])
    stats = enforce_reciprocity(base, 0, 1)
    assert stats == {"severed_next": 0, "severed_prev": 0}
    assert list(read_linkage(base, 0)[1]) == [0, 1]  # next unchanged


def test_enforce_reciprocity_severs_one_sided_links(tmp_path):
    base = str(tmp_path / "ptv_is")
    # frame0 next: i0->j0, i1->j1
    _write(base, 0, [-1, -1], [0, 1], [[0, 0, 0], [1, 0, 0]])
    # frame1 prev: BOTH claim i0 -> j1's back-link (0) disagrees with next0[1]=1
    _write(base, 1, [0, 0], [-2, -2], [[0, 1, 0], [1, 1, 0]])
    stats = enforce_reciprocity(base, 0, 1)
    # forward i1->j1 severed (j1.prev != 1); backward j1.prev=0 severed (n0[0]!=1)
    assert stats["severed_next"] == 1
    assert stats["severed_prev"] == 1
    prev1, next1, _ = read_linkage(base, 1)
    next0 = read_linkage(base, 0)[1]
    assert next0[0] == 0 and next0[1] == -2  # only the reciprocal link survives
    assert prev1[0] == 0 and prev1[1] == -1


def test_seed_cold_start_recovers_velocity_consistent_link(tmp_path):
    base = str(tmp_path / "ptv_is")
    # frame2: particle m at x=10 ; frame1: particle j at x=5 linked forward to m
    # (velocity v=+5), but NOT linked back; frame0 has a free particle at x=0
    # exactly where j came from (5 - 5).
    _write(base, 0, [-1], [-2], [[0, 0, 0]])
    _write(base, 1, [-1], [0], [[5, 0, 0]])
    _write(base, 2, [0], [-2], [[10, 0, 0]])
    before = count_links(base, 0, 2)
    stats = seed_cold_start(base, 0, 2, dv_max=15.5)
    after = count_links(base, 0, 2)
    assert stats["added"] == 1
    assert after == before + 1
    # bidirectional: frame0 particle now points to j, frame1 j points back
    assert read_linkage(base, 0)[1][0] == 0
    assert read_linkage(base, 1)[0][0] == 0


def test_seed_cold_start_rejects_out_of_tolerance(tmp_path):
    base = str(tmp_path / "ptv_is")
    # free frame0 particle is far from the predicted origin -> no link
    _write(base, 0, [-1], [-2], [[100, 0, 0]])
    _write(base, 1, [-1], [0], [[5, 0, 0]])
    _write(base, 2, [0], [-2], [[10, 0, 0]])
    stats = seed_cold_start(base, 0, 2, dv_max=15.5)
    assert stats["added"] == 0


def test_relink_trajectory_gaps_bridges_missing_frame(tmp_path):
    base = str(tmp_path / "ptv_is")
    # Frame 0: x=0, link to frame 1
    # Frame 1: x=2, end of track (next=-2)
    # Frame 2: missing detection (gap=1)
    # Frame 3: x=6, start of track (prev=-1, next=0) -> pred pos at frame 3 is 2 + 2*2 = 6!
    _write(base, 0, [-1], [0], [[0, 0, 0]])
    _write(base, 1, [0], [-2], [[2, 0, 0]])
    _write(base, 2, [], [], np.zeros((0, 3)))  # empty frame
    _write(base, 3, [-1], [0], [[6, 0, 0]])
    _write(base, 4, [0], [-2], [[8, 0, 0]])

    stats = relink_trajectory_gaps(
        base, first=0, last=4, max_gap=2, max_velocity_err=1.0
    )
    assert stats["bridged_gaps"] == 1

    # A bridged gap is a single cross-frame link (frame 1 -> frame 3), not a
    # fabricated measurement at the skipped frame: no point is invented where
    # the particle was never observed, and it matches how ground truth
    # represents a gap (a link of step > 1). Consumers recover the step via
    # link_step()/back_link_step().
    _prev1, next1, _ = read_linkage(base, 1)
    prev3, _next3, _ = read_linkage(base, 3)

    assert next1[0] == 0  # points straight into frame 3
    assert prev3[0] == 0
    assert read_linkage(base, 2) is None  # skipped frame left empty

    frames = {k: read_linkage(base, k) for k in range(5)}
    frames = {k: v for k, v in frames.items() if v is not None}
    assert link_step(lambda m: frames[m][0] if m in frames else None, 1, 0, 0) == 2


def test_enforce_reciprocity_keeps_gap_bridged_cross_frame_link(tmp_path):
    """A bridge written by relink_trajectory_gaps (next pointing 2 frames
    ahead) must survive reciprocity -- it used to be severed right back out."""
    base = str(tmp_path / "ptv_is")
    _write(base, 0, [-1], [0], [[0, 0, 0]])
    _write(base, 1, [0], [-2], [[2, 0, 0]])
    _write(base, 2, [], [], np.zeros((0, 3)))
    _write(base, 3, [-1], [0], [[6, 0, 0]])
    _write(base, 4, [0], [-2], [[8, 0, 0]])

    assert relink_trajectory_gaps(
        base, first=0, last=4, max_gap=2, max_velocity_err=1.0
    )["bridged_gaps"] == 1
    stats = enforce_reciprocity(base, 0, 4)

    assert stats == {"severed_next": 0, "severed_prev": 0}
    assert read_linkage(base, 1)[1][0] == 0  # bridge intact
    assert read_linkage(base, 3)[0][0] == 0
