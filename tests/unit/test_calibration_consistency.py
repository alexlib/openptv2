"""Calibration consistency report, self-calibration from a run store, and the explicit partial-point alignment."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from openptv2.calibration_consistency import (
    align_partial_points,
    consistency_report,
    self_calibrate_from_store,
)
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore
from openptv2.synthetic_rig import TruthRig

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"
FRAMES = list(range(1, 9))


def _drifted(cals, scale=1.0):
    """The cameras as the run really saw them: shifted/turned a little since the calibration."""
    rng = np.random.default_rng(3)
    out = [copy.deepcopy(c) for c in cals]
    for c in out:
        c.set_pos(c.get_pos() + rng.normal(0, 0.08 * scale, 3))
        c.set_angles(c.get_angles() + rng.normal(0, 2e-4 * scale, 3))
        c.int_par.xh += rng.normal(0, 0.004 * scale)
        c.int_par.yh += rng.normal(0, 0.004 * scale)
    return out


def _store(path, truth_cals, model_cals, cpar, vpar, drop=0.3, n=500, seed=0):
    """A run store of a rig whose true cameras are ``truth_cals``; its 3D points were triangulated with ``model_cals``;
    a share ``drop`` of the points lacks one camera."""
    from openptv2.loco import pixels_to_flat
    from openptv2.orientation import point_positions

    rng = np.random.default_rng(seed)
    rig = TruthRig.from_cals(truth_cals, cpar)
    st = RunStore(str(path), mode="a")
    truth = {}
    for f in FRAMES:
        pts = rng.uniform(-3.5, 3.5, (n * 3, 3))
        px = rig.observe(pts, 0.05, rng)
        ok = np.isfinite(px).all(axis=(1, 2))
        pts, px = pts[ok][:n], px[ok][:n]
        ids = np.tile(np.arange(len(pts))[:, None], (1, 4)).astype(np.int64)
        miss = rng.random(len(pts)) < drop
        ids[miss, rng.integers(0, 4, miss.sum())] = -1
        for c in range(4):
            tg = np.zeros((len(pts), 8))
            tg[:, 0] = np.arange(len(pts))
            tg[:, 1:3] = px[:, c]
            tg[:, 3] = 9
            st.write_targets(c, f, tg)
        flat = np.stack([pixels_to_flat(px[:, c], model_cals[c], cpar) for c in range(4)], axis=1)
        flat[ids < 0] = -1e10
        pos = np.asarray(point_positions(flat, cpar, model_cals, vpar, method="lsq")[0], float)
        st.write_correspondences(f, pos, ids)
        truth[f] = pts
    st.truth = truth
    return st


@pytest.fixture(scope="module")
def rig():
    cpar, vpar, cals, _ = load_parameters(YAML)
    return cpar, vpar, list(cals)


def test_a_consistent_calibration_is_reported_consistent(rig, tmp_path):
    cpar, vpar, cals = rig
    st = _store(tmp_path / "a.zarr", cals, cals, cpar, vpar)
    rep = consistency_report(st, FRAMES, cals, cpar, vpar, min_points=300)
    assert rep.applicable and rep.verdict == "consistent", rep.text()
    assert max(rep.jump_um) < 15


def test_a_drifted_calibration_is_reported_and_self_calibration_repairs_it(rig, tmp_path):
    cpar, vpar, cals = rig
    drifted = _drifted(cals)
    # the run was recorded with ``drifted`` cameras but is processed with the old calibration ``cals``
    st = _store(tmp_path / "b.zarr", drifted, cals, cpar, vpar)
    rep = consistency_report(st, FRAMES, cals, cpar, vpar, min_points=300)
    assert rep.verdict != "consistent", rep.text()
    res = self_calibrate_from_store(st, FRAMES, cals, cpar, vpar, max_nfev=120, min_yield=0)
    assert res.accepted, res.text()
    assert max(res.after.jump_um) < 0.5 * max(res.before.jump_um)
    # inputs untouched
    assert cals[0].int_par.xh != res.cals[0].int_par.xh


def test_self_calibration_declines_when_there_is_nothing_to_gain(rig, tmp_path):
    cpar, vpar, cals = rig
    st = _store(tmp_path / "c.zarr", cals, cals, cpar, vpar)
    res = self_calibrate_from_store(st, FRAMES, cals, cpar, vpar, max_nfev=120)
    # even if the fit shrinks the (already tiny) jump further it must not be accepted: it would only move the scale
    assert not res.accepted and "already consistent" in res.message, res.text()


def test_fewer_than_four_cameras_is_not_applicable(rig, tmp_path):
    cpar, vpar, cals = rig
    st = _store(tmp_path / "d.zarr", cals, cals, cpar, vpar)
    rep = consistency_report(st, FRAMES, cals[:3], cpar, vpar)
    assert not rep.applicable and "at least four" in rep.message


def test_too_few_points_is_said_not_guessed(rig, tmp_path):
    cpar, vpar, cals = rig
    st = _store(tmp_path / "e.zarr", cals, cals, cpar, vpar, n=20)
    rep = consistency_report(st, FRAMES[:1], cals, cpar, vpar)
    assert not rep.applicable and "too few" in rep.message


def test_alignment_is_explicit_marked_guarded_and_repeatable_only_by_force(rig, tmp_path):
    cpar, vpar, cals = rig
    drifted = _drifted(cals, 2.0)
    # processed with the same cameras the points were triangulated with, but the real cameras differ: partial points jump
    st = _store(tmp_path / "f.zarr", drifted, cals, cpar, vpar, drop=0.5, n=700)
    def partial_error():
        err = []
        for f in FRAMES:
            pos, ids = st.read_correspondences(f)
            seen = (np.asarray(ids)[:, :4] >= 0).sum(1)
            m = seen == 3
            err.append(np.linalg.norm(np.asarray(pos)[m] - st.truth[f][m], axis=1))
        return float(np.median(np.concatenate(err)))

    four_before = {f: np.asarray(st.read_correspondences(f)[0])[(np.asarray(st.read_correspondences(f)[1])[:, :4] >= 0).all(1)] for f in FRAMES}
    e0 = partial_error()
    res = align_partial_points(st, FRAMES, cals, cpar, vpar)
    assert res.applied and res.n_updated > 0, res.text()
    assert res.four_cam_changed == 0 and res.shift_median_um > 0
    for f in FRAMES:
        now = np.asarray(st.read_correspondences(f)[0])[(np.asarray(st.read_correspondences(f)[1])[:, :4] >= 0).all(1)]
        assert np.array_equal(now, four_before[f])
    assert partial_error() < e0
    assert "align_partial_points" in dict(st.root.attrs)
    again = align_partial_points(st, FRAMES, cals, cpar, vpar)
    assert not again.applied and "already applied" in again.message


def test_alignment_refuses_points_matched_with_other_cameras(rig, tmp_path):
    cpar, vpar, cals = rig
    # the stored points belong to cameras that are NOT the ones now given: a re-calibration without a new matching
    other = _drifted(cals, 6.0)
    st = _store(tmp_path / "g.zarr", other, other, cpar, vpar, drop=0.4)
    res = align_partial_points(st, FRAMES, cals, cpar, vpar, min_verdict="consistent")
    assert not res.applied and "Match again" in res.message
    assert "align_partial_points" not in dict(st.root.attrs)


def test_alignment_leaves_consistent_cameras_alone(rig, tmp_path):
    cpar, vpar, cals = rig
    st = _store(tmp_path / "h.zarr", cals, cals, cpar, vpar)
    res = align_partial_points(st, FRAMES, cals, cpar, vpar)
    assert not res.applied and "nothing to correct" in res.message


def test_a_small_sample_is_inconclusive_not_a_confident_verdict(rig, tmp_path):
    """Few points of a noisy sample cannot separate a systematic jump from noise: say so instead of guessing."""
    cpar, vpar, cals = rig
    drifted = _drifted(cals, 1.0)
    st = _store(tmp_path / "i.zarr", drifted, cals, cpar, vpar, drop=0.0, n=320)
    rep = consistency_report(st, FRAMES[:1], cals, cpar, vpar, min_points=300)
    assert rep.applicable
    assert rep.verdict in ("inconclusive", "consistent") or max(rep.jump_um) > 3 * max(rep.se_um), rep.text()
    assert all(np.isnan(c["field_explained"]) for c in rep.cameras)       # one frame: no held-out half


def test_wrong_matches_do_not_spoil_the_report(rig, tmp_path):
    """Matched quadruplets contain ghosts; their huge residuals must not turn a consistent calibration into a bad verdict."""
    cpar, vpar, cals = rig
    st = _store(tmp_path / "j.zarr", cals, cals, cpar, vpar, drop=0.0, n=600)
    clean = consistency_report(st, FRAMES, cals, cpar, vpar, min_points=300)
    rng = np.random.default_rng(5)
    for f in FRAMES:                                     # 15 % of the points get one camera's blob 20 px away: a wrong match
        for c in range(4):
            key = f"targets/cam_{c}/frame_{f:06d}"
            t = np.array(st.root[key])
            bad = rng.random(len(t)) < 0.04
            t[bad, 1:3] += rng.choice([-20.0, 20.0], (int(bad.sum()), 2))
            st.write_targets(c, f, t)
    dirty = consistency_report(st, FRAMES, cals, cpar, vpar, min_points=300)
    assert dirty.outlier_fraction > 0.08
    assert dirty.verdict == clean.verdict == "consistent", dirty.text()
    assert max(dirty.jump_um) < 3 * max(max(clean.jump_um), 5.0)


def test_an_anchor_is_reported_and_the_unanchored_warning_names_the_stretches(rig, tmp_path):
    """Without an anchor the result says the scale is not fixed (and gives the three stretches); with a known body it does not."""
    from openptv2.selfcal import KnownBody

    cpar, vpar, cals = rig
    drifted = _drifted(cals)
    st = _store(tmp_path / "k.zarr", drifted, cals, cpar, vpar)
    free = self_calibrate_from_store(st, FRAMES, cals, cpar, vpar, max_nfev=80, min_yield=0)
    assert not free.anchored and "SCALE NOT ANCHORED" in free.text()
    assert len(free.scale_axes) == 3 and free.stretch_dirs.shape == (3, 3)
    rng = np.random.default_rng(9)
    xyz = rng.uniform(-3.0, 3.0, (60, 3))
    obs = TruthRig.from_cals(drifted, cpar).project(xyz)
    ok = np.isfinite(obs).all(axis=(1, 2))
    anchored = self_calibrate_from_store(st, FRAMES, cals, cpar, vpar, max_nfev=80, min_yield=0, body=KnownBody(xyz[ok], obs[ok]))
    assert anchored.anchored and "SCALE NOT ANCHORED" not in anchored.text()


def test_a_calibration_that_matches_worse_is_refused_even_if_the_jump_fell(rig, tmp_path, monkeypatch):
    """The held-out jump is measured on points the old cameras matched; matching again with the new cameras must not lose points."""
    import openptv2.calibration_consistency as cc

    cpar, vpar, cals = rig
    st = _store(tmp_path / "m.zarr", _drifted(cals), cals, cpar, vpar)
    seen = []

    def fake_yield(store, frames, cs, cp, vp):
        seen.append(cs)
        return (1000, 2000) if len(seen) == 1 else (850, 2000)        # old cameras first, the new ones lose 15 %

    monkeypatch.setattr(cc, "matching_yield", fake_yield)
    res = cc.self_calibrate_from_store(st, FRAMES, cals, cpar, vpar, max_nfev=120)
    assert not res.accepted and abs(res.yield_ratio - 0.85) < 1e-9 and "fewer 4-camera matches" in res.message, res.text()
    monkeypatch.setattr(cc, "matching_yield", lambda *a: (1000, 2000))
    seen.clear()
    ok = cc.self_calibrate_from_store(st, FRAMES, cals, cpar, vpar, max_nfev=120)
    assert ok.accepted and ok.yield_ratio == 1.0
