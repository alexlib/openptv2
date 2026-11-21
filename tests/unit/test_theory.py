"""Tests for the theory module (ghost scaling law + trackability).

Reference: Raghunathan, Vradman & Shnapp (2026), "Time-marching
stereo-matching for 3D-PTV flow measurements", Appendix A Eq. 7 and Eq. 2.
"""

import math

import pytest

from openptv2.theory import (
    expected_ghosts,
    ghost_regime,
    prediction_search_radius,
    recommend_tolerance,
    theory_report,
    trackability,
)


class TestExpectedGhosts:
    def test_matches_paper_example_scaling(self):
        # n_ghost ~ n_p^nc * (d/L)^(2nc-3): doubling cameras (2 -> 3) at fixed
        # n_p, d/L should multiply ghosts by n_p * (d/L)^3, per Eq. 6 -> Eq. 7.
        n_p, ratio = 1000, 0.01
        g2 = expected_ghosts(n_p, 2, ratio, 1.0)
        g3 = expected_ghosts(n_p, 3, ratio, 1.0)
        assert g2 == pytest.approx(n_p**2 * ratio)
        assert g3 == pytest.approx(n_p**3 * ratio**3)
        assert g3 / g2 == pytest.approx(n_p * ratio**2)

    def test_edge_cases(self):
        assert expected_ghosts(0, 4, 0.01, 1.0) == 0.0
        assert expected_ghosts(100, 1, 0.01, 1.0) == float("inf")
        assert expected_ghosts(100, 4, 1.0, 1.0) == float("inf")
        assert expected_ghosts(100, 4, -0.1, 1.0) == 0.0

    def test_monotonicity(self):
        # More particles or a wider tolerance -> more ghosts. More cameras
        # multiply ghosts by n_p * (d/L)^2 (exponent delta 7-5): helpful
        # at tight tolerance, harmful when n_p * (d/L)^2 > 1 -- the paper's "too numerous
        # cameras can worsen the ghost issue" caveat.
        n_p, ratio = 500, 0.005
        base = expected_ghosts(n_p, 4, ratio, 1.0)
        assert expected_ghosts(1000, 4, ratio, 1.0) > base
        assert expected_ghosts(n_p, 4, 0.01, 1.0) > base
        g5 = expected_ghosts(n_p, 5, ratio, 1.0)
        assert g5 == pytest.approx(base * n_p * ratio**2)
        assert g5 < base  # n_p * (d/L)^2 = 0.0125 < 1: extra cameras help
        assert expected_ghosts(n_p, 5, 0.5, 1.0) > expected_ghosts(
            n_p, 4, 0.5, 1.0
        )  # but at wide tolerance they hurt


class TestRecommendTolerance:
    def test_inverts_expected_ghosts(self):
        n_p, n_c, L, budget = 5000, 4, 100.0, 1.0
        d = recommend_tolerance(n_p, n_c, L, ghost_budget=budget)
        assert expected_ghosts(n_p, n_c, d, L) == pytest.approx(budget)

    def test_tighter_budget_gives_smaller_tolerance(self):
        n_p, n_c, L = 5000, 4, 100.0
        assert recommend_tolerance(n_p, n_c, L, 0.1) < recommend_tolerance(
            n_p, n_c, L, 10.0
        )

    def test_overdense_gives_negligible_tolerance(self):
        # Budget below one ghost for a huge n_p: the inversion returns a
        # tolerance so small it is effectively zero (correct answer, not a
        # sentinel) -- the setup is over-dense for this camera count.
        assert recommend_tolerance(10**12, 4, 100.0, 0.001) < 1e-6 * 100.0


class TestGhostRegime:
    def test_sparse_intermediate_dense(self):
        # Sparse: essentially no ghosts at this tolerance.
        assert ghost_regime(100, 4, 0.001, 100.0) == "sparse"
        # Dense: tolerance so wide ghosts outnumber particles.
        assert ghost_regime(100, 4, 50.0, 100.0) == "dense"
        # Intermediate: in between.
        n_p, n_c, L = 5000, 4, 100.0
        d = recommend_tolerance(n_p, n_c, L, 100.0)  # 100 ghosts/frame
        assert ghost_regime(n_p, n_c, d, L) == "intermediate"


class TestTrackability:
    def test_definition(self):
        # Tr = dx * sqrt(n_p) / l
        assert trackability(2.0, 400, 1920) == pytest.approx(2.0 * 20 / 1920)

    def test_paper_operating_point(self):
        # Paper: ~95% temporal matches at Tr ~ 0.25 with the full predictor.
        # 0.005 PPP over a 1920 px image ~ 22112 particles? No -- PPP 0.005
        # on 1920x1920 is ~18k particles; instead check the boundary itself:
        assert trackability(0.25 * 1920 / math.sqrt(400), 400, 1920) == (
            pytest.approx(0.25)
        )

    def test_edge_cases(self):
        assert trackability(1.0, 0, 100) == float("inf")
        assert trackability(1.0, 100, 0) == float("inf")


class TestPredictionSearchRadius:
    def test_caps_at_safety_fraction_of_spacing(self):
        # Proposed radius larger than the safe fraction: the cap must win.
        assert prediction_search_radius(5.0, 2.0) == pytest.approx(1.0)
        # Modest proposed radius: passes through untouched (this is a cap,
        # not a target the radius is inflated to).
        assert prediction_search_radius(0.1, 2.0) == pytest.approx(0.1)

    def test_custom_safety(self):
        assert prediction_search_radius(5.0, 3.0, safety=0.8) == pytest.approx(2.4)


class TestTheoryReport:
    def test_report_fields_and_notes(self):
        n_p, n_c, L = 20000, 4, 100.0
        d = 1.0  # far above the <=1-ghost tolerance for this density
        rep = theory_report(n_p, n_c, d, L, displacement_px=3.0, image_size_px=1920)
        assert rep.expected_ghosts == pytest.approx(expected_ghosts(n_p, n_c, d, L))
        assert rep.regime in ("sparse", "intermediate", "dense")
        assert rep.recommended_tolerance > 0
        assert rep.recommended_tolerance < d  # suggests tightening
        assert any("d_min exceeds" in n for n in rep.notes)
        assert any("Tr=" in n for n in rep.notes)

    def test_summary_mentions_all_quantities(self):
        rep = theory_report(500, 4, 0.5, 100.0)
        s = rep.summary()
        for token in ("cameras=4", "expected ghosts", "regime", "Tr"):
            assert token in s


class TestSuggestParamsTrackabilityGuard:
    """The tracking_recommender integration: dacc capped in the ambiguous
    regime (rho > 1), untouched otherwise."""

    def _stats(self, p95, nn):
        from openptv2.tracking_recommender import DatasetStats

        stats = DatasetStats()
        stats.percentiles_displacement = (p95 * 0.5, p95 * 0.75, p95)
        stats.max_displacement = p95
        stats.mean_interparticle_distance = nn
        return stats

    def _params(self, stats):
        from openptv2.tracking_recommender import _suggest_params
        from openptv2.tracking_registry import TRACKER_REGISTRY

        return _suggest_params(TRACKER_REGISTRY["fast_3d"], stats)

    def test_guard_fires_when_displacement_exceeds_spacing(self):
        # rho = 4.0 / 2.0 = 2 > 1: dacc would be 0.6 * 3*4 = 7.2 (1.8x NN),
        # capped at the mean spacing 2.0.
        params = self._params(self._stats(p95=4.0, nn=2.0))
        assert params["dacc"] == pytest.approx(2.0)

    def test_no_guard_in_viable_regime(self):
        # rho = 1.858 / 2.788 < 1 (synthetic_turbulent-like): empirical
        # factor 0.6 preserved, regression test in
        # test_tracking_recommender.py pins the same expectation.
        params = self._params(self._stats(p95=1.858, nn=2.788))
        assert params["dacc"] == pytest.approx(params["dvxmax"] * 0.6)

    def test_guard_noop_without_spacing_stats(self):
        stats = self._stats(p95=4.0, nn=2.0)
        stats.mean_interparticle_distance = 0.0
        params = self._params(stats)
        assert params["dacc"] == pytest.approx(params["dvxmax"] * 0.6)
