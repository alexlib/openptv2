"""Candidate and Coord2d must be real dataclasses in BOTH the compiled and the interpreted
build. A bare ``pnr = cython.declare(...)`` field has no annotation, so interpreted
(the documented pure-Python fallback) @dataclass saw no fields and the constructors
rejected every argument (broke between 2026-08-22 and now, unnoticed because the
compiled build was fine)."""

import dataclasses

from openptv2.algorithms.epi import Candidate, Coord2d


def test_coord2d_has_pnr_x_y_fields_and_keyword_constructor():
    assert [f.name for f in dataclasses.fields(Coord2d)] == ["pnr", "x", "y"]
    c = Coord2d(pnr=3, x=1.5, y=-2.5)
    assert (c.pnr, c.x, c.y) == (3, 1.5, -2.5)
    assert Coord2d() == Coord2d(pnr=0, x=0.0, y=0.0)
    c.x = 9.0
    assert c.x == 9.0


def test_candidate_has_pnr_tol_corr_fields_and_keyword_constructor():
    assert [f.name for f in dataclasses.fields(Candidate)] == ["pnr", "tol", "corr"]
    k = Candidate(pnr=4, tol=0.25, corr=0.75)
    assert (k.pnr, k.tol, k.corr) == (4, 0.25, 0.75)
