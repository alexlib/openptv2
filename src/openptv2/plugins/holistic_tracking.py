"""Deprecated alias: ``holistic`` is ``two_phase`` with ``holistic_rounds`` defaulting to 2.

Prefer ``two_phase`` with ``track.holistic_rounds: 2`` (see docs/holistic-tracking.md).
"""

from __future__ import annotations

from openptv2.plugins.two_phase_tracking import Tracking as _TwoPhase


class Tracking(_TwoPhase):
    default_holistic_rounds = 2
