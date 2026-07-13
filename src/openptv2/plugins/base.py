"""Contract for openptv2 sequence/tracking plugins.

A plugin module exposes a ``Sequence`` class (with ``do_sequence``) and/or a
``Tracking`` class (with ``do_tracking``). Both are constructed as
``Cls(ptv=<openptv2.gui.ptv module>, exp=<experiment/ProcessingExperiment>)``
by :mod:`openptv2.plugins.loader` — plugins should not import ``ptv``
themselves.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SequencePlugin(Protocol):
    def __init__(self, ptv=None, exp=None) -> None: ...

    def do_sequence(self) -> None: ...


@runtime_checkable
class TrackingPlugin(Protocol):
    def __init__(self, ptv=None, exp=None) -> None: ...

    def do_tracking(self) -> None: ...
