"""Tiny synchronous publish/subscribe bus for the Tk GUI.

Views never reference each other directly. They publish typed events (a camera
click, a parameter change, a selection) and subscribe to the ones they care
about. This is what makes cross-view behaviour — e.g. a click in camera 1 drawing
the epipolar line in cameras 2..N — decoupled and testable without a display.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# --- event payloads -------------------------------------------------------- #

@dataclass(frozen=True)
class CameraClick:
    """A click inside a camera image, in IMAGE pixel coordinates."""
    cam: int
    x: float
    y: float
    button: int = 1


@dataclass(frozen=True)
class ParamsChanged:
    """A parameter section was edited and saved."""
    section: str


@dataclass(frozen=True)
class ExperimentLoaded:
    path: str
    num_cams: int


@dataclass(frozen=True)
class Status:
    """A message for the status bar."""
    text: str


class EventBus:
    """Synchronous topic bus keyed by event type.

    subscribe(EventType, handler) -> unsubscribe callable.
    publish(event) calls every handler registered for type(event).
    Handler exceptions are isolated (one bad subscriber can't break the rest)
    and collected on ``last_errors`` for tests/diagnostics.
    """

    def __init__(self) -> None:
        self._subs: dict[type, list[Callable[[Any], None]]] = {}
        self.last_errors: list[tuple[Callable, Exception]] = []

    def subscribe(self, event_type: type, handler: Callable[[Any], None]):
        self._subs.setdefault(event_type, []).append(handler)
        return lambda: self._subs.get(event_type, []).remove(handler)

    def publish(self, event: Any) -> int:
        """Deliver to all subscribers of the event's type. Returns #delivered."""
        self.last_errors = []
        handlers = list(self._subs.get(type(event), ()))
        for h in handlers:
            try:
                h(event)
            except Exception as exc:  # isolate faulty subscribers
                self.last_errors.append((h, exc))
        return len(handlers)

    def clear(self) -> None:
        self._subs.clear()
        self.last_errors = []
