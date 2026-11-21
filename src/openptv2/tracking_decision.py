"""Tracking decision audit record (Milestone F / F0).

Stores the resolved configuration, regime flags, conditioning ratio,
probe metrics, grid-search table, chosen configuration, post-run report
metrics, and provenance metadata (git version, timestamp, data hashes)
as ``tracking_decision.json`` and as a ``RunStore`` metadata attribute.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Union

if TYPE_CHECKING:
    from openptv2.storage.run_store import RunStore

DECISION_FILENAME = "tracking_decision.json"


def _is_run_store(obj: Any) -> bool:
    return hasattr(obj, "store_path") and hasattr(obj, "root")


def _resolve_json_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if p.suffix == ".json":
        return p
    return p / DECISION_FILENAME


@dataclass
class TrackingDecision:
    """Audit record of automatic parameter resolution and tracking outcome."""

    resolved_config: dict[str, Any] = field(default_factory=dict)
    regime_flags: dict[str, Any] = field(default_factory=dict)
    conditioning_ratio: float | None = None
    probe_metrics: list[dict[str, Any]] = field(default_factory=list)
    grid_search_table: list[dict[str, Any]] = field(default_factory=list)
    chosen_config: dict[str, Any] = field(default_factory=dict)
    report_metrics: dict[str, Any] = field(default_factory=dict)
    git_version: str = ""
    timestamp: str = ""
    data_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackingDecision":
        """Construct a ``TrackingDecision`` from a mapping, filling defaults."""
        raw_ratio = data.get("conditioning_ratio")
        ratio = float(raw_ratio) if raw_ratio is not None else None
        return cls(
            resolved_config=dict(data.get("resolved_config") or {}),
            regime_flags=dict(data.get("regime_flags") or {}),
            conditioning_ratio=ratio,
            probe_metrics=[dict(item) for item in (data.get("probe_metrics") or [])],
            grid_search_table=[
                dict(item) for item in (data.get("grid_search_table") or [])
            ],
            chosen_config=dict(data.get("chosen_config") or {}),
            report_metrics=dict(data.get("report_metrics") or {}),
            git_version=str(data.get("git_version") or ""),
            timestamp=str(data.get("timestamp") or ""),
            data_hashes={
                str(k): str(v) for k, v in (data.get("data_hashes") or {}).items()
            },
        )

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize this decision to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TrackingDecision":
        """Deserialize a ``TrackingDecision`` from a JSON string."""
        return cls.from_dict(json.loads(text))

    def save(self, store_or_path: Union["RunStore", str, Path]) -> Path:
        """Persist as ``tracking_decision.json`` and (for ``RunStore``) meta attr."""
        payload = self.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True)

        if _is_run_store(store_or_path):
            store: Any = store_or_path
            json_path = Path(store.store_path) / DECISION_FILENAME
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(text, encoding="utf-8")

            lock = getattr(store, "_lock", None)
            if lock is not None:
                with lock:
                    store.root["meta"].attrs["tracking_decision"] = payload
            else:
                store.root["meta"].attrs["tracking_decision"] = payload
            return json_path

        json_path = _resolve_json_path(store_or_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(text, encoding="utf-8")
        return json_path

    @classmethod
    def load(cls, store_or_path: Union["RunStore", str, Path]) -> "TrackingDecision":
        """Load a ``TrackingDecision`` from a ``RunStore`` or file/directory path."""
        if _is_run_store(store_or_path):
            store: Any = store_or_path
            json_path = Path(store.store_path) / DECISION_FILENAME
            if json_path.is_file():
                return cls.from_json(json_path.read_text(encoding="utf-8"))
            lock = getattr(store, "_lock", None)
            if lock is not None:
                with lock:
                    attr = store.root["meta"].attrs.get("tracking_decision")
            else:
                attr = store.root["meta"].attrs.get("tracking_decision")
            if attr is not None:
                return cls.from_dict(attr)
            raise FileNotFoundError(
                f"No {DECISION_FILENAME} or meta attribute found in RunStore at {store.store_path}"
            )

        json_path = _resolve_json_path(store_or_path)
        return cls.from_json(json_path.read_text(encoding="utf-8"))
