"""Temporary I/O helpers for tests that need writable data."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if missing, returning the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_empty_series(base: Path, first: int, last: int) -> None:
    """Create `<base>.<frame>` files containing `0` for a frame range."""
    ensure_dir(base.parent)
    for frame in range(first, last + 1):
        (base.parent / f"{base.name}.{frame}").write_text("0\n")
