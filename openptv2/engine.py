"""
Engine selection module for openptv2.

Provides functionality to switch between the C/Cython (optv) engine
and the Python fallback engine.

Example usage:
    >>> from openptv2 import get_engine, set_engine
    >>>
    >>> # Check current engine
    >>> print(get_engine())  # 'optv' or 'python'
    >>>
    >>> # Switch to Python engine for debugging
    >>> set_engine("python")
    >>>
    >>> # Set via environment variable
    >>> import os
    >>> os.environ["OPENPTV_ENGINE"] = "python"
"""

import os
from typing import Optional, Literal, Any
import threading

EngineType = Literal["optv", "python"]

# Global engine state (determined once at import time)
_default_engine = None
_engine_initialized = False


def _detect_engine():
    """
    Detect which engine to use based on environment and availability.

    Checks in order:
    1. OPENPTV_ENGINE environment variable
    2. Auto-detect: prefer optv if available, fallback to python

    Returns:
        str: Engine name ("optv" or "python")
    """
    global _default_engine, _engine_initialized

    if _engine_initialized:
        return _default_engine

    # Check environment variable
    env_engine = os.environ.get("OPENPTV_ENGINE", "").lower()
    if env_engine in ("python", "algorithms"):
        _default_engine = "python"
        _engine_initialized = True
        return _default_engine
    elif env_engine == "optv":
        _default_engine = "optv"
        _engine_initialized = True
        return _default_engine

    # Auto-detect: prefer optv if available
    try:
        import optv  # noqa: F401
        _default_engine = "optv"
    except ImportError:
        _default_engine = "python"

    _engine_initialized = True
    return _default_engine


def get_engine() -> EngineType:
    """
    Get the current default engine.

    Detects engine from environment variable or auto-detects on first call.

    Returns:
        Current default engine name ("optv" or "python")
    """
    return _detect_engine()


def set_engine(engine: EngineType) -> None:
    """
    Set the default engine.

    Note: Must be called before importing any openptv2.* modules.
    Once modules are imported, they cache the engine selection.

    Args:
        engine: Engine to use ("optv" or "python")

    Raises:
        ValueError: If unknown engine is specified
        RuntimeError: If called after engine already initialized
    """
    global _default_engine, _engine_initialized

    if engine not in ("optv", "python"):
        raise ValueError(f"Unknown engine: {engine}. Use 'optv' or 'python'.")

    if _engine_initialized and _default_engine != engine:
        import warnings
        warnings.warn(
            f"Engine already initialized to '{_default_engine}'. Changing to '{engine}' "
            "after importing openptv2 modules may have no effect on already loaded modules. "
            "Setting environment variable instead.",
            RuntimeWarning
        )

    _default_engine = engine
    _engine_initialized = True
    # Also set environment variable for subprocess consistency
    os.environ["OPENPTV_ENGINE"] = engine


def is_optv_available() -> bool:
    """
    Check if optv (C/Cython) engine is available.

    Returns:
        True if optv can be imported, False otherwise
    """
    try:
        import optv  # noqa: F401
        return True
    except ImportError:
        return False


def is_python_available() -> bool:
    """
    Check if python (algorithms) engine is available.

    Returns:
        True if algorithms package is available, False otherwise
    """
    try:
        import algorithms  # noqa: F401
        return True
    except ImportError:
        return False
