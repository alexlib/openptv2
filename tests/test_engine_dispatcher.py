"""
Tests for the dual-engine dispatcher in openptv2.engine.
"""

import os
import sys
import pytest
from unittest import mock
import openptv2.engine as engine


@pytest.fixture
def save_engine_state():
    """Fixture to save and restore engine global state and environment variables."""
    old_default = engine._default_engine
    old_initialized = engine._engine_initialized
    old_env = os.environ.get("OPENPTV_ENGINE")

    yield

    engine._default_engine = old_default
    engine._engine_initialized = old_initialized
    if old_env is not None:
        os.environ["OPENPTV_ENGINE"] = old_env
    elif "OPENPTV_ENGINE" in os.environ:
        del os.environ["OPENPTV_ENGINE"]


def test_engine_dispatcher_get_engine(save_engine_state):
    """Test that get_engine returns one of the valid engines."""
    # Reset initialization to force detection
    engine._engine_initialized = False
    engine._default_engine = None

    current = engine.get_engine()
    assert current in ("optv", "python")


def test_engine_dispatcher_set_engine(save_engine_state):
    """Test that set_engine programmatically sets the engine and sets the environment variable."""
    # Reset initialization
    engine._engine_initialized = False
    engine._default_engine = None

    engine.set_engine("python")
    assert engine.get_engine() == "python"
    assert os.environ.get("OPENPTV_ENGINE") == "python"

    # Reset initialization for optv test
    engine._engine_initialized = False
    engine._default_engine = None

    engine.set_engine("optv")
    assert engine.get_engine() == "optv"
    assert os.environ.get("OPENPTV_ENGINE") == "optv"


def test_engine_dispatcher_invalid_engine(save_engine_state):
    """Test that set_engine raises ValueError for invalid engine names."""
    with pytest.raises(ValueError):
        engine.set_engine("invalid_engine_name")  # type: ignore


def test_engine_dispatcher_env_override(save_engine_state):
    """Test that OPENPTV_ENGINE environment variable takes precedence."""
    # Reset initialization
    engine._engine_initialized = False
    engine._default_engine = None

    os.environ["OPENPTV_ENGINE"] = "python"
    assert engine.get_engine() == "python"

    # Reset initialization
    engine._engine_initialized = False
    engine._default_engine = None

    os.environ["OPENPTV_ENGINE"] = "optv"
    assert engine.get_engine() == "optv"


def test_engine_dispatcher_fallback_behavior(save_engine_state):
    """Test fallback to python when optv is not available."""
    # Reset initialization
    engine._engine_initialized = False
    engine._default_engine = None

    if "OPENPTV_ENGINE" in os.environ:
        del os.environ["OPENPTV_ENGINE"]

    # Mock the import of optv to fail with ImportError
    with mock.patch.dict(sys.modules, {"optv": None}):
        assert engine.get_engine() == "python"


def test_engine_dispatcher_availability_check(save_engine_state):
    """Test availability check helper functions."""
    assert isinstance(engine.is_python_available(), bool)
    assert isinstance(engine.is_optv_available(), bool)
