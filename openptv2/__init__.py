"""
openptv2 - Unified OpenPTV: Particle Tracking Velocimetry

This package provides a unified interface to OpenPTV algorithms with
dual-engine support:
- optv: C/Cython engine (default, fastest)
- python: Python/Numba engine (for debugging and visualization)

Example usage:
    >>> import openptv2
    >>> from openptv2 import Tracker, Calibration
    >>> 
    >>> # Use default engine (optv)
    >>> tracker = Tracker()
    >>> 
    >>> # Use Python engine for debugging
    >>> tracker = Tracker(engine="python")
    >>> 
    >>> # Global engine setting
    >>> openptv2.set_engine("python")
"""

__version__ = "1.0.0"
__author__ = "OpenPTV Community"
__email__ = "openptv@googlegroups.com"

# Engine selection
from .engine import get_engine, set_engine, EngineSelector

# Import main classes (will be populated by engine loader)
from . import tracking_framebuf
from . import tracker
from . import calibration
from . import correspondence

# Convenience imports
from .tracking_framebuf import Target, Targets
from .tracker import Tracker, TrackingParameters
from .calibration import Calibration, Orientation
from .correspondence import Correspondence

# Backward compatibility with optv API
# These aliases allow existing code to work without changes
# Note: optv is installed as a separate package from bindings/
try:
    import optv as _optv
    # Re-export optv bindings for backward compatibility
    from optv import *
except ImportError:
    # optv not available, will use Python engine
    pass

# Backward compatibility with pyptv API
# Import GUI components if available
try:
    from gui import pyptv_gui
    from gui.pyptv_gui import main as pyptv_main
except ImportError:
    pass


def get_version():
    """Return the version string."""
    return __version__


def get_engine_info():
    """
    Return information about available engines.
    
    Returns:
        dict: Engine availability and status
    """
    info = {
        "default_engine": get_engine(),
        "optv_available": False,
        "python_available": False,
    }

    try:
        import optv
        info["optv_available"] = True
        info["optv_version"] = getattr(optv, "__version__", "unknown")
    except ImportError:
        info["optv_error"] = "optv C/Cython engine not available"

    try:
        from .algorithms import numba_impl
        info["python_available"] = True
    except (ImportError, ModuleNotFoundError):
        info["python_error"] = "Python/Numba engine not available"

    return info


def validate_engines(tolerance=1e-10):
    """
    Validate that both engines produce identical results.
    
    Args:
        tolerance: Floating-point tolerance for comparison
        
    Returns:
        dict: Validation results for each algorithm
    """
    from .tests.engine_comparison import validate_all
    
    return validate_all(tolerance=tolerance)


# Module metadata
__all__ = [
    "Target",
    "Targets", 
    "Tracker",
    "TrackingParameters",
    "Calibration",
    "Orientation",
    "Correspondence",
    "get_engine",
    "set_engine",
    "EngineSelector",
    "get_version",
    "get_engine_info",
    "validate_engines",
]
