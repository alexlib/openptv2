"""
Engine selection module for openptv2.

Provides functionality to switch between the C/Cython (optv) engine
and the Python/Numba fallback engine.

Example usage:
    >>> from openptv2 import get_engine, set_engine
    >>> 
    >>> # Check current engine
    >>> print(get_engine())  # 'optv'
    >>> 
    >>> # Switch to Python engine for debugging
    >>> set_engine("python")
    >>> 
    >>> # Use per-call engine selection
    >>> from openptv2 import track_particles
    >>> result = track_particles(images, engine="python")
"""

from typing import Optional, Literal, Any
import threading

EngineType = Literal["optv", "python"]

# Thread-local storage for engine selection
_local = threading.local()
_local.default_engine = "optv"
_local._instance = None


class EngineSelector:
    """
    Engine selector for openptv2.
    
    Manages switching between the C/Cython (optv) engine and the
    Python/Numba fallback engine. Supports both global and per-call
    engine selection.
    
    Attributes:
        default_engine: Default engine to use ("optv" or "python")
        fallback_reason: Reason for falling back to Python engine
        
    Example:
        >>> selector = EngineSelector(default_engine="optv")
        >>> engine = selector.select()  # Returns optv engine
        >>> engine = selector.select("python")  # Returns Python engine
    """
    
    def __init__(self, default_engine: EngineType = "optv"):
        """
        Initialize the engine selector.
        
        Args:
            default_engine: Default engine to use ("optv" or "python")
        """
        self.default_engine = default_engine
        self.fallback_reason: Optional[str] = None
        self._optv_engine = None
        self._python_engine = None
        self._validate_engines()
    
    def _validate_engines(self):
        """Validate that at least one engine is available."""
        optv_ok = False
        python_ok = False

        try:
            import optv
            self._optv_engine = optv
            optv_ok = True
        except ImportError as e:
            self.fallback_reason = f"optv not available: {e}"
        
        try:
            from .algorithms import numba_impl
            self._python_engine = numba_impl
            python_ok = True
        except (ImportError, ModuleNotFoundError) as e:
            if not optv_ok:
                raise RuntimeError(
                    f"No engine available. optv: {self.fallback_reason}, "
                    f"python: {e}"
                )
        
        if not optv_ok and not python_ok:
            raise RuntimeError("No engine available")
    
    def select(
        self, 
        engine: Optional[EngineType] = None
    ) -> Any:
        """
        Select and return an engine instance.
        
        Args:
            engine: Engine to use. If None, uses default_engine.
            
        Returns:
            Engine instance (either optv or python)
            
        Raises:
            ValueError: If unknown engine is specified
            RuntimeError: If selected engine is not available
        """
        engine = engine or self.default_engine
        
        if engine == "optv":
            if self._optv_engine is not None:
                return self._optv_engine
            # Fall back to Python if optv not available
            if self._python_engine is not None:
                return self._python_engine
            raise RuntimeError("optv engine not available")
        
        elif engine == "python":
            if self._python_engine is not None:
                return self._python_engine
            raise RuntimeError("Python engine not available")
        
        else:
            raise ValueError(f"Unknown engine: {engine}. Use 'optv' or 'python'.")
    
    def get_available_engines(self) -> list[EngineType]:
        """
        Return list of available engines.
        
        Returns:
            List of available engine names
        """
        available = []
        if self._optv_engine is not None:
            available.append("optv")
        if self._python_engine is not None:
            available.append("python")
        return available


def get_engine() -> EngineType:
    """
    Get the current default engine.
    
    Returns:
        Current default engine name ("optv" or "python")
    """
    return getattr(_local, "default_engine", "optv")


def set_engine(engine: EngineType) -> None:
    """
    Set the default engine globally (for current thread).
    
    Args:
        engine: Engine to use ("optv" or "python")
        
    Raises:
        ValueError: If unknown engine is specified
    """
    if engine not in ("optv", "python"):
        raise ValueError(f"Unknown engine: {engine}. Use 'optv' or 'python'.")
    _local.default_engine = engine


def get_selector() -> EngineSelector:
    """
    Get or create the global engine selector instance.
    
    Returns:
        EngineSelector instance
    """
    if _local._instance is None:
        _local._instance = EngineSelector(default_engine=get_engine())
    return _local._instance


def select_engine(engine: Optional[EngineType] = None) -> Any:
    """
    Select and return an engine instance.
    
    Convenience function that uses the global selector.
    
    Args:
        engine: Engine to use. If None, uses default_engine.
        
    Returns:
        Engine instance
    """
    return get_selector().select(engine)
