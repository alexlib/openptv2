"""Version information for openptv2.

Version is managed in pyproject.toml and read dynamically.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("openptv2")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__version_tuple__ = tuple(int(x) for x in __version__.split(".")[:3])
VERSION = __version__
