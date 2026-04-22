"""Image coordinate module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.imgcoord import image_coordinates, flat_image_coordinates
    except ImportError:
        from algorithms.compat.imgcoord import image_coordinates, flat_image_coordinates
else:
    from algorithms.compat.imgcoord import image_coordinates, flat_image_coordinates

__all__ = ['image_coordinates', 'flat_image_coordinates']
