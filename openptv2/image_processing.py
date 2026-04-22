"""Image processing module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.image_processing import preprocess_image
    except ImportError:
        from algorithms.compat.image_processing import preprocess_image
else:
    from algorithms.compat.image_processing import preprocess_image

__all__ = ['preprocess_image']
