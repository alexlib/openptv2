"""
Engine comparison test package for openptv2.

This package contains tests that verify both engines (optv and python)
produce identical results within floating-point tolerance.
"""

from .test_tracking import validate_all_engines, compare_results

__all__ = ['validate_all_engines', 'compare_results']
