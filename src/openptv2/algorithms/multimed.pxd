# Cython declaration file for multimed.py
# Exposes C-level (nogil) helpers for cimport by other modules.

cdef double _mmlut_lookup_core(
    double tx,
    double ty,
    double tz,
    int mmlut_nr,
    int mmlut_nz,
    double mmlut_rw,
    double[:] mmlut_data,
) noexcept nogil
