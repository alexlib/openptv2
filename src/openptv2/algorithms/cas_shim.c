/* CAS shim for MSVC (Windows) — provides __sync_bool_compare_and_swap
 * which is a GCC/Clang builtin but not available in MSVC.
 *
 * On GCC/Clang this file produces an empty object (no #if block active).
 * On MSVC it provides the intrinsic via _InterlockedCompareExchange.
 *
 * Linked into the track_kernels_tracking extension as an extra source.
 * The Cython-compiled code calls __sync_bool_compare_and_swap() via
 * @cython.cfunc @cython.cname("__sync_bool_compare_and_swap") declaration.
 */

#if defined(_MSC_VER)
#include <intrin.h>

int __sync_bool_compare_and_swap(volatile int *ptr, int oldval, int newval) {
    return _InterlockedCompareExchange((volatile long *)ptr,
                                       (long)newval,
                                       (long)oldval) == (long)oldval;
}
#endif
