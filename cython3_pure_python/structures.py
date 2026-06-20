import cython
import math

# We declare the Target structure as a Cython Extension Class in Pure Python Mode
@cython.cclass
class Target:
    pnr: cython.int
    x: cython.double
    y: cython.double

    def __init__(self, pnr: cython.int, x: cython.double, y: cython.double):
        self.pnr = pnr
        self.x = x
        self.y = y


# Compiler directives to optimize execution when compiled
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_distances(
    targets: list,
    results: cython.double[:],
    ref_x: cython.double,
    ref_y: cython.double
) -> cython.void:
    """
    Refactored distance calculation in Cython 3 Pure Python Mode.
    Operates at native C speeds when compiled, but runs as standard
    interpreted Python with full debugging support when uncompiled.
    """
    # Declare local variables with Cython types for optimal performance
    n: cython.int = len(targets)
    i: cython.int
    t: Target
    dx: cython.double
    dy: cython.double

    for i in range(n):
        t = targets[i]
        dx = t.x - ref_x
        dy = t.y - ref_y
        results[i] = math.sqrt(dx * dx + dy * dy)


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled

