"""3D vector utilities using NumPy vectorized operations.

Translation of lib/src/vec_utils.c and lib/include/vec_utils.h.

Uses Structure-of-Arrays (SoA) layout: instead of an array of vec3d objects,
we store three separate arrays (x, y, z) for batch operations. This enables
full NumPy vectorization and future Numba acceleration.

For single-vector operations, plain float or length-3 arrays are accepted.
"""

import numpy as np
from typing import Tuple

# Sentinel value for empty/unused cells (matches C's EMPTY_CELL = NaN)
EMPTY_CELL = np.nan


def is_empty(x: float) -> bool:
    """Check if a value represents an empty cell (NaN)."""
    return np.isnan(x)


# ---------------------------------------------------------------------------
# Single-vector operations (direct translations of C functions)
# ---------------------------------------------------------------------------

def vec_init() -> np.ndarray:
    """Return a 3D vector initialized to NaN.

    Returns:
        ndarray of shape (3,) filled with NaN.
    """
    return np.full(3, np.nan, dtype=np.float64)


def vec_set(x: float, y: float, z: float) -> np.ndarray:
    """Create a 3D vector from components.

    Args:
        x, y, z: vector components.

    Returns:
        ndarray of shape (3,) with [x, y, z].
    """
    return np.array([x, y, z], dtype=np.float64)


def vec_copy(src: np.ndarray) -> np.ndarray:
    """Copy a 3D vector.

    Args:
        src: source vector of shape (3,).

    Returns:
        A new ndarray copy.
    """
    return src.copy()


def vec_subt(from_vec: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Subtract two 3D vectors.

    Args:
        from_vec: vector to subtract from.
        sub: vector to subtract.

    Returns:
        from_vec - sub as ndarray of shape (3,).
    """
    return from_vec - sub


def vec_add(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Add two 3D vectors.

    Args:
        vec1, vec2: vectors to add.

    Returns:
        vec1 + vec2 as ndarray of shape (3,).
    """
    return vec1 + vec2


def vec_scalar_mul(vec: np.ndarray, scalar: float) -> np.ndarray:
    """Multiply a vector by a scalar.

    Args:
        vec: vector of shape (3,).
        scalar: scalar multiplier.

    Returns:
        scalar * vec as ndarray of shape (3,).
    """
    return scalar * vec


def vec_norm(vec: np.ndarray) -> float:
    """Compute the Euclidean norm of a 3D vector.

    Args:
        vec: vector of shape (3,).

    Returns:
        ||vec|| as float.
    """
    return float(np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2))


def vec_diff_norm(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the norm of the difference between two vectors.

    This is optimized compared to calling vec_norm(vec_subt(...)).

    Args:
        vec1, vec2: vectors to compare.

    Returns:
        ||vec1 - vec2|| as float.
    """
    dx = vec1[0] - vec2[0]
    dy = vec1[1] - vec2[1]
    dz = vec1[2] - vec2[2]
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def vec_dot(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the dot product of two 3D vectors.

    Args:
        vec1, vec2: vectors.

    Returns:
        vec1 . vec2 as float.
    """
    return float(vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2])


def vec_cross(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Compute the cross product of two 3D vectors.

    Args:
        vec1, vec2: vectors.

    Returns:
        vec1 x vec2 as ndarray of shape (3,).
    """
    return np.array([
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
    ], dtype=np.float64)


def vec_cmp(vec1: np.ndarray, vec2: np.ndarray) -> bool:
    """Check exact equality of two vectors.

    Args:
        vec1, vec2: vectors to compare.

    Returns:
        True if all components are exactly equal.
    """
    return bool(np.all(vec1 == vec2))


def vec_approx_cmp(vec1: np.ndarray, vec2: np.ndarray, eps: float = 1e-10) -> bool:
    """Check approximate equality of two vectors.

    Args:
        vec1, vec2: vectors to compare.
        eps: maximum allowed difference per component.

    Returns:
        True if |vec1[i] - vec2[i]| <= eps for all i.
    """
    return bool(np.all(np.abs(vec1 - vec2) <= eps))


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    If the vector has zero norm, returns the original vector unchanged
    (matches C behavior).

    Args:
        vec: vector of shape (3,).

    Returns:
        vec / ||vec|| as ndarray of shape (3,).
    """
    norm = vec_norm(vec)
    if norm == 0.0:
        return vec.copy()
    return vec / norm


# ---------------------------------------------------------------------------
# Batch (SoA) vectorized operations
# ---------------------------------------------------------------------------

class Vec3dBatch:
    """Batch 3D vectors stored as Structure-of-Arrays.

    Instead of storing N vectors as an (N, 3) array, we store three
    separate (N,) arrays for x, y, z components. This enables fully
    vectorized NumPy operations and future Numba acceleration.

    Attributes:
        x: (N,) array of x components
        y: (N,) array of y components
        z: (N,) array of z components
    """

    __slots__ = ("x", "y", "z")

    def __init__(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
    ):
        """Initialize a batch vector collection.

        Args:
            x, y, z: 1D arrays of equal length N, or None for empty batch.
        """
        if x is None:
            self.x = np.empty(0, dtype=np.float64)
            self.y = np.empty(0, dtype=np.float64)
            self.z = np.empty(0, dtype=np.float64)
        else:
            self.x = np.asarray(x, dtype=np.float64)
            self.y = np.asarray(y, dtype=np.float64)
            self.z = np.asarray(z, dtype=np.float64)
            if not (self.x.shape == self.y.shape == self.z.shape):
                raise ValueError("x, y, z arrays must have equal shapes")

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Vec3dBatch":
        """Create from an (N, 3) array.

        Args:
            arr: ndarray of shape (N, 3).

        Returns:
            Vec3dBatch instance.
        """
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {arr.shape}")
        return cls(arr[:, 0], arr[:, 1], arr[:, 2])

    def __len__(self) -> int:
        """Number of vectors in the batch."""
        return len(self.x)

    def to_array(self) -> np.ndarray:
        """Convert to an (N, 3) array.

        Returns:
            ndarray of shape (N, 3).
        """
        if len(self) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.column_stack([self.x, self.y, self.z])

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get a single vector as a (3,) array."""
        return np.array([self.x[idx], self.y[idx], self.z[idx]])

    def __setitem__(self, idx: int, value: np.ndarray):
        """Set a single vector from a (3,) array."""
        self.x[idx] = value[0]
        self.y[idx] = value[1]
        self.z[idx] = value[2]

    # --- Vectorized operations ---

    def subtract(self, other: "Vec3dBatch") -> "Vec3dBatch":
        """Subtract another batch element-wise.

        Args:
            other: Vec3dBatch of the same length.

        Returns:
            New Vec3dBatch with self - other.
        """
        return Vec3dBatch(self.x - other.x, self.y - other.y, self.z - other.z)

    def add(self, other: "Vec3dBatch") -> "Vec3dBatch":
        """Add another batch element-wise.

        Args:
            other: Vec3dBatch of the same length.

        Returns:
            New Vec3dBatch with self + other.
        """
        return Vec3dBatch(self.x + other.x, self.y + other.y, self.z + other.z)

    def scalar_mul(self, scalar: float) -> "Vec3dBatch":
        """Multiply all vectors by a scalar.

        Args:
            scalar: float multiplier.

        Returns:
            New Vec3dBatch with scalar * self.
        """
        return Vec3dBatch(scalar * self.x, scalar * self.y, scalar * self.z)

    def norms(self) -> np.ndarray:
        """Compute the norm of each vector.

        Returns:
            (N,) array of norms.
        """
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def diff_norms(self, other: "Vec3dBatch") -> np.ndarray:
        """Compute ||self[i] - other[i]|| for each pair.

        Args:
            other: Vec3dBatch of the same length.

        Returns:
            (N,) array of difference norms.
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def dot(self, other: "Vec3dBatch") -> np.ndarray:
        """Compute dot product for each pair of vectors.

        Args:
            other: Vec3dBatch of the same length.

        Returns:
            (N,) array of dot products.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3dBatch") -> "Vec3dBatch":
        """Compute cross product for each pair of vectors.

        Args:
            other: Vec3dBatch of the same length.

        Returns:
            New Vec3dBatch with cross products.
        """
        return Vec3dBatch(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def unit(self) -> "Vec3dBatch":
        """Normalize all vectors to unit length.

        Zero-norm vectors are left unchanged (matches C behavior).

        Returns:
            New Vec3dBatch with unit vectors.
        """
        norms = self.norms()
        # Avoid division by zero: replace 0 with 1 so division is safe
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return Vec3dBatch(
            self.x / safe_norms,
            self.y / safe_norms,
            self.z / safe_norms,
        )

    def init_empty(self, n: int) -> "Vec3dBatch":
        """Return a new batch of n vectors initialized to NaN.

        Args:
            n: number of vectors.

        Returns:
            New Vec3dBatch filled with NaN.
        """
        return Vec3dBatch(
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
        )
