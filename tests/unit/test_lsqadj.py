import numpy as np

from openptv2.algorithms.lsqadj import ata, atl, matinv

EPS = 1e-5


def test_matmul():
    a = np.array([1.0, 1.0, 1.0])
    b = np.zeros(3)
    dm = np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.0], [-0.3, 0.0, 1.0]])
    expected_b = np.array([0.9, 1.2, 0.7])
    b[:] = dm @ a
    assert np.allclose(b, expected_b, atol=EPS)

    d = np.array([[1, 2, 3, 99], [4, 5, 6, 99], [7, 8, 9, 99], [99, 99, 99, 99]])
    e = np.array([10, 11, 12, 99])
    f = np.zeros(3)
    expected_f = np.array([68, 167, 266])
    f[:] = d[:3, :3] @ e[:3]
    assert np.allclose(f, expected_f, atol=EPS)


def test_ata():
    a = np.array([[1, 0, 1], [2, 2, 4], [1, 2, 3], [2, 4, 3]], dtype=np.float64)
    expected = np.array([[10, 14, 18], [14, 24, 26], [18, 26, 35]], dtype=np.float64)
    b = ata(a, 4, 3)
    assert np.allclose(b, expected, atol=EPS)


def test_atl():
    a = np.array([[1, 0, 1], [2, 2, 4], [1, 2, 3], [2, 4, 3]], dtype=np.float64)
    l = np.array([1, 2, 3, 4], dtype=np.float64)
    expected = np.array([16, 26, 30], dtype=np.float64)
    u = atl(a, l, 4, 3)
    assert np.allclose(u, expected, atol=EPS)


def test_matinv():
    c = np.array([[1, 2, 3], [0, 4, 5], [1, 0, 6]], dtype=np.float64)
    expected = np.array(
        [
            [1.090909, -0.545455, -0.090909],
            [0.227273, 0.136364, -0.227273],
            [-0.181818, 0.090909, 0.181818],
        ],
        dtype=np.float64,
    )
    inv = matinv(c, 3)
    assert np.allclose(inv, expected, atol=1e-5)
