import numpy as np


def test():
    residuals = np.array([0.1, 0.2, 0.3, 0.4])  # 2 points
    n = 3  # 3 points passed
    ret = np.empty((n, 2))
    for i in range(n):
        ret[i, 0] = residuals[2 * i]
        ret[i, 1] = residuals[2 * i + 1]


try:
    test()
except Exception as e:
    print(f"Exception: {e}")
