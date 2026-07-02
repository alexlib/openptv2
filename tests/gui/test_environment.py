import numpy as np
from packaging import version


def test_numpy_version():
    """Verify numpy version compatibility"""
    min_version = "2.0.0"
    max_version = "2.7"
    np_version = np.__version__
    assert version.parse(min_version) <= version.parse(np_version) < version.parse(max_version), \
        f"Expected numpy >={min_version} <{max_version}, got {np_version}"


def test_runtime_available():
    """Verify the single runtime reports sane metadata."""
    from openptv2 import get_runtime_info, is_compiled

    info = get_runtime_info()
    assert info["engine"] == "cython3-pure-python"
    assert info["compiled"] == is_compiled()


def test_numpy_functionality():
    """Test basic numpy operations used by PyPTV"""
    test_arr = np.zeros((10, 10))
    test_arr[5:8, 5:8] = 1
    assert test_arr.sum() == 9, "Basic numpy operations failed"
