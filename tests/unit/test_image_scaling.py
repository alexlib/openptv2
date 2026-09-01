"""Grey scaling: the rule that decides what a detection threshold means.

Every threshold in openptv2 is written on a 0-255 scale while cameras deliver
16-bit frames, so the 16->8 mapping is load-bearing.  These tests pin the two
historical mappings exactly, because the whole point of centralising them is
that no existing behaviour moved.
"""

import numpy as np
import pytest
from skimage.util import img_as_ubyte

from openptv2.image_scaling import (
    MODES,
    describe,
    from_parameters,
    suggest_range,
    to_uint8,
)


@pytest.mark.parametrize("seed", range(5))
def test_fixed_matches_img_as_ubyte(seed):
    """'fixed' must reproduce skimage exactly -- it is what the loaders and the
    GUI have always applied, so a one-grey-level drift would silently move every
    threshold in every existing parameter file."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 65536, size=(37, 53), dtype=np.uint16)
    np.testing.assert_array_equal(to_uint8(a, "fixed"), img_as_ubyte(a))


def test_fixed_is_a_bit_shift_not_a_rescale():
    """The distinction bites: 511 >> 8 == 1, while a rescale would give 2."""
    a = np.array([[0, 1, 255, 256, 511, 2112, 32768, 65520, 65535]], dtype=np.uint16)
    np.testing.assert_array_equal(to_uint8(a, "fixed"), img_as_ubyte(a))
    assert to_uint8(a, "fixed")[0, 4] == 1


def test_stretch_reproduces_detect_plate():
    """'stretch' must reproduce detect_plate's historical arithmetic byte for
    byte, including its truncating final cast."""
    rng = np.random.default_rng(0)
    w = (rng.random((64, 64)) * 60000 + 2000).astype(np.uint16)
    lo, hi = float(np.percentile(w, 1)), float(np.percentile(w, 99.5))
    expected = np.clip((w.astype(float) - lo) / (hi - lo) * 255, 0, 255).astype(
        np.uint8
    )
    np.testing.assert_array_equal(to_uint8(w, "stretch"), expected)


def test_nothing_ever_wraps():
    """The bug this module exists to prevent: an unsafe cast turns 65520 into
    240 and 32768 into 0, scrambling the image instead of scaling it."""
    a = np.array([[2112, 32768, 65520]], dtype=np.uint16)
    wrapped = a.astype(np.uint8)
    assert list(wrapped[0]) == [64, 0, 240]  # what NOT to do
    for mode in ("fixed", "stretch"):
        out = to_uint8(a, mode)
        assert out[0, 0] <= out[0, 1] <= out[0, 2], f"{mode} must stay monotone"


def test_range_is_absolute_and_clips():
    a = np.array([[0, 2112, 33816, 65520, 65535]], dtype=np.uint16)
    out = to_uint8(a, "range", lo=2112, hi=65520)
    assert out[0, 0] == 0 and out[0, 1] == 0  # at or below lo
    assert out[0, 3] == 255 and out[0, 4] == 255  # at or above hi clips
    assert 120 <= out[0, 2] <= 135  # midpoint lands mid-scale


def test_uint8_passes_through_untouched():
    a = np.array([[0, 7, 200, 255]], dtype=np.uint8)
    for mode in MODES:
        np.testing.assert_array_equal(to_uint8(a, mode, lo=0, hi=255), a)


def test_rgb_is_averaged():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    a[..., 0] = 30
    a[..., 1] = 60
    a[..., 2] = 90
    assert to_uint8(a, "fixed").shape == (4, 4)


def test_bad_mode_and_bad_range_are_rejected():
    a = np.zeros((2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="unknown grey-scaling mode"):
        to_uint8(a, "gamma")
    with pytest.raises(ValueError, match="needs both lo and hi"):
        to_uint8(a, "range")
    with pytest.raises(ValueError, match="hi > lo"):
        to_uint8(a, "range", lo=500, hi=100)
    with pytest.raises(ValueError, match="expected a 2D image"):
        to_uint8(np.zeros((2, 2, 2, 2)), "fixed")


def test_from_parameters_defaults_are_backward_compatible():
    """A parameter file that says nothing keeps the historical fixed map."""
    assert from_parameters(None)["mode"] == "fixed"
    assert from_parameters({})["mode"] == "fixed"
    assert from_parameters({"ptv": {}})["mode"] == "fixed"


def test_from_parameters_reads_and_validates():
    r = from_parameters({"ptv": {"grey_scaling": "range", "grey_range": [2112, 65520]}})
    assert (r["mode"], r["lo"], r["hi"]) == ("range", 2112.0, 65520.0)
    with pytest.raises(ValueError, match="expected one of"):
        from_parameters({"ptv": {"grey_scaling": "nonsense"}})
    with pytest.raises(ValueError, match="needs ptv.grey_range"):
        from_parameters({"ptv": {"grey_scaling": "range"}})


def test_suggest_range_brackets_the_data():
    rng = np.random.default_rng(3)
    a = (rng.random((128, 128)) * 40000 + 3000).astype(np.uint16)
    lo, hi = suggest_range(a)
    assert 3000 <= lo < hi <= 43000


def test_describe_names_the_tradeoff():
    assert "comparable" in describe("fixed")
    assert "every frame" in describe("stretch")
    assert "2112" in describe("range", 2112, 65520)


def test_targ_rec_refuses_uint16_with_an_actionable_message():
    """It used to wrap silently in the pure-Python path; the compiled kernel
    rejects it at the buffer.  Either way the caller needs to be told what to do."""
    from openptv2.algorithms.segmentation import targ_rec

    a = np.full((32, 32), 30000, dtype=np.uint16)
    with pytest.raises((TypeError, ValueError)) as e:
        targ_rec(a, 20, 80, 10, 5000, 10, 80, 10, 80, 5000)
    assert "uint8" in str(e.value) or "unsigned char" in str(e.value)


def test_targ_rec_scaled_accepts_uint16():
    """Step 3: the same call works once it goes through the scaling rule."""
    from openptv2.segmentation import targ_rec_scaled

    a = np.zeros((64, 64), dtype=np.uint16)
    a[28:34, 28:34] = 60000  # one bright blob
    out = targ_rec_scaled(a, 20, 80, 4, 5000, 2, 40, 2, 40, 100)
    assert len(out) >= 1
