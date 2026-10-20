"""
Tests for frame-level processing operations.
"""
import warnings

import numpy as np
import pytest

from spadlib.io import read_quanta_bin
from spadlib.processing import accumulate_frames
from tests.conftest import H, W


def _expected_sums(frames, n):
    n_out = frames.shape[0] // n
    return frames[:n_out * n].reshape(n_out, n, *frames.shape[1:]).sum(axis=1)


@pytest.mark.parametrize("t_chunk", [1, 3, 1000])  # chunks smaller than, unaligned with, larger than a group
def test_accumulate_frames(frames, t_chunk):
    out = accumulate_frames(frames, 2, t_chunk=t_chunk)
    assert out.shape == (5, H, W)
    np.testing.assert_array_equal(out, _expected_sums(frames, 2))


def test_accumulate_frames_drops_remainder_with_warning(frames):
    with pytest.warns(UserWarning, match="dropping the last 1 frame"):
        out = accumulate_frames(frames, 3)
    assert out.shape == (3, H, W)
    np.testing.assert_array_equal(out, _expected_sums(frames, 3))


def test_accumulate_frames_no_warning_when_divisible(frames):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        accumulate_frames(frames, 5)


def test_accumulate_frames_too_few_frames_raises(frames):
    with pytest.raises(ValueError, match="larger than the number of frames"):
        accumulate_frames(frames, 11)


@pytest.mark.parametrize("in_dtype, n, expected", [
    (np.uint8, 1, np.uint8),
    (np.uint8, 2, np.uint16),
    (np.uint16, 1000, np.uint32),
    (np.bool_, 255, np.uint8),
    (np.bool_, 256, np.uint16),
    (np.int16, 2, np.int32),
    (np.float32, 2, np.float32),
])
def test_accumulate_frames_dtype_cannot_overflow(in_dtype, n, expected):
    frames = np.ones((n, 2, 2), dtype=in_dtype)
    out = accumulate_frames(frames, n)
    assert out.dtype == expected
    np.testing.assert_array_equal(out, np.full((1, 2, 2), n))


def test_accumulate_frames_explicit_dtype(frames):
    out = accumulate_frames(frames, 2, dtype=np.uint8)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, _expected_sums(frames, 2))


def test_accumulate_frames_lazy_input(bin_path, frames):
    lazy = read_quanta_bin(bin_path, H=H, W=W)
    np.testing.assert_array_equal(accumulate_frames(lazy, 2, t_chunk=4), _expected_sums(frames, 2))
