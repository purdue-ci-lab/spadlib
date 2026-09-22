"""
Tests for the Piccolo 32x32 asynchronous SPAD reader.

A tiny synthetic acquisition (4 x 3 pixels, a few frames of coarse/fine counter values
per pixel) is written as ``timestamps_{n}.mat`` files and read back; the per-pixel
timestamps, the flat (t, y, x) events, the gap-preserving frame indexing and the
row/col subsetting are checked against the ground truth.
"""
import numpy as np
import pytest
import scipy.io

from spadlib.io import (
    PICCOLO_COARSE_TICK_S,
    PICCOLO_FINE_TICK_S,
    PICCOLO_ROLLOVER_S,
    read_async_spad_piccolo_dir,
)

H, W, ACQ_PER = 4, 3, 1e-3


def make_counters(h=H, w=W, seed=0):
    """
    Deterministic (h, w) object arrays of coarse/fine counter values. Pixel (1, 2) sees
    no photons in any frame.
    """
    rng = np.random.default_rng(seed)
    c1 = np.empty((h, w), dtype=object)
    c2 = np.empty((h, w), dtype=object)
    for r in range(h):
        for c in range(w):
            n = 0 if (r, c) == (1, 2) else int(rng.integers(1, 5))
            c1[r, c] = rng.integers(0, 2**16, size=n).astype(np.uint16)
            c2[r, c] = rng.integers(0, 100, size=n).astype(np.uint16)
    return c1, c2


def write_frame(path, c1, c2):
    scipy.io.savemat(str(path), {"timestampsCounter1": c1, "timestampsCounter2": c2})


def expected_pixel(c1, c2, frame_idxs, r, c, acq_per=ACQ_PER):
    """Ground-truth timestamps for pixel (r, c) across the given true frame indices."""
    parts = []
    for n, (a, b) in zip(frame_idxs, zip(c1, c2)):
        t0 = n * acq_per - np.mod(n * acq_per, PICCOLO_ROLLOVER_S)
        parts.append(
            PICCOLO_COARSE_TICK_S * np.atleast_1d(a[r, c]).astype("float64")
            - PICCOLO_FINE_TICK_S * np.atleast_1d(b[r, c]).astype("float64")
            + t0
        )
    return np.sort(np.concatenate(parts, dtype="float64"))


@pytest.fixture
def acquisition(tmp_path):
    """
    Three frames written at indices 3, 4, 6 -- a gap at 5, and a first index that is not
    0, so both the gap-preserving and the first-index-is-frame-0 rules are exercised.
    """
    disk_idxs = [3, 4, 6]
    counters = [make_counters(seed=i) for i in range(len(disk_idxs))]
    d = tmp_path / "piccolo"
    d.mkdir()
    for idx, (c1, c2) in zip(disk_idxs, counters):
        write_frame(d / f"timestamps_{idx}.mat", c1, c2)
    frame_idxs = [i - disk_idxs[0] for i in disk_idxs]  # 0, 1, 3
    return d, counters, frame_idxs


def test_pixels_and_meta(acquisition):
    d, counters, frame_idxs = acquisition
    c1s = [c1 for c1, _ in counters]
    c2s = [c2 for _, c2 in counters]
    points, pixels, meta = read_async_spad_piccolo_dir(d, ACQ_PER, return_meta=True)

    assert meta["num_frames"] == 4  # index span 3..6, not the 3 files on disk
    assert meta["T_exp"] == pytest.approx(4 * ACQ_PER)
    assert meta["first_frame_index"] == 3
    assert (meta["H"], meta["W"]) == (H, W)

    for r in range(H):
        for c in range(W):
            np.testing.assert_allclose(
                pixels[r][c], expected_pixel(c1s, c2s, frame_idxs, r, c)
            )
    assert pixels[1][2].size == 0  # read, but no photons
    assert points.shape[0] == sum(pixels[r][c].size for r in range(H) for c in range(W))


def test_events_match_pixels(acquisition):
    d, _, _ = acquisition
    points, pixels = read_async_spad_piccolo_dir(d, ACQ_PER)
    counts = np.zeros((H, W), dtype="int64")
    for t, y, x in points:
        assert t in pixels[int(y)][int(x)]
        counts[int(y), int(x)] += 1
    expected = np.array([[pixels[r][c].size for c in range(W)] for r in range(H)])
    np.testing.assert_array_equal(counts, expected)


def test_row_col_subset(acquisition):
    """Unselected pixels are None; selected ones match the full read."""
    d, _, _ = acquisition
    _, full = read_async_spad_piccolo_dir(d, ACQ_PER)
    _, subset = read_async_spad_piccolo_dir(d, ACQ_PER, rows=[0, 2], cols=[1])
    for r in range(H):
        for c in range(W):
            if r in (0, 2) and c == 1:
                np.testing.assert_array_equal(subset[r][c], full[r][c])
            else:
                assert subset[r][c] is None


def test_transpose(tmp_path):
    """A transposed capture reads back into output coordinates."""
    c1, c2 = make_counters()
    d = tmp_path / "t"
    d.mkdir()
    write_frame(d / "timestamps_0.mat", c1.T, c2.T)
    _, pixels = read_async_spad_piccolo_dir(d, ACQ_PER, transpose=True)
    for r in range(H):
        for c in range(W):
            np.testing.assert_allclose(pixels[r][c], expected_pixel([c1], [c2], [0], r, c))


def test_natural_sort_beats_string_sort(tmp_path):
    """timestamps_10 must be placed after timestamps_2, not before it."""
    d = tmp_path / "natural"
    d.mkdir()
    c1 = np.empty((H, W), dtype=object)
    c2 = np.empty((H, W), dtype=object)
    for r in range(H):
        for c in range(W):
            c1[r, c] = np.array([0] if (r, c) == (0, 0) else [], dtype=np.uint16)
            c2[r, c] = np.array([0] if (r, c) == (0, 0) else [], dtype=np.uint16)
    for idx in (2, 10):
        write_frame(d / f"timestamps_{idx}.mat", c1, c2)
    _, pixels, meta = read_async_spad_piccolo_dir(d, ACQ_PER, return_meta=True)
    assert meta["num_frames"] == 9  # span 2..10
    # frame 8's photon lands at its rollover-truncated start, after frame 0's
    np.testing.assert_allclose(pixels[0][0], expected_pixel([c1, c1], [c2, c2], [0, 8], 0, 0))
    assert pixels[0][0][1] > 0.0


def test_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_async_spad_piccolo_dir(tmp_path, ACQ_PER)
