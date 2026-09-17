"""
Roundtrip / sanity tests for asynchronous SPAD data (per-pixel photon timestamps).

A tiny synthetic acquisition (5 x 7 pixels of sorted timestamps in [0, T_exp), with
one photon-free pixel) is written to Zarr and read back; the per-pixel arrays, the
flat (t, y, x) event coordinates and the metadata attributes are all checked against
the ground truth.
"""
import numpy as np
import pytest

from spadlib.io import (
    read_async_spad_dir,
    read_async_spad_zarr,
    write_async_spad_npys,
    write_async_spad_zarr,
)

H, W, T_EXP = 5, 7, 0.1


def make_pixel_timeseries(h=H, w=W, T_exp=T_EXP):
    """
    Deterministic h x w list of lists of sorted timestamp arrays. Pixel (2, 3) is
    ``None`` (no photons), as :func:`read_async_spad_dir` leaves empty pixels.
    """
    rng = np.random.default_rng(0)
    pixel_timeseries = [
        [np.sort(rng.uniform(0, T_exp, rng.integers(1, 6))) for _ in range(w)]
        for _ in range(h)
    ]
    pixel_timeseries[2][3] = None
    return pixel_timeseries


def as_array(timestamps):
    """Ground-truth per-pixel array, with ``None`` meaning no photons."""
    return np.asarray([] if timestamps is None else timestamps, dtype="float64")


def n_photons(pixel_timeseries):
    return sum(as_array(ts).size for row in pixel_timeseries for ts in row)


@pytest.fixture
def pixel_timeseries():
    return make_pixel_timeseries()


@pytest.fixture
def zarr_path(tmp_path, pixel_timeseries):
    path = tmp_path / "async.zarr"
    write_async_spad_zarr(path, pixel_timeseries, T_exp=T_EXP)
    return path


def _check_pixels(read_pixels, expected):
    """Every pixel's timestamps must come back exactly, empty pixels included."""
    assert len(read_pixels) == len(expected)
    for i, row in enumerate(expected):
        assert len(read_pixels[i]) == len(row)
        for j, timestamps in enumerate(row):
            np.testing.assert_array_equal(read_pixels[i][j], as_array(timestamps))


def test_zarr_roundtrip(zarr_path, pixel_timeseries):
    (t, y, x), read_pixels, attrs = read_async_spad_zarr(zarr_path, load_data=True, return_meta=True)
    _check_pixels(read_pixels, pixel_timeseries)
    npoints = n_photons(pixel_timeseries)
    assert len(t) == len(y) == len(x) == npoints
    assert attrs["H"] == H and attrs["W"] == W
    assert attrs["T_exp"] == T_EXP
    assert attrs["npoints"] == npoints


def test_events_match_pixels(zarr_path, pixel_timeseries):
    """
    Each flat (t, y, x) event must be a photon of the pixel it points at, and the
    per-pixel counts must match.
    """
    (t, y, x), read_pixels = read_async_spad_zarr(zarr_path, load_data=True)
    counts = np.zeros((H, W), dtype="int64")
    for ti, yi, xi in zip(t, y, x):
        assert ti in read_pixels[int(yi)][int(xi)]
        counts[int(yi), int(xi)] += 1
    expected = np.array([[as_array(ts).size for ts in row] for row in pixel_timeseries])
    np.testing.assert_array_equal(counts, expected)


def test_rate_stats(zarr_path, pixel_timeseries):
    rates = np.array([[as_array(ts).size for ts in row] for row in pixel_timeseries]) / T_EXP
    _, _, attrs = read_async_spad_zarr(zarr_path, return_meta=True)
    assert attrs["avg_pts_persec_perpixel"] == pytest.approx(np.mean(rates))
    assert attrs["max_pts_persec_perpixel"] == pytest.approx(np.max(rates))
    assert attrs["min_pts_persec_perpixel"] == pytest.approx(0.0)  # pixel (2, 3)
    assert attrs["med_pts_persec_perpixel"] == pytest.approx(np.median(rates))
    assert attrs["stddev_pts_persec_perpixel"] == pytest.approx(np.std(rates, ddof=1))


def test_extra_attrs(tmp_path, pixel_timeseries):
    path = tmp_path / "extra.zarr"
    write_async_spad_zarr(path, pixel_timeseries, T_exp=T_EXP, attrs={"channel": "G"})
    _, _, attrs = read_async_spad_zarr(path, return_meta=True)
    assert attrs["channel"] == "G"


def test_npy_dir_roundtrip(tmp_path, pixel_timeseries):
    """
    The notebook flow: read a folder of per-pixel .npy files, then write it to Zarr
    with the reader's own ``points``.
    """
    npy_dir = tmp_path / "npys"
    write_async_spad_npys([[as_array(ts) for ts in row] for row in pixel_timeseries], npy_dir)
    points, read_pixels = read_async_spad_dir(npy_dir, h=H, w=W, T_exp=T_EXP)

    path = tmp_path / "from_dir.zarr"
    write_async_spad_zarr(path, read_pixels, T_exp=T_EXP, points=points)
    (t, y, x), zarr_pixels, attrs = read_async_spad_zarr(path, load_data=True, return_meta=True)

    _check_pixels(zarr_pixels, pixel_timeseries)
    assert attrs["npoints"] == points.shape[0] == n_photons(pixel_timeseries)
    # points= is stored as given, event for event
    np.testing.assert_array_equal(t, points[:, 0])
    np.testing.assert_array_equal(y, points[:, 1])
    np.testing.assert_array_equal(x, points[:, 2])


def test_no_photons(tmp_path):
    """An acquisition where no pixel saw a photon still writes and reads back."""
    path = tmp_path / "empty.zarr"
    write_async_spad_zarr(path, [[None] * 3 for _ in range(2)], T_exp=1.0)
    (t, y, x), read_pixels, attrs = read_async_spad_zarr(path, load_data=True, return_meta=True)
    assert len(t) == len(y) == len(x) == 0
    assert attrs["npoints"] == 0
    assert all(read_pixels[i][j].size == 0 for i in range(2) for j in range(3))


def test_ragged_rows_raise(tmp_path, pixel_timeseries):
    pixel_timeseries[1] = pixel_timeseries[1][:-1]
    with pytest.raises(ValueError):
        write_async_spad_zarr(tmp_path / "ragged.zarr", pixel_timeseries, T_exp=T_EXP)
