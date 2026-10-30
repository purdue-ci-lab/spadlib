"""
Roundtrip tests for the HDF5 readers/writers of async SPAD, quanta and gated quanta
data. Each is checked against the ground truth and against its Zarr counterpart: the
same input must produce the same metadata attributes in both formats.
"""
import h5py
import numpy as np
import pytest
import zarr

from spadlib.io import (
    HDF5_COMPRESSION_LEVEL,
    HDF5_MAX_CHUNK_BYTES,
    QuantaGatedDir,
    read_async_spad_hdf5,
    read_async_spad_zarr,
    read_quanta_auto,
    read_quanta_gated_hdf5,
    read_quanta_gated_zarr,
    read_quanta_hdf5,
    read_quanta_zarr,
    write_async_spad_hdf5,
    write_async_spad_zarr,
    write_quanta_gated_hdf5,
    write_quanta_gated_zarr,
    write_quanta_hdf5,
    write_quanta_zarr,
)
from tests.test_async_spad_io import T_EXP, as_array, make_pixel_timeseries, n_photons
from tests.test_async_spad_io import H as ASYNC_H, W as ASYNC_W
from tests.test_gated_quanta_io import (
    HEIGHT,
    N_FRAMES,
    N_GATE_STEPS,
    WIDTH,
    make_gated_images,
    write_gated_dir,
)


def _check_compression(ds, level=HDF5_COMPRESSION_LEVEL):
    assert ds.compression == "gzip"
    assert ds.compression_opts == level
    assert ds.shuffle


def test_default_compression_level():
    assert HDF5_COMPRESSION_LEVEL == 5


def _write_each_hdf5(tmp_path, gated_images, **kwargs):
    """Write one file with each HDF5 writer; returns their paths."""
    paths = {
        "async": tmp_path / "async.h5",
        "quanta": tmp_path / "quanta.h5",
        "gated": tmp_path / "gated.h5",
    }
    write_async_spad_hdf5(paths["async"], make_pixel_timeseries(), T_exp=T_EXP, **kwargs)
    write_quanta_hdf5(paths["quanta"], gated_images[:, 0], T_exp=1.0, **kwargs)
    write_quanta_gated_hdf5(paths["gated"], gated_images, **kwargs)
    return paths


@pytest.mark.parametrize("level", [0, 1, 9, np.int64(7)])
def test_hdf5_writers_compression_level(tmp_path, level):
    paths = _write_each_hdf5(tmp_path, make_gated_images(), compression_level=level)
    for path in paths.values():
        with h5py.File(path, "r") as f:
            f.visititems(lambda _, obj: isinstance(obj, h5py.Dataset) and _check_compression(obj, level))


@pytest.mark.parametrize("level", [-1, 10, 4.0, "5", True, None])
def test_hdf5_writers_reject_bad_compression_level(tmp_path, level):
    """A bad level is caught before the existing file is truncated."""
    paths = _write_each_hdf5(tmp_path, make_gated_images())
    before = {name: path.read_bytes() for name, path in paths.items()}
    for name, write in (
        ("async", lambda: write_async_spad_hdf5(
            paths["async"], make_pixel_timeseries(), T_exp=T_EXP, compression_level=level)),
        ("quanta", lambda: write_quanta_hdf5(
            paths["quanta"], make_gated_images()[:, 0], T_exp=1.0, compression_level=level)),
        ("gated", lambda: write_quanta_gated_hdf5(
            paths["gated"], make_gated_images(), compression_level=level)),
    ):
        with pytest.raises(ValueError, match="compression_level"):
            write()
        assert paths[name].read_bytes() == before[name]


# --- async SPAD -------------------------------------------------------------
@pytest.fixture
def pixel_timeseries():
    return make_pixel_timeseries()


def test_async_spad_hdf5_roundtrip(tmp_path, pixel_timeseries):
    path = tmp_path / "async.h5"
    write_async_spad_hdf5(path, pixel_timeseries, T_exp=T_EXP)
    (t, y, x), read_pixels, attrs = read_async_spad_hdf5(path, load_data=True, return_meta=True)

    for i, row in enumerate(pixel_timeseries):
        for j, timestamps in enumerate(row):
            np.testing.assert_array_equal(read_pixels[i][j], as_array(timestamps))
    assert all(isinstance(a, np.ndarray) for a in (t, y, x))
    assert len(t) == len(y) == len(x) == n_photons(pixel_timeseries)
    assert attrs["H"] == ASYNC_H and attrs["W"] == ASYNC_W

    zpath = tmp_path / "async.zarr"
    write_async_spad_zarr(zpath, pixel_timeseries, T_exp=T_EXP)
    (zt, zy, zx), _, zattrs = read_async_spad_zarr(zpath, load_data=True, return_meta=True)
    np.testing.assert_array_equal(t, zt)
    np.testing.assert_array_equal(y, zy)
    np.testing.assert_array_equal(x, zx)
    assert attrs == zattrs


def test_async_spad_hdf5_layout(tmp_path, pixel_timeseries):
    path = tmp_path / "async.h5"
    write_async_spad_hdf5(path, pixel_timeseries, T_exp=T_EXP)
    with h5py.File(path, "r") as f:
        assert set(f.keys()) == {"t", "y", "x", "offsets", "lengths"}
        for ds in f.values():
            _check_compression(ds)


def test_async_spad_hdf5_lazy(tmp_path, pixel_timeseries):
    path = tmp_path / "async.h5"
    write_async_spad_hdf5(path, pixel_timeseries, T_exp=T_EXP)
    (t, y, x), _ = read_async_spad_hdf5(path)
    assert isinstance(t, h5py.Dataset) and isinstance(y, h5py.Dataset)
    (et, _, _), _ = read_async_spad_hdf5(path, load_data=True)
    np.testing.assert_array_equal(t[:], et)


def test_async_spad_hdf5_no_photons(tmp_path):
    path = tmp_path / "empty.h5"
    write_async_spad_hdf5(path, [[None] * 3 for _ in range(2)], T_exp=1.0)
    (t, y, x), read_pixels, attrs = read_async_spad_hdf5(path, load_data=True, return_meta=True)
    assert len(t) == len(y) == len(x) == 0
    assert attrs["npoints"] == 0
    assert all(read_pixels[i][j].size == 0 for i in range(2) for j in range(3))


def test_async_spad_hdf5_overwrite(tmp_path, pixel_timeseries):
    path = tmp_path / "async.h5"
    write_async_spad_hdf5(path, pixel_timeseries, T_exp=T_EXP)
    with pytest.raises(FileExistsError):
        write_async_spad_hdf5(path, pixel_timeseries, T_exp=T_EXP, overwrite=False)


def test_hdf5_extra_attrs(tmp_path, pixel_timeseries):
    """Attribute values HDF5 can't store natively still get written."""
    path = tmp_path / "extra.h5"
    extra = {
        "channel": "G",
        "flag": True,
        "wavelengths": [450, 550],
        "names": ["a", "b"],
        "missing": None,
        "nested": {"a": 1},
    }
    write_async_spad_hdf5(path, pixel_timeseries, T_exp=T_EXP, attrs=extra)
    _, _, attrs = read_async_spad_hdf5(path, return_meta=True)
    assert attrs["channel"] == "G"
    assert attrs["flag"] is True
    assert attrs["wavelengths"] == [450, 550]
    assert attrs["names"] == ["a", "b"]
    assert attrs["missing"] is None
    assert attrs["nested"] == '{"a": 1}'  # no native HDF5 type: stored as JSON


# --- quanta -----------------------------------------------------------------
def test_quanta_hdf5_roundtrip(tmp_path, frames):
    path = tmp_path / "data.h5"
    write_quanta_hdf5(path, frames, T_exp=1.0)
    read_frames, attrs = read_quanta_hdf5(path, load_data=True, return_meta=True)
    assert isinstance(read_frames, np.ndarray)
    assert read_frames.dtype == np.uint8
    np.testing.assert_array_equal(read_frames, frames)

    zpath = tmp_path / "data.zarr"
    write_quanta_zarr(zpath, frames, T_exp=1.0)
    _, zattrs = read_quanta_zarr(zpath, return_meta=True)
    assert attrs == zattrs


def test_quanta_hdf5_lazy_and_chunked(tmp_path, frames):
    path = tmp_path / "data.h5"
    write_quanta_hdf5(path, frames, fps=10.0)
    read_frames = read_quanta_hdf5(path)
    assert isinstance(read_frames, h5py.Dataset)
    np.testing.assert_array_equal(read_frames[3:7], frames[3:7])
    _check_compression(read_frames)
    chunk_nbytes = np.prod(read_frames.chunks) * read_frames.dtype.itemsize
    assert chunk_nbytes <= HDF5_MAX_CHUNK_BYTES


@pytest.mark.parametrize("suffix", [".h5", ".hdf5"])
def test_read_quanta_auto_hdf5(tmp_path, frames, suffix):
    path = tmp_path / f"data{suffix}"
    write_quanta_hdf5(path, frames, T_exp=1.0)
    read_frames, meta = read_quanta_auto(path, load_data=True, return_meta=True)
    np.testing.assert_array_equal(read_frames, frames)
    assert meta == read_quanta_hdf5(path, return_meta=True)[1]
    assert isinstance(read_quanta_auto(path), h5py.Dataset)


def test_quanta_hdf5_missing_frames_raises(tmp_path):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f["data"] = np.zeros(3)
    with pytest.raises(KeyError, match="No frames dataset"):
        read_quanta_hdf5(path)


# --- gated quanta -----------------------------------------------------------
@pytest.fixture
def images():
    return make_gated_images()


@pytest.fixture
def gated_dir_path(tmp_path, images):
    return write_gated_dir(tmp_path / "gated", images)


def test_quanta_gated_hdf5_roundtrip(tmp_path, gated_dir_path, images):
    gd = QuantaGatedDir(gated_dir_path, integration_time_ms=1.0, gate_width_ns=6.0)
    path = tmp_path / "gated.h5"
    write_quanta_gated_hdf5(path, gd, frames_per_chunk=2, attrs={"note": "x"})

    frames, attrs = read_quanta_gated_hdf5(path, return_meta=True)
    assert isinstance(frames, h5py.Dataset)
    assert frames.shape == (N_FRAMES, N_GATE_STEPS, HEIGHT, WIDTH)
    assert frames.chunks == (2, 1, HEIGHT, WIDTH)
    _check_compression(frames)
    np.testing.assert_array_equal(frames[1, 2], images[1, 2])

    loaded = read_quanta_gated_hdf5(path, load_data=True)
    assert isinstance(loaded, np.ndarray)
    np.testing.assert_array_equal(loaded, images)

    zpath = tmp_path / "gated.zarr"
    write_quanta_gated_zarr(zpath, gd, frames_per_chunk=2, attrs={"note": "x"}, n_workers=2)
    assert attrs == dict(zarr.open_array(zpath, mode="r").attrs)
    assert attrs["source_dir"] == str(gated_dir_path)


def test_quanta_gated_hdf5_uint16(tmp_path):
    images16 = make_gated_images(dtype=np.uint16) * 300
    gd = QuantaGatedDir(write_gated_dir(tmp_path / "gated16", images16), image_bit_depth=12)
    path = tmp_path / "gated16.h5"
    write_quanta_gated_hdf5(path, gd)
    frames, attrs = read_quanta_gated_hdf5(path, load_data=True, return_meta=True)
    assert frames.dtype == np.uint16
    np.testing.assert_array_equal(frames, images16)
    assert attrs["image_bit_depth"] == 12


def test_write_quanta_gated_hdf5_rejects_a_path(tmp_path, gated_dir_path):
    out = tmp_path / "nope.h5"
    with pytest.raises(TypeError, match="must be a QuantaGatedDir"):
        write_quanta_gated_hdf5(out, gated_dir_path)
    assert not out.exists()


@pytest.fixture
def gated_zarr_path(tmp_path, gated_dir_path):
    gd = QuantaGatedDir(gated_dir_path, integration_time_ms=1.0, gate_width_ns=6.0)
    path = tmp_path / "gated.zarr"
    write_quanta_gated_zarr(path, gd, attrs={"note": "x"}, n_workers=2)
    return path


def test_quanta_gated_zarr_to_hdf5(tmp_path, gated_zarr_path, images):
    frames, meta = read_quanta_gated_zarr(gated_zarr_path, return_meta=True)
    path = tmp_path / "converted.h5"
    write_quanta_gated_hdf5(path, frames, attrs=meta)

    converted, converted_meta = read_quanta_gated_hdf5(path, load_data=True, return_meta=True)
    np.testing.assert_array_equal(converted, images)
    assert converted_meta == meta


def test_quanta_gated_hdf5_to_zarr(tmp_path, gated_dir_path, images):
    gd = QuantaGatedDir(gated_dir_path, integration_time_ms=1.0, gate_width_ns=6.0)
    h5_path = tmp_path / "gated.h5"
    write_quanta_gated_hdf5(h5_path, gd)

    frames, meta = read_quanta_gated_hdf5(h5_path, return_meta=True)
    path = tmp_path / "converted.zarr"
    write_quanta_gated_zarr(path, frames, attrs=meta, frames_per_chunk=1)

    converted, converted_meta = read_quanta_gated_zarr(path, load_data=True, return_meta=True)
    np.testing.assert_array_equal(converted, images)
    assert converted_meta == meta


@pytest.mark.parametrize("writer, reader, ext", [
    (write_quanta_gated_zarr, read_quanta_gated_zarr, ".zarr"),
    (write_quanta_gated_hdf5, read_quanta_gated_hdf5, ".h5"),
])
def test_quanta_gated_write_array(tmp_path, images, writer, reader, ext):
    path = tmp_path / f"array{ext}"
    writer(path, images, attrs={"integration_time_ms": 1.0, "source_dir": "elsewhere"})
    frames, meta = reader(path, load_data=True, return_meta=True)
    np.testing.assert_array_equal(frames, images)
    assert meta == {
        "n_frames": N_FRAMES,
        "n_gate_steps": N_GATE_STEPS,
        "image_height": HEIGHT,
        "image_width": WIDTH,
        "image_bit_depth": 8,  # storage bit depth, as QuantaGatedDir defaults to
        "integration_time_ms": 1.0,
        "source_dir": "elsewhere",
        "shape": "(n_frames, n_gate_steps, image_height, image_width)",
    }


@pytest.mark.parametrize("writer", [write_quanta_gated_zarr, write_quanta_gated_hdf5])
def test_quanta_gated_write_array_geometry_from_shape(tmp_path, images, writer):
    """Metadata from a bigger acquisition doesn't describe a slice of it."""
    path = tmp_path / "sliced"
    with pytest.warns(UserWarning, match="n_frames=99") as record:
        writer(path, images[:2], attrs={"n_frames": 99, "image_bit_depth": 12})
    assert record[0].filename == __file__
    reader = read_quanta_gated_zarr if writer is write_quanta_gated_zarr else read_quanta_gated_hdf5
    _, meta = reader(path, return_meta=True)
    assert meta["n_frames"] == 2
    assert meta["image_bit_depth"] == 12


@pytest.mark.parametrize("writer", [write_quanta_gated_zarr, write_quanta_gated_hdf5])
def test_quanta_gated_write_rejects_non_4d_and_non_arrays(tmp_path, images, writer):
    with pytest.raises(ValueError, match="4D"):
        writer(tmp_path / "3d", images[0])
    with pytest.raises(TypeError, match="must be a QuantaGatedDir or a 4D"):
        writer(tmp_path / "list", images.tolist())


def test_read_quanta_gated_hdf5_rejects_wrong_ndim(tmp_path, frames):
    path = tmp_path / "quanta.h5"
    write_quanta_hdf5(path, frames, T_exp=1.0)
    with pytest.raises(ValueError, match="expected a 4D"):
        read_quanta_gated_hdf5(path)
