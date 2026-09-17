"""
Tests for lazy (``load_data=False``) reading of ``.bin``/``.mat`` quanta files and
directories of them.

Every lazy index is checked against the same index applied to the ground-truth
``frames`` fixture, so results must match numpy semantics exactly.
"""
import numpy as np
import pytest

from spadlib.io import (
    LazyQuantaFrames,
    read_quanta_auto,
    read_quanta_bin,
    read_quanta_dir,
    read_quanta_mat,
)
from tests.conftest import T, H, W, MAT_KEY, write_bin, write_mat

# frames 0-2 | 3 | 4-9 in the uneven directories, to exercise file boundaries
UNEVEN_SPLITS = (3, 4)

KEYS = [
    0,
    3,
    T - 1,
    -1,
    -T,
    np.int64(4),
    slice(None),
    slice(2, 8),
    slice(3, 4),
    slice(2, 5),
    slice(4, 100),
    slice(-4, None),
    slice(None, None, 2),
    slice(1, None, 3),
    slice(None, None, 100),
    slice(None, None, -1),
    slice(8, 1, -2),
    slice(5, 5),
    slice(8, 2),
    (slice(1, 6), slice(10, 20), slice(30, 50)),
    (slice(None, None, -3), slice(None, None, 7), slice(100, 0, -9)),
    (4, 100),
    (slice(2, 9), 100, 200),
    (7, 100, 200),
    (-2, slice(None), -1),
    Ellipsis,
    (Ellipsis, 5),
    (2, Ellipsis),
    (slice(1, 9), Ellipsis, slice(0, 8)),
    (1, 2, 3, Ellipsis),
]


@pytest.fixture
def uneven_bin_dir(tmp_path, frames):
    d = tmp_path / "uneven_bin_dir"
    d.mkdir()
    for i, part in enumerate(np.split(frames, UNEVEN_SPLITS)):
        write_bin(d / f"RAW{i:05d}.bin", part)
    return d


@pytest.fixture
def uneven_mat_dir(tmp_path, frames):
    d = tmp_path / "uneven_mat_dir"
    d.mkdir()
    for i, part in enumerate(np.split(frames, UNEVEN_SPLITS)):
        write_mat(d / f"part_{i + 1}.mat", part)
    return d


@pytest.fixture
def mat73_path(tmp_path, frames):
    """A MATLAB v7.3 (HDF5) file holding the (H, W, T) volume under MAT_KEY."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data73.mat"
    with h5py.File(path, "w", userblock_size=512) as f:
        f[MAT_KEY] = np.transpose(frames, (1, 2, 0))
    header = b"MATLAB 7.3 MAT-file, HDF5 schema 1.00 .".ljust(116) + b"\x00" * 8 + b"\x00\x02IM"
    with open(path, "r+b") as f:
        f.write(header)
    return path


# (fixture name, reader) for every lazily readable source
SOURCES = {
    "bin": ("bin_path", lambda p, **kw: read_quanta_bin(p, H=H, W=W, **kw)),
    "mat": ("mat_path", read_quanta_mat),
    "mat73": ("mat73_path", read_quanta_mat),
    "bin_dir": ("bin_dir", lambda p, **kw: read_quanta_dir(p, H=H, W=W, **kw)),
    "mat_dir": ("mat_dir", read_quanta_dir),
    "uneven_bin_dir": ("uneven_bin_dir", lambda p, **kw: read_quanta_dir(p, H=H, W=W, **kw)),
    "uneven_mat_dir": ("uneven_mat_dir", read_quanta_dir),
    "auto_bin": ("bin_path", lambda p, **kw: read_quanta_auto(p, H=H, W=W, **kw)),
    "auto_mat": ("mat_path", read_quanta_auto),
    "auto_bin_dir": ("uneven_bin_dir", lambda p, **kw: read_quanta_auto(p, H=H, W=W, **kw)),
    "auto_mat_dir": ("uneven_mat_dir", read_quanta_auto),
}


@pytest.fixture(params=list(SOURCES))
def source(request):
    """(path, reader) for one lazily readable source."""
    fixture_name, reader = SOURCES[request.param]
    return request.getfixturevalue(fixture_name), reader


@pytest.fixture
def lazy(source):
    path, reader = source
    return reader(path)  # load_data defaults to False


def test_lazy_type_and_attributes(lazy):
    assert isinstance(lazy, LazyQuantaFrames)
    assert lazy.shape == (T, H, W)
    assert lazy.ndim == 3
    assert len(lazy) == T
    assert lazy.dtype == np.uint8


def test_lazy_full_read_matches_eager(source, frames):
    path, reader = source
    eager = reader(path, load_data=True)
    assert isinstance(eager, np.ndarray)
    np.testing.assert_array_equal(eager, frames)
    np.testing.assert_array_equal(reader(path, load_data=False)[:], eager)


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_lazy_indexing_matches_numpy(lazy, frames, key):
    out = lazy[key]
    expected = frames[key]
    assert type(out) is type(expected)
    assert np.shape(out) == expected.shape
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


def test_lazy_iteration(lazy, frames):
    np.testing.assert_array_equal(np.stack(list(lazy)), frames)


@pytest.mark.parametrize(
    "key",
    [T, -T - 1, (0, H), (slice(None), 0, -W - 1), (0, 0, 0, 0), (Ellipsis, Ellipsis)],
    ids=repr,
)
def test_lazy_out_of_range_raises_index_error(bin_dir, key):
    lazy = read_quanta_dir(bin_dir, H=H, W=W, load_data=False)
    with pytest.raises(IndexError):
        lazy[key]


@pytest.mark.parametrize(
    "key",
    [[0, 1], np.array([0, 1]), np.zeros(T, dtype=bool), True, None, (0, None), 1.0, "0"],
    ids=repr,
)
def test_lazy_unsupported_index_raises_type_error(bin_dir, key):
    lazy = read_quanta_dir(bin_dir, H=H, W=W, load_data=False)
    with pytest.raises(TypeError):
        lazy[key]


@pytest.mark.parametrize("dir_fixture", ["uneven_bin_dir", "uneven_mat_dir"])
def test_lazy_dir_only_reads_indexed_files(request, frames, dir_fixture):
    """Constructing reads no frame data, and indexing only opens the files it touches."""
    d = request.getfixturevalue(dir_fixture)
    lazy = read_quanta_dir(d, H=H, W=W, load_data=False)
    first_file = sorted(p for p in d.iterdir())[0]  # frames 0-2
    first_file.unlink()

    np.testing.assert_array_equal(lazy[3:], frames[3:])
    np.testing.assert_array_equal(lazy[-1, 5], frames[-1, 5])
    with pytest.raises(FileNotFoundError):
        lazy[2:5]


def test_lazy_bin_dir_garbage_file_raises(bin_dir):
    (bin_dir / "._RAW00000.bin").write_bytes(b"\x00" * 4096)  # macOS AppleDouble dotfile
    with pytest.raises(ValueError):
        read_quanta_dir(bin_dir, H=H, W=W, load_data=False)


def test_lazy_mat_dir_mismatched_frame_shapes_raises(tmp_path, frames):
    d = tmp_path / "mismatched"
    d.mkdir()
    write_mat(d / "part_1.mat", frames[:5])
    write_mat(d / "part_2.mat", frames[5:, :, :100])
    with pytest.raises(ValueError, match="mismatched frame shapes"):
        read_quanta_dir(d, load_data=False)


def test_lazy_mat_preserves_dtype(tmp_path, frames):
    path = tmp_path / "float.mat"
    write_mat(path, frames.astype(np.float32))
    lazy = read_quanta_mat(path, load_data=False)
    assert lazy.dtype == np.float32
    assert lazy[1:3].dtype == np.float32
    np.testing.assert_array_equal(lazy[1:3], frames[1:3])


def test_lazy_bin_nonsquare_shape(tmp_path):
    """.bin frames are rotated on read, so (H, W) come back as (W, H)."""
    h, w = 16, 32
    rot_frames = np.random.default_rng(1).integers(0, 2, size=(4, w, h), dtype=np.uint8)
    path = tmp_path / "nonsquare.bin"
    write_bin(path, rot_frames)
    lazy = read_quanta_bin(path, H=h, W=w, load_data=False)
    assert lazy.shape == (4, w, h)
    np.testing.assert_array_equal(lazy[1:3, 5], rot_frames[1:3, 5])
    np.testing.assert_array_equal(read_quanta_bin(path, H=h, W=w, load_data=True), rot_frames)
