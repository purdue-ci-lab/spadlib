"""
Shared fixtures for spadlib tests.

Builds tiny dummy quanta datasets (10 x 512 x 512 binary frames, every frame the
same pattern) in each on-disk format spadlib can read:

- ``.zarr``   : written with the library's own ``write_frames_zarr``.
- ``.bin``    : SPAD512 packed-bits format, synthesized by inverting ``read_quanta_bin``.
- ``.mat``    : MATLAB v7 volume stored as (H, W, T) under the single key ``OUTPUT``.
- dir of bin  : two ``RAW*.bin`` files, 5 frames each.
- dir of mat  : two ``.mat`` files, 5 frames each.

The ``frames`` fixture is the ground truth every reader is checked against.
"""
import numpy as np
import pytest

from spadlib.io import write_frames_zarr

T, H, W = 10, 512, 512
MAT_KEY = "OUTPUT"


def make_pattern_frames(t=T, h=H, w=W):
    """
    Deterministic (T, H, W) uint8 binary frames.
    """
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 2, size=(t, h, w), dtype=np.uint8)
    return pattern


def write_bin(path, frames):
    """
    Write frames as a SPAD512 packed-bits .bin file such that
    ``read_quanta_bin`` reconstructs ``frames`` exactly.

    Inverts the reader: read does ``unpackbits`` then ``rot90(k=1)``, so we
    apply ``rot90(k=-1)`` then ``packbits``.
    """
    unpacked = np.rot90(frames, k=-1, axes=(1, 2)).astype(np.uint8)
    packed = np.packbits(unpacked, axis=2)  # (T, H, W // 8)
    path.write_bytes(packed.tobytes())


def write_mat(path, frames, key=MAT_KEY):
    """Write frames as a MATLAB v7 (H, W, T) volume under ``key``."""
    import scipy.io

    arr = np.transpose(frames, (1, 2, 0))  # (T, H, W) -> (H, W, T)
    scipy.io.savemat(str(path), {key: arr})


@pytest.fixture
def frames():
    """Ground-truth (T, H, W) uint8 binary frames."""
    return make_pattern_frames()


@pytest.fixture
def zarr_path(tmp_path, frames):
    path = tmp_path / "data.zarr"
    write_frames_zarr(path, frames, T_exp=1.0)
    return path


@pytest.fixture
def bin_path(tmp_path, frames):
    path = tmp_path / "data.bin"
    write_bin(path, frames)
    return path


@pytest.fixture
def mat_path(tmp_path, frames):
    path = tmp_path / "data.mat"
    write_mat(path, frames)
    return path


@pytest.fixture
def bin_dir(tmp_path, frames):
    d = tmp_path / "bin_dir"
    d.mkdir()
    write_bin(d / "RAW00000.bin", frames[:5])
    write_bin(d / "RAW00001.bin", frames[5:])
    return d


@pytest.fixture
def mat_dir(tmp_path, frames):
    d = tmp_path / "mat_dir"
    d.mkdir()
    write_mat(d / "part_1.mat", frames[:5])
    write_mat(d / "part_2.mat", frames[5:])
    return d
