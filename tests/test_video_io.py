"""
Tests for the Phantom ``.cine`` reader and the HDF5 video reader/writer.

Tiny synthetic ``.cine`` files are built from pycine's own header structures, in both
supported image formats (uncompressed 16-bit and packed 10-bit). Frames are checked
against pycine's decoder, :func:`pycine.raw.create_raw_array`.
"""
import ctypes
import struct

import h5py
import numpy as np
import pytest
from pycine import cine
from pycine.file import read_header
from pycine.raw import create_raw_array

import spadlib.hdf5_utils as hdf5_utils
from spadlib.hdf5_utils import HDF5_MAX_CHUNK_BYTES
from spadlib.video_io import (
    LazyCineFrames,
    correct_hotpixels,
    read_video_cine,
    read_video_hdf5,
    write_video_hdf5,
)
from tests.test_quanta_lazy_io import KEYS, T

H, W = 240, 320  # non-square, so flips/transposes are caught; KEYS index up to 200
FPS = 1000


def _pack_10bit(vals):
    """Pack 10-bit values 4 pixels -> 5 bytes, the inverse of pycine's unpack_10bit."""
    v = vals.reshape(-1, 4).astype(np.uint16)
    packed = np.empty((len(v), 5), dtype=np.uint8)
    packed[:, 0] = v[:, 0] >> 2
    packed[:, 1] = ((v[:, 0] & 0b11) << 6) | (v[:, 1] >> 4)
    packed[:, 2] = ((v[:, 1] & 0b1111) << 4) | (v[:, 2] >> 6)
    packed[:, 3] = ((v[:, 2] & 0b111111) << 2) | (v[:, 3] >> 8)
    packed[:, 4] = v[:, 3] & 0xFF
    return packed.tobytes()


def write_cine(path, packed, gaps=False, t=T, bit_count=16):
    """
    Write a synthetic .cine file of `t` random frames. With `gaps`, junk bytes are put
    after every other frame, so frames are not stored back to back.
    """
    rng = np.random.default_rng(0)
    if packed:
        images = [_pack_10bit(rng.integers(0, 2**10, H * W)) for _ in range(t)]
    else:
        # uncompressed images are stored bottom-up
        images = [rng.integers(0, 2**12, (H, W), dtype=np.uint16)[::-1].tobytes() for _ in range(t)]

    setup = cine.SETUP()
    setup.FrameRate = FPS
    setup.RealBPP = 12
    setup.BlackLevel = 64
    setup.WhiteLevel = 4000
    setup.Description = b"synthetic"
    setup.BinName[0].value = b"bin0"

    header = cine.CINEFILEHEADER()
    header.Type = 18755  # "CI"
    header.Headersize = ctypes.sizeof(cine.CINEFILEHEADER)
    header.Version = 1
    header.ImageCount = t
    header.OffImageHeader = header.Headersize
    header.OffSetup = header.OffImageHeader + ctypes.sizeof(cine.BITMAPINFOHEADER)
    header.OffImageOffsets = header.OffSetup + ctypes.sizeof(setup)

    bih = cine.BITMAPINFOHEADER()
    bih.biSize = ctypes.sizeof(bih)
    bih.biWidth, bih.biHeight = W, H
    bih.biPlanes = 1
    bih.biBitCount = bit_count
    bih.biCompression = 256 if packed else 0
    bih.biSizeImage = len(images[0])

    frame_data, offsets = [], []
    pos = header.OffImageOffsets + 8 * t
    for i, image in enumerate(images):
        offsets.append(pos)
        frame = struct.pack("<II", 8, len(image)) + image  # 8-byte annotation, then the image
        if gaps and i % 2:
            frame += b"\xab" * 16
        frame_data.append(frame)
        pos += len(frame)

    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(bytes(bih))
        f.write(bytes(setup))
        f.write(struct.pack(f"<{t}q", *offsets))
        f.write(b"".join(frame_data))
    return path


def pycine_frames(path):
    """Every frame of `path`, decoded by pycine."""
    header = read_header(path)
    frames = []
    with open(path, "rb") as f:
        for offset in header["pImage"]:
            f.seek(offset)
            annotation_size = struct.unpack("<I", f.read(4))[0]
            f.seek(offset + annotation_size - 4)
            image_size = struct.unpack("<I", f.read(4))[0]
            frames.append(create_raw_array(f.read(image_size), header))
    return np.stack(frames)


CINE_KINDS = {
    "uncompressed": {"packed": False},
    "packed10": {"packed": True},
    "uncompressed_gaps": {"packed": False, "gaps": True},
    "packed10_gaps": {"packed": True, "gaps": True},
}


@pytest.fixture(params=list(CINE_KINDS))
def cine_path(request, tmp_path):
    return write_cine(tmp_path / "video.cine", **CINE_KINDS[request.param])


@pytest.fixture
def cine_frames(cine_path):
    """Ground truth: the frames of `cine_path` as decoded by pycine."""
    return pycine_frames(cine_path)


# --- .cine reading -------------------------------------------------------
def test_cine_lazy_type_and_attributes(cine_path):
    frames = read_video_cine(cine_path)
    assert isinstance(frames, LazyCineFrames)
    assert frames.shape == (T, H, W)
    assert frames.ndim == 3
    assert len(frames) == T
    assert frames.dtype == np.uint16


def test_cine_full_read_matches_pycine(cine_path, cine_frames):
    eager = read_video_cine(cine_path, load_data=True)
    assert isinstance(eager, np.ndarray)
    np.testing.assert_array_equal(eager, cine_frames)
    np.testing.assert_array_equal(read_video_cine(cine_path)[:], cine_frames)


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_cine_indexing_matches_numpy(cine_path, cine_frames, key):
    # small chunks, so reads span several chunks and threads
    frames = LazyCineFrames(cine_path, chunk_frames=3, n_workers=2)
    out = frames[key]
    expected = cine_frames[key]
    assert type(out) is type(expected)
    assert np.shape(out) == expected.shape
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("key", [[0, 1], None, 1.0, (0, None)], ids=repr)
def test_cine_unsupported_index_raises_type_error(tmp_path, key):
    frames = read_video_cine(write_cine(tmp_path / "video.cine", packed=True))
    with pytest.raises(TypeError):
        frames[key]


@pytest.mark.parametrize("packed, bit_depth", [(False, 12), (True, 12)])
def test_cine_meta_includes_full_header(tmp_path, packed, bit_depth):
    path = write_cine(tmp_path / "video.cine", packed=packed)
    frames, meta = read_video_cine(path, return_meta=True)
    assert isinstance(frames, LazyCineFrames)
    assert {k: meta[k] for k in ("fps", "T", "H", "W", "shape", "bit_depth")} == {
        "fps": FPS, "T": T, "H": H, "W": W, "shape": "(T, H, W)", "bit_depth": bit_depth,
    }
    for key, struct_type in [
        ("cinefileheader", cine.CINEFILEHEADER),
        ("bitmapinfoheader", cine.BITMAPINFOHEADER),
        ("setup", cine.SETUP),
    ]:
        assert list(meta[key]) == [field[0] for field in struct_type._fields_]
    assert "pImage" not in meta
    assert meta["cinefileheader"]["ImageCount"] == T
    assert meta["cinefileheader"]["TriggerTime"] == {"fractions": 0, "seconds": 0}  # nested struct
    assert meta["setup"]["Description"] == "synthetic"
    assert meta["setup"]["BinName"][:2] == ["bin0", ""]  # nested char arrays become strings
    assert meta["setup"]["WhiteLevel"] == 4000


def test_cine_unsupported_format_raises(tmp_path):
    path = write_cine(tmp_path / "video.cine", packed=False, bit_count=8)
    with pytest.raises(NotImplementedError, match="biBitCount=8"):
        read_video_cine(path)


# --- HDF5 video ----------------------------------------------------------
def test_cine_to_hdf5_roundtrip(tmp_path, cine_path, cine_frames):
    frames, meta = read_video_cine(cine_path, return_meta=True)
    path = tmp_path / "video.h5"
    write_video_hdf5(path, frames[2:8], attrs=meta)
    read_frames, attrs = read_video_hdf5(path, load_data=True, return_meta=True)
    np.testing.assert_array_equal(read_frames, cine_frames[2:8])
    # geometry is the written slice's, everything else is the .cine metadata
    assert {k: attrs[k] for k in ("T", "H", "W", "shape")} == {"T": 6, "H": H, "W": W, "shape": "(T, H, W)"}
    assert {k: v for k, v in attrs.items() if k != "T"} == {k: v for k, v in meta.items() if k != "T"}


def test_video_hdf5_lazy_and_chunked(tmp_path):
    frames = np.random.default_rng(0).integers(0, 2**12, (T, H, W), dtype=np.uint16)
    path = tmp_path / "video.h5"
    write_video_hdf5(path, frames, fps=30.0, compression_level=1)
    ds = read_video_hdf5(path)
    assert isinstance(ds, h5py.Dataset)
    np.testing.assert_array_equal(ds[3:7], frames[3:7])
    assert ds.compression == "gzip" and ds.compression_opts == 1 and ds.shuffle
    assert ds.chunks == (HDF5_MAX_CHUNK_BYTES // (H * W * 2), H, W)


def test_video_hdf5_rgb_and_fps(tmp_path):
    frames = np.random.default_rng(0).integers(0, 256, (T, 8, 12, 3), dtype=np.uint8)
    path = tmp_path / "video.h5"
    write_video_hdf5(path, frames, fps=24.0, attrs={"fps": 99.0, "source": "camera"})
    read_frames, attrs = read_video_hdf5(path, load_data=True, return_meta=True)
    np.testing.assert_array_equal(read_frames, frames)
    assert attrs == {
        "fps": 24.0,  # the fps argument wins over attrs
        "source": "camera",
        "T": T, "H": 8, "W": 12, "C": 3, "shape": "(T, H, W, C)",
    }


def test_video_hdf5_fps_defaults(tmp_path):
    frames = np.zeros((T, 4, 4), dtype=np.uint8)
    write_video_hdf5(tmp_path / "a.h5", frames)
    write_video_hdf5(tmp_path / "b.h5", frames, attrs={"fps": 50.0})
    assert read_video_hdf5(tmp_path / "a.h5", return_meta=True)[1]["fps"] is None
    assert read_video_hdf5(tmp_path / "b.h5", return_meta=True)[1]["fps"] == 50.0


def test_video_hdf5_writes_lazy_source_in_batches(tmp_path, monkeypatch, cine_path, cine_frames):
    """A lazy source is copied a run of whole chunks at a time, never read whole."""
    monkeypatch.setattr(hdf5_utils, "HDF5_WRITE_BATCH_BYTES", 1)  # one chunk per step
    frames = read_video_cine(cine_path)
    reads = []
    read = frames._read
    frames._read = lambda t_range, spatial_key: reads.append(len(t_range)) or read(t_range, spatial_key)
    write_video_hdf5(tmp_path / "video.h5", frames)
    frames_per_chunk = HDF5_MAX_CHUNK_BYTES // (H * W * 2)
    assert reads == [frames_per_chunk, T - frames_per_chunk]
    np.testing.assert_array_equal(read_video_hdf5(tmp_path / "video.h5", load_data=True), cine_frames)


@pytest.mark.parametrize("shape", [(T,), (T, 4), (T, 2, 2, 2, 2)])
def test_video_hdf5_rejects_bad_shape(tmp_path, shape):
    with pytest.raises(ValueError, match="frames must be"):
        write_video_hdf5(tmp_path / "video.h5", np.zeros(shape, dtype=np.uint8))


# --- hot pixel correction ------------------------------------------------
HOT = [(5, 7), (20, 30), (31, 0), (12, 13)]


def make_hot_frames(bayer=False):
    """
    (T, 32, 40) uint16 noisy flat frames with fixed hot pixels at HOT, plus a bright
    spot that moves every frame (which must not be flagged). With `bayer`, the four
    2x2 mosaic planes have different levels.
    """
    rng = np.random.default_rng(0)
    frames = rng.normal(1000, 10, (T, 32, 40))
    if bayer:
        frames[:, 0::2, 1::2] += 500
        frames[:, 1::2, 0::2] += 1000
        frames[:, 1::2, 1::2] += 1500
    clean = frames.copy()
    for t in range(T):
        frames[t, 3 * t, 2 * t] += 3000  # moving highlight
    for y, x in HOT:
        frames[:, y, x] += 3000
    return np.rint(frames).astype(np.uint16), np.rint(clean).astype(np.uint16)


def _expected_mask(shape):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(zip(*HOT))] = True
    return mask


@pytest.mark.parametrize("bayer", [False, True])
def test_correct_hotpixels_finds_and_replaces_hot_pixels(bayer):
    frames, clean = make_hot_frames(bayer)
    out, mask = correct_hotpixels(frames, bayer=bayer, return_mask=True)
    np.testing.assert_array_equal(mask, _expected_mask(frames.shape[1:]))
    assert out.dtype == frames.dtype
    np.testing.assert_array_equal(out[:, ~mask], frames[:, ~mask])  # only hot pixels change
    # replaced by neighbors from the same (Bayer) plane, so back at that plane's level
    for y, x in HOT:
        assert np.all(np.abs(out[:, y, x].astype(int) - clean[:, y, x]) < 60)
    np.testing.assert_array_equal(correct_hotpixels(frames, bayer=bayer), out)


def test_correct_hotpixels_single_frame():
    frames, _ = make_hot_frames()
    out, mask = correct_hotpixels(frames[0], return_mask=True)
    assert out.shape == frames.shape[1:]
    np.testing.assert_array_equal(out, correct_hotpixels(frames[:1])[0])
