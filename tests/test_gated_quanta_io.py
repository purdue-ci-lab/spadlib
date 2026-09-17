"""
Tests for gated quanta (SPAD512 PNG sequence) reading and writing, plus the
single-array Zarr helper.

A tiny synthetic acquisition (3 frames x 4 gate steps of 8 x 6 images) is written
out as ``IMG*-*.png`` files; every reader/writer is checked against the ground-truth
array, whose values encode (frame, gate step) so ordering errors are caught.
"""
import cv2
import numpy as np
import pytest
import zarr

from spadlib.io import (
    QuantaGatedDir,
    read_quanta_gated_zarr,
    save_arr_to_zarr,
    save_arrs_to_zarr,
    write_quanta_gated_zarr,
)

N_FRAMES, N_GATE_STEPS, HEIGHT, WIDTH = 3, 4, 8, 6


def make_gated_images(n_frames=N_FRAMES, n_gate_steps=N_GATE_STEPS, h=HEIGHT, w=WIDTH, dtype=np.uint8):
    """
    Ground-truth (n_frames, n_gate_steps, h, w) array; pixel values depend on the
    frame and gate step index so misordering is detectable.
    """
    images = np.zeros((n_frames, n_gate_steps, h, w), dtype=dtype)
    for frame_idx in range(n_frames):
        for gate_idx in range(n_gate_steps):
            images[frame_idx, gate_idx] = frame_idx * 10 + gate_idx + 1
    return images


def write_gated_dir(path, images, ext="png"):
    """Write an (n_frames, n_gate_steps, h, w) array as IMG{frame}-{gate}.png files."""
    path.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(images.shape[0]):
        for gate_idx in range(images.shape[1]):
            fpath = path / f"IMG{frame_idx:05d}-{gate_idx:04d}.{ext}"
            assert cv2.imwrite(str(fpath), images[frame_idx, gate_idx])
    return path


@pytest.fixture
def images():
    return make_gated_images()


@pytest.fixture
def gated_dir_path(tmp_path, images):
    return write_gated_dir(tmp_path / "gated", images)


def test_pre_convention_aliases_still_resolve():
    """
    Old names must keep working: they are plain aliases of the renamed objects.
    """
    import spadlib.io as io

    assert io.write_pixel_timeseries_npys is io.write_async_spad_npys
    assert io.write_frames_zarr is io.write_quanta_zarr
    assert io.write_frames_gated_zarr is io.write_quanta_gated_zarr
    assert io.read_frames_gated_zarr is io.read_quanta_gated_zarr


def test_save_arr_to_zarr_roundtrip(tmp_path):
    arr = np.arange(4 * 5 * 6, dtype="uint16").reshape(4, 5, 6)
    path = tmp_path / "arr.zarr"
    save_arr_to_zarr(arr, path, chunks=(2, 5, 6), attrs={"units": "counts", "n": np.int64(7)})

    z = zarr.open_array(path, mode="r")
    assert z.shape == arr.shape
    assert z.dtype == arr.dtype
    assert z.chunks == (2, 5, 6)
    np.testing.assert_array_equal(z[:], arr)
    assert z.attrs["units"] == "counts"
    assert z.attrs["n"] == 7


def test_gated_dir_geometry_and_dtype(gated_dir_path):
    gd = QuantaGatedDir(gated_dir_path)
    assert gd.shape == (N_FRAMES, N_GATE_STEPS, HEIGHT, WIDTH)
    assert len(gd) == N_FRAMES
    assert gd.dtype == np.uint8
    assert gd.image_bit_depth == 8
    assert gd.frame_nbytes == HEIGHT * WIDTH


def test_gated_dir_reads_correct_images(gated_dir_path, images):
    gd = QuantaGatedDir(gated_dir_path)
    np.testing.assert_array_equal(gd.read_image(2, 3), images[2, 3])
    np.testing.assert_array_equal(gd.read_block(), images)
    np.testing.assert_array_equal(gd.read_block(slice(1, 3), slice(0, 2)), images[1:3, 0:2])


def test_gated_dir_stream(gated_dir_path, images):
    gd = QuantaGatedDir(gated_dir_path)
    streamed = list(gd.stream(progress=False))
    assert len(streamed) == N_FRAMES * N_GATE_STEPS
    # frame-major order
    assert [(f, g) for f, g, _ in streamed] == [
        (f, g) for f in range(N_FRAMES) for g in range(N_GATE_STEPS)
    ]
    for frame_idx, gate_idx, img in streamed:
        np.testing.assert_array_equal(img, images[frame_idx, gate_idx])


def test_gated_dir_iter_chunks(gated_dir_path, images):
    gd = QuantaGatedDir(gated_dir_path)
    chunks = list(gd.iter_chunks(frames_per_chunk=2, progress=False))
    assert [(s.start, s.stop, g) for s, g, _ in chunks] == [
        (0, 2, 0), (2, 3, 0), (0, 2, 1), (2, 3, 1),
        (0, 2, 2), (2, 3, 2), (0, 2, 3), (2, 3, 3),
    ]
    for frame_slice, gate_idx, block in chunks:
        np.testing.assert_array_equal(block, images[frame_slice, gate_idx])


def test_gated_dir_default_chunking_respects_byte_cap(gated_dir_path):
    gd = QuantaGatedDir(gated_dir_path)
    # cap of 2 frames' worth of bytes -> 2 frames per chunk
    chunks = list(gd.iter_chunks(max_chunk_bytes=2 * gd.frame_nbytes, progress=False))
    assert max(block.shape[0] for _, _, block in chunks) == 2


def test_gated_dir_metadata(gated_dir_path):
    gd = QuantaGatedDir(
        gated_dir_path,
        image_bit_depth=8,
        image_width=WIDTH,
        integration_time_ms=1.0,
        laser_frequency=40e6,
        n_frames=N_FRAMES,
        n_gate_steps=N_GATE_STEPS,
        gate_step_size_ps=8.0,
        gate_width_ns=6.0,
        gate_offset_ps=100.0,
    )
    assert gd.metadata == {
        "n_frames": N_FRAMES,
        "n_gate_steps": N_GATE_STEPS,
        "image_height": HEIGHT,
        "image_width": WIDTH,
        "image_bit_depth": 8,
        "integration_time_ms": 1.0,
        "laser_frequency": 40e6,
        "gate_step_size_ps": 8.0,
        "gate_width_ns": 6.0,
        "gate_offset_ps": 100.0,
    }
    # unset fields are omitted rather than stored as None
    assert "gate_offset_ps" not in QuantaGatedDir(gated_dir_path).metadata


def test_gated_dir_warns_on_metadata_mismatch(gated_dir_path, caplog):
    with caplog.at_level("WARNING"):
        gd = QuantaGatedDir(gated_dir_path, n_frames=99, image_width=999)
    assert "n_frames=99" in caplog.text
    assert "image_width=999" in caplog.text
    # inferred values win
    assert gd.n_frames == N_FRAMES
    assert gd.image_width == WIDTH


def test_gated_dir_missing_files_read_as_zeros(tmp_path, images, caplog):
    path = write_gated_dir(tmp_path / "gaps", images)
    (path / f"IMG{1:05d}-{2:04d}.png").unlink()
    with caplog.at_level("WARNING"):
        gd = QuantaGatedDir(path)
    assert "1 of 12 images are missing" in caplog.text
    assert gd.shape == (N_FRAMES, N_GATE_STEPS, HEIGHT, WIDTH)
    np.testing.assert_array_equal(gd.read_image(1, 2), np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
    assert (1, 2) not in [(f, g) for f, g, _ in gd.stream(progress=False)]


def test_gated_dir_ignores_non_matching_files(tmp_path, images):
    path = write_gated_dir(tmp_path / "junk", images)
    (path / "._IMG00000-0000.png").write_bytes(b"applederp")
    (path / "notes.txt").write_text("hello")
    gd = QuantaGatedDir(path)
    assert gd.shape == (N_FRAMES, N_GATE_STEPS, HEIGHT, WIDTH)


def test_gated_dir_errors_on_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        QuantaGatedDir(empty)


def test_write_quanta_gated_zarr_roundtrip(tmp_path, gated_dir_path, images):
    gd = QuantaGatedDir(gated_dir_path, integration_time_ms=1.0, gate_width_ns=6.0)
    out = tmp_path / "gated.zarr"
    write_quanta_gated_zarr(out, gd, frames_per_chunk=2, n_workers=2)

    z = zarr.open_array(out, mode="r")
    assert z.shape == (N_FRAMES, N_GATE_STEPS, HEIGHT, WIDTH)
    assert z.chunks == (2, 1, HEIGHT, WIDTH)
    assert z.dtype == np.uint8
    np.testing.assert_array_equal(z[:], images)
    assert z.attrs["integration_time_ms"] == 1.0
    assert z.attrs["gate_width_ns"] == 6.0
    assert z.attrs["n_gate_steps"] == N_GATE_STEPS
    assert z.attrs["shape"] == "(n_frames, n_gate_steps, image_height, image_width)"


def test_read_quanta_gated_zarr(tmp_path, gated_dir_path, images):
    gd = QuantaGatedDir(gated_dir_path, integration_time_ms=1.0, gate_step_size_ps=8.0)
    out = tmp_path / "gated.zarr"
    write_quanta_gated_zarr(out, gd, frames_per_chunk=2, n_workers=2)

    # lazy by default
    frames, meta = read_quanta_gated_zarr(out, return_meta=True)
    assert isinstance(frames, zarr.Array)
    assert frames.shape == (N_FRAMES, N_GATE_STEPS, HEIGHT, WIDTH)
    np.testing.assert_array_equal(frames[1, 2], images[1, 2])

    assert isinstance(meta, dict)
    assert meta["integration_time_ms"] == 1.0
    assert meta["gate_step_size_ps"] == 8.0
    assert meta["n_frames"] == N_FRAMES
    assert meta["image_height"] == HEIGHT
    assert meta["source_dir"] == str(gated_dir_path)

    # eager
    frames_loaded, meta_loaded = read_quanta_gated_zarr(out, load_data=True, return_meta=True)
    assert isinstance(frames_loaded, np.ndarray)
    np.testing.assert_array_equal(frames_loaded, images)
    assert meta_loaded == meta


def test_read_quanta_gated_zarr_rejects_group_and_wrong_ndim(tmp_path):
    group_path = tmp_path / "group.zarr"
    save_arrs_to_zarr({"frames": np.zeros((2, 3, 4, 5), dtype="uint8")}, group_path)
    with pytest.raises(ValueError, match="is a Zarr group"):
        read_quanta_gated_zarr(group_path)

    arr_path = tmp_path / "3d.zarr"
    save_arr_to_zarr(np.zeros((2, 3, 4), dtype="uint8"), arr_path)
    with pytest.raises(ValueError, match="expected a 4D"):
        read_quanta_gated_zarr(arr_path)


def test_write_quanta_gated_zarr_rejects_a_path(tmp_path, gated_dir_path):
    out = tmp_path / "nope.zarr"
    # passing the image directory instead of the reader is the easy mistake to make
    for bad in (gated_dir_path, str(gated_dir_path), None):
        with pytest.raises(TypeError, match="must be a QuantaGatedDir"):
            write_quanta_gated_zarr(out, bad)
    assert not out.exists()


def test_write_quanta_gated_zarr_default_chunking(tmp_path, gated_dir_path):
    gd = QuantaGatedDir(gated_dir_path)
    out = tmp_path / "gated_auto.zarr"
    # tiny cap -> 1 frame per chunk; the gate axis is always chunked at 1
    write_quanta_gated_zarr(out, gd, max_chunk_bytes=1, n_workers=2)
    z = zarr.open_array(out, mode="r")
    assert z.chunks == (1, 1, HEIGHT, WIDTH)

    # huge cap -> capped by n_frames, never more
    out2 = tmp_path / "gated_auto_big.zarr"
    write_quanta_gated_zarr(out2, gd, max_chunk_bytes=10**12, n_workers=2)
    assert zarr.open_array(out2, mode="r").chunks == (N_FRAMES, 1, HEIGHT, WIDTH)


def test_write_quanta_gated_zarr_uint16(tmp_path):
    images16 = make_gated_images(dtype=np.uint16) * 300
    path = write_gated_dir(tmp_path / "gated16", images16)
    gd = QuantaGatedDir(path, image_bit_depth=12)
    assert gd.dtype == np.uint16
    assert gd.image_bit_depth == 12

    out = tmp_path / "gated16.zarr"
    write_quanta_gated_zarr(out, gd, n_workers=2)
    z = zarr.open_array(out, mode="r")
    assert z.dtype == np.uint16
    np.testing.assert_array_equal(z[:], images16)
    assert z.attrs["image_bit_depth"] == 12
