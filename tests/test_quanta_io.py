"""
Roundtrip / sanity tests for reading and writing quanta (synchronous binary
frame) data. Continuous/async SPAD event data is out of scope here.

Every reader is checked against the ground-truth ``frames`` fixture: correct
shape, dtype, and exact pixel values (which also verifies frame orientation and
ordering).
"""
import logging

import numpy as np
import pytest
import zarr

import spadlib.io as io

from spadlib.io import (
    read_quanta_auto,
    read_quanta_bin,
    read_quanta_dir,
    read_quanta_mat,
    read_quanta_zarr,
)
from tests.conftest import T, H, W, MAT_KEY


def _check(read_frames, expected):
    """Shared assertions: shape, binary dtype/values, exact equality."""
    read_frames = np.asarray(read_frames)
    assert read_frames.shape == expected.shape == (T, H, W)
    assert read_frames.dtype == np.uint8
    assert set(np.unique(read_frames)).issubset({0, 1})
    np.testing.assert_array_equal(read_frames, expected)


# --- single-file formats --------------------------------------------------
def test_zarr_roundtrip(zarr_path, frames):
    read_frames, attrs = read_quanta_zarr(zarr_path, load_data=True, return_meta=True)
    _check(read_frames, frames)
    assert attrs["T"] == T and attrs["H"] == H and attrs["W"] == W


def test_zarr_defaults_to_lazy_frames_only(zarr_path, frames):
    read_frames = read_quanta_zarr(zarr_path)
    assert isinstance(read_frames, zarr.Array)
    _check(read_frames[:], frames)


def test_zarr_stores_frames_only(zarr_path):
    """Event coordinates are no longer written alongside the frames."""
    assert set(zarr.open_group(str(zarr_path), mode="r").array_keys()) == {"frames"}


def test_bin_roundtrip(bin_path, frames):
    _check(read_quanta_bin(bin_path, H=H, W=W, load_data=True), frames)


def test_mat_roundtrip_autokey(mat_path, frames):
    _check(read_quanta_mat(mat_path, load_data=True), frames)


def test_mat_roundtrip_explicit_key(mat_path, frames):
    _check(read_quanta_mat(mat_path, key=MAT_KEY, load_data=True), frames)


# --- directory formats ----------------------------------------------------
def test_bin_dir_roundtrip(bin_dir, frames):
    _check(read_quanta_dir(bin_dir, H=H, W=W, load_data=True), frames)


def test_mat_dir_roundtrip(mat_dir, frames):
    _check(read_quanta_dir(mat_dir, load_data=True), frames)


# --- macOS dotfiles -------------------------------------------------------
def _add_dotfiles(d, names):
    """Drop AppleDouble-style junk into a directory the way macOS does."""
    for name in names:
        (d / name).write_bytes(b"\x00\x05\x16\x07" * 16)


def test_bin_dir_ignores_dotfiles(bin_dir, frames, caplog):
    _add_dotfiles(bin_dir, ["._RAW00000.bin", "._RAW00001.bin", ".DS_Store"])
    with caplog.at_level(logging.WARNING):
        _check(read_quanta_dir(bin_dir, H=H, W=W, load_data=True), frames)
    # .DS_Store never matches the *.bin glob, so only the AppleDouble files are reported
    assert "Ignoring 2 macOS dotfile(s)" in caplog.text


def test_mat_dir_ignores_dotfiles(mat_dir, frames, caplog):
    _add_dotfiles(mat_dir, ["._part_1.mat", "._info.json"])
    with caplog.at_level(logging.WARNING):
        _check(read_quanta_dir(mat_dir, load_data=True), frames)
    assert "Ignoring 1 macOS dotfile(s)" in caplog.text  # .mat and .json listed separately
    assert "Could not parse metadata file" not in caplog.text


@pytest.mark.parametrize("platform, expect_hint", [("darwin", True), ("linux", False)])
def test_dotfile_warning_only_suggests_dot_clean_on_macos(bin_dir, monkeypatch, caplog, platform, expect_hint):
    """`dot_clean` is a macOS-only command."""
    monkeypatch.setattr(io.sys, "platform", platform)
    _add_dotfiles(bin_dir, ["._RAW00000.bin"])
    with caplog.at_level(logging.WARNING):
        read_quanta_dir(bin_dir, H=H, W=W)
    assert ("dot_clean" in caplog.text) is expect_hint


# --- auto-detection dispatch ----------------------------------------------
def test_auto_zarr(zarr_path, frames):
    read_frames, meta = read_quanta_auto(zarr_path, load_data=True, return_meta=True)
    _check(read_frames, frames)
    assert meta["T"] == T


def test_auto_bin(bin_path, frames):
    read_frames, meta = read_quanta_auto(bin_path, H=H, W=W, load_data=True, return_meta=True)
    _check(read_frames, frames)
    assert meta is None


def test_auto_mat(mat_path, frames):
    read_frames, meta = read_quanta_auto(mat_path, load_data=True, return_meta=True)
    _check(read_frames, frames)
    assert meta is None


def test_auto_bin_dir(bin_dir, frames):
    _check(read_quanta_auto(bin_dir, H=H, W=W, load_data=True), frames)


def test_auto_mat_dir(mat_dir, frames):
    _check(read_quanta_auto(mat_dir, load_data=True), frames)
