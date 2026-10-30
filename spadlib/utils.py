"""
Miscellaneous utility helpers: temporary memmaps, macOS dotfile cleanup, JSON
serialization, natural sorting, number formatting, and colormap conversions.
"""
import atexit
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import weakref
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def dot_clean_dir(path):
    """
    Run macOS `dot_clean` on a directory to remove `._*` AppleDouble dotfiles,
    or things like .DS_store, which otherwise corrupt readers that glob a
    directory (e.g. Zarr stores).

    `dot_clean` only exists on macOS; on other platforms this is a no-op that
    returns None (no equivalent cleanup is performed yet).
    """
    path = Path(path)
    if sys.platform != "darwin":
        logger.debug(f"Skipping dot_clean on {path}: not available on platform {sys.platform!r}.")
        return None
    result = subprocess.run(
        ["dot_clean", "-mn", str(path.absolute())],
        capture_output=True, text=True, check=True
    )
    return result


def _ignore_dotfiles(paths):
    """
    Drop macOS dotfiles (``._*`` AppleDouble files, ``.DS_Store``) from a directory
    listing, warning about any that were found.
    """
    kept, dotfiles = [], []
    for p in paths:
        (dotfiles if p.name.startswith(".") else kept).append(p)
    if dotfiles:
        names = ", ".join(sorted(p.name for p in dotfiles)[:5])
        # dot_clean only exists on macOS, so only suggest it there
        hint = " Run `dot_clean` there to remove them." if sys.platform == "darwin" else ""
        logger.warning(
            f"Ignoring {len(dotfiles)} macOS dotfile(s) in {dotfiles[0].parent}: {names}"
            f"{'...' if len(dotfiles) > 5 else ''}.{hint}"
        )
    return kept


def _frames_per_chunk(n_frames, frame_nbytes, max_chunk_bytes):
    """
    Number of frames per chunk such that a chunk stays under `max_chunk_bytes`
    (at least 1 frame, at most all of them).
    """
    return int(min(max(1, max_chunk_bytes // frame_nbytes), n_frames))


def _index_frames(key, shape, read, name):
    """
    Index lazily read frames of `shape` with numpy semantics. Only integer and slice
    indexing (optionally with ``...``) is supported; the result is a numpy array.

    Args:
        key: the index.
        shape (tuple): shape of the frames, frames along axis 0.
        read (callable): ``read(t_range, spatial_key)`` returns the frames at the
            increasing frame indices `t_range` (a range), indexed along the remaining
            axes by `spatial_key` (whose first element is a full slice).
        name (str): class name for error messages.
    """
    ndim = len(shape)
    if not isinstance(key, tuple):
        key = (key,)
    for k in key:
        if not (k is Ellipsis or isinstance(k, slice) or (isinstance(k, (int, np.integer)) and not isinstance(k, bool))):
            raise TypeError(f"{name} only supports integer and slice indexing, got {k!r}")
    n_ellipsis = sum(k is Ellipsis for k in key)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if n_ellipsis:
        i = next(i for i, k in enumerate(key) if k is Ellipsis)
        key = key[:i] + (slice(None),) * (ndim - len(key) + 1) + key[i + 1:]
    if len(key) > ndim:
        raise IndexError(f"too many indices for array: array is {ndim}-dimensional, but {len(key)} were indexed")
    spatial_key = (slice(None), *key[1:])

    t_idx = range(shape[0])[key[0]]  # int (IndexError if out of range) or range
    is_int = not isinstance(t_idx, range)
    t_range = range(t_idx, t_idx + 1) if is_int else t_idx
    # read in increasing frame order, then flip back
    reverse = t_range.step < 0
    if reverse:
        t_range = t_range[::-1]

    frames = read(t_range, spatial_key)
    if reverse:
        frames = frames[::-1]
    if not is_int:
        return frames
    # like numpy, an all-integer index with an ellipsis gives a 0-d array, not a scalar
    return frames[0, ...] if n_ellipsis else frames[0]


def is_json_serializable(x):
    """
    Return True if `x` can be serialized by `json.dumps`, False otherwise.
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def make_json_serializable(value):
    """Convert a value to a JSON-serializable format."""

    # Numpy scalar / 0-d array (singleton)
    if isinstance(value, np.generic):
        return value.item()  # converts to native Python scalar

    # Numpy array
    if isinstance(value, np.ndarray):
        return value.tolist()  # converts to nested Python list

    # Recursively handle dicts
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}

    # Recursively handle lists/tuples
    if isinstance(value, (list, tuple)):
        converted = [make_json_serializable(v) for v in value]
        return converted if isinstance(value, list) else tuple(converted)

    return value  # already serializable (str, int, float, bool, None, etc.)


def natural_sort_key(s: str):
    """Splits string into text and integer chunks for natural ordering."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def format_number_unit(value, decimals=2):
    """
    Format a number to a human-friendly string using suffixes up to G (giga, 10^9).
    Examples:
        123        -> "123"
        1234       -> "1.23K"
        1500000    -> "1.5M"
        -2500000000-> "-2.5G"

    Parameters
        value: int or float-like value to format
        decimals: max number of decimal places for the scaled value (default 2)

    Returns
        A string with the value scaled and suffixed appropriately.
    """
    try:
        v = float(value)
    except Exception:
        return str(value)

    if not math.isfinite(v):
        return str(value)

    sign = "-" if v < 0 else ""
    v_abs = abs(v)

    # handle zero explicitly
    if v_abs == 0:
        return "0"

    # determine exponent in steps of 1000 (0 -> "", 1 -> K, 2 -> M, 3 -> G)
    exp = int(math.floor(math.log10(v_abs) / 3)) if v_abs >= 1 else 0
    exp = max(0, min(exp, 3))  # clamp to available hard-coded units

    # hard-coded units via conditionals
    if exp == 0:
        unit = ""
    elif exp == 1:
        unit = "K"
    elif exp == 2:
        unit = "M"
    elif exp == 3:
        unit = "G"
    else:
        unit = "G"

    scaled = v_abs / (1000 ** exp)
    fmt = f"{scaled:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{sign}{fmt}{unit}"


def vals_to_cmap(vals, cmap="viridis", vmin=None, vmax=None):
    """Map a 1D array of values to RGBA colors via a matplotlib colormap."""
    cmap_fn = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        vmax = np.max(vals)
    vals = np.interp(vals, (vmin, vmax), (0, 1))
    colors = cmap_fn(vals)
    return colors


def to_rgba_str(rgbas):
    """Convert array of RGBA values (0-255) to list of "rgba(r,g,b,a)" strings."""
    rgba_strs = []
    for rgba in rgbas:
        r, g, b, a = rgba
        rgba_strs.append(f"rgba({r},{g},{b},{a / 255:.2f})")  # a scaled to [0,1]
    return rgba_strs
