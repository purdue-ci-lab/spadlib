"""
Miscellaneous utility helpers: temporary memmaps, JSON serialization, natural
sorting, number formatting, and colormap conversions.
"""
import atexit
import json
import math
import os
import re
import subprocess
import tempfile
import weakref
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class TempMemmap:
    """
    Temporary np.memmap wrapper that cleans up automatically.
    """

    def __init__(self, shape, dtype=np.float32, mode="w+"):
        # Create a real temporary filename that np.memmap can use
        tmp = tempfile.NamedTemporaryFile(delete=False)
        self.filename = tmp.name
        tmp.close()  # close file handle so memmap can reopen it

        # Create the memory-mapped array
        self.data = np.memmap(self.filename, dtype=dtype, mode=mode, shape=shape)

        # Register cleanup handlers
        atexit.register(self._cleanup)
        # weakref ensures cleanup even if the object is deleted before interpreter exit
        self._finalizer = weakref.finalize(self, TempMemmap._delete_file, self.filename)

    def _cleanup(self):
        """Flush and delete on normal interpreter exit."""
        if hasattr(self, "data"):
            try:
                self.data.flush()
                self.data._mmap.close()
            except Exception:
                pass
        TempMemmap._delete_file(self.filename)

    @staticmethod
    def _delete_file(fname):
        """Safely delete file if it exists."""
        try:
            if os.path.exists(fname):
                os.remove(fname)
        except Exception:
            pass

    def close(self):
        """Manually close and remove file."""
        self._cleanup()
        self._finalizer.detach()  # prevent double cleanup

    def __enter__(self):
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def dot_clean_dir(path):
    """
    Run macOS `dot_clean` on a directory to remove `._*` AppleDouble dotfiles,
    which otherwise corrupt readers that glob a directory (e.g. Zarr stores).
    """
    path = Path(path)
    result = subprocess.run(
        ["dot_clean", "-mn", str(path.absolute())],
        capture_output=True, text=True, check=True
    )
    return result


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
