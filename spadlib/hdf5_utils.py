"""
Generic array <-> HDF5 (h5py) helpers shared by the HDF5 readers/writers in
``spadlib.io`` and ``spadlib.video_io``. Datasets are GZIP compressed with the shuffle
filter, and read/written single-threaded.
"""
import contextlib
import json
import logging
import math

import h5py
import numpy as np
from tqdm import tqdm

from spadlib.utils import make_json_serializable


logger = logging.getLogger(__name__)


# HDF5 decompresses a whole chunk on any partial read and by default caches only
# 1 MiB of chunks per dataset, so chunks are kept around that size; bigger chunks would
# be decompressed again on every small (e.g. frame-by-frame) read.
HDF5_MAX_CHUNK_BYTES = 1_048_576
HDF5_COMPRESSION_LEVEL = 5  # default GZIP level (0-9) of the HDF5 writers
# Source data copied per write step: a run of whole chunks about this big, so lazy
# sources (e.g. .cine files) are read in large blocks rather than one chunk at a time.
HDF5_WRITE_BATCH_BYTES = 256_000_000


def _hdf5_compression(level):
    """
    ``create_dataset`` kwargs for GZIP compression at `level` with the shuffle filter.
    Validated here, before any file is opened for writing, so a bad level can't leave an
    existing file truncated.
    """
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)) or not 0 <= level <= 9:
        raise ValueError(f"compression_level must be an integer from 0 to 9, got {level!r}")
    return {"compression": "gzip", "compression_opts": int(level), "shuffle": True}


def _set_hdf5_attrs(obj, attrs):
    """
    Set `attrs` on an h5py file/group/dataset. Values are made JSON-serializable first,
    like the Zarr writers do; None is stored as an empty attribute, and values HDF5 has
    no native type for (e.g. dicts, mixed-type lists) as JSON strings.
    """
    for key, value in attrs.items():
        value = make_json_serializable(value)
        if value is None:
            obj.attrs[key] = h5py.Empty("f")
            continue
        try:
            obj.attrs[key] = value
        except (TypeError, ValueError):
            obj.attrs[key] = json.dumps(value)


def _get_hdf5_attrs(obj):
    """
    Read the attributes of an h5py file/group/dataset as a dict of plain Python values
    (numpy scalars/arrays become Python scalars/lists, empty attributes become None).
    Attributes stored as JSON by :func:`_set_hdf5_attrs` are decoded back, so any
    string attribute holding a JSON object or array is returned decoded.
    """
    attrs = {}
    for key, value in obj.attrs.items():
        if isinstance(value, h5py.Empty):
            value = None
        else:
            value = make_json_serializable(value)
            if isinstance(value, str) and value[:1] in ("{", "["):
                with contextlib.suppress(json.JSONDecodeError):
                    value = json.loads(value)
        attrs[key] = value
    return attrs


def _write_hdf5_dataset(group, name, arr, compression, chunks=None):
    """
    Create a dataset `name` in the h5py `group`, compressed with the `compression`
    kwargs from :func:`_hdf5_compression`, and copy `arr` into it one run of chunks
    (along axis 0) at a time. `arr` can be any array-like with ``shape``, ``dtype`` and
    slicing along axis 0 (e.g. a lazy reader); it is never loaded whole.
    """
    ds = group.create_dataset(
        name,
        shape=arr.shape,
        dtype=arr.dtype,
        # h5py rejects a chunk shape larger than an empty dataset, so let it pick one
        chunks=chunks if math.prod(arr.shape) else None,
        **compression,
    )
    row_nbytes = max(1, math.prod(arr.shape[1:]) * ds.dtype.itemsize)
    step = ds.chunks[0] * max(1, HDF5_WRITE_BATCH_BYTES // (ds.chunks[0] * row_nbytes))
    for start in tqdm(range(0, arr.shape[0], step), desc=f"Writing {name}"):
        ds[start:start + step] = arr[start:start + step]
    return ds


def _save_arrs_to_hdf5(
    data_dict,
    path,
    chunks=None,
    attrs=None,
    compression_level=HDF5_COMPRESSION_LEVEL,
    overwrite=True,
):
    """
    Save multiple arrays as datasets at the root of an HDF5 file, with `attrs` set on
    the root. HDF5 counterpart of :func:`spadlib.io.save_arrs_to_zarr`.

    Args:
        data_dict (dict): mapping of name->array to save (see :func:`_write_hdf5_dataset`).
        path (str or Path): path to the output HDF5 file.
        chunks (dict or None): mapping of name->chunk tuple; h5py picks the chunk
            shape for arrays not in it.
        attrs (dict or None): attribute metadata to set on the file's root group.
        compression_level (int): GZIP compression level, 0-9.
        overwrite (bool): if True, overwrite an existing file at `path`.
    """
    chunks = chunks or {}
    compression = _hdf5_compression(compression_level)
    with h5py.File(path, "w" if overwrite else "w-") as f:
        _set_hdf5_attrs(f, attrs or {})
        for key, arr in data_dict.items():
            _write_hdf5_dataset(f, key, arr, compression, chunks.get(key))
    logger.info(f"Saved arrays to HDF5 at: {path}")


def _open_hdf5_frames(path):
    """
    Open the HDF5 file at `path` read-only and return its ``frames`` dataset. The file
    stays open for as long as the dataset is referenced.
    """
    f = h5py.File(path, "r")
    if "frames" not in f:
        keys = sorted(f.keys())
        f.close()
        raise KeyError(f"No frames dataset found in {path}. Available datasets: {keys}")
    return f["frames"]


def _read_hdf5_frames(path, load_data=False, return_meta=False):
    """
    Read the ``frames`` dataset at the root of an HDF5 file, and optionally the root
    attributes; see :func:`spadlib.io.read_quanta_hdf5`.
    """
    frames = _open_hdf5_frames(path)
    attrs = _get_hdf5_attrs(frames.file)
    if load_data:
        f = frames.file
        frames = frames[:]
        f.close()
    return (frames, attrs) if return_meta else frames
