"""
I/O for SPAD/quanta data plus conversions between the three representations used
throughout spadlib:

- ``frames``           : (T, H, W) synchronous binary/count quanta frames
- ``events``           : asynchronous continuous (t, y, x) event coordinates
- ``pixel_timeseries`` : H x W list-of-lists of per-pixel timestamp arrays

Also includes generic array <-> Zarr helpers used by the SPAD writers.
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import warnings

import cv2
import numpy as np
import zarr
from tqdm import tqdm

from spadlib.processing import correct_hotpixels_conv, thin_events_uniform, thin_frames_uniform
from spadlib.utils import dot_clean_dir, is_json_serializable, make_json_serializable, natural_sort_key

try:
    import cupy as cp
except ImportError:
    cp = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic array <-> Zarr helpers
# ---------------------------------------------------------------------------
def save_arrs_to_zarr(
    data_dict,
    zarr_path,
    chunks=None,
    attrs=None,
    compressor_dict=None,
    n_workers=8,
    overwrite=True,
):
    """
    Save multiple numpy arrays into a single Zarr group concurrently.
    Uses Group.create_array (newer zarr API).

    Args:
        data_dict : dict
            Mapping of name->numpy array to save.
        zarr_path : path-like
            Path to output Zarr directory (e.g., "freqinfo.zarr").
        chunks : dict or None
            - If dict: mapping name->chunk tuple
            - If None: auto applied to all
        attrs : dict or None
            Global attribute metadata to set on the Zarr group.
        compressor_dict: dict of codec or None
            Codec for compression (default: whatever auto is in zarr).
        n_workers : int
            Number of worker threads to write large arrays concurrently.
        overwrite : bool
            If True, overwrite an existing zarr at zarr_path.
    """
    zarr_path = Path(zarr_path)
    if compressor_dict is None:
        compressor_dict = {}
        for key, arr in data_dict.items():
            # if it's uint8 or boolean, use bitshuffle; otherwise, use shuffle
            if arr.itemsize == 1:
                compressor_dict[key] = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
            else:
                compressor_dict[key] = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    if chunks is None:
        chunks = {}

    try:
        root = zarr.open_group(zarr_path, mode="w" if overwrite else "w-")
    except FileNotFoundError:
        dot_clean_dir(zarr_path)
        root = zarr.open_group(zarr_path, mode="w" if overwrite else "w-")

    if attrs is not None:
        for key, value in attrs.items():
            root.attrs[key] = make_json_serializable(value)

    for key, arr in data_dict.items():
        arr = np.asarray(arr)
        ds_chunks = chunks.get(key, "auto")
        ds_compressor = compressor_dict.get(key, "auto")
        ds = root.create_array(
            name=key,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=ds_chunks,
            compressors=ds_compressor,
            overwrite=True
        )

        # Generate slice tuples for all chunks
        slices_list = []
        for dim, chunk_size in zip(arr.shape, ds.chunks):
            starts = list(range(0, dim, chunk_size))
            slices_list.append(starts)

        # Cartesian product over all chunk starts -> all chunk positions
        chunk_starts = list(product(*slices_list))

        def write_chunk(start_indices):
            slc = tuple(
                slice(start, min(start + cs, dim))
                for start, cs, dim in zip(start_indices, ds.chunks, arr.shape)
            )
            ds[slc] = arr[slc]

        # Write chunks concurrently
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(tqdm(executor.map(write_chunk, chunk_starts), total=len(chunk_starts), desc=f"Writing {key}"))
    logger.info(f"Saved arrays to Zarr at: {zarr_path}")
    return root  # return the zarr group (useful for immediate reading)


# ---------------------------------------------------------------------------
# SPAD writers
# ---------------------------------------------------------------------------
def write_pixel_timeseries_npys(pixel_timeseries, output_dir):
    """
    Save a list of lists of pixel timeseries to .npy files, where pixel_timeseries[y][x] is an array of timestamps for that pixel.
    The files will be named with the pattern "scan_posX{X}_posY{Y}.npy" where {X} and {Y} are the pixel coordinates.

    Originally pixel_timeseries_to_npys(pixel_timeseries, output_dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(pixel_timeseries) * len(pixel_timeseries[0]), desc="Saving pixel timeseries to .npy files")
    for y in range(len(pixel_timeseries)):
        for x in range(len(pixel_timeseries[y])):
            if pixel_timeseries[y][x] is not None:
                np.save(output_dir / f"scan_posX{x:03d}_posY{y:03d}.npy", pixel_timeseries[y][x])
                pbar.update(1)


def write_frames_zarr(path, frames, T_exp=None, fps=None, save_coords=False):
    """
    Write binary quanta frames to a Zarr group.

    Originally write_quantaframes_zarr(path, frames, T_exp, fps, save_coords).

    Args:
        path (str or Path): Path to the output Zarr file.
        frames (np.ndarray): Array of shape (T, H, W) with binary frames.
        T_exp (float): exposure time in seconds.
        fps (float): if T_exp is None, fps must be provided to calculate T_exp.
        save_coords (bool): if True, also saves the (t, y, x) coordinates of events (for
            speed purposes so they don't have to be recomputed). Set to False to save
            disk space.
    """
    if T_exp is None:
        if fps is not None:
            T_exp = frames.shape[0] / fps
        else:
            raise ValueError("Either T_exp or fps must be provided")
    elif fps is not None and T_exp is not None:
        raise ValueError("Only one of T_exp or fps should be provided")
    elif T_exp is not None and fps is None:
        fps = frames.shape[0] / T_exp
    frames = np.asarray(frames, dtype="uint8")
    npoints = np.count_nonzero(frames)
    npts_persec_perpix = np.count_nonzero(frames, axis=0) / T_exp
    # statistics
    avg_pts_persec_perpixel = npoints / (T_exp * frames.shape[1] * frames.shape[2])
    max_pts_persec_perpixel = np.max(npts_persec_perpix)
    min_pts_persec_perpixel = np.min(npts_persec_perpix)
    med_pts_persec_perpixel = np.median(npts_persec_perpix)
    stddev_pts_persec_perpixel = np.std(npts_persec_perpix, ddof=1)
    fps = fps or (frames.shape[0] / T_exp)
    frame_size_bytes = frames.shape[1] * frames.shape[2]  # H * W for uint8
    max_chunk_bytes = 200_000_000
    max_frames_per_chunk = max(1, max_chunk_bytes // frame_size_bytes)
    if save_coords:
        t, y, x = frames_to_events(frames, T_exp=T_exp, normalize=True)
        quantadata = {
            "frames": frames,
            "t": t,
            "y": y,
            "x": x
        }
        chunks = {
            "frames": (max_frames_per_chunk, frames.shape[1], frames.shape[2]),
            "t": (100_000_000,),
            "y": (100_000_000,),
            "x": (100_000_000,),
        }
    else:
        quantadata = {
            "frames": frames
        }
        chunks = {
            "frames": (max_frames_per_chunk, frames.shape[1], frames.shape[2]),
        }
    compressors = {
        "frames": zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    }
    save_arrs_to_zarr(
        quantadata, path,
        chunks=chunks,
        n_workers=48,
        overwrite=True,
        attrs={
            "fps": fps,
            "T_exp": T_exp,
            "T": frames.shape[0],
            "H": frames.shape[1],
            "W": frames.shape[2],
            "shape": "(T, H, W)",
            "npoints": npoints,
            "avg_pts_persec_perpixel": avg_pts_persec_perpixel,
            "max_pts_persec_perpixel": max_pts_persec_perpixel,
            "min_pts_persec_perpixel": min_pts_persec_perpixel,
            "med_pts_persec_perpixel": med_pts_persec_perpixel,
            "stddev_pts_persec_perpixel": stddev_pts_persec_perpixel,
        },
        compressor_dict=compressors,
    )


# ---------------------------------------------------------------------------
# SPAD readers
# ---------------------------------------------------------------------------
def read_async_spad_dir(dirpath, h, w, T_exp, keep_prob=1.0, normalize=False):
    """
    Reads a folder of .npy files containing per-pixel timestamp arrays.
    Each .npy file is expected to be named with the pattern "posX{X}_posY{Y}.npy"
    where {X} and {Y} are the pixel coordinates.

    Originally read_spadfolder(dirpath, h, w, T_exp, keep_prob, normalize).

    If there are multiple channels, there will be multiple folders. Run this for each
    folder.

    Returns:
        tuple:
            - points (np.ndarray): array of shape (N, 3) with columns [t, y, x].
            - pixel_timeseries (list of lists): list of lists of arrays.
    """
    dirpath = Path(dirpath)
    pixel_timeseries = [[None] * w for _ in range(h)]
    all_points = []
    for fn in tqdm(sorted(dirpath.glob("*.npy")), desc="Reading pixel timestamps from files"):
        m = re.search(r"posX(\d+)_posY(\d+)", fn.stem)
        if m is None:
            logger.debug(f"skipping file with unexpected name: {fn.name}")
            continue
        px = int(m.group(1))
        py = int(m.group(2))

        timestamps = np.load(fn)
        timestamps = timestamps[timestamps <= T_exp]
        if timestamps.size == 0:
            continue
        if normalize:
            timestamps /= T_exp  # normalize to [0, 1]

        if normalize:
            # normalize to [0, 1]
            x_vals = np.full(timestamps.shape[0], float(px) / w)
            y_vals = np.full(timestamps.shape[0], float(py) / h)
        else:
            x_vals = np.full(timestamps.shape[0], float(px))
            y_vals = np.full(timestamps.shape[0], float(py))

        pts = np.stack([
            timestamps.astype("float64"),
            y_vals,
            x_vals
        ], axis=1)
        all_points.append(pts)
        pixel_timeseries[py][px] = timestamps
    points = np.vstack(all_points)
    logger.info(f"Read {points.shape[0]} points from folder {dirpath}")
    if keep_prob < 1.0:
        nevents_orig = points.shape[0]
        points = thin_events_uniform(points, keep_prob=keep_prob, seed=42)
        logger.info(f"Kept {points.shape[0]} of {nevents_orig} points after thinning")
    return points, pixel_timeseries


def read_async_spad_zarr(path, load_data=True):
    """
    Read asynchronous SPAD event data (per-pixel timestamps + flat coords) from a
    Zarr group.

    Originally read_asyncspad_zarr(path, load_data).
    """
    root = zarr.open_group(path, mode="r")

    # can't really lazily load this data
    timestamps = root["timestamps"][:]
    offsets = root["offsets"][:]
    lengths = root["lengths"][:]

    H, W = offsets.shape
    pixel_timeseries = [[None for _ in range(W)] for _ in range(H)]

    for i in range(H):
        for j in range(W):
            start = offsets[i, j]
            length = lengths[i, j]
            pixel_timeseries[i][j] = timestamps[start: start + length]
    t, y, x = root["t"], root["y"], root["x"]
    if load_data:
        t = t[:]
        y = y[:]
        x = x[:]
    return (t, y, x), pixel_timeseries, root.attrs


def read_quanta_bin(path, H=512, W=512):
    """
    Read a single binary SPAD512 file.

    Ripped from spadtools (https://github.com/lyehe/spadtools).
    """
    if H is None:
        H = 512
    if W is None:
        W = 512
    with open(path, "rb") as f:
        raw_data = f.read()
    binframe_length = H * W // 8
    frame_count = len(raw_data) // binframe_length
    bits_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(
        frame_count, H, W // 8
    )

    frames = np.unpackbits(bits_array, axis=2)
    frames = np.rot90(frames, k=1, axes=(1, 2))
    return frames


def _load_mat_array(mat_path, key=None):
    """
    Load a single 3D array from a MATLAB ``.mat`` file, as stored (``(H, W, T)``).

    Supports both MATLAB v7/v7.2 files (via ``scipy.io.loadmat``) and v7.3 HDF5-based
    files (via ``h5py``, imported lazily so it is only required for v7.3 data).

    Args:
        mat_path (str or Path): Path to the .mat file.
        key (str or None): Variable name (or HDF5 dataset path for v7.3) holding the
            array. If None, expects exactly one 3D array (v7.3) or one non-reserved
            variable (v7/v7.2) in the file.
    """
    mat_path = Path(mat_path)

    # Try scipy first (handles v7/v7.2; raises NotImplementedError for v7.3 files).
    scipy_error = None
    try:
        import scipy.io
        d = scipy.io.loadmat(mat_path.as_posix())
        candidates = [k for k in d.keys() if not k.startswith("__")]
        if not candidates:
            raise ValueError(f"No array variables found in {mat_path.name}")
        if key is None:
            if len(candidates) != 1:
                raise ValueError(
                    f"{mat_path.name}: multiple variables found {candidates}. "
                    f"Specify one with key=."
                )
            key_use = candidates[0]
        else:
            if key not in d:
                raise KeyError(f"{mat_path.name}: key '{key}' not found. Available: {candidates}")
            key_use = key
        return np.asarray(d[key_use])
    except NotImplementedError:
        pass  # v7.3 file; fall through to h5py
    except Exception as e:
        scipy_error = e

    # Try h5py (v7.3 / HDF5-backed .mat).
    try:
        import h5py
        with h5py.File(mat_path.as_posix(), "r") as f:
            if key is None:
                datasets = []

                def _visit(name, obj):
                    if hasattr(obj, "shape") and hasattr(obj, "dtype") and len(getattr(obj, "shape", ())) == 3:
                        datasets.append(name)

                f.visititems(_visit)
                if not datasets:
                    raise ValueError(f"{mat_path.name}: no 3D datasets found (v7.3).")
                if len(datasets) != 1:
                    raise ValueError(
                        f"{mat_path.name}: multiple 3D datasets found {datasets}. "
                        f"Specify one with key= (use full path inside the .mat)."
                    )
                key_use = datasets[0]
            else:
                key_use = key
                if key_use not in f:
                    raise KeyError(f"{mat_path.name}: dataset '{key_use}' not found in v7.3 file.")
            return np.array(f[key_use])  # loads this part into RAM (one .mat at a time)
    except Exception as e:
        if scipy_error is not None:
            raise RuntimeError(
                f"Failed to load {mat_path.name} with scipy ({scipy_error}) and h5py ({e})."
            ) from e
        raise


def read_quanta_mat(path, key=None):
    """
    Read a single MATLAB ``.mat`` quanta volume and return it as ``(T, H, W)`` frames
    (consistent with :func:`read_quanta_bin`).

    The volume is assumed to be stored as ``(H, W, T)`` (MATLAB WxHxT convention) and is
    transposed to ``(T, H, W)`` on read. Dtype is preserved from the file.

    Args:
        path (str or Path): Path to the .mat file.
        key (str or None): Variable name (or HDF5 dataset path for v7.3) holding the
            volume. If None, expects exactly one 3D array in the file.
    """
    arr = _load_mat_array(path, key=key)
    if arr.ndim != 3:
        raise ValueError(f"{Path(path).name}: expected 3D array, got shape {arr.shape}")
    # (H, W, T) -> (T, H, W)
    return np.transpose(arr, (2, 0, 1))


def _read_mat_meta(path):
    """
    Read the optional MATLAB-volume metadata sidecar from a directory.

    Looks for ``info.json`` first, then falls back to any single ``*.json`` file. The
    JSON is expected to carry ``no_frames_total``, ``no_parts`` and ``no_frames`` keys
    (as produced by the concat-to-zarr tooling) but any/all may be absent.

    Returns:
        dict or None: parsed metadata, or None if no readable sidecar is found.
    """
    path = Path(path)
    meta_path = path / "info.json"
    if not meta_path.is_file():
        candidates = sorted(path.glob("*.json"))
        if not candidates:
            return None
        meta_path = candidates[0]
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not parse metadata file {meta_path.name}: {e}")
        return None


def _read_quanta_mat_dir(path, matpaths, key=None):
    """
    Read and concatenate a directory of MATLAB ``.mat`` quanta volumes into ``(T, H, W)``.

    Files are read in the caller-provided (natural sorted) order and concatenated along
    T. If an ``info.json`` sidecar is present, its ``no_parts``/``no_frames_total`` are
    used to validate the result (mismatches are logged as warnings, not fatal).
    """
    meta = _read_mat_meta(path)
    no_parts = meta.get("no_parts") if meta else None
    if no_parts is not None and len(matpaths) != int(no_parts):
        logger.warning(f"Metadata says no_parts={no_parts}, but found {len(matpaths)} .mat files.")

    all_frames = []
    for f in tqdm(matpaths, desc="Reading .mat files from directory"):
        all_frames.append(read_quanta_mat(f, key=key))
    all_frames = np.concatenate(all_frames, axis=0)

    no_frames_total = meta.get("no_frames_total") if meta else None
    if no_frames_total is not None and all_frames.shape[0] != int(no_frames_total):
        logger.warning(
            f"Metadata says no_frames_total={no_frames_total}, but read {all_frames.shape[0]} frames."
        )
    return all_frames


def read_quanta_dir(path, H=512, W=512, key=None):
    """
    Read a directory of quanta files and concatenate them into a single ``(T, H, W)``
    array.

    Two directory layouts are supported (detected automatically):

    - **SPAD512 binary**: ``*.bin`` files; frame dimensions are given by ``H``, ``W``,
      read in natural sorted order.
    - **MATLAB volumes**: ``*.mat`` files, each a ``(H, W, T)`` volume, read in natural
      sorted order (e.g. ``part2.mat`` before ``part10.mat``) and concatenated along T.
      An optional ``info.json`` sidecar (with ``no_frames_total``, ``no_parts``,
      ``no_frames``) is used to validate the result if present.

    Args:
        H, W (int): Height and width of the frames where it cannot be inferred (e.g.
            SPAD512 .bin files). Ignored for .mat files, where they are inferred.
        key (str or None): For .mat files, the variable/dataset name holding the volume.
            If None, expects exactly one 3D array per file.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Path {path} is not a directory")

    binpaths = sorted(path.glob("*.bin"), key=lambda p: natural_sort_key(p.name))
    if binpaths:
        try:
            all_frames = []
            for f in tqdm(binpaths, desc="Reading .bin files from directory"):
                all_frames.append(read_quanta_bin(f, H=H, W=W))
            return np.concatenate(all_frames, axis=0)
        except ValueError as e:
            logger.error("Error with reading likely due to MacOS dotfiles or other garbage. Run `dot_clean` in the directory to clean up dotfiles.")
            raise e

    matpaths = sorted(path.glob("*.mat"), key=lambda p: natural_sort_key(p.name))
    if matpaths:
        return _read_quanta_mat_dir(path, matpaths, key=key)

    raise FileNotFoundError(f"No *.bin or *.mat files found in {path}")


def read_quanta_zarr(path, load_data=True):
    """
    Reads quanta data from a Zarr v3 group and returns:
      - frames: (T,H,W) array (lazy unless load_data=True)
      - (t,y,x): optional coordinate arrays if present
      - attrs: group attributes

    Works with:
      - "original" QuantaBurst zarr (frames/quantaframes + rich attrs)
      - your concatenated zarr (data + different attrs)
    """
    zarrdata = zarr.open_group(str(path), mode="r")

    # Zarr v3 safe listing of arrays
    keys = set(zarrdata.array_keys())

    # Accept multiple possible dataset names
    candidate_keys = ("frames", "quantaframes", "data")
    framekey = next((k for k in candidate_keys if k in keys), None)
    if framekey is None:
        raise KeyError(
            f"No frames array found in {path}. "
            f"Available arrays: {sorted(keys)}"
        )

    frames = zarrdata[framekey]

    # Optional coordinates
    coord_keys = set(zarrdata.array_keys())  # arrays again
    contains_coords = all(k in coord_keys for k in ("t", "y", "x"))
    if contains_coords:
        t = zarrdata["t"]
        y = zarrdata["y"]
        x = zarrdata["x"]
    else:
        t = y = x = None

    if load_data:
        frames = frames[:]
        if contains_coords:
            t = t[:]
            y = y[:]
            x = x[:]

    # attrs differ between the two formats; just return them as-is
    return frames, (t, y, x), dict(zarrdata.attrs)


def read_quanta_auto(path, load_data=True, H=None, W=None, key=None):
    """
    Automatically detects the format of the input path and reads the quanta data accordingly:
    a .zarr group, a .bin file, a .mat file, or a directory of .bin/.mat files.

    Args:
        load_data (bool): If True, load the data into memory. If False, returns lazy data if supported.
        H, W (int): Height and width of the frames where it cannot be inferred (e.g. SPAD512 .bin files).
            Otherwise, these are ignored and inferred from data.
        key (str or None): For .mat inputs, the variable/dataset name holding the volume.
            If None, expects exactly one 3D array per file.

    Returns:
        tuple:
            - quantaframes: array of shape (T, H, W) with binary frames.
            - metadata (dict or None): relevant metadata if available, else None.
    """
    path = Path(path)
    if path.suffix == ".zarr":
        logger.info(f"Reading quanta from Zarr file: {path}")
        frames, _, metadata = read_quanta_zarr(path, load_data=load_data)
        return frames, metadata
    elif path.suffix == ".bin":
        logger.info(f"Reading quanta from binary file: {path}")
        frames = read_quanta_bin(path, H=H, W=W)
        return frames, None
    elif path.suffix == ".mat":
        logger.info(f"Reading quanta from MATLAB file: {path}")
        frames = read_quanta_mat(path, key=key)
        return frames, None
    # otherwise, assume it's a directory of .bin or .mat files
    logger.info(f"Reading quanta from directory of quanta files: {path}")
    frames = read_quanta_dir(path, H=H, W=W, key=key)
    return frames, None


# ---------------------------------------------------------------------------
# Format conversions: frames <-> events <-> pixel_timeseries
# ---------------------------------------------------------------------------
def frames_to_events(frames: np.ndarray, T_exp=None, fps=None, keep_prob=1.0, normalize=True, cuda=False):
    """
    Convert (T, H, W) binary frames (0/1 per pixel) to spatiotemporal
    events (t, y, x).

    Originally binframes_to_spt(frames, T_exp, fps, keep_prob, normalize, cuda).

    Args:
        frames (np.ndarray): array of shape (nframes, height, width) with
            binary pixel values.
        T_exp (float): exposure time in seconds.
        keep_prob (float): probability of keeping each event (for thinning).
        normalize (bool): if True, normalize x, y, t to [0, 1].

    Returns:
        tuple:
            - t: 1D array of time
            - y: 1D array of y coordinates
            - x: 1D array of x coordinates
    """
    xp = cp if cuda else np
    if T_exp is None:
        if fps is not None:
            T_exp = frames.shape[0] / fps
        else:
            raise ValueError("Either T_exp or fps must be provided")
    elif fps is not None and T_exp is not None:
        raise ValueError("Only one of T_exp or fps should be provided")
    # Get coordinates where pixel == 1
    t_coords, y_coords, x_coords = xp.nonzero(frames)
    t_coords = t_coords.astype("float64")
    y_coords = y_coords.astype("float64")
    x_coords = x_coords.astype("float64")

    t_coords *= T_exp / frames.shape[0]  # convert frame index to time in seconds
    if keep_prob < 1.0:
        nevents_orig = t_coords.shape[0]
        idxs = thin_events_uniform(t_coords, return_idx=True, keep_prob=keep_prob, seed=42)
        t_coords = t_coords[idxs]
        y_coords = y_coords[idxs]
        x_coords = x_coords[idxs]
        logger.info(f"Kept {len(idxs)} of {nevents_orig} points after thinning")
    if normalize:
        # Normalize x, y, t to [0, 1]
        x_coords /= frames.shape[2]  # x
        y_coords /= frames.shape[1]  # y
        t_coords /= T_exp  # t
    return t_coords, y_coords, x_coords


def events_to_frames(x, y, t, nframes, H, W, accumulate=False, cuda=False):
    """
    Convert spatiotemporal events (x, y, t) to binary frames of shape
    (nframes, H, W) where each pixel is 0/1.

    Originally events_to_binframes(x, y, t, nframes, H, W, accumulate, cuda).

    Args:
        x (np.ndarray): x coordinates in [0, 1].
        y (np.ndarray): y coordinates in [0, 1].
        t (np.ndarray): time coordinates in [0, 1].
        nframes (int): number of frames.
        H (int): height of each frame.
        W (int): width of each frame.

    Returns:
        frames (np.ndarray): binary frames of shape (nframes, H, W).
    """
    if cuda:
        frames = cp.zeros((nframes, H, W), dtype="uint8")
        # Convert normalized coordinates to pixel indices and frame indices
        x_idx = cp.clip((x * W).astype(int), 0, W - 1)
        y_idx = cp.clip((y * H).astype(int), 0, H - 1)
        t_idx = cp.clip((t * nframes).astype(int), 0, nframes - 1)

        if accumulate:
            frames = frames.astype("uint32")
            cp.add.at(frames, (t_idx, y_idx, x_idx), 1)
        else:
            frames[t_idx, y_idx, x_idx] = 1
    else:
        frames = np.zeros((nframes, H, W), dtype="uint8")
        # Convert normalized coordinates to pixel indices and frame indices
        x_idx = np.clip((x * W).astype(int), 0, W - 1)
        y_idx = np.clip((y * H).astype(int), 0, H - 1)
        t_idx = np.clip((t * nframes).astype(int), 0, nframes - 1)

        if accumulate:
            frames = frames.astype("uint32")
            np.add.at(frames, (t_idx, y_idx, x_idx), 1)
        else:
            frames[t_idx, y_idx, x_idx] = 1
    return frames


def events_to_pixel_timeseries(x, y, t, H=60, W=60):
    """
    Bin the spatiotemporal events by pixel, so each pixel contains a sorted
    timeseries of event times.

    Originally bin_spt_events(x, y, t, H, W).

    x, y, t are assumed to be normalized to [0, 1].

    Returns:
        pixel_timeseries: 2D list of sorted event times for each pixel (so H x W
            list of arrays)
    """
    x_bins = np.linspace(0, 1, W + 1)
    y_bins = np.linspace(0, 1, H + 1)

    # Digitize x and y to get pixel indices
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1

    # Ensure indices are within bounds
    x_idx = np.clip(x_idx, 0, W - 1)
    y_idx = np.clip(y_idx, 0, H - 1)

    # Prepare a 2D array of lists to hold event times for each pixel
    pixel_events = [[[] for _ in range(W)] for _ in range(H)]

    # Assign each event time to the corresponding pixel
    for xi, yi, ti in tqdm(zip(x_idx, y_idx, t), total=len(t), desc="Assigning events to pixels"):
        pixel_events[yi][xi].append(ti)

    # Convert lists to sorted numpy arrays for each pixel
    pixel_timeseries = [[np.sort(np.array(times)) for times in row] for row in pixel_events]
    return pixel_timeseries


def frames_to_pixel_timeseries(frames, T_exp):
    """
    Converts a (T, H, W) binary frame array to a (H, W) list of sorted event times per
    pixel.

    Originally timeseries_from_binframes(binframes, T_exp).
    """
    nframes, H, W = frames.shape
    pixel_timeseries = [[[] for _ in range(W)] for _ in range(H)]

    pbar = tqdm(total=H * W, desc="Extracting event times per pixel")
    for y in range(H):
        for x in range(W):
            t_idx = np.nonzero(frames[:, y, x])[0]
            if t_idx.size > 0:
                times = t_idx * (T_exp / nframes)
                pixel_timeseries[y][x] = times
            pbar.update(1)
    pbar.close()
    return pixel_timeseries
