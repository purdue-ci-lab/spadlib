"""
I/O for SPAD/quanta data plus conversions between the three representations used
throughout spadlib:

- ``frames``           : (T, H, W) synchronous binary/count quanta frames
- ``events``           : asynchronous continuous (t, y, x) event coordinates
- ``pixel_timeseries`` : H x W list-of-lists of per-pixel timestamp arrays

Also includes generic array <-> Zarr helpers used by the SPAD writers.
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import zarr
from tqdm import tqdm

from spadlib.processing import correct_hotpixels_conv, thin_events_uniform, thin_frames_uniform
from spadlib.utils import dot_clean_dir, is_json_serializable, make_json_serializable

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
def read_pixel_timeseries_dir(dirpath, h, w, T_exp, keep_prob=1.0, normalize=False):
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


def read_spad_bin(
    path, nframes, h, w, keep_prob=1.0, downsample=1, hotpix_thresh=None, hotpix_kernel_size=5,
    downsample_method="resize",
):
    """
    Reads a .bin file from a SPAD 512 camera and returns a numpy array of shape
    (nframes, h, w) containing the binary pixel data, with optional hot-pixel
    correction, thinning, and downsampling.

    Originally read_spadbin(path, nframes, h, w, keep_prob, downsample, hotpix_thresh,
    hotpix_kernel_size, downsample_method).

    There are probably hundreds of thousands of frames in the .bin, so set nframes
    accordingly.

    Args:
        hotpix_thresh (float or None): if not None, will do hot pixel correction based on
            the given stddev threshold and kernel size.
        downsample_method (str): "resize" to use cv2 resize, "slice" to use slicing.
    """
    nbytes_per_frame = h * w // 8
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(nframes * nbytes_per_frame), dtype=np.uint8)
    databit = np.unpackbits(data)
    wholetotalbit = databit.reshape((nframes, w, h))
    # transpose width/height and set origin to top-left
    wholetotalbit = wholetotalbit.transpose(0, 2, 1)[:, ::-1, :]
    if hotpix_thresh is not None:
        arr_hotpixel_corrected, hot_pixels, neighbor_mean = correct_hotpixels_conv(
            wholetotalbit, thresh_std=hotpix_thresh, kernel_size=hotpix_kernel_size,
        )
        nhotpixs = np.sum(hot_pixels)
        wholetotalbit = arr_hotpixel_corrected
        logger.info(f"Replaced {nhotpixs} hot pixels by neighbor comparison (threshold={hotpix_thresh})")
    if keep_prob < 1.0:
        nevents_orig = int(np.sum(wholetotalbit))
        wholetotalbit = thin_frames_uniform(wholetotalbit, keep_prob, seed=42)
        total_kept = int(np.sum(wholetotalbit))
        logger.info(f"Kept {total_kept} of {nevents_orig} points after thinning")
    if downsample > 1:
        if downsample_method == "resize":
            downsampled = []
            for frame in tqdm(wholetotalbit, desc="Downsampling frames"):
                downsampled.append(cv2.resize(frame, (w // downsample, h // downsample), interpolation=cv2.INTER_NEAREST))
            wholetotalbit = np.array(downsampled)
        elif downsample_method == "slice":
            wholetotalbit = wholetotalbit[:, ::int(downsample), ::int(downsample)]
    return wholetotalbit


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


def read_quanta_dir(path, H=512, W=512):
    """
    Read a directory of binary SPAD512 files and concatenate them into a single array.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Path {path} is not a directory")
    try:
        binpaths = sorted(path.glob("RAW*.bin"))
        all_frames = []
        for f in tqdm(binpaths, desc="Reading .bin files from directory"):
            all_frames.append(read_quanta_bin(f, H=H, W=W))
        all_frames = np.concatenate(all_frames, axis=0)
        return all_frames
    except ValueError as e:
        logger.error("Error with reading likely due to MacOS dotfiles or other garbage. Run `dot_clean` in the directory to clean up dotfiles.")
        raise e


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


def read_quanta_auto(path, load_data=True, H=None, W=None):
    """
    Automatically detects the format of the input path and reads the quanta data accordingly.
    Priority: .zarr > .bin > directory of .bin files.

    Returns:
        tuple:
            - quantaframes: array of shape (T, H, W) with binary frames.
            - metadata (dict or None): relevant metadata if available, else None.
    """
    path = Path(path)
    # first check if the path ends in .zarr, or if you append .zarr to the dir name it exists
    if path.suffix == ".zarr":
        if path.is_dir():
            logger.info(f"Reading quanta from Zarr file: {path}")
            frames, _, metadata = read_quanta_zarr(path, load_data=load_data)
            return frames, metadata
    elif (path.parent / f"{path.name}.zarr").is_dir():
        logger.info(f"Reading quanta from Zarr file: {path.parent / f'{path.name}.zarr'}")
        frames, _, metadata = read_quanta_zarr(path.parent / f"{path.name}.zarr", load_data=load_data)
        return frames, metadata
    # check if the path ends in .bin, or if you append .bin to the name it exists
    elif path.suffix == ".bin":
        if path.is_file():
            logger.info(f"Reading quanta from binary file: {path}")
            frames = read_quanta_bin(path, H=H, W=W)
            return frames, None
    elif (path.parent / f"{path.name}.bin").is_file():
        logger.info(f"Reading quanta from binary file: {path.parent / f'{path.name}.bin'}")
        frames = read_quanta_bin(path.parent / f"{path.name}.bin", H=H, W=W)
        return frames, None
    # otherwise, assume it's a directory of .bin files
    logger.info(f"Reading quanta from directory of binary files: {path}")
    frames = read_quanta_dir(path, H=H, W=W)
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
