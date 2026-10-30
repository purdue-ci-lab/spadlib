"""
Read/write utilities for color video and image data (as opposed to the SPAD
quanta data handled by ``spadlib.io``): video files, image directories, lazily read
Phantom high-speed camera ``.cine`` files, and HDF5 video. Also includes hot pixel
correction and simple frame resizing.
"""
import contextlib
import ctypes
import logging
import math
import os
import struct
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycine.file import read_header
from pycine.linLUT import linLUT
from tqdm import tqdm

from spadlib.hdf5_utils import (
    HDF5_COMPRESSION_LEVEL,
    HDF5_MAX_CHUNK_BYTES,
    _read_hdf5_frames,
    _save_arrs_to_hdf5,
)
from spadlib.utils import _frames_per_chunk, _index_frames, natural_sort_key


logger = logging.getLogger(__name__)


def read_video(path, return_framerate=False, grayscale=False):
    """
    Read a video file and return frames as a numpy array of
    shape (num_frames, height, width, 3) and the framerate.

    Args:
        path (str): Path to the video file.

    Returns:
        tuple:
        - frames (np.ndarray): Array of video frames.
        - framerate (float): video FPS.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    framerate = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    pbar = tqdm(desc="Number of frames read")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        pbar.update(1)
    pbar.close()

    cap.release()

    frames = np.stack(frames, axis=0)

    if return_framerate:
        return frames, framerate
    return frames


def read_image_dir(path, grayscale: bool = False) -> np.ndarray:
    """
    Read a directory of images (natural-sorted) into a single stacked array.

    Originally read_imgdir(path, grayscale).
    """
    path = Path(path)
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    tiff_extensions = {".tiff", ".tif"}

    image_files = sorted(
        [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in supported_extensions],
        key=natural_sort_key
    )

    images = []
    for filename in tqdm(image_files):
        ext = os.path.splitext(filename)[1].lower()
        img = Image.open(path / filename)

        if ext in tiff_extensions:
            # tiff files may have higher bit depth; preserve it by converting to numpy array directly
            arr = np.array(img)  # preserves native bit depth (e.g. uint16 for 12-bit)
            if not grayscale and arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)  # expand to RGB if needed
        else:
            img = img.convert("L" if grayscale else "RGB")
            arr = np.array(img)  # uint8

        images.append(arr)

    return np.array(images)


# ---------------------------------------------------------------------------
# Phantom .cine files
# ---------------------------------------------------------------------------
# biCompression values of the .cine image formats that can be read
_CINE_UNCOMPRESSED = 0
_CINE_PACKED_10BIT = 256  # 10-bit log-encoded values, 4 pixels packed into 5 bytes


def _cine_header_value(value):
    """
    Convert a value of a pycine (ctypes) header to plain Python: structures become
    dicts, arrays lists, and char arrays strings.
    """
    if isinstance(value, ctypes.Structure):
        return {field[0]: _cine_header_value(getattr(value, field[0])) for field in value._fields_}
    if isinstance(value, ctypes.Array):
        if value._type_ is not ctypes.c_char:
            return [_cine_header_value(v) for v in value]
        value = value.value  # ctypes only gives bytes for top-level char arrays (not e.g. setup.BinName)
    if isinstance(value, bytes):
        # unused text fields can hold uninitialized junk, so don't fail on invalid UTF-8
        return value.decode("utf-8", errors="replace")
    return value


class LazyCineFrames:
    """
    Lazy ``(T, H, W)`` frames of a Phantom ``.cine`` file. Returned by
    :func:`read_video_cine` when ``load_data=False``.

    Only the header is read at construction. Frames are read from disk and decoded when
    indexed, `chunk_frames` at a time in `n_workers` threads; frames stored back to back
    in the file are read as one block.

    Frames are decoded exactly like :func:`pycine.raw.create_raw_array`: uncompressed
    frames are flipped upright and rescaled from [BlackLevel, WhiteLevel] to
    [0, 2**RealBPP - 1]; packed 10-bit frames are linearized with Phantom's lookup table
    and rescaled to 12 bits. Frames are raw sensor values (a Bayer mosaic for color
    cameras), not demosaiced.

    Only integer and slice indexing (optionally with ``...``) is supported, with numpy
    semantics; the result is a uint16 numpy array.

    Args:
        path (str or Path): path to the ``.cine`` file.
        chunk_frames (int): number of frames read and decoded per thread task.
        n_workers (int): number of threads reading/decoding frames.

    Attributes:
        header (dict): the pycine header (see :func:`pycine.file.read_header`).
        bit_depth (int): bit depth of the decoded values (12 for packed 10-bit files).
    """

    def __init__(self, path, chunk_frames=64, n_workers=4):
        self.path = Path(path)
        self.chunk_frames = chunk_frames
        self.n_workers = n_workers
        self.header = read_header(self.path)
        bih, setup = self.header["bitmapinfoheader"], self.header["setup"]
        self.shape = (self.header["cinefileheader"].ImageCount, bih.biHeight, bih.biWidth)
        self.dtype = np.dtype(np.uint16)
        self._frame_offsets = np.asarray(self.header["pImage"], dtype=np.int64)

        self._packed = bih.biCompression == _CINE_PACKED_10BIT
        if self._packed:
            self.bit_depth = 12
            self._lut = np.interp(linLUT.astype(np.uint16), [64, 4064], [0, 2**12 - 1]).astype(np.uint16)
            expected_image_size = bih.biHeight * bih.biWidth * 5 // 4
        elif bih.biCompression == _CINE_UNCOMPRESSED and bih.biBitCount == 16:
            self.bit_depth = setup.RealBPP
            self._lut = np.interp(
                np.arange(2**16), [setup.BlackLevel, setup.WhiteLevel], [0, 2**setup.RealBPP - 1]
            ).astype(np.uint16)
            expected_image_size = bih.biHeight * bih.biWidth * 2
        else:
            raise NotImplementedError(
                f"Unsupported .cine image format: biCompression={bih.biCompression}, "
                f"biBitCount={bih.biBitCount}"
            )

        # Each frame is an annotation, whose first 4 bytes are its size and last 4 bytes
        # the image size, followed by the image. Phantom writes the same annotation for
        # every frame, so the sizes are read from the first.
        with open(self.path, "rb") as f:
            f.seek(self._frame_offsets[0])
            self._annotation_size = struct.unpack("<I", f.read(4))[0]
            f.seek(self._frame_offsets[0] + self._annotation_size - 4)
            self._image_size = struct.unpack("<I", f.read(4))[0]
        if self._image_size != expected_image_size:
            raise ValueError(
                f"{self.path} has {self._image_size}-byte images, expected {expected_image_size} "
                f"for {bih.biHeight}x{bih.biWidth} frames with biCompression={bih.biCompression}"
            )

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"{type(self).__name__}(path={str(self.path)!r}, shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, key):
        return _index_frames(key, self.shape, self._read, type(self).__name__)

    def _read(self, t_range, spatial_key):
        """Frames at the increasing frame indices `t_range`, indexed by `spatial_key`."""
        idxs = np.asarray(t_range, dtype=np.int64)
        frame_shape = np.empty((0, *self.shape[1:]), dtype=self.dtype)[spatial_key].shape[1:]
        out = np.empty((len(idxs), *frame_shape), dtype=self.dtype)

        def load(start):
            chunk = idxs[start:start + self.chunk_frames]
            out[start:start + self.chunk_frames] = self._decode(self._read_raw(chunk))[spatial_key]

        starts = range(0, len(idxs), self.chunk_frames)
        with ThreadPoolExecutor(self.n_workers) as ex:
            # only show progress for reads that take more than one round of threads
            list(tqdm(
                ex.map(load, starts), total=len(starts), desc="Reading .cine frames",
                leave=False, disable=len(starts) <= self.n_workers,
            ))
        return out

    def _read_raw(self, idxs):
        """Raw image bytes of the frames `idxs`, as an (n_frames, image_size) uint8 array."""
        offsets = self._frame_offsets[idxs]
        frame_size = self._annotation_size + self._image_size
        if np.all(np.diff(offsets) == frame_size):  # stored back to back: one read
            raw = np.fromfile(self.path, dtype=np.uint8, count=len(idxs) * frame_size, offset=offsets[0])
            raw = raw.reshape(len(idxs), frame_size)
        else:
            raw = np.stack([
                np.fromfile(self.path, dtype=np.uint8, count=frame_size, offset=offset) for offset in offsets
            ])
        return raw[:, self._annotation_size:]

    def _decode(self, raw):
        """Decode raw image bytes from :meth:`_read_raw` to (n_frames, H, W) uint16 frames."""
        n, (H, W) = len(raw), self.shape[1:]
        if self._packed:
            p = raw.reshape(n, -1, 5).astype(np.uint16)
            vals = np.empty((n, p.shape[1], 4), dtype=np.uint16)
            vals[..., 0] = (p[..., 0] << 2) | (p[..., 1] >> 6)
            vals[..., 1] = ((p[..., 1] & 0b00111111) << 4) | (p[..., 2] >> 4)
            vals[..., 2] = ((p[..., 2] & 0b00001111) << 6) | (p[..., 3] >> 2)
            vals[..., 3] = ((p[..., 3] & 0b00000011) << 8) | p[..., 4]
            return self._lut[vals.reshape(n, H, W)]
        # uncompressed images are stored bottom-up
        return self._lut[np.ascontiguousarray(raw).view(np.uint16).reshape(n, H, W)[:, ::-1]]


def read_video_cine(path, load_data=False, return_meta=False, n_workers=4):
    """
    Read frames from a Phantom high-speed camera ``.cine`` file.

    Args:
        path (str or Path): path to the ``.cine`` file.
        load_data (bool): if True, read all frames into memory as a numpy array. If False
            (default), return a :class:`LazyCineFrames`, which reads frames when indexed.
        return_meta (bool): if True, also return the metadata: ``fps``, ``T``, ``H``,
            ``W``, ``shape`` and ``bit_depth`` (of the decoded values), plus the full
            file header as the ``cinefileheader``, ``bitmapinfoheader`` and ``setup``
            dicts. The header's table of frame offsets in the file is left out.
        n_workers (int): number of threads reading/decoding frames.

    Returns:
        frames: (T, H, W) uint16 frames, lazy unless load_data=True; or
        ``(frames, meta)`` if return_meta=True.
    """
    frames = LazyCineFrames(path, n_workers=n_workers)
    if return_meta:
        T, H, W = frames.shape
        meta = {
            "fps": frames.header["setup"].FrameRate,
            "T": T,
            "H": H,
            "W": W,
            "shape": "(T, H, W)",
            "bit_depth": frames.bit_depth,
            **{
                key: _cine_header_value(frames.header[key])
                for key in ("cinefileheader", "bitmapinfoheader", "setup")
            },
        }
    if load_data:
        frames = frames[:]
    return (frames, meta) if return_meta else frames


# ---------------------------------------------------------------------------
# HDF5 video
# ---------------------------------------------------------------------------
def write_video_hdf5(path, frames, fps=None, attrs=None, compression_level=HDF5_COMPRESSION_LEVEL):
    """
    Write video frames to an HDF5 file: a ``frames`` dataset at the root of the file,
    GZIP compressed with the shuffle filter, with the metadata as root attributes.

    Frames are copied a run of chunks at a time, so a lazy source (e.g. from
    :func:`read_video_cine`, or an h5py/zarr array) is never loaded whole. Each chunk
    holds as many whole frames as fit in ``HDF5_MAX_CHUNK_BYTES`` (at least one).

    To convert a ``.cine`` file, pass the lazy frames (or a slice of them) and their
    metadata::

        frames, meta = read_video_cine("video.cine", return_meta=True)
        write_video_hdf5("video.h5", frames[1000:2000], attrs=meta)

    Args:
        path (str or Path): path to the output HDF5 file.
        frames: (T, H, W) or (T, H, W, C) array, or a lazy array sliceable along T.
        fps (float or None): frame rate. If None, the ``fps`` in `attrs` is kept
            (stored as empty if there is none).
        attrs (dict or None): extra attributes to store. The geometry (``T``, ``H``,
            ``W``, ``C`` for 4D frames, and ``shape``) is always taken from `frames`.
        compression_level (int): GZIP compression level, 0-9.

    Raises:
        ValueError: if `frames` is not 3D or 4D.
    """
    if len(frames.shape) not in (3, 4):
        raise ValueError(f"frames must be (T, H, W) or (T, H, W, C), got shape {frames.shape}")
    geometry = dict(zip("THWC", frames.shape))
    geometry["shape"] = f"({', '.join(geometry)})"
    attrs = {"fps": None, **(attrs or {}), **geometry}
    if fps is not None:
        attrs["fps"] = fps

    frame_nbytes = math.prod(frames.shape[1:]) * np.dtype(frames.dtype).itemsize
    frames_per_chunk = _frames_per_chunk(frames.shape[0], frame_nbytes, HDF5_MAX_CHUNK_BYTES)
    _save_arrs_to_hdf5(
        {"frames": frames}, path,
        chunks={"frames": (frames_per_chunk, *frames.shape[1:])},
        attrs=attrs,
        compression_level=compression_level,
    )


def read_video_hdf5(path, load_data=False, return_meta=False):
    """
    Read video frames from an HDF5 file written by :func:`write_video_hdf5`.

    Args:
        path (str or Path): path to the HDF5 file.
        load_data (bool): if True, read the frames into memory as a numpy array. If
            False (default), return the lazy h5py dataset, which keeps the file open
            (read-only) for as long as it is referenced.
        return_meta (bool): if True, also return the root attributes.

    Returns:
        frames: (T, H, W) or (T, H, W, C) array, lazy unless load_data=True; or
        ``(frames, attrs)`` if return_meta=True.
    """
    return _read_hdf5_frames(path, load_data=load_data, return_meta=return_meta)


# ---------------------------------------------------------------------------
# Hot pixel correction
# ---------------------------------------------------------------------------
def _locate_hotpixels(frames, hp_threshold, kernel_size, n_detect):
    """Flag pixels that sit far above their local median in (the temporal median of) every frame."""
    frames = frames[:: max(1, len(frames) // n_detect)]
    residual = np.empty(frames.shape, dtype=np.float32)
    for t, f in enumerate(frames):
        f = np.ascontiguousarray(f, dtype=np.float32)
        residual[t] = f - cv2.medianBlur(f, kernel_size)
    # temporal median rejects transient bright features (moving highlights), keeps fixed-pattern ones
    residual = np.median(residual, axis=0)
    # sigma-clipped stats, so the threshold doesn't depend on how many hot pixels there are
    keep = np.ones(residual.shape, dtype=bool)
    for _ in range(5):
        mean, std = residual[keep].mean(), residual[keep].std()
        keep = residual < mean + 5 * std
    return residual > mean + hp_threshold * std


def _replace_hotpixels(sub, mask, kernel_size, chunk=1024):
    """Replace masked pixels in `sub` (T, h, w), in place, with the per-frame median of non-hot neighbors."""
    h, w = mask.shape
    r = kernel_size // 2
    ys, xs = np.nonzero(mask)
    oy, ox = np.mgrid[-r : r + 1, -r : r + 1].reshape(2, -1)
    center = (oy == 0) & (ox == 0)
    oy, ox = oy[~center], ox[~center]
    ny = np.clip(ys[:, None] + oy, 0, h - 1)  # (n_hot, n_neighbors)
    nx = np.clip(xs[:, None] + ox, 0, w - 1)
    invalid = mask[ny, nx]  # don't borrow from other hot pixels
    for t0 in range(0, sub.shape[0], chunk):
        vals = sub[t0 : t0 + chunk][:, ny, nx].astype(np.float32)
        vals[:, invalid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(vals, axis=-1)
        med = np.where(np.isnan(med), sub[t0 : t0 + chunk][:, ys, xs], med)  # all neighbors hot: leave as is
        if np.issubdtype(sub.dtype, np.integer):
            med = np.rint(med)
        sub[t0 : t0 + chunk, ys, xs] = med.astype(sub.dtype)


def correct_hotpixels(frames, hp_threshold: float = 20.0, kernel_size: int = 3, bayer: bool = False, n_detect: int = 500, return_mask: bool = False):
    """
    Correct hot pixels in scalar-valued frames (e.g. Phantom cine data).

    Hot pixels are located by subtracting a median-filtered background from each frame and taking
    the temporal median of the residual, so only pixels that are bright in (nearly) every frame are
    flagged, not moving highlights. Pixels whose residual exceeds mean + hp_threshold * std
    (sigma-clipped) are hot. Each hot pixel is then replaced, frame by frame, by the median of its
    non-hot neighbors.

    Args:
        frames: (T, H, W) or (H, W) array. Use raw (pre-debayer) frames.
        hp_threshold: Number of standard deviations above the background to consider a pixel "hot".
        kernel_size: Size of the kernel for background estimation and neighbor replacement (3 or 5).
        bayer: If True, treat frames as a 2x2 Bayer mosaic and correct each of the four
            sub-planes independently. Kernel size applies to the sub-planes.
        n_detect: Max number of (evenly spaced) frames used for detection.
        return_mask: If True, also return the (H, W) boolean hot pixel mask.
    """
    frames = np.asarray(frames)
    squeeze = frames.ndim == 2
    if squeeze:
        frames = frames[None]
    out = frames.copy()
    mask = np.zeros(frames.shape[-2:], dtype=bool)
    s = 2 if bayer else 1
    for dy in range(s):
        for dx in range(s):
            sub = out[:, dy::s, dx::s]
            m = _locate_hotpixels(sub, hp_threshold, kernel_size, n_detect)
            _replace_hotpixels(sub, m, kernel_size)
            mask[dy::s, dx::s] = m
    if squeeze:
        out = out[0]
    return (out, mask) if return_mask else out


def write_video(
    frames: np.ndarray, path, res_scale=1.0, playback_fps=None, gamma=1.0, cmap=None, fileformat=None,
    vmin=None, vmax=None, qmin=None, qmax=None, framenames=None, verbose=False
):
    """
    Saves video frame arrays to a video file or sequence of PNGs. If path has no extension,
    it is treated as a directory and individual image files are saved.

    Originally to_video(frames, path, res_scale, playback_fps, gamma, cmap, fileformat,
    vmin, vmax, quantile, framenames).

    Args:
        frames (np.ndarray): (T x H x W x C) (RGB) or (T x H x W) (intensity) video frames.
        path (str or Path): output video file path or directory for image files.
        res_scale (float): resolution scaling factor with nearest neighbor interpolation.
        cmap: ignored if frames are RGB; otherwise, matplotlib colormap name or object.
        fileformat (str or None): video format (e.g., "mp4", "avi"), or image format (e.g., "png");
            if None, inferred from path suffix.
        qmin (float or None): if not None, use this quantile of frames as vmin (ignored if vmin is specified).
        qmax (float or None): if not None, use this quantile of frames as vmax (ignored if vmax is specified).
    """
    path = Path(path)
    if cmap is None:
        cmap = "viridis"
    cmap_fn = plt.get_cmap(cmap)
    is_rgb = False
    if frames.ndim == 4:
        if frames.shape[3] == 3:
            is_rgb = True
        else:
            raise ValueError("4D frames array must have shape (T, H, W, 3) for RGB video")
    elif frames.ndim == 3:
        is_rgb = False
    else:
        raise ValueError("frames must be a 3D or 4D numpy array")

    # compute a normalized intensity in [0,1] for colormap input
    if vmax is None:
        if qmax is not None:
            vmax = float(np.quantile(frames, qmax))
        else:
            vmax = float(np.max(frames))
    if vmin is None:
        if qmin is not None:
            vmin = float(np.quantile(frames, qmin))
        else:
            vmin = float(np.min(frames))
            if vmin >= 0:
                if verbose:
                    logger.info(f"vmin was not specified and frames have non-negative values, so using vmin=0 for more accurate scaling")
                vmin = 0.0

    H, W = frames.shape[1], frames.shape[2]
    if res_scale != 1.0:
        out_W = int(W * res_scale)
        out_H = int(H * res_scale)
    else:
        out_W = W
        out_H = H
    # if path is a directory, write individual image files
    is_video_file = path.suffix in [".mp4", ".avi", ".mov", ".mkv"]
    if not is_video_file:
        path.mkdir(parents=True, exist_ok=True)
        if fileformat is None:
            fileformat = "png"
    else:
        if playback_fps is None:
            raise ValueError("playback_fps must be specified if saving a video file")
        path.parent.mkdir(parents=True, exist_ok=True)
        if fileformat is None:
            fileformat = path.suffix[1:].lower()
        codec = get_codec_for_format(fileformat)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vidwriter = cv2.VideoWriter(str(path), fourcc, playback_fps, (out_W, out_H), isColor=True)

    max_frames = len(frames)

    if not is_video_file:
        allpaths = []
    if verbose:
        frameiterator = tqdm(range(max_frames), desc="Writing video frames")
    else:
        frameiterator = range(max_frames)
    for i in frameiterator:
        intensity = (np.clip(frames[i], vmin, vmax) - vmin) / (vmax - vmin)  # normalize to [0,1]
        if gamma != 1:
            intensity = intensity ** gamma
        if is_rgb:
            rgb_mapped = (intensity * 255.0).astype(np.uint8)  # (H,W,3) in RGB
        else:
            # apply matplotlib colormap -> returns RGBA in [0,1]
            rgba_mapped = cmap_fn(intensity)  # shape (H,W,4)
            rgb_mapped = (rgba_mapped[..., :3] * 255.0).astype(np.uint8)  # (H,W,3) in RGB
        bgr_mapped = rgb_mapped[..., ::-1]  # convert to BGR for OpenCV
        if res_scale != 1.0:
            bgr_mapped = cv2.resize(bgr_mapped, (out_W, out_H), interpolation=cv2.INTER_NEAREST)

        if is_video_file:
            vidwriter.write(bgr_mapped)
        else:
            if framenames is None:
                frame_path = path / f"frame_{i:05d}.{fileformat}"
            else:
                frame_path = path / f"{framenames[i]}.{fileformat}"
            if fileformat.lower() == "png":
                # higher compression level because there's thousands of frames
                # reminder for anyone reading here; IT'S LOSSLESS COMPRESSION BECAUSE IT'S A PNG
                cv2.imwrite(str(frame_path), bgr_mapped, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            else:
                cv2.imwrite(str(frame_path), bgr_mapped)
            allpaths.append(frame_path)
    if is_video_file:
        vidwriter.release()
    if not is_video_file:
        return allpaths
    return path


def write_frames_tiled(frames, path, sep=1, cmap='viridis'):
    """Save a [N, H, W] float numpy array as a tiled contact-sheet PNG.

    Frames are laid out in a roughly square grid separated by white lines.
    The colormap range is anchored at 0 if all values are non-negative.
    """
    cmap_fn = plt.get_cmap(cmap)
    vmin_g = float(frames.min())
    vmax_g = float(frames.max())
    if vmin_g >= 0:
        vmin_g = 0.0
    vmax_g = max(vmax_g, vmin_g + 1e-8)
    n, Ph, Pw = frames.shape
    ncols = math.ceil(math.sqrt(2.0 * n))
    nrows = math.ceil(n / ncols)
    ch = nrows * Ph + (nrows - 1) * sep
    cw = ncols * Pw + (ncols - 1) * sep
    canvas = np.full((ch, cw), np.nan, dtype=np.float32)
    for f in range(n):
        r, c = divmod(f, ncols)
        y0 = r * (Ph + sep)
        x0 = c * (Pw + sep)
        canvas[y0:y0 + Ph, x0:x0 + Pw] = frames[f]
    norm = (np.clip(canvas, vmin_g, vmax_g) - vmin_g) / (vmax_g - vmin_g)
    norm = np.where(np.isnan(norm), 0.0, norm)
    rgb = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)
    rgb[np.isnan(canvas)] = 255
    cv2.imwrite(str(path), rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 5])


def avi_to_mov(avi_path, mov_path=None):
    """
    Converts an AVI video file to a MOV file using ffmpeg with ProRes codec.
    """
    avi_path = Path(avi_path)
    if mov_path is None:
        mov_path = avi_path.with_suffix(".mov")
    subprocess.run([
        "ffmpeg",
        "-i", str(avi_path),
        "-y",  # override
        "-c:v", "prores_ks",
        "-profile:v", "3",  # 3 = HQ
        "-c:a", "copy",
        str(mov_path)
    ], check=True)
    logger.info(f"Converted {avi_path} to {mov_path}")
    return mov_path


def resize_video(frames, w, h, interpolation=cv2.INTER_NEAREST):
    """Resize every frame of a (T, H, W) stack to (h, w)."""
    Nt, H, W = frames.shape
    resized_frames = np.zeros((Nt, h, w), dtype=frames.dtype)
    for t in tqdm(range(Nt), desc="Resizing video frames"):
        resized_frames[t] = cv2.resize(frames[t], (w, h), interpolation=interpolation)
    return resized_frames


def get_codec_for_format(format: str):
    """
    Get appropriate fourcc codec string for given video format.
    For MP4, tries avc1 first and falls back to mp4v if unavailable.
    """
    format = format.lower()
    if format == "mp4":
        return _get_mp4_codec()
    elif format == "avi":
        return "FFV1"
    elif format == "mov":
        return "avc1"
    else:
        raise ValueError(f"I haven't added the codec for: {format}")


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect C-level stderr (fd 2) to /dev/null for the duration of the block.

    Needed for FFMPEG codec probing: failed codecs print directly to the C stderr
    file descriptor, bypassing Python's logging and sys.stderr entirely.
    """
    old_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull_fd)


_mp4_codec_cache: str | None = None


def _get_mp4_codec() -> str:
    """Select the best available H.264/MP4 codec, cached after the first call.

    Probes avc1 then mp4v. FFMPEG error output during the probe is suppressed
    at the C file-descriptor level since it bypasses Python logging entirely.
    """
    global _mp4_codec_cache
    if _mp4_codec_cache is not None:
        return _mp4_codec_cache
    import tempfile
    test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for codec in ["avc1", "mp4v"]:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            with _suppress_c_stderr():
                writer = cv2.VideoWriter(tmp_path, fourcc, 24, (64, 64), isColor=True)
                writer.write(test_frame)
                writer.release()
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                logger.info(f"MP4 codec selected: {codec}")
                _mp4_codec_cache = codec
                return codec
            else:
                logger.debug(f"MP4 codec '{codec}' produced no output, trying next...")
        except Exception as e:
            logger.debug(f"MP4 codec '{codec}' raised an error: {e}, trying next...")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    raise RuntimeError("No working MP4 codec found (tried avc1, mp4v). Consider using imageio+ffmpeg instead.")
