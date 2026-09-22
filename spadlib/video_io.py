"""
Read/write utilities for color video and image data (as opposed to the SPAD
quanta data handled by ``spadlib.io``). Also includes simple frame resizing.
"""
import contextlib
import logging
import math
import os
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from spadlib.utils import natural_sort_key


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
