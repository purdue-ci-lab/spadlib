"""
Read/write utilities for color video and image data (as opposed to the SPAD
quanta data handled by ``spadlib.io``). Also includes simple frame resizing.
"""
import logging
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
    vmin=None, vmax=None, quantile=None, framenames=None
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
        quantile (float or None): if not None, use quantiles to determine vmin and vmax for normalization
            (ignored if vmin or vmax are specified).
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
        if quantile is not None:
            vmax = float(np.quantile(frames, quantile))
        else:
            vmax = float(np.max(frames))
    if vmin is None:
        if quantile is not None:
            vmin = float(np.quantile(frames, 1 - quantile))
        else:
            vmin = float(np.min(frames))
            if vmin >= 0:
                logger.info("vmin was not specified and frames have non-negative values, so using vmin=0 for more accurate scaling")
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
    for i in tqdm(range(max_frames), desc="Writing video frames"):
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
                # (still lossless because it's a PNG)
                cv2.imwrite(str(frame_path), bgr_mapped, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            else:
                cv2.imwrite(str(frame_path), bgr_mapped)
            allpaths.append(frame_path)
    if is_video_file:
        vidwriter.release()
    if not is_video_file:
        return allpaths
    return path


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


def _get_mp4_codec() -> str:
    """
    Test whether avc1 (H.264) is available in this OpenCV build by writing a small
    test video. Falls back to mp4v if not (e.g. Colab silently fails avc1 but mp4v
    works). Raises RuntimeError if neither works.
    """
    import os
    import tempfile
    test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for codec in ["avc1", "mp4v"]:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(tmp_path, fourcc, 24, (64, 64), isColor=True)
            writer.write(test_frame)
            writer.release()
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                logger.info(f"MP4 codec selected: {codec}")
                return codec
            else:
                logger.warning(f"MP4 codec '{codec}' produced no output, trying next...")
        except Exception as e:
            logger.warning(f"MP4 codec '{codec}' raised an error: {e}, trying next...")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    raise RuntimeError("No working MP4 codec found (tried avc1, mp4v). Consider using imageio+ffmpeg instead.")
