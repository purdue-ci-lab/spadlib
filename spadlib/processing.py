"""
Data-modifying processing operations for SPAD/quanta data: photon thinning,
dark-count-rate (DCR) injection, and hot/dead pixel correction.
"""
import logging

import cv2
import dask.array as da
import numpy as np
from tqdm import tqdm

try:
    import cupy as cp
except ImportError:
    cp = None


logger = logging.getLogger(__name__)


def prob_from_dcr(dcr_rate_hz, fps):
    """
    Convert a dark count rate in Hz to a per-frame probability of a dark count photon.
    """
    return 1 - np.exp(-dcr_rate_hz / fps)


def thin_events_uniform(events, keep_prob, dcr_rate=None, return_idx=False, seed=None, cuda=False):
    """
    Thin events uniformly with probability keep_prob.
    """
    xp = cp if cuda else np
    rng = xp.random.default_rng(seed=seed)
    mask = rng.random(events.shape[0]) < keep_prob
    if return_idx:
        return xp.nonzero(mask)
    return events[mask]


def thin_pixel_timeseries(pixel_timeseries, keep_prob, dcr_rate=None, seed=None, cuda=False):
    """
    Thin events uniformly with probability keep_prob, applied to a list of lists of pixel timeseries.

    Originally thin_events_pixel_timeseries(pixel_timeseries, keep_prob, dcr_rate, seed, cuda).

    Args:
        pixel_timeseries: list of lists of arrays, where pixel_timeseries[y][x] is an array of timestamps for that pixel.
        keep_prob: probability to keep each event.
        dcr_rate: if not None, will add dark count events with the given rate (in Hz) to each pixel after thinning.
    """
    xp = cp if cuda else np
    rng = xp.random.default_rng(seed=seed)
    thinned_pixel_timeseries = [[None] * len(row) for row in pixel_timeseries]
    pbar = tqdm(total=len(pixel_timeseries) * len(pixel_timeseries[0]), desc="Thinning events in pixels")
    for y in range(len(pixel_timeseries)):
        for x in range(len(pixel_timeseries[y])):
            if pixel_timeseries[y][x] is not None:
                events = pixel_timeseries[y][x]
                mask = rng.random(events.shape[0]) < keep_prob
                thinned_events = events[mask]
                if dcr_rate is not None:
                    dcr_prob = prob_from_dcr(dcr_rate, fps=1 / np.median(np.diff(events)))  # approximate fps from median inter-event time
                    n_dcr_events = rng.binomial(len(thinned_events), dcr_prob)
                    dcr_events = rng.uniform(0, events.max(), size=n_dcr_events)
                    thinned_events = xp.sort(xp.concatenate([thinned_events, dcr_events]))
                thinned_pixel_timeseries[y][x] = thinned_events
                pbar.update(1)
    return thinned_pixel_timeseries


def thin_frames_uniform(frames, p, dcr_prob=None, t_chunk=1000, seed=None):
    """
    Thin binary quanta frames uniformly, keeping each photon with probability `p`
    and optionally injecting dark counts. Processes the stack in temporal chunks.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if p == 1.0:
        return
    frames = np.copy(frames)  # avoid modifying in-place
    P = frames.shape[0]
    rng = np.random.default_rng(seed=seed)
    for t0 in tqdm(range(0, P, t_chunk), desc="Thinning frames in chunks"):
        t1 = min(P, t0 + t_chunk)
        slab = frames[t0:t1]  # (Bt,H,W) uint8
        U = rng.random(slab.shape)  # float64, ephemeral
        keep = (slab > 0) & (U < p)
        if dcr_prob is not None:
            dcr_mask = rng.random(slab.shape) < dcr_prob
            keep = keep | dcr_mask
        slab[...] = 0
        slab[keep] = 1
        frames[t0:t1] = slab
    return frames


def thin_frames_uniform_dask(frames, keep_prob, dcr_prob=None, seed=None):
    """
    Thin binary frames uniformly with probability keep_prob. Also adds dark
    count photons to lower the SNR.

    Originally thin_frames_uniform_da(frames, keep_prob, dcr_prob, seed).

    This is an expensive operation on SPAD data, so dask is used for
    multiprocessing.

    Args:
        dcr_prob: dark count photon probability (not the rate itself)
    """
    T, H, W = frames.shape
    # convert to a dask array with automatic chunking and apply a lazy random mask
    frames = da.from_array(frames, chunks=(400, H, W))
    rs = da.random.RandomState(seed)
    mask = rs.random(frames.shape, chunks=frames.chunks) < keep_prob
    frames = (frames.astype("uint8") & mask.astype("uint8"))
    if dcr_prob is not None:
        dcr_photons = rs.binomial(1, dcr_prob, size=frames.shape, chunks=frames.chunks).astype("uint8")
        frames = frames | dcr_photons
    out = frames.compute()
    del frames
    return out


def thin_frames_counts_dask(frames, keep_prob, dcr_rate=None, seed=None):
    """
    Thin photon-count frames by independently dropping each photon with
    probability (1 - keep_prob). Also adds dark count photons to lower the SNR.

    Originally thin_frames_counts_da(frames, keep_prob, dcr_rate, seed).

    For each pixel with count k, the surviving count is drawn from
    Binomial(k, keep_prob). Dark counts are drawn from Poisson(dcr_rate).

    This is an expensive operation on SPAD data, so dask is used for
    multiprocessing.

    Args:
        frames:     Array of shape (T, H, W) with non-negative integer photon counts.
        keep_prob:  Probability that any individual photon survives thinning.
        dcr_rate:   Expected number of dark count photons added per pixel per frame.
                    (Poisson-distributed; pass None or 0 to skip.)
        seed:       Optional integer seed for reproducibility.
    """
    T, H, W = frames.shape
    frames = da.from_array(frames, chunks=(400, H, W))
    rs = da.random.RandomState(seed)

    # Each photon survives independently with keep_prob → Binomial thinning
    thinned = rs.binomial(frames, keep_prob, size=frames.shape, chunks=frames.chunks)

    # Dark counts: Poisson(dcr_rate) per pixel
    if dcr_rate is not None and dcr_rate > 0:
        dcr_photons = rs.poisson(dcr_rate, size=frames.shape, chunks=frames.chunks)
        thinned = thinned + dcr_photons

    out = thinned.compute()
    del thinned
    del frames
    return out


def correct_hotpixels(frames, probablity_base="hotpixels", hp_threshold: float = 2.0, kernel_size: int = 3):
    """
    Correct hot pixels in SPAD data using the SPADHotpixelTool.

    Args:
        frames: 3D numpy array of shape (T, H, W) representing the SPAD data.
        probablity_base: Base for probability calculation, either "hotpixels" or "all".
        hp_threshold: Threshold multiplier for hot pixel detection.
        kernel_size: Size of the kernel for background estimation.
    """
    hotpixtool = SPADHotpixelTool(frames)
    return hotpixtool.correct_hotpixels(probablity_base=probablity_base, hp_threshold=hp_threshold, kernel_size=kernel_size)


def correct_hotpixels_conv(arr, thresh_std=5.0, kernel_size=5):
    """
    Hot pixel correction via convolutional neighbor statistics.
    """
    arr = arr.copy()
    # count events per pixel over the whole timespan
    pixel_event_counts = np.sum(arr, axis=0).astype(float)

    # neighborhood sums (including center)
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    total = cv2.filter2D(pixel_event_counts, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # neighbors = total - center
    sum_neighbors = total - pixel_event_counts

    # how many neighbors each pixel has (edges will have fewer than nxn-1)
    neighbor_counts = cv2.filter2D(np.ones_like(pixel_event_counts, dtype=float), -1, kernel, borderType=cv2.BORDER_CONSTANT) - 1.0
    neighbor_counts_safe = np.where(neighbor_counts == 0, 1.0, neighbor_counts)

    # neighbor mean
    neighbor_mean = sum_neighbors / neighbor_counts_safe

    # neighbor std via mean of squares
    total_sq = cv2.filter2D(pixel_event_counts ** 2, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    sumsq_neighbors = total_sq - pixel_event_counts ** 2
    neighbor_mean_sq = sumsq_neighbors / neighbor_counts_safe
    neighbor_var = np.maximum(neighbor_mean_sq - neighbor_mean ** 2, 0.0)
    neighbor_std = np.sqrt(neighbor_var)

    # mark as hot if significantly above neighbors (hotpix_thresh acts as multiplier of std)
    hot_pixels = pixel_event_counts > (neighbor_mean + thresh_std * neighbor_std)
    # if neighbor_std is zero, mark only if strictly greater than neighbor mean
    hot_pixels = hot_pixels | ((neighbor_std == 0) & (pixel_event_counts > neighbor_mean))

    # zero out all hot pixels across time
    # arr is expected shape (T, H, W)
    try:
        arr[:, hot_pixels] = 0
    except Exception:
        # fallback: assign per-frame if boolean advanced indexing fails
        for t in range(arr.shape[0]):
            arr[t][hot_pixels] = 0

    # For each hot pixel, iterate over time and probabilistically revive events
    # so that pixels with more active neighbors have a higher chance of revival.
    # Probability is proportional to the number of active neighbors, capped at 1.0.
    rng = np.random.default_rng()
    nframes = arr.shape[0]
    for t in tqdm(range(nframes), total=nframes, desc="Reviving hot pixels for each frame"):
        frame = arr[t].astype(float)
        neigh_total = cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        neigh_count = (neigh_total - frame)

        # probability of revival per pixel: fraction of neighbors that are active
        # (neighbor_counts_safe is from earlier; avoids division by zero)
        probs = neigh_count / neighbor_counts_safe
        probs = np.clip(probs, 0.0, 1.0)

        # draw random numbers and revive hot pixels according to probs
        rnd = rng.random(size=probs.shape)
        revive_mask = (rnd < probs) & hot_pixels

        if revive_mask.any():
            arr[t][revive_mask] = 1

    return arr, hot_pixels, neighbor_mean


class SPADHotpixelTool:
    """
    Ripped directly from bit2bit.

    Class to facilitate the hotpixel correction of SPAD data.

    :param data: 3D np array
    :param kwargs: Additional parameters
    """

    def __init__(self, data: np.ndarray, **kwargs):
        """Constructor method."""
        self.data = data
        self.flattend_data = self._flatten(self.data)
        if (zero_count := self._zero_count()) > 512:
            logger.warning(f"Possible insufficient data for correction. Zero count: {zero_count}")
        self.reset()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self, *args: str) -> "SPADHotpixelTool":
        """Reset the intermediate results of the hotpixel correction.

        :param args: Attributes to reset
        :return: The instance of the class
        """
        if not args:
            self.background = None
            self.hotpixel_image = None
            self.hotpixel_locations = None
            self.deadpixel_locations = None
            self.hotpixel_values = None
            self.expected_values = None
            self.probability_list = None
            self.corrected_image = None
        for arg in args:
            try:
                setattr(self, arg, None)
            except AttributeError:
                logger.error(f"Attribute {arg} not found")
        return self

    def _zero_count(self) -> int:
        """Return the number of zero in the flattened data."""
        count = np.sum(self.flattend_data < 1).item()
        return count

    @staticmethod
    def _flatten(data: np.ndarray) -> np.ndarray:
        """Reduce the dimension of the input data to 2D and save it as a np array."""
        if data.ndim > 2:
            data = np.sum(data, axis=tuple(range(data.ndim - 2)))
        return data.astype(np.float32)

    def get_background(self, method: str = "open", kernel_size: int = 3) -> np.ndarray:
        """Apply background filter to the input data and return the last result.

        :param method: Method for background filter, defaults to "open"
        :return: Data after background filter
        """
        image = self.flattend_data
        methods = {
            "open": lambda i, k: cv2.morphologyEx(
                i,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            ),
            "mean": lambda i, k: cv2.blur(i, (k, k)),
            "median": lambda i, k: cv2.medianBlur(i.astype(np.float32), k),
            "median+mean": lambda i, k: methods["median"](methods["mean"](i, k), k),
        }
        if method in methods:
            self.background = methods[method](image, kernel_size)
        else:
            logger.error(f"Method {method} not implemented")
            raise NotImplementedError
        return self.background

    def get_hotpixels(self, kernel_size: int = 3) -> np.ndarray:
        """Subtract background from the input data to get hot pixels.

        :return: Data after hot pixel filter
        """
        if self.hotpixel_image is None:
            background_image = self.get_background(kernel_size=kernel_size)
            self.hotpixel_image = self.flattend_data - background_image
        return self.hotpixel_image

    def locate_hotpixels(
        self,
        hp_threshold: float = 2,
        **kwargs,
    ) -> np.ndarray:
        """Find hot pixels in the input data based on the standard deviation threshold.

        :param hp_threshold: Multiplyer for the standard deviation, defaults to 2
        :return: The locations of the hot pixels
        """
        if self.hotpixel_locations is None:
            hp_threshold = kwargs.get("hp_threshold", hp_threshold)
            hotpixel_image = self.get_hotpixels(**kwargs)
            mean = np.mean(hotpixel_image).item()
            std = np.std(hotpixel_image).item()
            hp_threshold = mean + hp_threshold * std
            self.hotpixel_locations = np.argwhere(hotpixel_image > hp_threshold)
        return self.hotpixel_locations

    def locate_deadpixels(self, **kwargs) -> np.ndarray:
        """Find dead pixels in the input data based on the standard deviation dp_threshold.

        :param dp_threshold: Multiplyer for the standard deviation, defaults to 1
        :return: The locations of the dead pixels
        """
        if self.deadpixel_locations is None:
            dp_threshold = kwargs.get("dp_threshold", 1)
            self.deadpixel_locations = np.argwhere(self.flattend_data < dp_threshold)
        return self.deadpixel_locations

    def correct_deadpixels(self, **kwargs) -> np.ndarray:
        """Correct the input data for dead pixels.

        :return: The corrected image
        """
        image = np.array(self.data) if self.corrected_image is None else self.corrected_image
        index = self.locate_deadpixels(**kwargs)
        kernel_size: int = kwargs.get("kernel_size", 3)
        background_image = self.get_background(kernel_size=kernel_size, method="median")
        expected_values = background_image[index[:, 0], index[:, 1]]
        for (x, y), value in zip(index, expected_values, strict=False):
            prob = value / image.shape[0]
            image[:, x, y] = np.random.binomial(1, prob, image.shape[0])
        self.corrected_image = image
        return self.corrected_image

    def get_hotpixel_values(self, **kwargs) -> np.ndarray:
        """Get the values of the hot pixels in the input data.

        :return: The values of the hot pixels
        """
        if self.hotpixel_values is None:
            image = self.flattend_data
            index = self.locate_hotpixels(**kwargs)
            self.hotpixel_values = image[index[:, 0], index[:, 1]]
        return self.hotpixel_values

    def get_expected_values(self, **kwargs) -> np.ndarray:
        """Get the values of the hot pixels in the background image.

        :return: The values of the hot pixels
        """
        if self.expected_values is None:
            kernel_size: int = kwargs.get("kernel_size", 3)
            background_image = self.get_background(kernel_size=kernel_size)
            index = self.locate_hotpixels(**kwargs)
            self.expected_values = background_image[index[:, 0], index[:, 1]]
        return self.expected_values

    def get_probablity(self, **kwargs) -> np.ndarray:
        """Get the probability of a hot pixel.

        :param probablity_base: Probability base, defaults to "hotpixels"
        :return: The probability of a hot pixel
        """
        if self.probability_list is None:
            probablity_base = kwargs.get("probablity_base", "hotpixels")
            expected_values = self.get_expected_values(**kwargs)
            if probablity_base == "all":
                self.probability_list = expected_values / self.data.shape[0]
            elif probablity_base == "hotpixels":
                hotpixel_values = self.get_hotpixel_values(**kwargs)
                self.probability_list = expected_values / hotpixel_values
            else:
                logger.error(f"Probability base {probablity_base} not implemented")
                raise NotImplementedError
        return self.probability_list

    def correct_hotpixels(self, probablity_base="hotpixels", hp_threshold: float = 2.0, kernel_size: int = 3) -> np.ndarray:
        """Correct the input data for hot pixels.

        :param probablity_base: Probability base, defaults to "hotpixels"
        :return: The corrected image
        """
        image = np.array(self.data) if self.corrected_image is None else self.corrected_image
        index = self.locate_hotpixels(hp_threshold=hp_threshold, kernel_size=kernel_size)
        p = self.get_probablity(probablity_base=probablity_base, hp_threshold=hp_threshold, kernel_size=kernel_size)
        n_iters = len(index)
        # should make np.nonzero faster... I think?
        image = np.asarray(image, dtype="bool")
        for (x, y), prob in tqdm(zip(index, p, strict=False), total=n_iters):
            if probablity_base == "all":
                filling = np.random.binomial(1, prob, image.shape[0])
                image[:, x, y] = filling
            elif probablity_base == "hotpixels":
                z_idx = np.flatnonzero(image[:, x, y])
                image[z_idx, x, y] = np.random.binomial(1, prob, z_idx.size)
            else:
                logger.error(f"Probability base {probablity_base} not implemented")
        self.corrected_image = np.asarray(image, dtype="uint8")
        return self.corrected_image

    def inspect(self, **kwargs):
        """Show image results of the hotpixel correction."""
        try:
            import matplotlib.pyplot as plt

            deadpixel = kwargs.get("deadpixel", False)
            if deadpixel:
                self.correct_deadpixels(**kwargs)
            _, ax = plt.subplots(1, 2, figsize=(20, 40))
            ax[0].imshow(self.flattend_data, cmap="gray")
            ax[0].scatter(
                self.locate_hotpixels(**kwargs)[:, 1] + 3,
                self.locate_hotpixels(**kwargs)[:, 0],
                c="r",
                s=5,
                marker="_",
            )
            if deadpixel:
                ax[0].scatter(
                    self.locate_deadpixels(**kwargs)[:, 1] + 3,
                    self.locate_deadpixels(**kwargs)[:, 0],
                    c="g",
                    s=5,
                    marker="_",
                )
            title = f"Hotpixels (count: {self.locate_hotpixels(**kwargs).shape[0]})"
            if deadpixel:
                title += f" & Deadpixels (count: {self.locate_deadpixels(**kwargs).shape[0]})"
            ax[0].set_title(title)
            ax[0].set_axis_off()
            ax[1].imshow(np.sum(self.correct_hotpixels(**kwargs), axis=0), cmap="gray")
            ax[1].scatter(
                self.locate_hotpixels(**kwargs)[:, 1] + 3,
                self.locate_hotpixels(**kwargs)[:, 0],
                c="r",
                s=5,
                alpha=0.5,
                marker="_",
            )
            if deadpixel:
                ax[1].scatter(
                    self.locate_deadpixels(**kwargs)[:, 1] + 3,
                    self.locate_deadpixels(**kwargs)[:, 0],
                    c="g",
                    s=5,
                    alpha=0.5,
                    marker="_",
                )
            ax[1].set_title("Projection without hotpixels")
            ax[1].set_axis_off()
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.show()
        except ImportError:
            logger.error("Matplotlib not installed")
