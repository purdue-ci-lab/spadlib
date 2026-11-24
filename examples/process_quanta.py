"""
Read raw quanta frames, correct hot pixels, thin them and add dark counts, and save the
result to HDF5 along with an mp4 of the frames summed in groups.
"""
import spadlib.io as spio
import spadlib.processing as spproc
import spadlib.video_io as vio

quanta_path = "path/to/quanta/data"  # a .bin/.mat file, a directory of them, or .h5/.zarr
out_h5_path = "path/to/processed.h5"
out_video_path = "path/to/summed.mp4"
fps = 57_000  # quanta frame rate of the capture
keep_prob = 0.1  # probability of keeping each photon when thinning
sum_every = 100  # quanta frames summed into each video frame

# Lazy (T, H, W) array: only headers are read here, frames are read from disk when indexed
frames = spio.read_quanta_auto(quanta_path, H=1024, W=1024)  # .bin files need H, W given
print(frames.shape, frames.dtype)

# Reads only the first 10k frames into memory (frames[:] loads everything)
frames = frames[:10_000]

# Corrects each 2x2 Bayer plane separately for the SPAD Alpha; use bayer=False for SPAD512
corrected = spproc.correct_hotpixels(frames, bayer=True)

# Keep each photon with probability `keep_prob`, simulating a dimmer scene
thinned = spproc.thin_frames_uniform(corrected, keep_prob)

# Add SPAD Alpha dark counts in place, with per-pixel lognormal probabilities
thinned = spproc.add_darkcounts_lognormal_frames(
    thinned, spproc.SPADALPHA_DARK_MEAN, spproc.SPADALPHA_DARK_SIGMA, copy=False
)

# HDF5 is compressed, stores metadata, and is much quicker to read than raw sensor files
spio.write_quanta_hdf5(out_h5_path, thinned, fps=fps)

# Sum every `sum_every` binary frames into one photon-count frame: (T // sum_every, H, W)
summed = spproc.accumulate_frames(thinned, sum_every)

vio.write_video(
    summed,
    out_video_path,  # leave off the extension to save a directory of PNG frames instead
    playback_fps=30,
    gamma=1 / 2.2,
    cmap="gray",  # any matplotlib colormap
)
