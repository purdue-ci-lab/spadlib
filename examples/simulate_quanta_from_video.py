"""
Simulate SPAD Alpha quanta frames from a regular intensity video, including the sensor's
dark counts and hot pixels.
"""
import numpy as np

import spadlib.io as spio
import spadlib.processing as spproc
import spadlib.sim as spsim
import spadlib.video_io as vio

video_path = "path/to/intensity_video.mp4"
out_h5_path = "path/to/simulated_quanta.h5"
fps = 57_000  # physical frame rate the video was captured at (not its playback fps); quanta inherit it
target_ppp = 0.1  # mean photons per pixel per quanta frame at the brightest pixel
H, W = 1024, 1024  # SPAD Alpha resolution

# (T, H, W) uint8 grayscale frames, resized to the sensor resolution
video = vio.read_video(video_path, grayscale=True)
video = vio.resize_video(video, W, H)

# Linearize: undo the video's gamma encoding (roughly sRGB)
intensity = (video.astype(np.float32) / 255) ** 2.2

# Normalize to [0, 1]
intensity /= intensity.max()

# Scale to the target photon level, as a photon flux in photons/sec/pixel
flux = intensity * target_ppp * fps

# Per-pixel probability of detecting at least one photon in a quanta frame
prob = spsim.photons_to_prob(flux, fps=fps)

# Bernoulli-sample one binary quanta frame per video frame
quanta = spsim.simulate_quanta(prob)

# Add SPAD Alpha dark counts and hot pixels in place (measured in complete darkness)
quanta = spproc.add_darkcounts_lognormal_frames(
    quanta, spproc.SPADALPHA_DARK_MEAN, spproc.SPADALPHA_DARK_SIGMA, copy=False
)
quanta = spproc.add_hotpixels_frames(
    quanta, spproc.SPADALPHA_HOT_PROB, spproc.SPADALPHA_HOT_MIN_PROB, copy=False
)

spio.write_quanta_hdf5(out_h5_path, quanta, fps=fps)
