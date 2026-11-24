"""
Convert a Phantom high-speed camera .cine file to HDF5, and save an mp4 preview of every
`skip`-th frame.
"""
import spadlib.video_io as vio

cine_path = "path/to/video.cine"
out_h5_path = "path/to/video.h5"
out_video_path = "path/to/video_preview.mp4"
skip = 10  # keep every `skip`-th frame in the mp4

# Lazy (T, H, W) raw sensor frames: only the header is read here, frames are read when indexed
frames, meta = vio.read_video_cine(cine_path, return_meta=True)
print(frames, meta["fps"], meta["bit_depth"])

# Copies a chunk of frames at a time, storing the camera's header as HDF5 attributes
vio.write_video_hdf5(out_h5_path, frames, attrs=meta)

# Reads only every `skip`-th frame from disk, into a numpy array
preview = frames[::skip]

vio.write_video(
    preview,
    out_video_path,
    playback_fps=30,
    vmin=0,
    vmax=2 ** meta["bit_depth"] - 1,  # full range of the decoded values
    cmap="gray",
)
