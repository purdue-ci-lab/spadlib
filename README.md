# spadlib

Collection of utilities for reading/writing single-photon data from quanta and asynchronous SPAD sensors.

## Installation

```shell
pip install git+https://github.com/purdue-ci-lab/spadlib.git
```

While this repo is private, you should install it with ssh instead.

```shell
pip install git+ssh://git@github.com:purdue-ci-lab/spadlib.git
```

CUDA support with CuPy depends on your CUDA Toolkit version

```shell
# CUDA 12.x
pip install cupy-cuda12x
# CUDA 13.x
pip install cupy-cuda13x
```

If the CUDA Toolkit is not installed already, install CuPy like the following:


```shell
pip install "cupy-cuda12x[ctk]"
```

## Usage

```python
import spadlib.io as spio
import spadlib.video_io as vio
import spadlib.processing as spproc

quanta_path = "path/to/quanta/data"
frames = spio.read_quanta_auto(
    quanta_path,
    load_data=False,  # lazily load array
    H=1024, W=1024  # for SPAD Alpha
)

summed_frames = []
sum_every = 100
for i in range(0, frames.shape[0], sum_every):
    summed_frames.append(np.sum(frames[i:i+sum_every], axis=0))
summed_frames = np.array(summed_frames)

vio.write_video(
    summed_frames,
    "path/to/video.mp4",  # exclude the extension to save as frames
    playback_fps=30,
    gamma=1/2.2,
    cmap="grey"  # supports any matplotlib colormap
)

hotpixel_corrected_frames = spproc.correct_hotpixels(
    frames[:],
    bayer=True  # set to False if correcting SPAD 512 data
)

# convert to HDF5, which is quicker to read, has metadata, and stores compressed
spio.write_quanta_hdf5("path/to/hotpixcorrected.h5", hotpixel_corrected_frames, fps=57_000)
```

## Data file formats

The file format that the sensors spit out (`.bin` files, etc) are uncompressed and awful, so
for our purposes we standardize it as other formats.

HDF5 (`.h5`, `.hdf5`) is now my preferred format to convert these files, since it is
compressed, fits into a single file, and stores metadata.

Previously we used Zarr files, which were also great for compression and metadata but
unfortunately it splits chunks into many files which plays poorly with file sharing.

This library can read and convert between the sensor output formats and the
alternative HDF5 and Zarr formats.

## Contributing

```shell
uv sync
uv run pytest
```
