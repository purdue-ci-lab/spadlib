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

### Updating to the latest version of spadlib

I will not bother with releases and versions. I push to main you get the main.

```shell
uv lock --upgrade-package spadlib
uv sync
```

## Quick start

```python
import spadlib.io as spio
import spadlib.processing as spproc
import spadlib.video_io as vio

# lazy (T, H, W) array: frames are only read from disk when indexed
frames = spio.read_quanta_auto("path/to/quanta/data", H=1024, W=1024)  # SPAD Alpha

corrected = spproc.correct_hotpixels(frames[:], bayer=True)  # bayer=False for SPAD512
spio.write_quanta_hdf5("path/to/hotpixel_corrected.h5", corrected, fps=57_000)

summed = spproc.accumulate_frames(corrected, 100)  # sum every 100 frames
vio.write_video(summed, "path/to/summed.mp4", playback_fps=30, gamma=1/2.2, cmap="gray")
```

## Examples

More complete scripts are in [`examples/`](examples/):

- [`process_quanta.py`](examples/process_quanta.py): read quanta frames lazily, correct
  hot pixels, thin them and add dark counts, and save the result to HDF5 and an mp4 of
  the summed frames.
- [`simulate_quanta_from_video.py`](examples/simulate_quanta_from_video.py): simulate
  quanta frames from an intensity video, with SPAD Alpha dark counts and hot pixels.
- [`convert_cine.py`](examples/convert_cine.py): convert a Phantom `.cine` file to HDF5
  and save an mp4 preview of every Nth frame.

The paths in the examples are placeholders; replace them with your own data.

## Data file formats

The file format that the sensors spit out (`.bin` files, etc) are uncompressed and awful, so
for our purposes we standardize it as other formats.

HDF5 (`.h5`, `.hdf5`) is now my preferred format to convert these files, since it is
compressed, fits into a single file, and stores metadata.

Previously we used Zarr files, which were also great for compression and metadata but
unfortunately it splits chunks into many files which plays poorly with file sharing.

| HDF5 | Zarr |
| ---- | ---- |
| saves as one file, easy to share | saves as thousands of files, awful to transfer and share |
| slightly worse compression for some reason | slightly better compression for some reason |
| kind of slow to save | fast to save (can multithread) |
| reading is not super fast | reading is fast |
| reading supported by many languages | mainly just Python |

So Zarr is generally better, but the fact to even share one, it needs to be zipped (or even worse, its files are uploaded one-by-one to the cloud), it just makes data sharing and publishing infinitely annoying. Thus, let's use HDF5.

This library can read and convert between the sensor output formats and the
alternative HDF5 and Zarr formats.

## Contributing

```shell
uv sync
uv run pytest
```

Open issues, pull requests, whatever.
