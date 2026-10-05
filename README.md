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
```

## Contributing

```shell
uv sync
uv run pytest
```
