# spadlib

Collection of utilities for reading/writing single-photon data from quanta and asynchronous SPAD sensors.

## Installation

```shell
pip install git+https://github.com/jerukan/spadlib.git
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
