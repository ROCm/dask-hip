Dask HIP 
=========

> [!CAUTION] 
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.
***

> [!NOTE]
> This README is derived from the original RAPIDSAI project's README. More care is necessary to remove/modify parts that are only applicable to the original version.

> [!NOTE]
> This repository will be eventually moved to the [ROCm-DS](https://github.com/rocm-ds) Github organization.

> [!NOTE]
> This ROCm&trade; port is derived from the NVIDIA RAPIDS&reg; dask-cuda project. It aims to
follow the latter's directory structure, file naming and API naming as closely as possible to minimize porting friction for users that are interested in using both projects.


Dask HIP is derived from DASK CUDA. 

It contains various utilities to improve deployment and management of Dask workers on
HIP-enabled systems.

This library is experimental, and its API is subject to change at any time
without notice.

Example
-------

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster()
client = Client(cluster)
```

Documentation is available [here](https://docs.rapids.ai/api/dask-cuda/nightly/).

What this is not
----------------

This library does not automatically convert your Dask code to run on GPUs.

It only helps with deployment and management of Dask workers in multi-GPU
systems.  Parallelizing GPU libraries like [RAPIDS](https://rapids.ai) and
[CuPy](https://cupy.chainer.org) with Dask is an ongoing effort.  You may wish
to read about this effort at [blog.dask.org](https://blog.dask.org) for more
information.  Additional information about Dask-CUDA can also be found in the
[docs](https://docs.rapids.ai/api/dask-cuda/nightly/).
