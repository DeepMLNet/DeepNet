# Status

The library is nearly feature complete, but some limitations exists that we hope to resolve in the near future.

## Current limitations

* Although we use 64-bit indices, tensors stored on the host are internally limited to 32-bit indices and thus to a maximum of 2,147,483,647 elements.
This limitation is imposed by using .NET arrays as the storage backend for tensors stored on the host.
We will implement a backend using native (unmanaged) host memory to work around this limitation.
This limitation does not apply to tensors stored on CUDA GPUs.
* On CUDA GPUs only primitive data types (like single, float, boolean, int) are supported.
* The following operations are not yet implemented on CUDA GPUs.
Invoking them will raise an exception.
  * masking
  * countTrue
  * trueIdx
  * SVD
  * symmetricEigendecomposition
* Some useful linear algebra functions like LU and QR decomposition are missing.
* No standarized interface for exposing tensors to native code is yet available.
However, getting pointers to their memory already works fine.
* MacOS is unsupported due to the lack of development hardware.
* No backend exists yet for [OpenCL](https://en.wikipedia.org/wiki/OpenCL) hardware and thus AMD and Intel GPUs are not supported.
* The host backend is currently hard coded to use [Intel MKL](https://software.intel.com/en-us/mkl) as its BLAS library.
It would be nice, if we could dynamically switch the BLAS library, for example to use the free [OpenBLAS](https://www.openblas.net/).

Don't be shy to get involved with development and submit pull requests at <https://github.com/DeepMLNet/DeepNet/pulls>.

## Issues

Issues are tracked via GitHub at <https://github.com/DeepMLNet/DeepNet/issues>.

## API stability

Until we reach version 1.0 the API is *not* guaranteed to be stable.
Function signatures may change in incompatible ways or functions may be removed altogether.
