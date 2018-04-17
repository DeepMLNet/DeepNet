# Tensor

A *tensor* is an n-dimensional array of an arbitrary data type (for example `single` or `double`).
Tensors of data type `'T` are implemented by the [Tensor<'T>](xref:Tensor.Tensor`1) type.

A tensor can be either stored in host memory or in the memory of a GPU computing device.
Currenty only nVidia cards implementing the [CUDA API](https://developer.nvidia.com/cuda-zone) are supported.
The API for host and GPU stored tensors is mostly equal, thus a program can make use of GPU accelerated operations without porting effort.

The tensor library provides functionality similar to [Numpy's Ndarray](http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.html) and [MATLAB arrays](http://www.mathworks.com/help/matlab/matrices-and-arrays.html), including vector-wise operations, reshaping, slicing, broadcasting, masked assignment, reduction operations and BLAS operations.

This open source library is written in [F#](http://fsharp.org/) and targets the [.NET Standard 2.0 platform](https://github.com/dotnet/standard/blob/master/docs/versions/netstandard2.0.md) with Linux and Microsoft Windows as supported operating systems.

### Features provided by the core Tensor library

* Core features
  * n-dimensional arrays (tensors) in host memory or on CUDA GPUs 
  * element-wise operations (addition, multiplication, absolute value, etc.)
  * basic linear algebra operations (dot product, SVD decomposition, matrix inverse, etc.)
  * reduction operations (sum, product, average, maximum, arg max, etc.)
  * logic operations (comparision, and, or, etc.)
  * views, slicing, reshaping, broadcasting (similar to NumPy) 
  * scatter and gather by indices
  * standard functional operations (map, fold, etc.)
* Data exchange
  * read/write support for HDF5 (.h5)
  * interop with standard F# types (Seq, List, Array, Array2D, Array3D, etc.)
* Performance
  * host: SIMD and Intel MKL accelerated operations 
  * CUDA GPU: all operations performed locally on GPU and cuBLAS used for matrix operations

### Additional features provided by Tensor.Algorithm

* Matrix algebra (integer, rational)
  * Row echelon form
  * Smith normal form
  * Kernel, cokernel and (pseudo-)inverse
* Matrix decomposition (floating point)
  * Principal component analysis (PCA)
  * ZCA whitening
* Misc
  * Bezout's identity
  * Loading of NumPy's .npy and .npz files.

## Documentation

To get an overview of available functions by category, see [Tensor at a glance](articles/Tensor.md).
We also provide [full reference documentation](xref:Tensor).

To start using the library, follow the [installation guide](articles/Guide-Installation.md) and the [getting started guide](articles/Guide-Creation.md).

Current limitations are documented on the [status page](articles/Status.md).
