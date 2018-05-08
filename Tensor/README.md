# Tensor for F#

[![Build Status](https://www.travis-ci.org/DeepMLNet/DeepNet.svg?branch=master)](https://www.travis-ci.org/DeepMLNet/DeepNet)
[![Build status](https://ci.appveyor.com/api/projects/status/7qrfufbj0mvb6llv/branch/master?svg=true)](https://ci.appveyor.com/project/surban/deepnet/branch/master)

A *tensor* is an n-dimensional array of an arbitrary data type (for example `single` or `double`).
Tensors of data type `'T` are implemented by the [Tensor<'T>](xref:Tensor.Tensor`1) type.

A tensor can be either stored in host memory or in the memory of a GPU computing device.
Currently only nVidia cards implementing the [CUDA API](https://developer.nvidia.com/cuda-zone) are supported.
The API for host and GPU stored tensors is mostly equal, thus a program can make use of GPU accelerated operations without porting effort.

The tensor library provides functionality similar to [Numpy's Ndarray](http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.html) and [MATLAB arrays](http://www.mathworks.com/help/matlab/matrices-and-arrays.html), including vector-wise operations, reshaping, slicing, broadcasting, masked assignment, reduction operations and BLAS operations.

This open source library is written in [F#](http://fsharp.org/) and targets the [.NET Standard 2.0 platform](https://github.com/dotnet/standard/blob/master/docs/versions/netstandard2.0.md) with Linux, MacOS and Microsoft Windows as supported operating systems.

### Features provided by the core Tensor library

* Core features
  * n-dimensional arrays (tensors) in host memory or on CUDA GPUs
  * element-wise operations (addition, multiplication, absolute value, etc.)
  * basic linear algebra operations (dot product, SVD decomposition, matrix inverse, etc.)
  * reduction operations (sum, product, average, maximum, arg max, etc.)
  * logic operations (comparison, and, or, etc.)
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
  * Kernel, co-kernel and (pseudo-)inverse
* Matrix decomposition (floating point)
  * Principal component analysis (PCA)
  * ZCA whitening
* Misc
  * Bezout's identity
  * Loading of NumPy's .npy and .npz files.

## Release notes

Release notes are at <http://www.deepml.net/Tensor/articles/ReleaseNotes.html>.

## NuGet packages

The following NuGet packages are available for download.

* [Tensor NuGet package](https://www.nuget.org/packages/Tensor)
* [Tensor.Algorithm NuGet package](https://www.nuget.org/packages/Tensor.Algorithm)

## Documentation

Documentation is provided at <http://www.deepml.net/Tensor>.
