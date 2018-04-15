# Overview

A *tensor* is an n-dimensional array of an arbitrary data type (for example `single` or `double`).
Tensors of data type `'T` are implemented by the [Tensor<'T>](xref:Tensor.Tensor`1) type.

A tensor can be either stored in host memory or in the memory of a GPU computing device.
Currenty only nVidia cards implementing the [CUDA API](https://developer.nvidia.com/cuda-zone) are supported.
The API for host and GPU stored tensors is mostly equal, thus a program can make use of GPU accelerated operations without porting effort.

The tensor library provides functionality similar to [Numpy's Ndarray](http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.html) and [MATLAB arrays](http://www.mathworks.com/help/matlab/matrices-and-arrays.html), including vector-wise operations, reshaping, slicing, broadcasting, masked assignment, reduction operations and BLAS operations.

This open source library is written in [F#](http://fsharp.org/) and targets the [.NET Standard 2.0 platform](https://github.com/dotnet/standard/blob/master/docs/versions/netstandard2.0.md) with Linux and Microsoft Windows as supported operating systems.

## Documentation

To get an overview of what the library can do, see [Tensor at a glance](Tensor.md).

To start using the library, follow the [installation guide](Guide-Installation.md) and the [getting started guide](Guide-Getting-Started.md).
