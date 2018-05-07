# Installation and testing

This page guides you through installing the Tensor library and creating a skeleton project for experimentation.

## System requirements

The following system requirements must be met.

* System architecture: x86-64 (AMD64 or Intel 64)
* Operating system: Linux, MacOS or Microsoft Windows
* Microsoft .NET Standard 2.0 implementation
  * Recommended platform is [.NET Core >= 2.0](https://www.microsoft.com/net/learn/get-started)
  * .NET Framework >= 4.7 is supported
  * [Mono](https://www.mono-project.com/download/stable/) >= 5.10 is supported, but significantly slower
* For Linux
  * The library `libgomp.so.1` must be installed. (install on Ubuntu by running `sudo apt install libgomp1`)
* For GPU acceleration (optional)
  * nVidia GPU supporting [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) 3.5 or higher
  * [nVidia GPU driver](http://www.nvidia.com/Download/index.aspx) 387.92 or higher

## Installation

The library is delivered in two NuGet packages.
The [Tensor NuGet package](https://www.nuget.org/packages/Tensor) provides the [Tensor<'T>](xref:Tensor.Tensor`1) type and all core functions.
Additional algorithms and data exchange methods are provided in the [Tensor.Algorithm NuGet package](https://www.nuget.org/packages/Tensor.Algorithm).

The packages can be installed into your project by installing the `Tensor` and `Tensor.Algorithm` packages using the NuGet package manager (either via command line or graphical interface).

## Skeleton project for .NET Core

In the course of this tutorial you will use the following skeleton project for experimentation.
We assume that you are using [.NET Core 2.0](https://www.microsoft.com/net/learn/get-started) on either Linux or Windows for the rest of the tutorial.

To create the skeleton project run the following commands.
```
$ mkdir tutorial
$ cd tutorial
$ dotnet new console -lang F#
```

Then, run the following commands to install the Tensor library into your project.
```
$ dotnet add package Tensor
$ dotnet add package Tensor.Algorithm
```

### Basic verificiation test

To verify that the installation was successful you can perform a basic test of the library.
Place the following code into `Program.fs`.
```fsharp
open Tensor
[<EntryPoint>]
let main argv =
    let x = HostTensor.counting 6L
    printfn "x = %A" x
    0
```
If everything works fine, `dotnet run` automatically builds your project and produces the following output.
```
$ dotnet run
x = [   0    1    2    3    4    5]
```

### GPU acceleration verification test

By changing `HostTensor` to `CudaTensor` inside `Program.fs` and executing `dotnet run`, you can test if GPU acceleration works properly.

## Source code and issues

The source code of the Tensor library is available at <https://github.com/DeepMLNet/DeepNet>.

You can also directly reference the `Tensor.fsproj` and `Tensor.Algorithm.fsproj` projects inside the source tree from your project by using `dotnet add reference <path>`.
This is useful if you want to modify the Tensor library itself or for debugging.

Please report issues via <https://github.com/DeepMLNet/DeepNet/issues> and submit your pull requests via <https://github.com/DeepMLNet/DeepNet/pulls>.
