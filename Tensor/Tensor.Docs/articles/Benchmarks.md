# Benchmarks

The following benchmarks are available.

- [F# Tensor vs. Anaconda NumPy on Linux](../benchmarks/benchmark-linux.html)
- [F# Tensor vs. Anaconda NumPy on Windows](../benchmarks/benchmark-windows.html)

Since the Windows and Linux machines have different hardware, results between them are *not* comparable.

## Methodology

We benchmarked a variety of functions using different data types and tensor shapes.
Each benchmark was run twice, once for tensors stored in host memory and a second time using a CUDA GPU.
When benchmarking on the GPU a call to `cudaDeviceSynchronize` was made to ensure that the operation is complete before the elapsed time was measured.
If an operation is not yet implemented for a particular combination of data type and device, the result is replaced by a dash.

Benchmarking was driven by [BenchmarkDotNet](https://benchmarkdotnet.org) using the following options.

```ini
BenchmarkDotNet=v0.10.14
MinInvokeCount=1  IterationTime=250.0000 ms  LaunchCount=1
TargetCount=4  UnrollFactor=1  WarmupCount=0
```

## Comparison to NumPy

For comparison, we run equivalent operations using [NumPy](http://www.numpy.org) from the [Anaconda](https://anaconda.org/)  Python 3 distribution.
Since it has no inherent support for GPU acceleration, we report only timings for operations executed on the host.
The NumPy benchmarks are work in progress and not every equivalent function has been evaluated yet.

Benchmarking was driven by [pytest-benchmark](https://pytest-benchmark.readthedocs.io) using the default settings.

## Source code

Source code for the benchmarks is available in the [Tensor/Tensor.Benchmark](https://github.com/DeepMLNet/DeepNet/tree/master/Tensor/Tensor.Benchmark) folder of the repository.
