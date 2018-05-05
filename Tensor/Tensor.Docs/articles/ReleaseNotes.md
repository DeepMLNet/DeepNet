# Release notes

* 0.4.10
  * Update to CUDA 9.1
  * CUDA SDK no longer required. 

* 0.4.9
  * Remove type constraints on Tensor.arange and Tensor.linspace.

* 0.4.8
  * Support for dynamic loading of BLAS libraries for host tensors.
    Use Tensor.Host.Cfg.BLASLib to configure.

* 0.4.7
  * Experimental support for macOS (without CUDA).

* 0.4.6
  * Fix double-precision BLAS operations on CUDA GPU.

* 0.4.5
  * No changes.

* 0.4.4
  * Fix tensor_mkl.dll not being found.
  * Update Tensor.Sample sample project.
  * Add SourceLink support (experimental).

* 0.4.3
  * Documentation improvements.
  * Add Tensor.Sample sample project.

* 0.4.2
  * Improve CUDA and CUBLAS initialization.

* 0.4.1
  * First version to support .NET Standard 2.0.
  * Optimized host backend.
  * Redesigned API.
