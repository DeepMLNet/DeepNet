namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor.Utils
open Tensor
open Tensor.Cuda


/// Test that only runs when CUDA is available.
type CudaFactAttribute() as this =
    inherit FactAttribute()
    do
        if TensorCudaDevice.count = 0 then
            this.Skip <- "TensorCudaDevice.count = 0"