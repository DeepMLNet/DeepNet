module HDF5Tests

open Xunit
open FsUnit.Xunit

open Tensor.Utils
open Tensor


[<Fact>]
let ``Open HDF5 file for writing`` () =
    use f = HDF5.OpenWrite "test.h5"
    ()

    