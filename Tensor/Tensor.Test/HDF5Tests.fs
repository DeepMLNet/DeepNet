namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor.Utils
open Tensor



type HDF5Tests (output: ITestOutputHelper) =

    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``Open HDF5 file for writing`` () =
        use f = HDF5.OpenWrite "test.h5"
        ()

    