module TensorCudaTests

open Xunit
open FsUnit.Xunit

open Tensor.Utils
open Tensor



[<Fact>]
let ``Tensor transfer to Cuda``() =
    
    let data = HostTensor.arange 30L |> Tensor.float |> Tensor.reshape [3L; 10L]
    let cuda = CudaTensor.transfer data
    let back = HostTensor.transfer cuda

    printfn "data:\n%A" data
    printfn "back:\n%A" back

    Tensor.almostEqual data back |> should equal true


[<Fact>]
let ``Tensor transfer to Cuda 2``() =
    
    let data = HostTensor.arange 30L |> Tensor.float |> Tensor.reshape [3L; 2L; 5L]
    let data = data.Copy (order=CustomOrder [1; 0; 2])
    printfn "data layout:%A" data.Layout

    let cuda = CudaTensor.transfer data
    let back = HostTensor.transfer cuda

    printfn "data:\n%A" data
    printfn "back:\n%A" back

    Tensor.almostEqual data back |> should equal true

