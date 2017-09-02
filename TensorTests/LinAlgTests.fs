module LinAlgTests

open Xunit
open FsUnit.Xunit

open Tensor.Utils
open Tensor
open Tensor.Algorithms


[<Fact>]
let ``Wikipedia example`` () =
    let M = HostTensor.ofList2D [[  2;  1; -1]
                                 [ -3; -1;  2]
                                 [ -2;  1;  2]]

    let Mf = Tensor.convert<float> M
    let Mfr = RowEchelonForm.compute Mf []
    printfn "Mf=\n%A" Mf
    printfn "row echelon:\n%A" Mfr

    let Mr = Tensor.convert<Rat> M
    let Mrr = RowEchelonForm.compute Mr []
    printfn "Mr=\n%A" Mr
    printfn "row echelon:\n%A" Mrr


    