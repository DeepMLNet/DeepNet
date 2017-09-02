module LinAlgTests

open Xunit
open FsUnit.Xunit

open Tensor.Utils
open Tensor
open Tensor.Algorithms


let calcRowEchelon M =
    let Mf = Tensor.convert<float> M
    let Mfr = RowEchelonForm.compute Mf
    //printfn "Mf=\n%A" Mf
    //printfn "row echelon:\n%A" Mfr

    let Mr = Tensor.convert<Rat> M
    let Mrr = RowEchelonForm.compute Mr 

    let A = HostTensor.identity (List.max Mr.Shape)
    let A = A.[0L .. Mr.Shape.[0]-1L, *]
    let Mrr, Arr = RowEchelonForm.computeAugmented Mr A
    printfn "---------------------------"
    printfn "M=\n%A" Mr
    printfn "A=\n%A" A
    printfn "row echelon:\n%A" Mrr
    printfn "augmented row echelon:\n%A" Arr
    if Mr.Shape.[0] = Mr.Shape.[1] then
        let id = Arr .* Mr
        printfn "identity:\n%A" id
    printfn "---------------------------"

[<Fact>]
let ``Row echelon: Wikipedia example`` () =
    let M = HostTensor.ofList2D [[  2;  1; -1]
                                 [ -3; -1;  2]
                                 [ -2;  1;  2]]
    calcRowEchelon M


[<Fact>]
let ``Row echelon: Colinear 1`` () =
    let M = HostTensor.ofList2D [[  2;  0; -1]
                                 [ -3;  0;  2]
                                 [ -2;  0;  2]]
    calcRowEchelon M

[<Fact>]
let ``Row echelon: Colinear 2`` () =
    let M = HostTensor.ofList2D [[  0;  0;  0]
                                 [ -3; -1;  2]
                                 [ -2;  1;  2]]
    calcRowEchelon M

[<Fact>]
let ``Row echelon: Colinear 3`` () =
    let M = HostTensor.ofList2D [[  2;  1; -1; 4]
                                 [ -3; -1;  2; 5]
                                 [ -2;  1;  2; 6]]
    calcRowEchelon M

[<Fact>]
let ``Row echelon: Colinear 4`` () =
    let M = HostTensor.ofList2D [[  2;  1; -1]
                                 [ -3; -1;  2]
                                 [ -3; -1;  8]
                                 [ -2;  1;  2]]
    calcRowEchelon M

