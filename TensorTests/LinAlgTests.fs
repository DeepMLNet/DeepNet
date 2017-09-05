module LinAlgTests

open Xunit
open FsUnit.Xunit

open Tensor.Utils
open Tensor
open Tensor.Algorithms


/// Checks general inverse.
let checkGeneralInverse (M: Tensor<_>) minConstraints minNullspace =
    printfn "checkGeneralInverse"
    printfn "==================="
    printfn "Input       M=\n%A" M

    let I, S, N = LinAlg.generalInverse M

    printfn "Inverse     I=\n%A" I
    printfn "Solvability S=\n%A" S
    printfn "Nullspace   N=\n%A" N
    printfn ""

    // check that inversion is okay
    let MIM = M .* I .* M
    printfn "M .* I .* M=\n%A" MIM
    MIM ==== M |> Tensor.all |> should equal true
    printfn "Inverse okay."

    // check that solvability is okay
    S.Shape.[0] >= minConstraints |> should equal true
    for r in 0L .. S.Shape.[0]-1L do
        S.[r, *] <<>> Tensor.zerosLike S.[r,*] |> Tensor.any |> should equal true
    let SM  = S .* M
    printfn "S .* M=\n%A" SM
    SM ==== Tensor.zerosLike SM |> Tensor.all |> should equal true
    printfn "Solvability okay."

    // check that nullspace is okay
    N.Shape.[1] >= minNullspace |> should equal true
    for c in 0L .. N.Shape.[1]-1L do
        N.[*, c] <<>> Tensor.zerosLike N.[*,c] |> Tensor.any |> should equal true
    let MN = M .* N
    printfn "M .* N=\n%A" MN
    MN ==== Tensor.zerosLike MN |> Tensor.all |> should equal true   
    printfn "Nullspace okay."

    printfn "==================="
    printfn ""

/// Generates a rational matrix with the specified number of rows and cols.
let generateRandomRatMatrix (rnd: System.Random) rows cols =
    let maxValue = 20
    let ratSeq = seq {
        while true do
            let num = rnd.Next (-maxValue, maxValue)
            let dnm = rnd.Next (1, maxValue)
            yield Rat (num, dnm)
    }
    HostTensor.ofSeqWithShape [rows; cols] ratSeq

/// Generates a rational matrix with the specified number of rows and cols and
/// with the specified rank.
let generateRandomMatrixWithRank rnd rows cols rank =
    if rank > rows || rank > cols then
        invalidArg "rank" "rank cannot be bigger than rows or cols"
    let R = generateRandomRatMatrix rnd rank cols
    let L = generateRandomRatMatrix rnd rows rank
    L .* R

/// Generate a random rational matrix with random size and rank.
let generateMatrixWithRandomRank (rnd: System.Random) maxSize =
    let rows = rnd.Next(1, maxSize)
    let cols = rnd.Next(1, maxSize)
    let rank = rnd.Next(0, min rows cols)
    generateRandomMatrixWithRank rnd (int64 rows) (int64 cols) (int64 rank), int64 rank

[<Fact>]
let ``Random matrix generation`` () =
    let rnd = System.Random 100
    for n in 0 .. 30 do
        let M, rank = generateMatrixWithRandomRank rnd 10
        printfn "M.[%d]= (rank %d)\n%A" n rank M

[<Fact>]
let ``Matrix general inversion`` () =
    let rnd = System.Random 100
    let maxSize = 6L
    for rows in 0L .. maxSize do
        for cols in 0L .. maxSize do
            for rank in 0L .. (min rows cols) do
                let M = generateRandomMatrixWithRank rnd rows cols rank
                printfn "Rows: %d   Cols: %d    Rank: %d" rows cols rank
                checkGeneralInverse M (rows - rank) (cols - rank)


let calcRowEchelon M =
    let Mf = Tensor.convert<float> M
    let Mfr = LinAlg.rowEchelon Mf
    //printfn "Mf=\n%A" Mf
    //printfn "row echelon:\n%A" Mfr

    let Mr = Tensor.convert<Rat> M
    let Mrr = LinAlg.rowEchelon Mr 

    let A = HostTensor.identity (List.max Mr.Shape)
    let A = A.[0L .. Mr.Shape.[0]-1L, *]
    let Mrr, nzRows, unCols, Arr = LinAlg.rowEchelonAugmented Mr A
    let Minv, S, N = LinAlg.generalInverse Mr
    printfn "---------------------------"
    printfn "M=\n%A" Mr
    //printfn "A=\n%A" A
    printfn "row echelon:\n%A" Mrr
    printfn "non-zero rows:        %d" nzRows
    printfn "unnormalized columns: %A" unCols
    printfn "Minv:\n%A" Minv
    printfn "Solvability constraint:\n%A" S
    printfn "Nullspace of M:\n%A" N
    printfn "M .* Minv:\n%A" (Mr .* Minv)
    printfn "***************************"
    printfn "M^-1:\n%A" Arr
    if Mr.Shape.[0] = Mr.Shape.[1] then
        let id = Arr .* Mr
        let idT = Mr .* Arr
        let idf = Tensor.convert<float> Mr .* Tensor.convert<float> Arr
        printfn "M^-1 .* M:\n%A" id
        printfn "M .* M^-1:\n%A" idT
        //printfn "M^-1 .* M (float):\n%A" idf
        if Tensor.all (Mrr ==== A) then
            Tensor.all (id ==== A) |> should equal true
            printfn "Inverse is okay."
        else
            printfn "Matrix not inverible."
    printfn "---------------------------"

[<Fact>]
let ``Row echelon: Wikipedia example`` () =
    let M = HostTensor.ofList2D [[  2;  1; -1]
                                 [ -3; -1;  2]
                                 [ -2;  1;  2]]
    calcRowEchelon M 
    checkGeneralInverse (Tensor.convert<Rat> M) 0L 0L

[<Fact>]
let ``Row echelon: Wikipedia example 2`` () =
    let M = HostTensor.ofList2D [[  2; -1;  0]
                                 [ -1;  2; -1]
                                 [  0; -1;  2]]
    calcRowEchelon M
    checkGeneralInverse (Tensor.convert<Rat> M) 0L 0L

[<Fact>]
let ``Row echelon: Colinear 1`` () =
    let M = HostTensor.ofList2D [[  2;  0; -1]
                                 [ -3;  0;  2]
                                 [ -2;  0;  2]]
    calcRowEchelon M
    checkGeneralInverse (Tensor.convert<Rat> M) 1L 1L

[<Fact>]
let ``Row echelon: Colinear 2`` () =
    let M = HostTensor.ofList2D [[  0;  0;  0]
                                 [ -3; -1;  2]
                                 [ -2;  1;  2]]
    calcRowEchelon M
    checkGeneralInverse (Tensor.convert<Rat> M) 1L 1L

[<Fact>]
let ``Row echelon: Colinear 3`` () =
    let M = HostTensor.ofList2D [[  2;  1; -1;  4]
                                 [ -3; -1;  2;  5]
                                 [ -2;  1;  2;  6]]
    calcRowEchelon M
    checkGeneralInverse (Tensor.convert<Rat> M) 0L 1L

[<Fact>]
let ``Row echelon: Colinear 4`` () =
    let M = HostTensor.ofList2D [[  2;  1; -1]
                                 [ -3; -1;  2]
                                 [ -3; -1;  8]
                                 [ -2;  1;  2]]
    calcRowEchelon M
    checkGeneralInverse (Tensor.convert<Rat> M) 1L 0L

