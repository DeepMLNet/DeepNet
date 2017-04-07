module ArrayNDTests

open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS
open Datasets


[<Fact>]
let ``Basic arithmetic works`` () =
    let v1val : ArrayNDHostT<float> = ArrayNDHost.ones [3L]
    let v2val = ArrayNDHost.scalar 3.0 |> ArrayND.padLeft |> ArrayND.broadcastToShape [3L]
    let v3val = -v2val
    let v4val = v1val + v3val

    printfn "v1val: %A" v1val.[[0L]]
    printfn "v2val: %A" v2val.[[0L]]
    printfn "v3val: %A" v3val.[[0L]]
    printfn "v4val: %A" v4val.[[0L]]
    

[<Fact>]
let ``Slicing works`` () =
    let ary : ArrayNDHostT<single> = ArrayNDHost.ones [5L; 7L; 4L]
    //printfn "ary=\n%A" ary

    let slc1 = ary.[0L..1L, 1L..3L, 2L..4L]
    printfn "slc1=\n%A" slc1

    let slc1b = ary.[0L..1L, 1L..3L, *]
    printfn "slc1b=\n%A" slc1b

    let slc2 = ary.[1L, 1L..3L, 2L..4L]
    printfn "slc2=\n%A" slc2

    let ary2 : ArrayNDHostT<single> = ArrayNDHost.ones [5L; 4L]
    //printfn "ary2=\n%A" ary2

    let slc3 = ary2.[NewAxis, 1L..3L, 2L..4L]
    printfn "slc3=\n%A" slc3

    let slc4 = ary2.[Fill, 1L..3L, 2L..4L]
    printfn "slc4=\n%A" slc4

    ary2.[NewAxis, 1L..3L, 2L..4L] <- slc3

[<Fact>]
let ``Pretty printing works`` () =
    printfn "3x4 one matrix:       \n%A" (ArrayNDHost.ones [3L; 4L] :> ArrayNDT<float>)
    printfn "6 zero vector:        \n%A" (ArrayNDHost.zeros [6L] :> ArrayNDT<float>)
    printfn "5x5 identity matrix:  \n%A" (ArrayNDHost.identity 5L :> ArrayNDT<float>)

[<Fact>]
let ``Batched matrix-matrix dot product`` () =
    let N, M = 2L, 3L
    let rng = System.Random(123)
    let a = rng.UniformArrayND (-3., 3.) [N; M; 4L; 3L]
    let b = rng.UniformArrayND (-1., 1.) [N; M; 3L; 2L]
    let c = a .* b

    let cr = ArrayNDHost.zeros<float> [N; M; 4L; 2L]
    for n in 0L .. N-1L do
        for m in 0L .. M-1L do
            cr.[n, m, Fill] <- a.[n, m, Fill] .* b.[n, m, Fill]
    ArrayND.almostEqual c cr |> ArrayND.value |> Assert.True
    
[<Fact>]
let ``Batched matrix-vector dot product`` () =
    let N, M = 2L, 3L
    let rng = System.Random(123)
    let a = rng.UniformArrayND (-3., 3.) [N; M; 4L; 3L]
    let b = rng.UniformArrayND (-1., 1.) [N; M; 3L]
    let c = a .* b

    let cr = ArrayNDHost.zeros<float> [N; M; 4L]
    for n in 0L .. N-1L do
        for m in 0L .. M-1L do
            cr.[n, m, Fill] <- a.[n, m, Fill] .* b.[n, m, Fill]
    ArrayND.almostEqual c cr |> ArrayND.value |> Assert.True
    

[<Fact>]
let ``Build and extract diagonal`` () =
    let N = 3L
    let rng = System.Random 123
    let v = rng.UniformArrayND (-1., 1.) [N]
    let dm = ArrayND.diagMat v
    
    printfn "v=%A" v
    printfn "diag(v)=\n%A" dm

    let vv = ArrayND.diag dm
    ArrayND.almostEqual v vv |> ArrayND.value |> Assert.True


[<Fact>]
let ``Batched build and extract diagonal`` () =
    let S1, S2 = 2L, 3L
    let N = 4L
    let rng = System.Random 123
    let v = rng.UniformArrayND (-1., 1.) [S1; S2; N]
    let dm = ArrayND.diagMat v
    
    printfn "v=\n%A" v
    printfn "diag(v)=\n%A" dm

    let vv = ArrayND.diag dm
    ArrayND.almostEqual v vv |> ArrayND.value |> Assert.True


[<Fact>]
let ``Batched trace`` () =
    let v = [[1; 2; 3]; [4; 5; 6]] |> ArrayNDHost.ofList2D
    let dm = ArrayND.diagMat v
    
    let tr = ArrayND.trace dm
    let trv = ArrayND.sumAxis 1 v

    printfn "v=\n%A" v
    printfn "trace(diag(v))=%A" tr
    printfn "sum(v)=%A" trv

    ArrayND.almostEqual tr trv |> ArrayND.value |> Assert.True



[<Fact>]
let ``Invert diagonal matrix`` () =
    let v = [1.0; 2.0; 3.0] |> ArrayNDHost.ofList

    let dm = ArrayND.diagMat v
    let dmInv = ArrayND.invert dm
    let dmInvInv = ArrayND.invert dmInv

    printfn "dm=\n%A" dm
    printfn "dm^-1=\n%A" dmInv
    printfn "dm^-1^-1=\n%A" dmInvInv

    ArrayND.almostEqual dm dmInvInv |> ArrayND.value |> Assert.True

[<Fact>]
let ``Invert random matrix`` () =
    let rng = System.Random 123

    let dm = rng.UniformArrayND (-1.0, 1.0) [4L; 4L]
    let dmInv = ArrayND.invert dm
    let dmInvInv = ArrayND.invert dmInv

    printfn "dm=\n%A" dm
    printfn "dm^-1=\n%A" dmInv
    printfn "dm^-1^-1=\n%A" dmInvInv

    ArrayND.almostEqual dm dmInvInv |> ArrayND.value |> Assert.True

[<Fact>]
let ``Batch invert random matrices`` () =
    let rng = System.Random 123

    let dm = rng.UniformArrayND (-1.0, 1.0) [2L; 4L; 3L; 3L]
    let dmInv = ArrayND.invert dm
    let dmInvInv = ArrayND.invert dmInv

    printfn "dm=\n%A" dm
    printfn "dm^-1=\n%A" dmInv
    printfn "dm^-1^-1=\n%A" dmInvInv

    ArrayND.almostEqual dm dmInvInv |> ArrayND.value |> Assert.True


[<Fact>]
let ``Invert singular matrix`` () =
    let dm = ArrayNDHost.ofList2D [[1.0; 0.0; 0.0]
                                   [1.0; 2.0; 0.0]
                                   [1.0; 0.0; 0.0]]

    shouldFail (fun () -> ArrayND.invert dm |> ignore)


[<Fact>]
let ``Invert Kk matrix`` () =
    use hdf = HDF5.OpenRead "MatInv.h5"
    let Kk : ArrayNDHostT<single> = ArrayNDHDF.read hdf "Kk"

    let Kkinv = ArrayND.invert Kk   
    let id = Kkinv .* Kk

    printfn "Kk=\n%A" Kk
    printfn "Kkinv=\n%A" Kkinv
    printfn "id=\n%A" id

    let s = Kk.Shape.[0]
    let n = Kk.Shape.[1]

    let ids = ArrayND.concat 0 [for i in 0L .. s-1L do yield (ArrayNDHost.identity n).[NewAxis, *, *]]

    let diff = id - ids
    printfn "maxdiff: %f" (ArrayND.max diff |> ArrayND.value)
    ArrayND.almostEqualWithTol 1e-5f 1e-5f id ids |> ArrayND.value |> Assert.True


[<Fact>]
let ``Select`` () =

    let a = Seq.counting |> ArrayNDHost.ofSeqWithShape [4L; 3L] |> ArrayND.float
    let i0 = [1L; 2L; 0L; 3L] |> ArrayNDHost.ofList |> ArrayND.padRight |> ArrayND.broadcastDim 1 2L
    let idxs = [Some i0; None]

    let s = a |> ArrayND.gather idxs

    printfn "a=\n%A" a
    printfn "idxs=\n%A" idxs
    printfn "select idxs a=\n%A" s

[<Fact>]
let ``Select 2`` () =

    let a = Seq.counting |> ArrayNDHost.ofSeqWithShape [4L; 3L] |> ArrayND.float
    let i0 = [1L; 2L; 2L] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let idxs = [Some i0; None]

    let s = a |> ArrayND.gather idxs

    printfn "a=\n%A" a
    printfn "idxs=\n%A" idxs
    printfn "select idxs a=\n%A" s

[<Fact>]
let ``Select 3`` () =

    let a = Seq.counting |> ArrayNDHost.ofSeqWithShape [4L; 3L] |> ArrayND.float
    let i0 = [1L; 2L; 2L] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let i1 = [0L; 0L; 1L] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let idxs = [Some i0; Some i1]

    let s = a |> ArrayND.gather idxs

    printfn "a=\n%A" a
    printfn "idxs=\n%A" idxs
    printfn "select idxs a=\n%A" s


[<Fact>]
let ``Disperse 1`` () =

    let a = Seq.counting |> ArrayNDHost.ofSeqWithShape [4L; 3L] |> ArrayND.float
    let i0 = [1L; 2L; 2L] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let i1 = [0L; 0L; 1L] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let idxs = [Some i0; Some i1]
    let shp = [3L; 3L]

    let s = a |> ArrayND.scatter idxs shp

    printfn "a=\n%A" a
    printfn "shp=%A" shp
    printfn "idxs=\n%A" idxs
    printfn "disperse idxs shp a=\n%A" s


[<Fact>]
let ``Disperse 2`` () =

    let a = Seq.counting |> ArrayNDHost.ofSeqWithShape [4L; 3L] |> ArrayND.float
    let i0 = [1L; 2L; 2L] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let idxs = [Some i0; None]
    let shp = [3L; 3L]

    let s = a |> ArrayND.scatter idxs shp

    printfn "a=\n%A" a
    printfn "shp=%A" shp
    printfn "idxs=\n%A" idxs
    printfn "disperse idxs shp a=\n%A" s


[<Fact>]
let ``Disperse 3`` () =

    let a = Seq.counting |> ArrayNDHost.ofSeqWithShape [4L; 3L] |> ArrayND.float
    let idxs = [None; None]
    let shp = [5L; 3L]

    let s = a |> ArrayND.scatter idxs shp

    printfn "a=\n%A" a
    printfn "shp=%A" shp
    printfn "idxs=\n%A" idxs
    printfn "disperse idxs shp a=\n%A" s

[<Fact>]
let ``Log determinant 1`` () =
    let a = ArrayNDHost.ofList2D [[3.0;2.0;3.0]
                                  [4.0;8.0;9.0]
                                  [1.0;4.0;5.0]]
    let logDet = ArrayND.logDeterminant a
    let ld = ArrayNDHost.scalar (log 14.0)
    printfn "logDet =\n %A" logDet
    printfn "ld = \n%A" ld
    ArrayND.almostEqual logDet ld |> ArrayND.value |> Assert.True

[<Fact>]
let ``Log determinant 2`` () =
    let aofAofA = [|[|[|2.0;6.0|];
                      [|1.0;5.0|]|];
                    [|[|3.0;4.0|];
                      [|2.0;6.0|]|];
                    [|[|5.0;4.0|];
                      [|2.0;2.0|]|]|]
    let a = ArrayNDHost.ofArray3D (Array3D.init 3 2 2 (fun i j k -> aofAofA.[i].[j].[k]) )
    let logDet = ArrayND.logDeterminant a
    let ld = ArrayNDHost.ofList [log(4.0); log(10.0);log(2.0)]
    printfn "logDet =\n %A"  logDet
    printfn "ld = \n%A" ld
    ArrayND.almostEqual logDet ld |> ArrayND.value |> Assert.True

[<Fact>]
let ``Log determinant 3`` () =
    let aofAofA = [|[|[|2.0;6.0;2.0|];
                      [|2.0;3.0;1.0|];
                      [|1.0;5.0;3.0|]|];
                    [|[|3.0;4.0;3.0|];
                      [|2.0;4.0;1.0|];
                      [|2.0;6.0;1.0|]|];
                    [|[|5.0;4.0;3.0|];
                      [|1.0;4.0;3.0|];
                      [|2.0;2.0;2.0|]|]|]
    let a = ArrayNDHost.ofArray3D (Array3D.init 3 3 3 (fun i j k -> aofAofA.[i].[j].[k]) )
    let logDet = ArrayND.logDeterminant a
    let ld = ArrayNDHost.ofList [log(8.0); log(6.0);log(8.0)]
    printfn "logDet =\n %A"  logDet
    printfn "ld = \n%A" ld
    ArrayND.almostEqual logDet ld |> ArrayND.value |> Assert.True