namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open Tensor.Utils
open Tensor


type BaseTests (output: ITestOutputHelper) =

    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``Basic arithmetic works`` () =
        let v1val : Tensor<float> = HostTensor.ones [3L]
        let v2val = HostTensor.scalar 3.0 |> Tensor.padLeft |> Tensor.broadcastTo [3L]
        let v3val = -v2val
        let v4val = v1val + v3val

        printfn "v1val: %A" v1val.[[0L]]
        printfn "v2val: %A" v2val.[[0L]]
        printfn "v3val: %A" v3val.[[0L]]
        printfn "v4val: %A" v4val.[[0L]]


    [<Fact>]
    let ``Slicing works`` () =
        let ary : Tensor<single> = HostTensor.ones [5L; 7L; 4L]
        //printfn "ary=\n%A" ary

        let slc1 = ary.[0L..1L, 1L..3L, 2L..4L]
        printfn "slc1=\n%A" slc1

        let slc1b = ary.[0L..1L, 1L..3L, *]
        printfn "slc1b=\n%A" slc1b

        let slc2 = ary.[1L, 1L..3L, 2L..4L]
        printfn "slc2=\n%A" slc2

        let ary2 : Tensor<single> = HostTensor.ones [5L; 4L]
        //printfn "ary2=\n%A" ary2

        let slc3 = ary2.[NewAxis, 1L..3L, 2L..4L]
        printfn "slc3=\n%A" slc3

        let slc4 = ary2.[Fill, 1L..3L, 2L..4L]
        printfn "slc4=\n%A" slc4

        ary2.[NewAxis, 1L..3L, 2L..4L] <- slc3
        
    [<Fact>]
    let ``Boolean slicing`` () =
        let ary = HostTensor.arange 1 1 10
        let ary = Tensor.concat 0 [ary.[NewAxis, *]; ary.[NewAxis, *]]
        ary.[[1L; 0L]] <- 3 
        printfn "ary=\n%A" ary
                
        printfn "ary ==== 3=\n%A" (ary====3)
        printfn "ary.[*,0L] ==== 1=\n%A" (ary.[*,0L] ==== 1)
        printfn "ary.M(ary====3)=\n%A" (ary.M(ary====3))
        printfn "ary.M(ary.[*,0L] ==== 1, NoMask)=\n%A" (ary.M(ary.[*,0L] ==== 1, NoMask)) 
        printfn "ary.M(ary.[*,0L] ==== 1, ary.[0L, *]====3)=\n%A" (ary.M(ary.[*,0L]====1, ary.[0L,*]====3))
        ary.M(ary====3) <- ary.M(ary====3) + 2
        printfn "ary.[ary====3] <- ary.[ary====3] + 2:\n%A" ary
        ary.M(ary====5) <- HostTensor.scalar 11
        printfn "ary.[ary====5] <- 11:\n%A" ary               
        
    [<Fact>]
    let ``True indices`` () =
        let m = [[true; true; false; true]
                 [false; true; true; true]] |> HostTensor.ofList2D
        printfn "m=\n%A" m
        let i = Tensor.trueIdx m
        printfn "idx where m ==== true:\n%A" i                          

    [<Fact>]
    let ``Pretty printing works`` () =
        printfn "3x4 one matrix:       \n%A" (HostTensor.ones [3L; 4L] :> Tensor<float>)
        printfn "6 zero vector:        \n%A" (HostTensor.zeros [6L] :> Tensor<float>)
        printfn "5x5 identity matrix:  \n%A" (HostTensor.identity 5L :> Tensor<float>)

    [<Fact>]
    let ``Batched matrix-matrix dot product`` () =
        let N, M = 2L, 3L
        let rng = System.Random(123)
        let a = HostTensor.randomUniform rng (-3., 3.) [N; M; 4L; 3L]
        let b = HostTensor.randomUniform rng (-1., 1.) [N; M; 3L; 2L]
        let c = a .* b

        let cr = HostTensor.zeros<float> [N; M; 4L; 2L]
        for n in 0L .. N-1L do
            for m in 0L .. M-1L do
                cr.[n, m, Fill] <- a.[n, m, Fill] .* b.[n, m, Fill]
        Tensor.almostEqual c cr |> Assert.True
        
    [<Fact>]
    let ``Batched matrix-vector dot product`` () =
        let N, M = 2L, 3L
        let rng = System.Random(123)
        let a = HostTensor.randomUniform rng (-3., 3.) [N; M; 4L; 3L]
        let b = HostTensor.randomUniform rng (-1., 1.) [N; M; 3L]
        let c = a .* b

        let cr = HostTensor.zeros<float> [N; M; 4L]
        for n in 0L .. N-1L do
            for m in 0L .. M-1L do
                cr.[n, m, Fill] <- a.[n, m, Fill] .* b.[n, m, Fill]

        printfn "c=\n%A" c
        printfn "cr=\n%A" cr

        Tensor.almostEqual c cr |> Assert.True
        

    [<Fact>]
    let ``Build and extract diagonal`` () =
        let N = 3L
        let rng = System.Random 123
        let v = HostTensor.randomUniform rng (-1., 1.) [N]
        let dm = Tensor.diagMat v
        
        printfn "v=%A" v
        printfn "diag(v)=\n%A" dm

        let vv = Tensor.diag dm
        Tensor.almostEqual v vv |> Assert.True


    [<Fact>]
    let ``Batched build and extract diagonal`` () =
        let S1, S2 = 2L, 3L
        let N = 4L
        let rng = System.Random 123
        let v = HostTensor.randomUniform rng (-1., 1.) [S1; S2; N]
        let dm = Tensor.diagMat v
        
        printfn "v=\n%A" v
        printfn "diag(v)=\n%A" dm

        let vv = Tensor.diag dm
        Tensor.almostEqual v vv |> Assert.True


    [<Fact>]
    let ``Batched trace`` () =
        let v = [[1; 2; 3]; [4; 5; 6]] |> HostTensor.ofList2D
        let dm = Tensor.diagMat v
        
        let tr = Tensor.trace dm
        let trv = Tensor.sumAxis 1 v

        printfn "v=\n%A" v
        printfn "trace(diag(v))=%A" tr
        printfn "sum(v)=%A" trv

        Tensor.almostEqual tr trv |> Assert.True



    [<Fact>]
    let ``Invert diagonal matrix`` () =
        let v = [1.0; 2.0; 3.0] |> HostTensor.ofList

        let dm = Tensor.diagMat v
        let dmInv = Tensor.invert dm
        let dmInvInv = Tensor.invert dmInv

        printfn "dm=\n%A" dm
        printfn "dm^-1=\n%A" dmInv
        printfn "dm^-1^-1=\n%A" dmInvInv

        Tensor.almostEqual dm dmInvInv |> Assert.True

    [<Fact>]
    let ``Invert random matrix`` () =
        let rng = System.Random 123

        let dm = HostTensor.randomUniform rng (-1.0, 1.0) [4L; 4L]
        let dmInv = Tensor.invert dm
        let dmInvInv = Tensor.invert dmInv

        printfn "dm=\n%A" dm
        printfn "dm^-1=\n%A" dmInv
        printfn "dm^-1^-1=\n%A" dmInvInv

        Tensor.almostEqual dm dmInvInv |> Assert.True

    [<Fact>]
    let ``Batch invert random matrices`` () =
        let rng = System.Random 123

        let dm = HostTensor.randomUniform rng (-1.0, 1.0) [2L; 4L; 3L; 3L]
        let dmInv = Tensor.invert dm
        let dmInvInv = Tensor.invert dmInv

        printfn "dm=\n%A" dm
        printfn "dm^-1=\n%A" dmInv
        printfn "dm^-1^-1=\n%A" dmInvInv

        Tensor.almostEqual dm dmInvInv |> Assert.True


    [<Fact>]
    let ``Invert singular matrix`` () =
        let dm = HostTensor.ofList2D [[1.0; 0.0; 0.0]
                                      [1.0; 2.0; 0.0]
                                      [1.0; 0.0; 0.0]]

        shouldFail (fun () -> Tensor.invert dm |> ignore)


    [<Fact>]
    let ``Invert Kk matrix`` () =
        use hdf = HDF5.OpenRead (Util.assemblyDirectory + "/TestData/MatInv.h5")
        let Kk : Tensor<single> = HostTensor.read hdf "Kk"

        let Kkinv = Tensor.invert Kk   
        let id = Kkinv .* Kk

        printfn "Kk=\n%A" Kk
        printfn "Kkinv=\n%A" Kkinv
        printfn "id=\n%A" id

        let s = Kk.Shape.[0]
        let n = Kk.Shape.[1]

        let ids = Tensor.concat 0 [for i in 0L .. s-1L do yield (HostTensor.identity n).[NewAxis, *, *]]

        let diff = id - ids
        printfn "maxdiff: %f" (Tensor.max diff)
        Tensor.almostEqualWithTol (id, ids, absTol=1e-5f, relTol=1e-5f)  |> Assert.True

    [<Fact>]
    let ``Pseudo Invert random matrix`` () =
        let rng = System.Random 123

        let dm = HostTensor.randomUniform rng (-1.0, 1.0) [4L; 4L]
        let dmInv = Tensor.pseudoInvert dm
        let dmInvInv = Tensor.pseudoInvert dmInv

        printfn "dm=\n%A" dm
        printfn "dm^-1=\n%A" dmInv
        printfn "dm^-1^-1=\n%A" dmInvInv 

        Tensor.almostEqual dm dmInvInv |> Assert.True

    [<Fact>]
    let ``Pseudo Invert singular matrix`` () =
        let dm = HostTensor.ofList2D [[1.0; 0.0; 0.0]
                                      [1.0; 2.0; 0.0]
                                      [1.0; 0.0; 0.0]]

        let dmInv = Tensor.pseudoInvert dm

        printfn "dm=\n%A" dm
        printfn "dm^-1=\n%A" dmInv


    [<Fact>]
    let ``Select`` () =

        let a = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor<float>.convert
        let i0 = [1L; 2L; 0L; 3L] |> HostTensor.ofList |> Tensor.padRight |> Tensor.broadcastDim 1 2L
        let idxs = [Some i0; None]

        let s = a |> Tensor.gather idxs

        printfn "a=\n%A" a
        printfn "idxs=\n%A" idxs
        printfn "select idxs a=\n%A" s

    [<Fact>]
    let ``Select 2`` () =

        let a = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor<float>.convert
        let i0 = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
        let idxs = [Some i0; None]

        let s = a |> Tensor.gather idxs

        printfn "a=\n%A" a
        printfn "idxs=\n%A" idxs
        printfn "select idxs a=\n%A" s

    [<Fact>]
    let ``Select 3`` () =

        let a = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor<float>.convert
        let i0 = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
        let i1 = [0L; 0L; 1L] |> HostTensor.ofList |> Tensor.padLeft
        let idxs = [Some i0; Some i1]

        let s = a |> Tensor.gather idxs

        printfn "a=\n%A" a
        printfn "idxs=\n%A" idxs
        printfn "select idxs a=\n%A" s


    [<Fact>]
    let ``Disperse 1`` () =

        let a = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor<float>.convert
        let i0 = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
        let i1 = [0L; 0L; 1L] |> HostTensor.ofList |> Tensor.padLeft
        let idxs = [Some i0; Some i1]
        let shp = [3L; 3L]

        let s = a |> Tensor.scatter idxs shp

        printfn "a=\n%A" a
        printfn "shp=%A" shp
        printfn "idxs=\n%A" idxs
        printfn "disperse idxs shp a=\n%A" s


    [<Fact>]
    let ``Disperse 2`` () =

        let a = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor<float>.convert
        let i0 = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
        let idxs = [Some i0; None]
        let shp = [3L; 3L]

        let s = a |> Tensor.scatter idxs shp

        printfn "a=\n%A" a
        printfn "shp=%A" shp
        printfn "idxs=\n%A" idxs
        printfn "disperse idxs shp a=\n%A" s


    [<Fact>]
    let ``Disperse 3`` () =

        let a = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor<float>.convert
        let idxs = [None; None]
        let shp = [5L; 3L]

        let s = a |> Tensor.scatter idxs shp

        printfn "a=\n%A" a
        printfn "shp=%A" shp
        printfn "idxs=\n%A" idxs
        printfn "disperse idxs shp a=\n%A" s


    [<Fact>]
    let ``Normal distribution sampling`` () =
        let rng = System.Random 124
        let mean, var = 3.0, 1.5
        let samples = 10000L
        let x = HostTensor.randomNormal rng (mean, var) [samples]
        printfn "Generating mean=%f variance=%f" mean var
        let xMean, xVar = Tensor.mean x, Tensor.var x
        printfn "Caluclated mean=%f variance=%f" xMean xVar
        abs (mean - xMean) < 0.06 |> should equal true
        abs (var - xVar) < 0.06 |> should equal true



    
    