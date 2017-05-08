module OpTests
#nowarn "25"

open Xunit
open FsUnit.Xunit

open Basics
open Tensor
open SymTensor
open SymTensor.Compiler.Cuda
open TestUtils




[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix-matrix dot`` () =   
    requireEqualTracesWithRandomData [[6L; 3L]; [3L; 2L]] (fun [a; b] ->
        a .* b
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix-vector dot`` () =   
    requireEqualTracesWithRandomData [[6L; 3L]; [3L]] (fun [a; b] ->
        a .* b
    )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix-matrix dot`` () =   
    requireEqualTracesWithRandomData [[7L; 5L; 6L; 3L]; [7L; 5L; 3L; 2L]] (fun [a; b] ->
        a .* b
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix-matrix dot with broadcasting`` () =   
    requireEqualTracesWithRandomData [[7L; 5L; 6L; 3L]; [7L; -1L; 3L; 2L]] (fun [a; b] ->
        a .* b
    )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched build diagonal`` () =
    requireEqualTracesWithRandomData [[7L; 5L; 3L]] (fun [a] ->
        Expr.diagMat a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched extract diagonal`` () =
    requireEqualTracesWithRandomData [[7L; 5L; 4L; 4L]] (fun [a] ->
        Expr.diag a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix inverse`` () =
    requireEqualTracesWithRandomData [[3L; 3L]] (fun [a] ->
        Expr.invert a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: transposed matrix inverse`` () =
    requireEqualTracesWithRandomData [[5L; 5L]] (fun [a] ->
        Expr.invert a.T
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix inverse`` () =
    requireEqualTracesWithRandomData [[7L; 3L; 4L; 4L]] (fun [a] ->
        Expr.invert a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: sum`` () =
    requireEqualTracesWithRandomData [[7L; 3L; 4L; 5L]] (fun [a] ->
        Expr.sum a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: sum axis 1`` () =
    requireEqualTracesWithRandomData [[7L; 3L; 4L; 5L]] (fun [a] ->
        Expr.sumAxis 1 a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: sum axis 2`` () =
    requireEqualTracesWithRandomData [[7L; 3L; 4L; 5L]] (fun [a] ->
        a |> Expr.sumAxis 3 |> Expr.sumAxis 0
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: large sum axis`` () =
    requireEqualTracesWithRandomData [[7L; 200L]] (fun [a] ->
        a |> Expr.sumAxis 0
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: product`` () =
    requireEqualTracesWithRandomData [[7L; 3L; 4L; 5L]] (fun [a] ->
        Expr.product a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: product axis 1`` () =
    requireEqualTracesWithRandomData [[7L; 3L; 4L; 5L]] (fun [a] ->
        Expr.productAxis 1 a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Singular matrix inverse`` () =
    let a = Expr.var<single> "a" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    let expr = Expr.invert a
    let fn = Func.make<single> DevCuda.DefaultFactory expr |> arg1 a
    let av = CudaTensor.zeros<single> [3L; 3L]
    let iav = fn av
    printfn "a=\n%A" av
    printfn "a^-1=\n%A" iav

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Replicate`` () =
    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 3L]
    let expr0 = Expr.replicate 0 (SizeSpec.fix 2L) a
    let expr1 = Expr.replicate 1 (SizeSpec.fix 3L) a
    let fns = Func.make2<single, single> DevCuda.DefaultFactory expr0 expr1 |> arg1 a
    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> HostTensor.ofList2D 
    let av0, av1 = fns av
    printfn "a=\n%A" av 
    printfn "rep 0 2 a=\n%A" av0
    printfn "rep 1 3 a=\n%A" av1

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``ReplicateTo on CUDA`` () =
    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 3L]
    let expr0 = Expr.replicateTo 0 (SizeSpec.fix 6L) a
    let expr1 = Expr.replicateTo 1 (SizeSpec.fix 7L) a
    let fns = Func.make2<single, single> DevCuda.DefaultFactory expr0 expr1 |> arg1 a
    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> HostTensor.ofList2D 
    let av0, av1 = fns av
    printfn "a=\n%A" av 
    printfn "repTo 0 6 a=\n%A" av0
    printfn "repTo 1 7 a=\n%A" av1

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Derivative of ReplicateTo on CUDA`` () =
    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 3L]
    let expr0 = Expr.replicateTo 0 (SizeSpec.fix 6L) a
    let expr1 = Expr.replicateTo 1 (SizeSpec.fix 7L) a
    let da0 = Deriv.compute expr0 |> Deriv.ofVar a
    let da1 = Deriv.compute expr1 |> Deriv.ofVar a
    let fns = Func.make2<single, single> DevCuda.DefaultFactory da0 da1 |> arg1 a
    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> HostTensor.ofList2D 
    let dav0, dav1 = fns av
    printfn "a=\n%A" av 
    printfn "d(repTo 0 7 a) / da=\n%A" dav0.Full
    printfn "d(repTo 1 5 a) / da=\n%A" dav1.Full

[<Fact>]
let ``Derivative of ReplicateTo on host`` () =
    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 3L]
    let expr0 = Expr.replicateTo 0 (SizeSpec.fix 6L) a
    let expr1 = Expr.replicateTo 1 (SizeSpec.fix 7L) a
    let da0 = Deriv.compute expr0 |> Deriv.ofVar a
    let da1 = Deriv.compute expr1 |> Deriv.ofVar a
    let fns = Func.make2<single, single> DevHost.DefaultFactory da0 da1 |> arg1 a
    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> HostTensor.ofList2D 
    let dav0, dav1 = fns av
    printfn "a=\n%A" av 
    printfn "d(repTo 0 7 a) / da=\n%A" dav0.Full
    printfn "d(repTo 1 5 a) / da=\n%A" dav1.Full



[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: max, min elemwise`` () =
    requireEqualTracesWithRandomData [[3L; 3L]; [3L; 3L]; [3L; 3L]] (fun [a; b; c]  ->
        Expr.minElemwise (Expr.maxElemwise a b) c
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: max, min elemwise derivative`` () =
    requireEqualTracesWithRandomData [[2L; 2L]; [2L; 2L]; [2L; 2L]] (fun [a; b; c]  ->
        let expr = Expr.minElemwise (Expr.maxElemwise a b) c
        let dexpr = Deriv.compute expr
        let da = dexpr |> Deriv.ofVar a
        let db = dexpr |> Deriv.ofVar b
        let dc = dexpr |> Deriv.ofVar c
        Expr.discard [expr; da; db; dc]
   )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: max, min reduction`` () =
    requireEqualTracesWithRandomData [[4L; 5L; 3L]] (fun [a]  ->
        a |> Expr.maxAxis 2 |> Expr.minAxis 1
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: max, min reduction derivative`` () =
    requireEqualTracesWithRandomData [[4L; 5L; 3L]] (fun [a]  ->
        let expr = a |> Expr.maxAxis 2 |> Expr.minAxis 1
        let dexpr = Deriv.compute expr
        let da = dexpr |> Deriv.ofVar a
        Expr.discard [expr; da]
   )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: argMax reduction`` () =
    requireEqualTracesWithRandomDataIdx [[4L; 5L; 3L]] (fun [a]  ->
        a |> Expr.argMaxAxis 1
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: argMin reduction`` () =
    requireEqualTracesWithRandomDataIdx [[4L; 5L; 3L]] (fun [a]  ->
        a |> Expr.argMinAxis 2
    )



[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: comparison`` () =
    requireEqualTracesWithRandomDataLogic [[3L; 3L]; [3L; 3L]] (fun [a; b] ->
        a >>== b
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: comparison, logics`` () =
    requireEqualTracesWithRandomDataLogic [[3L; 3L]; [3L; 3L]; [3L; 3L]] (fun [a; b; c] ->
        a >>== b &&&& ~~~~(b <<<< c)
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: comparison, logics, conditionals`` () =
    requireEqualTracesWithRandomData [[5L; 5L]; [5L; 5L]; [5L; 5L]; [5L; 5L]] (fun [a; b; c; d] ->
        Expr.ifThenElse ((a <<== b) &&&& (b >>>> c)) (d) (a) 
    )


let conditionalsTest (device: IDevice) =
    let a = Expr.var<single> "a" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    let b = Expr.var<single> "b" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    let c = Expr.var<single> "c" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    let d = Expr.var<single> "d" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    let expr = Expr.ifThenElse ((a <<== b) &&&& (b >>>> c)) (d) (a) 
    let fn = Func.make<single> device.DefaultFactory expr |> arg4 a b c d
    let rng = System.Random (123)
    let av = rng.UniformTensor (-1.0f, 1.0f) [3L; 3L] |> post device
    let bv = rng.UniformTensor (-1.0f, 1.0f) [3L; 3L] |> post device
    let cv = rng.UniformTensor (-1.0f, 1.0f) [3L; 3L] |> post device
    let dv = rng.UniformTensor (-1.0f, 1.0f) [3L; 3L] |> post device
    let res = fn av bv cv dv
    printfn "a=\n%A" av
    printfn "b=\n%A" bv
    printfn "c=\n%A" cv
    printfn "d=\n%A" dv
    printfn "res=\n%A" res

[<Fact>]
let ``Comparison, logics, conditionals on host`` () =
    conditionalsTest DevHost

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Comparison, logics, conditionals on CUDA`` () =
    SymTensor.Compiler.Cuda.Debug.DumpCode <- true
    conditionalsTest DevCuda
    

let ``Interpolate1D: simple test`` device =
    let tbl = [1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f]
                |> HostTensor.ofList |> post device
    let minVal = 1.0
    let maxVal = 6.0

    let ip = Interpolator.create tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp = Expr.var<single> "inp" [nSmpls]
    let expr = Expr.interpolate1D ip inp
    let fn = Func.make device.DefaultFactory expr |> arg1 inp

    let inpVal = [-0.5f; 0.9f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.5f; 200.0f]
                    |> HostTensor.ofList |> post device
    let expVal = [ 1.0f; 1.0f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.0f; 6.0f]
                    |> HostTensor.ofList |> post device
    let resVal = fn inpVal

    printfn "tbl=\n%A" tbl
    printfn "inp=\n%A" inpVal
    printfn "res=\n%A" resVal

    Tensor.almostEqualWithTol (resVal, expVal, absTol=0.005f, relTol=1e-5f) |> should equal true

let ``Interpolate2D: simple test`` device =
    let tbl = [[1.0f; 2.0f; 3.0f]
               [4.0f; 5.0f; 6.0f]
               [7.0f; 8.0f; 9.0f]]
              |> HostTensor.ofList2D |> post device
    let minVal = [0.0; 0.0]
    let maxVal = [2.0; 2.0]

    let ip = Interpolator.create tbl minVal maxVal [Nearest; Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp1 = Expr.var<single> "inp1" [nSmpls]
    let inp2 = Expr.var<single> "inp2" [nSmpls]
    let expr = Expr.interpolate2D ip inp1 inp2
    let fn = Func.make device.DefaultFactory expr |> arg2 inp1 inp2

    let inpVal1 = [-0.1f; 0.0f; 0.5f; 1.5f; 2.0f; 2.3f;] |> HostTensor.ofList |> post device
    let inpVal2 = [-0.1f; 0.0f; 0.8f; 4.5f; 2.0f; 2.3f;] |> HostTensor.ofList |> post device
    let expVal =  [ 1.0f; 1.0f; 3.3f; 7.5f; 9.0f; 9.0f;] |> HostTensor.ofList |> post device
    let resVal = fn inpVal1 inpVal2

    printfn "tbl=\n%A" tbl
    printfn "inp1=\n%A" inpVal1
    printfn "inp2=\n%A" inpVal2
    printfn "res=\n%A" resVal

    Tensor.almostEqualWithTol (resVal, expVal, absTol=0.005f, relTol=1e-5f) |> should equal true

[<Fact>]
let ``Interpolate1D: simple test on host`` () =    
    ``Interpolate1D: simple test`` DevHost

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Interpolate1D: simple test on CUDA`` () =    
    ``Interpolate1D: simple test`` DevCuda

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Interpolate2D: simple test on host`` () =    
    ``Interpolate2D: simple test`` DevHost

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Interpolate2D: simple test on CUDA`` () =    
    ``Interpolate2D: simple test`` DevCuda



let ``Interpolate1D: derivative test`` device =
    let tbl = [1.0f; 2.0f; 4.0f; 7.0f; 11.0f; 16.0f]
                |> HostTensor.ofList |> post device
    let minVal = 1.0
    let maxVal = 6.0

    let ip = Interpolator.create tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp = Expr.var<single> "inp" [nSmpls]
    let expr = Expr.interpolate1D ip inp
    let dexpr = Deriv.compute expr
    let dinp = dexpr |> Deriv.ofVar inp
    let fn = Func.make device.DefaultFactory dinp |> arg1 inp

    let inpVal = [-0.5f; 0.9f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.5f; 200.0f]
                    |> HostTensor.ofList |> post device
    let expVal = [ 0.0f; 0.0f; 1.0f; 1.0f; 2.0f; 5.0f; 0.0f; 0.0f; 0.0f]
                    |> HostTensor.ofList |> Tensor.diagMat |> post device
    let resVal = fn inpVal

    printfn "derivative:"
    printfn "tbl=\n%A" tbl
    printfn "inp=\n%A" inpVal
    printfn "res=\n%A" resVal

    Tensor.almostEqualWithTol (resVal, expVal, absTol=0.005f, relTol=1e-5f) |> should equal true


[<Fact>]
let ``Interpolate1D: derivative test on host`` () =    
    ``Interpolate1D: derivative test`` DevHost

[<Trait("Category", "Skip_CI")>]
[<Fact>]
let ``Interpolate1D: derivative test on CUDA`` () =    
    ``Interpolate1D: derivative test`` DevCuda


let checkFiniteOpTest diagVal offDiagVal =
    let a = Expr.var<single> "a" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    let b = Expr.var<single> "b" [SizeSpec.fix 3L; SizeSpec.fix 3L]
    let expr = a / b |> Expr.checkFinite "a / b"
    let fn = Func.make<single> DevCuda.DefaultFactory expr |> arg2 a b
    let av = CudaTensor.ones<single> [3L; 3L]
    let dv = diagVal * CudaTensor.ones<single> [3L]
    let bv = offDiagVal * CudaTensor.ones<single> [3L; 3L]
    (Tensor.diag bv).[*] <- dv
    printfn "a=\n%A" av
    printfn "b=\n%A" bv
    let iav = fn av bv
    printfn "a / b=\n%A" iav

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Check finite on CUDA failing`` () =
    SymTensor.Compiler.Cuda.Debug.TerminateWhenNonFinite <- false
    printfn "failing:"
    checkFiniteOpTest 1.0f 0.0f

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Check finite on CUDA passing`` () =
    printfn "passing:"
    checkFiniteOpTest 1.0f 0.5f

[<Fact>]
let ``ReverseAxis on host`` () =
    let a = Expr.var<int> "a" [SizeSpec.fix 3L; SizeSpec.fix 2L]
    let expr0 = Expr.reverseAxis 0 a
    let expr1 = Expr.reverseAxis 1 a
    let fn = Func.make2<int, int> DevHost.DefaultFactory expr0 expr1 |> arg1 a

    let av = [0 .. 5] |> HostTensor.ofList |> Tensor.reshape [3L; 2L]
    printfn "av=\n%A" av

    let rav0, rav1 = fn av
    printfn "rev 0 av=\n%A" rav0
    printfn "rev 1 av=\n%A" rav1

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: Gather 1`` () =
    requireEqualTraces (fun device ->
        let a = Expr.var<single> "a" [SizeSpec.fix 4L; SizeSpec.fix 3L]
        let i0 = Expr.var<int64> "i0" [SizeSpec.broadcastable; SizeSpec.fix 3L]
        let i1 = Expr.var<int64> "i1" [SizeSpec.broadcastable; SizeSpec.fix 3L]

        let expr = a |> Expr.gather [Some i0; Some i1]
        let exprFn = Func.make<single> device.DefaultFactory expr |> arg3 a i0 i1

        let av = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor.single
        let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
        let i1v = [0L; 0L; 1L] |> HostTensor.ofList |> Tensor.padLeft

        let sv = exprFn av i0v i1v
        printfn "a=\n%A" av
        printfn "idxs=\n%A\n%A" i0v i1v
        printfn "select idxs a=\n%A" sv
    )
    

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: Gather 2`` () =
    requireEqualTraces (fun device ->
        let a = Expr.var<single> "a" [SizeSpec.fix 4L; SizeSpec.fix 3L]
        let i0 = Expr.var<int64> "i0" [SizeSpec.broadcastable; SizeSpec.fix 3L]

        let expr = a |> Expr.gather [Some i0; None]
        let exprFn = Func.make<single> device.DefaultFactory expr |> arg2 a i0 

        let av = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor.single
        let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft

        let sv = exprFn av i0v 
        printfn "a=\n%A" av
        printfn "idxs=\n%A" i0v 
        printfn "select idxs a=\n%A" sv
    )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: Scatter 1`` () =
    requireEqualTraces (fun device ->
        let a = Expr.var<single> "a" [SizeSpec.fix 4L; SizeSpec.fix 3L]
        let i0 = Expr.var<int64> "i0" [SizeSpec.broadcastable; SizeSpec.fix 3L]
        let shp = [SizeSpec.fix 5L; SizeSpec.fix 5L]

        let expr = a |> Expr.scatter [Some i0; None] shp
        let exprFn = Func.make<single> device.DefaultFactory expr |> arg2 a i0 

        let av = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor.single
        let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft

        let sv = exprFn av i0v 
        printfn "a=\n%A" av
        printfn "idxs=\n%A" i0v 
        printfn "scatter idxs a=\n%A" sv
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: Scatter 2`` () =
    requireEqualTraces (fun device ->
        let a = Expr.var<single> "a" [SizeSpec.fix 4L; SizeSpec.fix 3L]
        let i0 = Expr.var<int64> "i0" [SizeSpec.broadcastable; SizeSpec.fix 3L]
        let i1 = Expr.var<int64> "i1" [SizeSpec.broadcastable; SizeSpec.fix 3L]
        let shp = [SizeSpec.fix 5L; SizeSpec.fix 5L]

        let expr = a |> Expr.scatter [Some i0; Some i1] shp
        let exprFn = Func.make<single> device.DefaultFactory expr |> arg3 a i0 i1

        let av = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor.single
        let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
        let i1v = [0L; 0L; 0L] |> HostTensor.ofList |> Tensor.padLeft

        let sv = exprFn av i0v i1v
        printfn "a=\n%A" av
        printfn "idxs=\n%A\n%A" i0v i1v
        printfn "scatter idxs a=\n%A" sv
    )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: arange int`` () =
    requireEqualTraces (fun device ->
        let expr = Expr.arange<int> (SizeSpec.fix 10L)
        let exprFn = Func.make<int> device.DefaultFactory expr |> arg0
        let av = exprFn ()
        printfn "arange<int> 10 =%A"av
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: arange single`` () =
    requireEqualTraces (fun device ->
        let expr = Expr.arange<single> (SizeSpec.fix 10L)
        let exprFn = Func.make<single> device.DefaultFactory expr |> arg0
        let av = exprFn ()
        printfn "arange<single> 10 =%A"av
    )
