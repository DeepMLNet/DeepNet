module OpTests
#nowarn "25"

open Xunit
open FsUnit.Xunit

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open TestUtils




[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix-matrix dot`` () =   
    requireEqualTracesWithRandomData [[6; 3]; [3; 2]] (fun [a; b] ->
        a .* b
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix-vector dot`` () =   
    requireEqualTracesWithRandomData [[6; 3]; [3]] (fun [a; b] ->
        a .* b
    )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix-matrix dot`` () =   
    requireEqualTracesWithRandomData [[7; 5; 6; 3]; [7; 5; 3; 2]] (fun [a; b] ->
        a .* b
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix-matrix dot with broadcasting`` () =   
    requireEqualTracesWithRandomData [[7; 5; 6; 3]; [7; -1; 3; 2]] (fun [a; b] ->
        a .* b
    )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched build diagonal`` () =
    requireEqualTracesWithRandomData [[7; 5; 3]] (fun [a] ->
        Expr.diagMat a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched extract diagonal`` () =
    requireEqualTracesWithRandomData [[7; 5; 4; 4]] (fun [a] ->
        Expr.diag a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix inverse`` () =
    requireEqualTracesWithRandomData [[3; 3]] (fun [a] ->
        Expr.invert a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: transposed matrix inverse`` () =
    requireEqualTracesWithRandomData [[5; 5]] (fun [a] ->
        Expr.invert a.T
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix inverse`` () =
    requireEqualTracesWithRandomData [[7; 3; 4; 4]] (fun [a] ->
        Expr.invert a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: sum`` () =
    requireEqualTracesWithRandomData [[7; 3; 4; 5]] (fun [a] ->
        Expr.sum a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: sum axis 1`` () =
    requireEqualTracesWithRandomData [[7; 3; 4; 5]] (fun [a] ->
        Expr.sumAxis 1 a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: sum axis 2`` () =
    requireEqualTracesWithRandomData [[7; 3; 4; 5]] (fun [a] ->
        a |> Expr.sumAxis 3 |> Expr.sumAxis 0
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: large sum axis`` () =
    requireEqualTracesWithRandomData [[7; 200]] (fun [a] ->
        a |> Expr.sumAxis 0
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Singular matrix inverse`` () =
    let a = Expr.var<single> "a" [SizeSpec.fix 3; SizeSpec.fix 3]
    let expr = Expr.invert a
    let fn = Func.make<single> DevCuda.DefaultFactory expr |> arg1 a
    let av = ArrayNDCuda.zeros<single> [3; 3]
    let iav = fn av
    printfn "a=\n%A" av
    printfn "a^-1=\n%A" iav

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Replicate`` () =
    let a = Expr.var<single> "a" [SizeSpec.fix 2; SizeSpec.fix 3]
    let expr0 = Expr.replicate 0 (SizeSpec.fix 2) a
    let expr1 = Expr.replicate 1 (SizeSpec.fix 3) a
    let fns = Func.make2<single, single> DevCuda.DefaultFactory expr0 expr1 |> arg1 a
    let av = [[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]] |> ArrayNDHost.ofList2D 
    let av0, av1 = fns av
    printfn "a=\n%A" av 
    printfn "rep 0 2 a=\n%A" av0
    printfn "rep 1 3 a=\n%A" av1

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: max, min`` () =
    requireEqualTracesWithRandomData [[3; 3]; [3; 3]; [3; 3]] (fun [a; b; c]  ->
        Expr.minElemwise (Expr.maxElemwise a b) c
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: max, min derivative`` () =
    requireEqualTracesWithRandomData [[2; 2]; [2; 2]; [2; 2]] (fun [a; b; c]  ->
        let expr = Expr.minElemwise (Expr.maxElemwise a b) c
        let dexpr = Deriv.compute expr
        let da = dexpr |> Deriv.ofVar a
        let db = dexpr |> Deriv.ofVar b
        let dc = dexpr |> Deriv.ofVar c
        Expr.discard [expr; da; db; dc]
   )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: comparison`` () =
    requireEqualTracesWithRandomDataLogic [[3; 3]; [3; 3]] (fun [a; b] ->
        a >>== b
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: comparison, logics`` () =
    requireEqualTracesWithRandomDataLogic [[3; 3]; [3; 3]; [3; 3]] (fun [a; b; c] ->
        a >>== b &&&& ~~~~(b <<<< c)
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: comparison, logics, conditionals`` () =
    requireEqualTracesWithRandomData [[5; 5]; [5; 5]; [5; 5]; [5; 5]] (fun [a; b; c; d] ->
        Expr.ifThenElse ((a <<== b) &&&& (b >>>> c)) (d) (a) 
    )


let conditionalsTest (device: IDevice) =
    let a = Expr.var<single> "a" [SizeSpec.fix 3; SizeSpec.fix 3]
    let b = Expr.var<single> "b" [SizeSpec.fix 3; SizeSpec.fix 3]
    let c = Expr.var<single> "c" [SizeSpec.fix 3; SizeSpec.fix 3]
    let d = Expr.var<single> "d" [SizeSpec.fix 3; SizeSpec.fix 3]
    let expr = Expr.ifThenElse ((a <<== b) &&&& (b >>>> c)) (d) (a) 
    let fn = Func.make<single> device.DefaultFactory expr |> arg4 a b c d
    let rng = System.Random (123)
    let av = rng.UniformArrayND (-1.0f, 1.0f) [3; 3] |> post device
    let bv = rng.UniformArrayND (-1.0f, 1.0f) [3; 3] |> post device
    let cv = rng.UniformArrayND (-1.0f, 1.0f) [3; 3] |> post device
    let dv = rng.UniformArrayND (-1.0f, 1.0f) [3; 3] |> post device
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
                |> ArrayNDHost.ofList |> post device
    let minVal = 1.0
    let maxVal = 6.0

    let ip = Interpolator.createInterpolator tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp = Expr.var<single> "inp" [nSmpls]
    let expr = Expr.interpolate1D ip inp
    let fn = Func.make device.DefaultFactory expr |> arg1 inp

    let inpVal = [-0.5f; 0.9f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.5f; 200.0f]
                    |> ArrayNDHost.ofList |> post device
    let expVal = [ 1.0f; 1.0f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.0f; 6.0f]
                    |> ArrayNDHost.ofList |> post device
    let resVal = fn inpVal

    printfn "tbl=\n%A" tbl
    printfn "inp=\n%A" inpVal
    printfn "res=\n%A" resVal

    ArrayND.almostEqualWithTol 0.005f 1e-5f resVal expVal |> ArrayND.value |> should equal true

let ``Interpolate2D: simple test`` device =
    let tbl = [[1.0f; 2.0f; 3.0f]
               [4.0f; 5.0f; 6.0f]
               [7.0f; 8.0f; 9.0f]]
              |> ArrayNDHost.ofList2D |> post device
    let minVal = [0.0; 0.0]
    let maxVal = [2.0; 2.0]

    let ip = Interpolator.createInterpolator tbl minVal maxVal [Nearest; Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp1 = Expr.var<single> "inp1" [nSmpls]
    let inp2 = Expr.var<single> "inp2" [nSmpls]
    let expr = Expr.interpolate2D ip inp1 inp2
    let fn = Func.make device.DefaultFactory expr |> arg2 inp1 inp2

    let inpVal1 = [-0.1f; 0.0f; 0.5f; 1.5f; 2.0f; 2.3f;] |> ArrayNDHost.ofList |> post device
    let inpVal2 = [-0.1f; 0.0f; 0.8f; 4.5f; 2.0f; 2.3f;] |> ArrayNDHost.ofList |> post device
    let expVal =  [ 1.0f; 1.0f; 3.3f; 7.5f; 9.0f; 9.0f;] |> ArrayNDHost.ofList |> post device
    let resVal = fn inpVal1 inpVal2

    printfn "tbl=\n%A" tbl
    printfn "inp1=\n%A" inpVal1
    printfn "inp2=\n%A" inpVal2
    printfn "res=\n%A" resVal

    ArrayND.almostEqualWithTol 0.005f 1e-5f resVal expVal |> ArrayND.value |> should equal true

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
                |> ArrayNDHost.ofList |> post device
    let minVal = 1.0
    let maxVal = 6.0

    let ip = Interpolator.createInterpolator tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp = Expr.var<single> "inp" [nSmpls]
    let expr = Expr.interpolate1D ip inp
    let dexpr = Deriv.compute expr
    let dinp = dexpr |> Deriv.ofVar inp
    let fn = Func.make device.DefaultFactory dinp |> arg1 inp

    let inpVal = [-0.5f; 0.9f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.5f; 200.0f]
                    |> ArrayNDHost.ofList |> post device
    let expVal = [ 0.0f; 0.0f; 1.0f; 1.0f; 2.0f; 5.0f; 0.0f; 0.0f; 0.0f]
                    |> ArrayNDHost.ofList |> ArrayND.diagMat |> post device
    let resVal = fn inpVal

    printfn "derivative:"
    printfn "tbl=\n%A" tbl
    printfn "inp=\n%A" inpVal
    printfn "res=\n%A" resVal

    ArrayND.almostEqualWithTol 0.005f 1e-5f resVal expVal |> ArrayND.value |> should equal true


[<Fact>]
let ``Interpolate1D: derivative test on host`` () =    
    ``Interpolate1D: derivative test`` DevHost

[<Trait("Category", "Skip_CI")>]
[<Fact>]
let ``Interpolate1D: derivative test on CUDA`` () =    
    ``Interpolate1D: derivative test`` DevCuda


let checkFiniteOpTest diagVal offDiagVal =
    let a = Expr.var<single> "a" [SizeSpec.fix 3; SizeSpec.fix 3]
    let b = Expr.var<single> "b" [SizeSpec.fix 3; SizeSpec.fix 3]
    let expr = a / b |> Expr.checkFinite "a / b"
    let fn = Func.make<single> DevCuda.DefaultFactory expr |> arg2 a b
    let av = ArrayNDCuda.ones<single> [3; 3]
    let dv = diagVal * ArrayNDCuda.ones<single> [3]
    let bv = offDiagVal * ArrayNDCuda.ones<single> [3; 3]
    (ArrayND.diag bv).[*] <- dv
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
