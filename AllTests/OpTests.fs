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
let ``Singular matrix inverse`` () =
    let a = Expr.var "a" [SizeSpec.fix 3; SizeSpec.fix 3]
    let expr = Expr.invert a
    let fn = Func.make DevCuda.DefaultFactory expr |> arg1 a
    let av = ArrayNDCuda.zeros<single> [3; 3]
    let iav = fn av
    printfn "a=\n%A" av
    printfn "a^-1=\n%A" iav


[<Fact>]
let ``Interpolate1D: simple test on host`` () =    
        let tbl = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
                  |> ArrayNDHost.ofList
        let minVal = 1.0
        let maxVal = 6.0

        let ip = Expr.createInterpolator tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

        let nSmpls = SizeSpec.symbol "nSmpls"
        let inp = Expr.var "inp" [nSmpls]
        let expr = Expr.interpolate1D ip inp
        let fn = Func.make DevHost.DefaultFactory expr |> arg1 inp

        let inpVal = [-0.5; 0.9; 1.0; 1.5; 2.3; 5.9; 6.0; 6.5; 200.0]
                     |> ArrayNDHost.ofList
        let expVal = [ 1.0; 1.0; 1.0; 1.5; 2.3; 5.9; 6.0; 6.0; 6.0]
                     |> ArrayNDHost.ofList
        let resVal = fn inpVal

        printfn "tbl=\n%A" tbl
        printfn "inp=\n%A" inpVal
        printfn "res=\n%A" resVal

        ArrayND.almostEqual resVal expVal |> ArrayND.value |> should equal true

[<Fact>]
let ``Interpolate1D: simple test on CUDA`` () =    
        let tbl = [1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f]
                  |> ArrayNDHost.ofList |> ArrayNDCuda.toDev
        let minVal = 1.0f
        let maxVal = 6.0f

        let ip = Expr.createInterpolator tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

        let nSmpls = SizeSpec.symbol "nSmpls"
        let inp = Expr.var "inp" [nSmpls]
        let expr = Expr.interpolate1D ip inp
        let fn = Func.make DevCuda.DefaultFactory expr |> arg1 inp

        let inpVal = [-0.5f; 0.9f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.5f; 200.0f]
                     |> ArrayNDHost.ofList |> ArrayNDCuda.toDev
        let expVal = [ 1.0f; 1.0f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.0f; 6.0f]
                     |> ArrayNDHost.ofList |> ArrayNDCuda.toDev
        let resVal = fn inpVal

        printfn "tbl=\n%A" tbl
        printfn "inp=\n%A" inpVal
        printfn "res=\n%A" resVal

        ArrayND.almostEqualWithTol 0.005f 1e-5f resVal expVal |> ArrayND.value |> should equal true



[<Fact>]
let ``Interpolate1D: derivative test on host`` () =    
        let tbl = [1.0; 2.0; 4.0; 7.0; 11.0; 16.0]
                  |> ArrayNDHost.ofList
        let minVal = 1.0
        let maxVal = 6.0

        let ip = Expr.createInterpolator tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

        let nSmpls = SizeSpec.symbol "nSmpls"
        let inp = Expr.var "inp" [nSmpls]
        let expr = Expr.interpolate1D ip inp
        let dexpr = Deriv.compute expr
        let dinp = dexpr |> Deriv.ofVar inp
        let fn = Func.make DevHost.DefaultFactory dinp |> arg1 inp

        let inpVal = [-0.5; 0.9; 1.0; 1.5; 2.3; 5.9; 6.0; 6.5; 200.0]
                     |> ArrayNDHost.ofList
        let expVal = [ 0.0; 0.0; 1.0; 1.0; 2.0; 5.0; 0.0; 0.0; 0.0]
                     |> ArrayNDHost.ofList |> ArrayND.diagMat
        let resVal = fn inpVal

        printfn "derivative:"
        printfn "tbl=\n%A" tbl
        printfn "inp=\n%A" inpVal
        printfn "res=\n%A" resVal

        ArrayND.almostEqual resVal expVal |> ArrayND.value |> should equal true


[<Fact>]
let ``Interpolate1D: derivative test on CUDA`` () =    
        let tbl = [1.0f; 2.0f; 4.0f; 7.0f; 11.0f; 16.0f]
                  |> ArrayNDHost.ofList |> ArrayNDCuda.toDev
        let minVal = 1.0f
        let maxVal = 6.0f

        let ip = Expr.createInterpolator tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

        let nSmpls = SizeSpec.symbol "nSmpls"
        let inp = Expr.var "inp" [nSmpls]
        let expr = Expr.interpolate1D ip inp
        let dexpr = Deriv.compute expr
        let dinp = dexpr |> Deriv.ofVar inp
        let fn = Func.make DevCuda.DefaultFactory dinp |> arg1 inp

        let inpVal = [-0.5f; 0.9f; 1.0f; 1.5f; 2.3f; 5.9f; 6.0f; 6.5f; 200.0f]
                     |> ArrayNDHost.ofList |> ArrayNDCuda.toDev
        let expVal = [ 0.0f; 0.0f; 1.0f; 1.0f; 2.0f; 5.0f; 0.0f; 0.0f; 0.0f]
                     |> ArrayNDHost.ofList |> ArrayND.diagMat |> ArrayNDCuda.toDev
        let resVal = fn inpVal

        printfn "derivative:"
        printfn "tbl=\n%A" tbl
        printfn "inp=\n%A" inpVal
        printfn "res=\n%A" resVal

        ArrayND.almostEqualWithTol 0.005f 1e-5f resVal expVal |> ArrayND.value |> should equal true







