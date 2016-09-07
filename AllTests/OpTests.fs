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


let ``Interpolate1D: simple test`` device =
    let tbl = [1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f]
                |> ArrayNDHost.ofList |> post device
    let minVal = 1.0f
    let maxVal = 6.0f

    let ip = Expr.createInterpolator tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp = Expr.var "inp" [nSmpls]
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
    let minVal = [0.0f; 0.0f]
    let maxVal = [2.0f; 2.0f]

    let ip = Expr.createInterpolator tbl minVal maxVal [Nearest; Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp1 = Expr.var "inp1" [nSmpls]
    let inp2 = Expr.var "inp2" [nSmpls]
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
    let minVal = 1.0f
    let maxVal = 6.0f

    let ip = Expr.createInterpolator tbl [minVal] [maxVal] [Nearest] InterpolateLinearaly None

    let nSmpls = SizeSpec.symbol "nSmpls"
    let inp = Expr.var "inp" [nSmpls]
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

