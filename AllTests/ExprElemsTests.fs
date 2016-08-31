module ExprElemsTests

open Xunit
open FsUnit.Xunit

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open TestUtils


[<Fact>]
let ``Evaluating an element expression`` () =   
    // input  x[i, j]
    //        y[m, n]
    // output  [k]

    printfn "======= Testing evaluation:"

    let k = ElemExpr.idx 0   
    let x = ElemExpr.argElem 0
    let y = ElemExpr.argElem 1
    let expr = 2.0 * (x [k; k]) + y [SizeSpec.zero; k]

    let xVal = [1.0; 2.0; 3.0] |> ArrayNDHost.ofList |> ArrayND.diagMat
    let yVal = [[4.0; 5.0; 6.0]
                [7.0; 8.0; 9.0]] |> ArrayNDHost.ofList2D
    let res = ElemExpr.eval expr [xVal; yVal] [xVal.Shape.[0]]

    printfn "Expr:\n%A" expr
    printfn "x=\n%A" xVal
    printfn "y=\n%A" yVal
    printfn "result=\n%A" res

    let expected = [6.0; 9.0; 12.0] |> ArrayNDHost.ofList
    ArrayND.almostEqual res expected |> ArrayND.value |> should equal true


[<Fact>]
let ``Derivative of an element expression`` () =   
    // input  x[i, j]
    //        y[m, n]
    // output  [k]

    printfn "======= Testing derivatives:"

    let ks = SizeSpec.symbol "ks"

    let k = ElemExpr.idx 0   
    let x = ElemExpr.argElem 0
    let y = ElemExpr.argElem 1
    let expr = 2.0 * (x [k; k]) + y [SizeSpec.zero; k]
    let dExpr = ElemExprDeriv.buildDerivElemExpr expr [ks] 2

    printfn "Expr:\n%A" expr
    printfn "dExpr / dx:\n%A" dExpr.[0]
    printfn "dExpr / dy:\n%A" dExpr.[1]


[<Fact>]
let ``Derivative of an element expression 2`` () =   
    // input  x[i, j]
    //        y[m, n]
    // output  [k, l]

    printfn "======= Testing derivatives 2:"

    let ks = SizeSpec.symbol "ks"
    let ls = SizeSpec.symbol "ls"
    let k = ElemExpr.idx 0   
    let l = ElemExpr.idx 1

    let x = ElemExpr.argElem 0
    let y = ElemExpr.argElem 1
    let expr = 2.0 * (x [k; k]) + y [l; k]
    let dExpr = ElemExprDeriv.buildDerivElemExpr expr [ks; ls] 2

    printfn "Expr:\n%A" expr
    printfn "dExpr / dx:\n%A" dExpr.[0]
    printfn "dExpr / dy:\n%A" dExpr.[1]
