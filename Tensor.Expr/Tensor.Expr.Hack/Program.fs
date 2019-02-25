﻿
open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Expr

let dumpExpr (expr: Expr) =
    printfn "Expr: %A" expr
    printfn "==== DataType:           %A" expr.DataType
    printfn "==== Shape:              %A" expr.Shape
    printfn "==== CanEvalAllSymSizes: %A" expr.CanEvalAllSymSizes
    printfn "==== Vars:               %A" expr.Vars
    printfn ""


let ExprAPB () =
    let a = Var.make<float32> ("a", [SizeSpec.fix 10L; SizeSpec.fix 20L])
    let b = Var.make<float32> ("b", [SizeSpec.fix 10L; SizeSpec.fix 20L])
    printfn "Building expr..."
    let expr = Expr a + Expr b
    printfn "a+b:"
    dumpExpr expr


module Vars =
    let a = Var.make<float32> ("a", [SizeSpec.fix 10L; SizeSpec.fix 20L])
    let b = Var.make<float32> ("b", [SizeSpec.fix 10L; SizeSpec.fix 20L])


let ``Deriv: a + b`` () =
    printfn "Deriv a+b:"
    let expr = Expr Vars.a + Expr Vars.b
    let derivs = Deriv.compute expr
    printfn "wrt a: %A" derivs.[Vars.a]  
    printfn "wrt b: %A" derivs.[Vars.b]


[<EntryPoint>]
let main argv =
    ``Deriv: a + b`` ()
    0



