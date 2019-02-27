
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


module Vars =
    let a = Var.make<float32> ("a", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])
    let b = Var.make<float32> ("b", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])


let ``Deriv: a + b`` () =
    printfn "Deriv a+b:"
    let expr = Expr Vars.a + Expr Vars.b
    let derivs = Deriv.compute expr
    printfn "wrt a: %A" derivs.[Vars.a]  
    printfn "wrt b: %A" derivs.[Vars.b]


let ``Deriv: sin a * exp b`` () =
    printfn "Deriv sin a * exp b:"
    let expr = sin (Expr Vars.a) * exp (Expr Vars.b)
    let derivs = Deriv.compute expr
    printfn "wrt a: %A" derivs.[Vars.a]  
    printfn "wrt b: %A" derivs.[Vars.b]

[<EntryPoint>]
let main argv =
    ``Deriv: a + b`` ()
    ``Deriv: sin a * exp b`` ()
    0



