
open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Expr

let dumpExpr (expr: UExpr) =
    printfn "Expr: %A" expr
    printfn "==== DataType:           %A" expr.DataType
    printfn "==== Shape:              %A" expr.Shape
    printfn "==== CanEvalAllSymSizes: %A" expr.CanEvalAllSymSizes
    printfn "==== Vars:               %A" expr.Vars
    printfn ""


module Vars =
    let ctx = Context.root HostTensor.Dev
    let a = Var<float32> (ctx / "a", [SizeSpec.fix 10L; SizeSpec.fix 20L])
    let b = Var<float32> (ctx / "b", [SizeSpec.fix 10L; SizeSpec.fix 20L])


let ``Deriv: a + b`` () =
    printfn "Deriv a+b:"
    let expr = Expr Vars.a + Expr Vars.b
    let derivs = Deriv.compute expr
    printfn "wrt a: %s" ((derivs.Wrt Vars.a).ToString())
    printfn "wrt b: %s" ((derivs.Wrt Vars.a).ToString())


let ``Deriv: sin a * exp b`` () =
    printfn "Deriv sin a * exp b:"
    let expr = sin (Expr Vars.a) * exp (Expr Vars.b)
    let derivs = Deriv.compute expr
    printfn "wrt a: %s" ((derivs.Wrt Vars.a).ToString())
    printfn "wrt b: %s" ((derivs.Wrt Vars.a).ToString())


[<EntryPoint>]
let main argv =
    ``Deriv: a + b`` ()
    ``Deriv: sin a * exp b`` ()
    0



