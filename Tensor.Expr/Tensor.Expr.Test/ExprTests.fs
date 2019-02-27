module ExprTests

open Xunit
open FsUnit.Xunit

open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Expr
open Utils


module Vars =
    let a = Var.make<float32> ("a", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])
    let b = Var.make<float32> ("b", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])


[<Fact>]
let ``Expr is reference unique`` () =
    printfn "==== Expr is reference unique:"

    let a1 = Var.make<float32> ("a", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])
    let b1 = Var.make<float32> ("b", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])
    let a2 = Var.make<float32> ("a", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])
    let b2 = Var.make<float32> ("b", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])

    let expr1 = sin (Expr a1) - cos (Expr b1)
    let expr2 = sin (Expr a2) - cos (Expr b2)
    let expr3 = sin (Expr a2) - cos (Expr a2)

    printfn "expr1: %A" expr1
    printfn "expr2: %A" expr2
    printfn "expr3: %A" expr3

    printfn "expr1 = expr2: %A" (expr1 = expr2)
    printfn "Reference equals of BaseExpr: %A" (obj.ReferenceEquals(expr1.BaseExpr, expr2.BaseExpr))
    assert (expr1 = expr2)
    assert (obj.ReferenceEquals(expr1.BaseExpr, expr2.BaseExpr))

    printfn "expr1 <> expr3: %A" (expr1 <> expr3)
    printfn "Reference equals of BaseExpr: %A" (obj.ReferenceEquals(expr1.BaseExpr, expr3.BaseExpr))
    assert (expr1 <> expr3)
    assert (not (obj.ReferenceEquals(expr1.BaseExpr, expr3.BaseExpr)))

    printfn "expr2 <> expr3: %A" (expr2 <> expr3)
    printfn "Reference equals of BaseExpr: %A" (obj.ReferenceEquals(expr2.BaseExpr, expr3.BaseExpr))
    assert (expr2 <> expr3)
    assert (not (obj.ReferenceEquals(expr2.BaseExpr, expr3.BaseExpr)))



[<Fact>]
let ``Expr: a + b`` () =
    printfn "==== a+b:"
    let expr = Expr Vars.a + Expr Vars.b
    dumpExpr expr  
    assert (expr.DataType = typeof<float32>)
    assert (expr.Shape = [SizeSpec.fix 10L; SizeSpec.fix 20L])
    assert (expr.CanEvalAllSymSizes = true)
    assert (expr.Vars = Set [Vars.a; Vars.b])

[<Fact>]
let ``Expr: sin a + cos b`` () =
    printfn "==== sin a + cos b:"
    let expr = sin (Expr Vars.a) + cos (Expr Vars.b)
    dumpExpr expr  
    assert (expr.DataType = typeof<float32>)
    assert (expr.Shape = [SizeSpec.fix 10L; SizeSpec.fix 20L])
    assert (expr.CanEvalAllSymSizes = true)
    assert (expr.Vars = Set [Vars.a; Vars.b])





