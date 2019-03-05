﻿namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit


open Tensor
open Tensor.Expr


type BaseExprTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``Expr equality`` () =
        let a1 = Var<float32> ("a", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])
        let b1 = Var<float32> ("b", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])
        let a2 = Var<float32> ("a", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])
        let b2 = Var<float32> ("b", HostTensor.Dev, [SizeSpec.fix 10L; SizeSpec.fix 20L])

        let expr1 = sin (Expr.var a1) - cos (Expr.var b1)
        let expr2 = sin (Expr.var a2) - cos (Expr.var b2)
        let expr3 = sin (Expr.var a2) - cos (Expr.var a2)

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