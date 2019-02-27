namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr
open Utils


module Vars =
    let a = Var.make<float32> ("a", HostTensor.Dev, [SizeSpec.fix 2L; SizeSpec.fix 3L])
    let b = Var.make<float32> ("b", HostTensor.Dev, [SizeSpec.fix 2L; SizeSpec.fix 3L])


module VarVals =
    let a = HostTensor.counting 6L |> Tensor<float32>.convert |> Tensor.reshape [2L; 3L]
    let b = 10L + HostTensor.counting 6L |> Tensor<float32>.convert |> Tensor.reshape [2L; 3L]

    let varEnv =
        VarEnv.empty
        |> VarEnv.add Vars.a a
        |> VarEnv.add Vars.b b 



type ExprTestCase = {
    Expr:       Expr
    DataType:   System.Type
    Dev:        ITensorDevice
    Shape:      ShapeSpec
    Value:      ITensor
}


module ExprTestCase =
    let test (output: ITestOutputHelper) (tc: ExprTestCase) =
        let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 
        printfn "Expr: %s" (tc.Expr.ToString())
        printfn "==== DataType:           %A" tc.Expr.DataType
        printfn "==== Device:             %A" tc.Expr.Dev
        printfn "==== Shape:              %A" tc.Expr.Shape
        printfn "==== CanEvalAllSymSizes: %A" tc.Expr.CanEvalAllSymSizes
        printfn "==== Vars:               %A" tc.Expr.Vars
        let value = tc.Expr |> Expr.eval VarVals.varEnv 
        printfn "==== Value:              \n%A" value

        assert (tc.Expr.DataType = tc.DataType)
        assert (tc.Expr.Dev = tc.Dev)
        assert (tc.Expr.Shape = tc.Shape)
        assert (value.AlmostEqual tc.Value)



module ExprTestCases =
    let ``a + b`` () = {
        Expr = Expr Vars.a + Expr Vars.b
        DataType = typeof<float32>
        Dev = HostTensor.Dev
        Shape = [SizeSpec.fix 2L; SizeSpec.fix 3L]
        Value = VarVals.a + VarVals.b
    }

    let ``sin a + cos b`` () = {
        Expr = sin (Expr Vars.a) + cos (Expr Vars.b)
        DataType = typeof<float32>
        Dev = HostTensor.Dev
        Shape = [SizeSpec.fix 2L; SizeSpec.fix 3L]
        Value = sin VarVals.a + cos VarVals.b
    }


type ExprTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

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
    let ``a + b`` () = ExprTestCase.test output (ExprTestCases.``a + b`` ())

    [<Fact>]
    let ``sin a + cos b`` () = ExprTestCase.test output (ExprTestCases.``sin a + cos b`` ())
