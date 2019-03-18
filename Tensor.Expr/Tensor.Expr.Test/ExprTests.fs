namespace global

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor.Utils
open Tensor
open Tensor.Backend
open Tensor.Expr
open TestUtils



type ExprTestCase = {
    Expr:       UExpr
    DataType:   System.Type
    Dev:        ITensorDevice
    Shape:      Shape
    Value:      ITensor
}


module ExprTestCase =
    let test (output: ITestOutputHelper) (varEnv: VarEnv) (tc: ExprTestCase) =
        let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

        printfn "Expr: %s" (tc.Expr.ToString())
        printfn "==== DataType:           %A" tc.Expr.DataType
        printfn "==== Device:             %A" tc.Expr.Dev
        printfn "==== Shape:              %A" tc.Expr.Shape
        printfn "==== CanEvalAllSymSizes: %A" tc.Expr.CanEvalAllSymSizes
        printfn "==== Vars:               %A" tc.Expr.Vars
        assert (tc.Expr.DataType = tc.DataType)
        assert (tc.Expr.Dev = tc.Dev)
        assert (tc.Expr.Shape = tc.Shape)

        printfn ""
        let tracer = TextTracer (output.WriteLine)
        let evalEnv: Ops.EvalEnv = {VarEnv=varEnv; Tracer=tracer}
        let value = tc.Expr |> UExpr.evalWithEnv evalEnv      
        printfn ""
        assert (value.AlmostEqual tc.Value)

    let deriv (output: ITestOutputHelper) (varEnv: VarEnv) (tc: ExprTestCase) =
        let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

        printfn "Expr: %s" (tc.Expr.ToString())
        let derivs = Deriv.compute tc.Expr
        for var in tc.Expr.Vars do
            printfn "wrt %A:   %A" var (derivs.Wrt var)
            let value = derivs.Wrt var |> UExpr.eval varEnv 
            printfn "evaled:   %A" value
            printfn ""

        DerivCheck.expr (tc.Expr, varEnv, log=output.WriteLine)


module Vars =
    let ctx = Context.root HostTensor.Dev
    let a = Var<float> (ctx / "a", [Size.fix 2L; Size.fix 3L])
    let b = Var<float> (ctx / "b", [Size.fix 2L; Size.fix 3L])


module VarVals =
    let a = HostTensor.counting 6L |> Tensor<float>.convert |> Tensor.reshape [2L; 3L]
    let b = 10L + HostTensor.counting 6L |> Tensor<float>.convert |> Tensor.reshape [2L; 3L]

    let varEnv =
        VarEnv.empty
        |> VarEnv.add Vars.a a
        |> VarEnv.add Vars.b b 



module ExprTestCases =
    let ``a + b`` () = {
        Expr = Expr Vars.a + Expr Vars.b |> Expr.untyped
        DataType = typeof<float>
        Dev = HostTensor.Dev
        Shape = [Size.fix 2L; Size.fix 3L]
        Value = VarVals.a + VarVals.b
    }

    let ``sin a + cos b`` () = {
        Expr = sin (Expr Vars.a) + cos (Expr Vars.b) |> Expr.untyped
        DataType = typeof<float>
        Dev = HostTensor.Dev
        Shape = [Size.fix 2L; Size.fix 3L]
        Value = sin VarVals.a + cos VarVals.b
    }


type ExprTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>] 
    let ``a + b`` () = 
        ExprTestCase.test output VarVals.varEnv (ExprTestCases.``a + b`` ())

    [<Fact>] 
    let ``Deriv: a + b`` () = 
        ExprTestCase.deriv output VarVals.varEnv (ExprTestCases.``a + b`` ())

    [<Fact>]
    let ``sin a + cos b`` () = 
        ExprTestCase.test output VarVals.varEnv (ExprTestCases.``sin a + cos b`` ())

    [<Fact>]
    let ``Deriv: sin a + cos b`` () = 
        ExprTestCase.deriv output VarVals.varEnv (ExprTestCases.``sin a + cos b`` ())
