namespace global
#nowarn "25"

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor
open Tensor.Expr
open TestUtils


type DerivTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 
    let check typShps exprFn =
        DerivCheck.random (typShps, exprFn, log=output.WriteLine)
    let checkProduct typShps exprFn =
        DerivCheck.random (typShps, exprFn, maxDeviation=1e-2, log=output.WriteLine)

    [<Fact>]
    let ``Plus`` () =
        check [typeof<float>, [3L; 3L]; typeof<float>, [3L; 3L]] (fun [a; b] ->
            a + b 
        )

    [<Fact>]
    let ``Sum`` () =
        check [typeof<float>, [3L; 3L]] (fun [a] ->
            UExpr.sum a 
        )

    [<Fact>]
    let ``Sum Axis 1`` () =
        check [typeof<float>, [4L; 3L; 2L]] (fun [a] ->
            UExpr.sumAxis 1 a 
        )

    [<Fact>]
    let ``Product`` () =
        checkProduct [typeof<float>, [3L; 3L]] (fun [a] ->
            UExpr.product a 
        )

    [<Fact>]
    let ``Product Axis 1`` () =
        checkProduct [typeof<float>, [4L; 3L; 2L]] (fun [a] ->
            UExpr.productAxis 1 a 
        )

    [<Fact>]
    let ``Product Axis 2`` () =
        checkProduct [typeof<float>, [3L; 2L]] (fun [a] ->
            UExpr.productAxis 1 a 
        )

    [<Fact>]
    let ``Inverse`` () =
        check [typeof<float>, [2L; 2L]] (fun [a] ->
            UExpr.invert a
        ) 

    [<Fact>]
    let ``Batch inverse`` () =
        check [typeof<float>, [3L; 2L; 2L]] (fun [a] ->
            UExpr.invert a
        ) 

    [<Fact>]
    let ``Dot`` () =
        check [typeof<float>, [2L; 3L]; typeof<float>, [3L; 2L]] (fun [a; b] ->
            a .* b 
        )

    [<Fact>]
    let ``Batched Dot`` () =
        check [typeof<float>, [2L; 3L; 3L]; typeof<float>, [2L; 3L]] (fun [a; b] ->
            a .* b 
        )

    [<Fact>]
    let ``Max, min`` () =
        check  [typeof<float>, [3L; 3L]; typeof<float>, [3L; 3L]; typeof<float>, [3L; 3L]] (fun [a; b; c]  ->
            UExpr.minElemwise (UExpr.maxElemwise a b) c
        )

    //[<Fact>]
    //let ``ReplicateTo`` () =
    //    check [typeof<float>, [7L; 5L]] (fun [a]  ->
    //        a |> Expr.replicateTo 0 (SizeSpec.fix 21L) |> Expr.replicateTo 1 (SizeSpec.fix 13L)
    //    )

    [<Fact>]
    let ``Max, min output`` () =
        runOnAllDevs output (fun ctx ->
            let a = Var<single> (ctx / "a", [Size.fix 2L; Size.fix 2L])
            let b = Var<single> (ctx / "b", [Size.fix 2L; Size.fix 2L])
            let c = Var<single> (ctx / "c", [Size.fix 2L; Size.fix 2L])    
            let expr = Expr.minElemwise (Expr.maxElemwise (Expr a) (Expr b)) (Expr c)
            let fn = ExprFunc.make expr |> ExprFunc.arg3 a b c
            let dexpr = Deriv.compute expr
            let da = dexpr.Wrt a
            let db = dexpr.Wrt b
            let dc = dexpr.Wrt c
            let fda = ExprFunc.make da |> ExprFunc.arg3 a b c
            let fdb = ExprFunc.make db |> ExprFunc.arg3 a b c
            let fdc = ExprFunc.make dc |> ExprFunc.arg3 a b c
            let rng = System.Random (123)
            let av = HostTensor.randomUniform rng (-1.0f, 1.0f) [2L; 2L] |> Tensor.transfer ctx.Dev
            let bv = HostTensor.randomUniform rng (-1.0f, 1.0f) [2L; 2L] |> Tensor.transfer ctx.Dev
            let cv = HostTensor.randomUniform rng (-1.0f, 1.0f) [2L; 2L] |> Tensor.transfer ctx.Dev
            let res = fn av bv cv
            let dav = fda av bv cv
            let dbv = fdb av bv cv
            let dcv = fdc av bv cv
            printfn "a=\n%A" av
            printfn "b=\n%A" bv
            printfn "c=\n%A" cv
            printfn "res=\n%A" res
            printfn "da=\n%A" dav
            printfn "db=\n%A" dbv
            printfn "dc=\n%A" dcv
        )
           
    [<Fact>]
    let ``Max reduction output`` () =
        runOnAllDevs output (fun ctx ->
            let a = Var<single> (ctx / "a", [Size.fix 3L; Size.fix 4L])
            let expr = Expr.maxAxis 0 (Expr a)
            let fn = ExprFunc.make expr |> ExprFunc.arg1 a
            let dexpr = Deriv.compute expr
            let da = dexpr.Wrt a
            let fda = ExprFunc.make da |> ExprFunc.arg1 a
            let rng = System.Random (123)
            let av = HostTensor.randomUniform rng (-1.0f, 1.0f) [3L; 4L] |> Tensor.transfer ctx.Dev
            let res = fn av 
            let dav = fda av 
            printfn "a=\n%A" av
            printfn "res=\n%A" res
            printfn "da=\n%A" dav
        )

    [<Fact>]
    let ``Gather`` () =
        runOnAllDevs output (fun ctx ->
            let a = Var<float> (ctx / "a", [Size.fix 4L; Size.fix 3L])
            let i0 = Var<int64> (ctx / "i0", [Size.broadcastable; Size.fix 3L])
            let i1 = Var<int64> (ctx / "i1", [Size.broadcastable; Size.fix 3L])

            let expr = Expr a |> Expr.gather [Some (Expr i0); Some (Expr i1)]

            let av = HostTensor.counting 12L |> Tensor.reshape [4L; 3L] |> Tensor<float>.convert
            let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
            let i1v = [0L; 0L; 1L] |> HostTensor.ofList |> Tensor.padLeft
            let varEnv = VarEnv.ofSeq [
                a.Untyped, av :> ITensor
                i0.Untyped, i0v :> ITensor
                i1.Untyped, i1v :> ITensor
            ]

            DerivCheck.expr (expr.Untyped, varEnv, maxDeviation=1e-6, epsilon=1e-7, log=output.WriteLine)
        )

    [<Fact>]
    let ``Scatter`` () =
        runOnAllDevs output (fun ctx ->
            let a = Var<float> (ctx / "a", [Size.fix 4L; Size.fix 3L])
            let i0 = Var<int64> (ctx / "i0", [Size.broadcastable; Size.fix 3L])
            let i1 = Var<int64> (ctx / "i1", [Size.broadcastable; Size.fix 3L])
            let trgtShp = [Size.fix 3L; Size.fix 4L]

            let expr = Expr a |> Expr.scatter [Some (Expr i0); Some (Expr i1)] trgtShp

            let av = HostTensor.counting 12L |> Tensor.reshape [4L; 3L] |> Tensor<float>.convert
            let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
            let i1v = [0L; 0L; 1L] |> HostTensor.ofList |> Tensor.padLeft
            let varEnv = VarEnv.ofSeq [
                a.Untyped, av :> ITensor
                i0.Untyped, i0v :> ITensor
                i1.Untyped, i1v :> ITensor
            ]

            DerivCheck.expr (expr.Untyped, varEnv, maxDeviation=1e-6, epsilon=1e-7, log=output.WriteLine)
        )
