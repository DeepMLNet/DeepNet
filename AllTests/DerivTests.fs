module DerivTests
#nowarn "25"

open Xunit
open FsUnit.Xunit

open Basics
open Tensor
open SymTensor
open SymTensor.Compiler.Cuda
open TestUtils


[<Fact>]
let ``Plus`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[3L; 3L]; [3L; 3L]] (fun [a; b] ->
        a + b 
    )

[<Fact>]
let ``Sum`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[3L; 3L]] (fun [a] ->
        Expr.sum a 
    )

[<Fact>]
let ``Sum Axis 1`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[4L; 3L; 2L]] (fun [a] ->
        Expr.sumAxis 1 a 
    )

[<Fact>]
let ``Product`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[3L; 3L]] (fun [a] ->
        Expr.product a 
    )

[<Fact>]
let ``Product Axis 1`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[4L; 3L; 2L]] (fun [a] ->
        Expr.productAxis 1 a 
    )

[<Fact>]
let ``Product Axis 2`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[3L; 2L]] (fun [a] ->
        Expr.productAxis 1 a 
    )

[<Fact>]
let ``Inverse`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[2L; 2L]] (fun [a] ->
        Expr.invert a
    ) 

[<Fact>]
let ``Batch inverse`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[3L; 2L; 2L]] (fun [a] ->
        Expr.invert a
    ) 

[<Fact>]
let ``Dot`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[2L; 3L]; [3L; 2L]] (fun [a; b] ->
        a .* b 
    )

[<Fact>]
let ``Batched Dot`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[2L; 3L; 3L]; [2L; 3L]] (fun [a; b] ->
        a .* b 
    )

[<Fact>]
let ``Max, min`` () =
    randomDerivativeCheckTreeOnHost 1e-4  [[3L; 3L]; [3L; 3L]; [3L; 3L]] (fun [a; b; c]  ->
        Expr.minElemwise (Expr.maxElemwise a b) c
    )

[<Fact>]
let ``ReplicateTo`` () =
    randomDerivativeCheckTreeOnHost 1e-4 [[7L; 5L]] (fun [a]  ->
        a |> Expr.replicateTo 0 (SizeSpec.fix 21L) |> Expr.replicateTo 1 (SizeSpec.fix 13L)
    )


let ``Max, min output`` (device: IDevice) =
    let a = Expr.var<single> "a" [SizeSpec.fix 2L; SizeSpec.fix 2L]
    let b = Expr.var<single> "b" [SizeSpec.fix 2L; SizeSpec.fix 2L]
    let c = Expr.var<single> "c" [SizeSpec.fix 2L; SizeSpec.fix 2L]    
    let expr = Expr.minElemwise (Expr.maxElemwise a b) c
    let fn = Func.make<single> device.DefaultFactory expr |> arg3 a b c
    let dexpr = Deriv.compute expr
    let da = dexpr |> Deriv.ofVar a
    let db = dexpr |> Deriv.ofVar b
    let dc = dexpr |> Deriv.ofVar c
    let fda = Func.make<single> device.DefaultFactory da |> arg3 a b c
    let fdb = Func.make<single> device.DefaultFactory db |> arg3 a b c
    let fdc = Func.make<single> device.DefaultFactory dc |> arg3 a b c
    let rng = System.Random (123)
    let av = rng.UniformTensor (-1.0f, 1.0f) [2L; 2L] |> post device
    let bv = rng.UniformTensor (-1.0f, 1.0f) [2L; 2L] |> post device
    let cv = rng.UniformTensor (-1.0f, 1.0f) [2L; 2L] |> post device
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
           
[<Fact>]
let ``Max, min output on host`` () =
    ``Max, min output`` DevHost

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Max, min output on CUDA`` () =
    ``Max, min output`` DevCuda

let ``Max reduction output`` (device: IDevice) =
    let a = Expr.var<single> "a" [SizeSpec.fix 3L; SizeSpec.fix 4L]
    let expr = Expr.maxAxis 0 a
    let fn = Func.make<single> device.DefaultFactory expr |> arg1 a
    let dexpr = Deriv.compute expr
    let da = dexpr |> Deriv.ofVar a
    let fda = Func.make<single> device.DefaultFactory da |> arg1 a
    let rng = System.Random (123)
    let av = rng.UniformTensor (-1.0f, 1.0f) [3L; 4L] |> post device
    let res = fn av 
    let dav = fda av 
    printfn "a=\n%A" av
    printfn "res=\n%A" res
    printfn "da=\n%A" dav

[<Fact>]
let ``Max reduction output on host`` () =
    ``Max reduction output`` DevHost

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Max reduction output on CUDA`` () =
    ``Max reduction output`` DevCuda

[<Fact>]
let ``Gather`` () =
    let a = Expr.var<float> "a" [SizeSpec.fix 4L; SizeSpec.fix 3L]
    let i0 = Expr.var<int64> "i0" [SizeSpec.broadcastable; SizeSpec.fix 3L]
    let i1 = Expr.var<int64> "i1" [SizeSpec.broadcastable; SizeSpec.fix 3L]

    let expr = a |> Expr.gather [Some i0; Some i1]

    let av = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor.float
    let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
    let i1v = [0L; 0L; 1L] |> HostTensor.ofList |> Tensor.padLeft
    let varEnv = VarEnv.ofSeq [a, av :> ITensor; i0, i0v :> ITensor; i1, i1v :> ITensor]

    DerivCheck.checkExprTree DevHost 1e-6 1e-7 varEnv expr

[<Fact>]
let ``Scatter`` () =
    let a = Expr.var<float> "a" [SizeSpec.fix 4L; SizeSpec.fix 3L]
    let i0 = Expr.var<int64> "i0" [SizeSpec.broadcastable; SizeSpec.fix 3L]
    let i1 = Expr.var<int64> "i1" [SizeSpec.broadcastable; SizeSpec.fix 3L]
    let trgtShp = [SizeSpec.fix 3L; SizeSpec.fix 4L]

    let expr = a |> Expr.scatter [Some i0; Some i1] trgtShp

    let av = Seq.counting |> HostTensor.ofSeqWithShape [4L; 3L] |> Tensor.float
    let i0v = [1L; 2L; 2L] |> HostTensor.ofList |> Tensor.padLeft
    let i1v = [0L; 0L; 1L] |> HostTensor.ofList |> Tensor.padLeft
    let varEnv = VarEnv.ofSeq [a, av :> ITensor; i0, i0v :> ITensor; i1, i1v :> ITensor]

    DerivCheck.checkExprTree DevHost 1e-6 1e-7 varEnv expr
