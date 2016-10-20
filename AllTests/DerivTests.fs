module DerivTests
#nowarn "25"

open Xunit
open FsUnit.Xunit

open Basics
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open TestUtils


[<Fact>]
let ``Plus`` () =
    randomDerivativeCheck 1e-4 [[3; 3]; [3; 3]] (fun [a; b] ->
        a + b 
    )

[<Fact>]
let ``Inverse`` () =
    randomDerivativeCheck 1e-4 [[2; 2]] (fun [a] ->
        Expr.invert a
    ) 

[<Fact>]
let ``Batch inverse`` () =
    randomDerivativeCheck 1e-4 [[3; 2; 2]] (fun [a] ->
        Expr.invert a
    ) 


[<Fact>]
let ``Dot`` () =
    randomDerivativeCheck 1e-4 [[2; 3]; [3;2]] (fun [a; b] ->
        a .* b 
    )

[<Fact>]
let ``Batched Dot`` () =
    randomDerivativeCheck 1e-4 [[2;3; 3]; [2;3]] (fun [a; b] ->
        a .* b 
    )

[<Fact>]
let ``Max, min`` () =
    randomDerivativeCheck 1e-4  [[3; 3]; [3; 3]; [3; 3]] (fun [a; b; c]  ->
        Expr.minElemwise (Expr.maxElemwise a b) c
    )

[<Fact>]
let ``ReplicateTo`` () =
    randomDerivativeCheck 1e-4 [[7; 5]] (fun [a]  ->
        a |> Expr.replicateTo 0 (SizeSpec.fix 21) |> Expr.replicateTo 1 (SizeSpec.fix 13)
    )


let ``Max, min output`` (device: IDevice) =
    let a = Expr.var<single> "a" [SizeSpec.fix 2; SizeSpec.fix 2]
    let b = Expr.var<single> "b" [SizeSpec.fix 2; SizeSpec.fix 2]
    let c = Expr.var<single> "c" [SizeSpec.fix 2; SizeSpec.fix 2]    
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
    let av = rng.UniformArrayND (-1.0f, 1.0f) [2; 2] |> post device
    let bv = rng.UniformArrayND (-1.0f, 1.0f) [2; 2] |> post device
    let cv = rng.UniformArrayND (-1.0f, 1.0f) [2; 2] |> post device
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
    let a = Expr.var<single> "a" [SizeSpec.fix 3; SizeSpec.fix 4]
    let expr = Expr.maxAxis 0 a
    let fn = Func.make<single> device.DefaultFactory expr |> arg1 a
    let dexpr = Deriv.compute expr
    let da = dexpr |> Deriv.ofVar a
    let fda = Func.make<single> device.DefaultFactory da |> arg1 a
    let rng = System.Random (123)
    let av = rng.UniformArrayND (-1.0f, 1.0f) [3; 4] |> post device
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
    let a = Expr.var<float> "a" [SizeSpec.fix 4; SizeSpec.fix 3]
    let i0 = Expr.var<int> "i0" [SizeSpec.broadcastable; SizeSpec.fix 3]
    let i1 = Expr.var<int> "i1" [SizeSpec.broadcastable; SizeSpec.fix 3]

    let expr = a |> Expr.gather [Some i0; Some i1]

    let av = Seq.counting |> ArrayNDHost.ofSeqWithShape [4; 3] |> ArrayND.float
    let i0v = [1; 2; 2] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let i1v = [0; 0; 1] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let varEnv = VarEnv.ofSeq [a, av :> IArrayNDT; i0, i0v :> IArrayNDT; i1, i1v :> IArrayNDT]

    DerivCheck.checkExprTree DevHost 1e-6 1e-7 varEnv expr


[<Fact>]
let ``Scatter`` () =
    let a = Expr.var<float> "a" [SizeSpec.fix 4; SizeSpec.fix 3]
    let i0 = Expr.var<int> "i0" [SizeSpec.broadcastable; SizeSpec.fix 3]
    let i1 = Expr.var<int> "i1" [SizeSpec.broadcastable; SizeSpec.fix 3]
    let trgtShp = [SizeSpec.fix 3; SizeSpec.fix 4]

    let expr = a |> Expr.scatter [Some i0; Some i1] trgtShp

    let av = Seq.counting |> ArrayNDHost.ofSeqWithShape [4; 3] |> ArrayND.float
    let i0v = [1; 2; 2] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let i1v = [0; 0; 1] |> ArrayNDHost.ofList |> ArrayND.padLeft
    let varEnv = VarEnv.ofSeq [a, av :> IArrayNDT; i0, i0v :> IArrayNDT; i1, i1v :> IArrayNDT]

    DerivCheck.checkExprTree DevHost 1e-6 1e-7 varEnv expr
