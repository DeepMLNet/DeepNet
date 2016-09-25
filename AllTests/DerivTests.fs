module DerivTests
#nowarn "25"

open Xunit
open FsUnit.Xunit

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

