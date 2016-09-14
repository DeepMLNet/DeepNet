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
let ``Dot2`` () =
    randomDerivativeCheck 1e-4 [[2;3; 3]; [2;3]] (fun [a; b] ->
        a .* b 
    )