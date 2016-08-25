module OpTests
#nowarn "25"

open Xunit
open FsUnit.Xunit

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open TestUtils




[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix-matrix dot`` () =   
    requireEqualTracesWithRandomData [[6; 3]; [3; 2]] (fun [a; b] ->
        a .* b
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix-vector dot`` () =   
    requireEqualTracesWithRandomData [[6; 3]; [3]] (fun [a; b] ->
        a .* b
    )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix-matrix dot`` () =   
    requireEqualTracesWithRandomData [[7; 5; 6; 3]; [7; 5; 3; 2]] (fun [a; b] ->
        a .* b
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix-matrix dot with broadcasting`` () =   
    requireEqualTracesWithRandomData [[7; 5; 6; 3]; [7; -1; 3; 2]] (fun [a; b] ->
        a .* b
    )


[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched build diagonal`` () =
    requireEqualTracesWithRandomData [[7; 5; 3]] (fun [a] ->
        Expr.diagMat a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched extract diagonal`` () =
    requireEqualTracesWithRandomData [[7; 5; 4; 4]] (fun [a] ->
        Expr.diag a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: matrix inverse`` () =
    requireEqualTracesWithRandomData [[3; 3]] (fun [a] ->
        Expr.invert a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: transposed matrix inverse`` () =
    requireEqualTracesWithRandomData [[5; 5]] (fun [a] ->
        Expr.invert a.T
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Trace compare: batched matrix inverse`` () =
    requireEqualTracesWithRandomData [[7; 3; 4; 4]] (fun [a] ->
        Expr.invert a
    )

[<Fact>]
[<Trait("Category", "Skip_CI")>]
let ``Singular matrix inverse`` () =
    let a = Expr.var "a" [SizeSpec.fix 3; SizeSpec.fix 3]
    let expr = Expr.invert a
    let fn = Func.make DevCuda.DefaultFactory expr |> arg1 a
    let av = ArrayNDCuda.zeros<single> [3; 3]
    let iav = fn av
    printfn "a=\n%A" av
    printfn "a^-1=\n%A" iav

    



