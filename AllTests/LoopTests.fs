module LoopTests
#nowarn "25"

open Xunit
open FsUnit.Xunit

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open TestUtils




[<Fact>]
let ``Simple loop`` () =
    // adding one every iteration to channel A
    
    let nIters = SizeSpec.symbol "nIters"
    let m = SizeSpec.symbol "m"
    let n = SizeSpec.symbol "n"

    let prevA = Expr.var<single> "prevA" [m; n]
    let initialA = Expr.var<single> "initialA" [m; n]

    let ch = "A"
    let chExpr = prevA + 1.0f

    let loopSpec = {
        Expr.Length = nIters
        Expr.Vars = Map [Expr.extractVar prevA, 
                         Expr.PreviousChannel {Channel=ch; Delay=SizeSpec.fix 1; Initial=Expr.InitialArg 0}]
        Expr.Channels = Map [ch,
                            {LoopValueT.Expr=chExpr; LoopValueT.SliceDim=0}]    
    }
    printfn "Loop specification:\n%A" loopSpec

    let result = Expr.loop loopSpec ch [initialA]
    printfn "result :\n%A" result

    let symSizes = Map [SizeSpec.extractSymbol nIters, SizeSpec.fix 5; 
                        SizeSpec.extractSymbol m, SizeSpec.fix 3; 
                        SizeSpec.extractSymbol n, SizeSpec.fix 2]
    let result = result |> Expr.substSymSizes symSizes
    printfn "result after substitution:\n%A" result

    let resultFn = Func.make<single> DevHost.DefaultFactory result |> arg1 initialA

    let initialAv = ArrayNDHost.zeros<single> [3; 2]
    printfn "initialAv=\n%A" initialAv

    let resultVal = resultFn initialAv
    printfn "result value=\n%A" resultVal 

