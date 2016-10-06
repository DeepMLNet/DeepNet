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
    let initialA = Expr.var<single> "initialA" [SizeSpec.fix 1; m; n]

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

    let initialAv = ArrayNDHost.zeros<single> [1; 3; 2]
    printfn "initialAv=\n%A" initialAv

    let resultVal = resultFn initialAv
    printfn "result value=\n%A" resultVal 


[<Fact>]
let ``Complicated loop 1`` () =
    // adding one every iteration to channel A
    
    let nIters = SizeSpec.symbol "nIters"
    let m = SizeSpec.symbol "m"
    let n = SizeSpec.symbol "n"
    let delayA = SizeSpec.symbol "delayA"
    let delayB = SizeSpec.symbol "delayB"

    let prevA = Expr.var<single> "prevA" [m; n]
    let initialA = Expr.var<single> "initialA" [delayA; m; n]
    let prevB = Expr.var<single> "prevB" [m; n]
    let initialB = Expr.var<single> "initialA" [m; n; delayB]
    let sliceA = Expr.var<single> "sliceA" [n; m]
    let seqA = Expr.var<single> "seqA" [n; nIters; m]
    let constA = Expr.var<single> "constA" [n]
    let constAExt = Expr.var<single> "constAExt" [n]

    let chA = "A"
    let chAExpr = prevA + 1.0f + sliceA.T
    let chB = "B"
    let chBExpr = prevB + prevA + constA

    let loopSpec = {
        Expr.Length = nIters
        Expr.Vars = Map [Expr.extractVar prevA,  Expr.PreviousChannel {Channel=chA; Delay=delayA; Initial=Expr.InitialArg 0}
                         Expr.extractVar prevB,  Expr.PreviousChannel {Channel=chB; Delay=delayB; Initial=Expr.InitialArg 1}
                         Expr.extractVar sliceA, Expr.SequenceArgSlice {ArgIdx=2; SliceDim=1}
                         Expr.extractVar constA, Expr.ConstArg 3]
        Expr.Channels = Map [chA, {LoopValueT.Expr=chAExpr; LoopValueT.SliceDim=0}
                             chB, {LoopValueT.Expr=chBExpr; LoopValueT.SliceDim=2}]    
    }
    printfn "Loop specification:\n%A" loopSpec

    let resultA = Expr.loop loopSpec chA [initialA; initialB; seqA; constAExt]
    let resultB = Expr.loop loopSpec chB [initialA; initialB; seqA; constAExt]
    printfn "resultA:\n%A" resultA
    printfn "resultB:\n%A" resultB

    let symSizes = Map [SizeSpec.extractSymbol nIters, SizeSpec.fix 5
                        SizeSpec.extractSymbol m,      SizeSpec.fix 3 
                        SizeSpec.extractSymbol n,      SizeSpec.fix 2
                        SizeSpec.extractSymbol delayA, SizeSpec.fix 1
                        SizeSpec.extractSymbol delayB, SizeSpec.fix 2
                        ]
    let resultA = resultA |> Expr.substSymSizes symSizes
    let resultB = resultB |> Expr.substSymSizes symSizes   

    let resultFn = 
        Func.make2<single, single> DevHost.DefaultFactory resultA resultB 
        |> arg4 initialA initialB seqA constAExt

    let initialAv = ArrayNDHost.zeros<single> [1; 3; 2]
    let initialBv = ArrayNDHost.ones<single>  [3; 2; 2]
    let seqAv     = ArrayNDHost.linSpaced 0.0f 50.0f (5 * 3 * 2) |> ArrayND.reshape [2; 5; 3]
    let constAv   = ArrayNDHost.ofList [3.0f; 7.0f] 

    printfn "initialAv=\n%A" initialAv
    printfn "initialBv=\n%A" initialBv
    printfn "seqAv=\n%A" seqAv
    printfn "constAv=\n%A" constAv

    let resultAv, resultBv = resultFn initialAv initialBv seqAv constAv
    printfn "resultAv=\n%A" resultAv
    printfn "resultBv=\n%A" resultBv



