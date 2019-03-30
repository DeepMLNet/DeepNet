namespace global
#nowarn "25"

open Xunit
open Xunit.Abstractions
open FsUnit.Xunit

open DeepNet.Utils
open Tensor
open Tensor.Expr
open TestUtils


type LoopTests (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``Simple loop`` () =
        runOnAllDevs output (fun ctx ->
            let nIters = SizeSym "nIters"
            let m = SizeSym "m"
            let n = SizeSym "n"

            let prevA = Var<single> (ctx / "prevA", [m; n])
            let initialA = Var<single> (ctx / "initialA", [Size.fix 1L; m; n])

            let ch = "A"
            let chExpr = Expr prevA + 1.0f
            
            let loopSpec = {
                Expr.Length = nIters
                Expr.Vars = Map [Expr.extractVar prevA, 
                                 Expr.PreviousChannel {Channel=ch; Delay=Size.fix 1L; InitialArg=0}]
                Expr.Channels = Map [ch,
                                    {LoopValueT.Expr=chExpr; LoopValueT.SliceDim=0}]    
            }
            printfn "Loop specification:\n%A" loopSpec

            let result = Expr.loop loopSpec ch [initialA]
            printfn "result :\n%A" result

            let symSizes = Map [Size.extractSymbol nIters, Size.fix 5L; 
                                Size.extractSymbol m, Size.fix 3L; 
                                Size.extractSymbol n, Size.fix 2L]
            let result = result |> Expr.substSymSizes symSizes
            printfn "result after substitution:\n%A" result

            let resultFn = Func.make<single> device.DefaultFactory result |> arg1 initialA
    
            let initialAv = HostTensor.zeros<single> [1L; 3L; 2L]
            printfn "initialAv=\n%A" initialAv

            let resultVal = resultFn initialAv
            printfn "result value=\n%A" resultVal 
    )

    let ``Build complicated loop 1`` () =
        let nIters = Size.symbol "nIters"
        let m = Size.symbol "m"
        let n = Size.symbol "n"
        let delayA = Size.symbol "delayA"
        let delayB = Size.symbol "delayB"

        let prevA = Expr.var<single> "prevA" [m; n]
        let initialA = Expr.var<single> "initialA" [delayA; m; n]
        let prevB = Expr.var<single> "prevB" [m; n]
        let initialB = Expr.var<single> "initialB" [m; n; delayB]
        let sliceA = Expr.var<single> "sliceA" [n; m]
        let seqA = Expr.var<single> "seqA" [n; nIters; m]
        let constA = Expr.var<single> "constA" [n]
        let constAExt = Expr.var<single> "constAExt" [n]

        let chA = "A"
        let chAExpr = prevA + sliceA.T + 0.1f
        let chB = "B"
        let chBExpr = prevB + prevA + constA + 0.01f

        let loopSpec = {
            Expr.Length = nIters
            Expr.Vars = Map [Expr.extractVar prevA,  Expr.PreviousChannel {Channel=chA; Delay=delayA; InitialArg=0}
                             Expr.extractVar prevB,  Expr.PreviousChannel {Channel=chB; Delay=delayB; InitialArg=1}
                             Expr.extractVar sliceA, Expr.SequenceArgSlice {ArgIdx=2; SliceDim=1}
                             Expr.extractVar constA, Expr.ConstArg 3]
            Expr.Channels = Map [chA, {LoopValueT.Expr=chAExpr; LoopValueT.SliceDim=0}
                                 chB, {LoopValueT.Expr=chBExpr; LoopValueT.SliceDim=2}]    
        }
        printfn "Loop specification:\n%A" loopSpec

        let resultA = Expr.loop loopSpec chA [initialA; initialB; seqA; constAExt]
        let resultB = Expr.loop loopSpec chB [initialA; initialB; seqA; constAExt]
        //printfn "resultA:\n%A" resultA
        //printfn "resultB:\n%A" resultB

    //    let symSizes = Map [Size.extractSymbol nIters, Size.fix 5
    //                        Size.extractSymbol m,      Size.fix 3 
    //                        Size.extractSymbol n,      Size.fix 2
    //                        Size.extractSymbol delayA, Size.fix 1
    //                        Size.extractSymbol delayB, Size.fix 2
    //                        ]
    //    let resultA = resultA |> Expr.substSymSizes symSizes
    //    let resultB = resultB |> Expr.substSymSizes symSizes   
    //
    //    let initialA = initialA |> Expr.substSymSizes symSizes
    //    let initialB = initialB |> Expr.substSymSizes symSizes
    //    let seqA = seqA |> Expr.substSymSizes symSizes
    //    let constAExt = constAExt |> Expr.substSymSizes symSizes

        resultA, resultB, initialA, initialB, seqA, constAExt

    
    let ``Values for complicated loop 1`` () =
        let initialAv = Seq.countingFrom 0 |> Seq.map single |> HostTensor.ofSeqWithShape [1L; 3L; 2L]
        let initialBv = Seq.countingFrom 100 |> Seq.map single |> HostTensor.ofSeqWithShape [3L; 2L; 2L]
        let seqAv     = Seq.countingFrom 1000 |> Seq.map single |> HostTensor.ofSeqWithShape [2L; 5L; 3L]
        let constAv   = HostTensor.ofList [0.001f; 0.0004f] 
        printfn "initialAv=\n%A" initialAv
        printfn "initialBv=\n%A" initialBv
        printfn "seqAv=\n%A" seqAv
        printfn "constAv=\n%A" constAv
        initialAv, initialBv, seqAv, constAv

    let ``Complicated loop 1`` (device: IDevice) =   
        let resultA, resultB, initialA, initialB, seqA, constAExt = ``Build complicated loop 1`` ()

        let resultFn = 
            Func.make2<single, single> device.DefaultFactory resultA resultB 
            |> arg4 initialA initialB seqA constAExt

        //let ses = Trace.startSession "cloop"

        let initialAv, initialBv, seqAv, constAv = ``Values for complicated loop 1`` ()
        let resultAv, resultBv = resultFn initialAv initialBv seqAv constAv
        printfn "resultAv=\n%A" resultAv
        printfn "resultBv=\n%A" resultBv

        //let ts = ses.End ()
        //ts |> Trace.dumpToFile "ComplicatedLoop1.txt"

    [<Fact>]
    let ``Complicated loop 1 on host`` () =   
        ``Complicated loop 1`` DevHost

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Complicated loop 1 on CUDA`` () =   
        ``Complicated loop 1`` DevCuda

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Trace compare: Complicated loop 1`` () =   
        requireEqualTraces ``Complicated loop 1``

    let ``Derivative of complicated loop 1`` (device: IDevice) =   
        let resultA, resultB, initialA, initialB, seqA, constAExt = ``Build complicated loop 1`` ()

        let result = Expr.sum resultA + Expr.sum resultB
        printfn "result:\n%A" result
        let dResult = Deriv.compute result
        let dInitialA = dResult |> Deriv.ofVar initialA
        let dInitialB = dResult |> Deriv.ofVar initialB
        let dSeqA = dResult |> Deriv.ofVar seqA
        let dConstAExt = dResult |> Deriv.ofVar constAExt

    //    printfn "result:\n%A" result
    //    printfn "dresult / dInitialA:\n%A" dInitialA
    //    printfn "dresult / dInitialB:\n%A" dInitialB
    //    printfn "dresult / dSeqA:\n%A" dSeqA
    //    printfn "dresult / dConstAExt:\n%A" dConstAExt

        let resultFn = 
            Func.make5<single, single, single, single, single> device.DefaultFactory 
                result dInitialA dInitialB dSeqA dConstAExt
            |> arg4 initialA initialB seqA constAExt

        let initialAv, initialBv, seqAv, constAv = ``Values for complicated loop 1`` ()
        let resultV, dInitialAV, dInitialBV, dSeqAV, dConstAExtV = resultFn initialAv initialBv seqAv constAv
        let dInitialAV = dInitialAV |> Tensor.reshape initialAv.Shape
        let dInitialBV = dInitialBV |> Tensor.reshape initialBv.Shape
        let dSeqAV = dSeqAV |> Tensor.reshape seqAv.Shape
        let dConstAExtV = dConstAExtV |> Tensor.reshape constAv.Shape
        printfn "resultV=\n%A" resultV
        printfn "dInitialAV=\n%A" dInitialAV.Full
        printfn "dInitialBV=\n%A" dInitialBV.Full
        printfn "dSeqA=\n%A" dSeqAV.Full
        printfn "dConstAExt=\n%A" dConstAExtV.Full

    
    [<Fact>]
    let ``Derivative of complicated loop 1 on host`` () =   
        ``Derivative of complicated loop 1`` DevHost

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Derivative of complicated loop 1 on CUDA`` () =   
        ``Derivative of complicated loop 1`` DevCuda

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    let ``Trace compare: Derivative of complicated loop 1`` () =   
        requireEqualTraces ``Derivative of complicated loop 1``


    [<Fact>]
    let ``Derivative compare: Complicated loop 1`` () =
        randomDerivativeCheckTreeOnHost 1e-4 [[1L; 3L; 2L]; [3L; 2L; 2L]; [2L; 5L; 3L]; [2L]] 
            (fun [initialA; initialB; seqA; constAExt] ->
                let nIters = Size.fix 5L
                let m = Size.fix 3L
                let n = Size.fix 2L
                let delayA = Size.fix 1L
                let delayB = Size.fix 2L

                let prevA = Expr.var<float> "prevA" [m; n]
                let prevB = Expr.var<float> "prevB" [m; n]
                let sliceA = Expr.var<float> "sliceA" [n; m]
                let constA = Expr.var<float> "constA" [n]

                let chA = "A"
                let chAExpr = prevA + 1.0 + sliceA.T
                let chB = "B"
                let chBExpr = prevB + prevA + constA

                let loopSpec = {
                    Expr.Length = nIters
                    Expr.Vars = Map [Expr.extractVar prevA,  Expr.PreviousChannel {Channel=chA; Delay=delayA; InitialArg=0}
                                     Expr.extractVar prevB,  Expr.PreviousChannel {Channel=chB; Delay=delayB; InitialArg=1}
                                     Expr.extractVar sliceA, Expr.SequenceArgSlice {ArgIdx=2; SliceDim=1}
                                     Expr.extractVar constA, Expr.ConstArg 3]
                    Expr.Channels = Map [chA, {LoopValueT.Expr=chAExpr; LoopValueT.SliceDim=0}
                                         chB, {LoopValueT.Expr=chBExpr; LoopValueT.SliceDim=2}]    
                }
                let resultA = Expr.loop loopSpec chA [initialA; initialB; seqA; constAExt]
                let resultB = Expr.loop loopSpec chB [initialA; initialB; seqA; constAExt]
                Expr.sum resultA + Expr.sum resultB            
            )


    [<Fact>]
    let ``Derivative compare: Simple loop 1`` () =
        randomDerivativeCheckTreeOnHost 1e-4 [[1L; 2L]] 
            (fun [initialA] ->
                let nIters = Size.fix 3L
                let n = Size.fix 2L
                let delayA = Size.fix 1L

                let prevA = Expr.var<float> "prevA" [n]

                let chA = "A"
                let chAExpr = 2.0 * prevA + 1.0

                let loopSpec = {
                    Expr.Length = nIters
                    Expr.Vars = Map [Expr.extractVar prevA,  Expr.PreviousChannel {Channel=chA; Delay=delayA; InitialArg=0}]
                    Expr.Channels = Map [chA, {LoopValueT.Expr=chAExpr; LoopValueT.SliceDim=0}]
                }
                let resultA = Expr.loop loopSpec chA [initialA]
                Expr.sum resultA
            )



