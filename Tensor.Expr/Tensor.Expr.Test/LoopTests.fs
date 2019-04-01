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

            let ch = Ch.Custom "A"
            let initialA = Var<single> (ctx / "initialA", [Size.fix 1L; Size.sym m; Size.sym n])
            let prevA = Expr.loopPrevCh ch (Expr initialA) 0
            let chExpr = prevA + 1.0f          

            printfn "initialA shape: %A" initialA.Shape
            printfn "initialA dev:   %A" initialA.Dev
            printfn "prevA shape:    %A" prevA.Shape
            printfn "prevA dev:      %A" prevA.Dev
            printfn "chExpr shape:   %A" chExpr.Shape
            printfn "chExpr dev:     %A" chExpr.Dev

            let loopChs = Map [
                ch, (chExpr.Untyped, 0)
            ]
            printfn "Loop channels:\n%A" (chExpr.ToString())

            let loopExpr = MultiChannelExpr.loop (Size.sym nIters) loopChs
            printfn "Loop expression:\n%A" (loopExpr.ToString())

            let symSizes = Map [
                nIters, Size.fix 5L
                m, Size.fix 3L
                n, Size.fix 2L
            ]
            let loopExprSubst = loopExpr |> MultiChannelExpr.substSymSizes symSizes
            printfn "Loop expression after size substitution:\n%A" loopExprSubst

            let initialAv = HostTensor.zeros<single> [1L; 3L; 2L] |> Tensor.transfer ctx.Dev
            let varEnv = VarEnv.ofSeq [
                initialA, initialAv
            ]

            printfn "initialA=\n%A" initialAv
            let resultVal = loopExprSubst |> MultiChannelExpr.eval varEnv
            printfn "Loop result value=\n%A" resultVal 
    )


    let ``Build complicated loop 1`` ctx =
        let nIters = SizeSym "nIters"
        let m = SizeSym "m"
        let n = SizeSym "n"
        let delayA = SizeSym "delayA"
        let delayB = SizeSym "delayB"

        let initialA = Var<single> (ctx / "initialA", [Size.sym delayA; Size.sym m; Size.sym n])
        let initialB = Var<single> (ctx / "initialB", [Size.sym m; Size.sym n; Size.sym delayB])
        let seqA = Var<single> (ctx / "seqA", [Size.sym n; Size.sym nIters; Size.sym m])
        let constAExt = Var<single> (ctx / "constAExt", [Size.sym n])

        let chA, chASliceDim = Ch.Custom "A", 0
        let chB, chBSliceDim = Ch.Custom "B", 2
        let prevA = Expr.loopPrevCh chA (Expr initialA) chASliceDim
        let prevB = Expr.loopPrevCh chB (Expr initialB) chBSliceDim
        let sliceA = Expr.loopInputSlice (Expr seqA) 1
        let chAExpr = prevA + sliceA.T + 0.1f
        let chBExpr = prevB + prevA + Expr constAExt + 0.01f

        let loopChs = Map [
            chA, (chAExpr.Untyped, chASliceDim)
            chB, (chBExpr.Untyped, chBSliceDim)
        ]

        let loop = MultiChannelExpr.loop (Size.sym nIters) loopChs
        printfn "Loop specification:\n%A" loop

        let resultA: Expr<single> = loop.Ch chA
        let resultB: Expr<single> = loop.Ch chB
        printfn "resultA:\n%A" resultA
        printfn "resultB:\n%A" resultB

        resultA, resultB, initialA, initialB, seqA, constAExt

    let ``Values for complicated loop 1`` (dev: ITensorDevice) =
        let initialAv = Seq.initInfinite id |> Seq.map single |> HostTensor.ofSeqWithShape [1L; 3L; 2L] |> Tensor.transfer dev
        let initialBv = Seq.initInfinite ((+) 100) |> Seq.map single |> HostTensor.ofSeqWithShape [3L; 2L; 2L] |> Tensor.transfer dev
        let seqAv     = Seq.initInfinite ((+) 1000) |> Seq.map single |> HostTensor.ofSeqWithShape [2L; 5L; 3L] |> Tensor.transfer dev
        let constAv   = HostTensor.ofList [0.001f; 0.0004f] |> Tensor.transfer dev
        printfn "initialAv=\n%A" initialAv
        printfn "initialBv=\n%A" initialBv
        printfn "seqAv=\n%A" seqAv
        printfn "constAv=\n%A" constAv
        initialAv, initialBv, seqAv, constAv

    let ``Complicated loop 1`` (ctx: Context) =   
        let resultA, resultB, initialA, initialB, seqA, constAExt = ``Build complicated loop 1`` ctx

        let resultFn = 
            ExprFunc.make (resultA, resultB) 
            |> ExprFunc.arg4 initialA initialB seqA constAExt

        let initialAv, initialBv, seqAv, constAv = ``Values for complicated loop 1`` ctx.Dev
        let resultAv, resultBv = resultFn initialAv initialBv seqAv constAv
        printfn "resultAv=\n%A" resultAv
        printfn "resultBv=\n%A" resultBv

    [<Fact>]
    let ``Evaluate complicated loop 1`` () =   
        runOnAllDevs output ``Complicated loop 1``

    let ``Complicated loop 1 Expr`` (ctx: Context) =   
        let resultA, resultB, initialA, initialB, seqA, constAExt = ``Build complicated loop 1`` ctx
        let expr = UExpr.discard [resultA.Untyped; resultB.Untyped]

        let initialAv, initialBv, seqAv, constAv = ``Values for complicated loop 1`` ctx.Dev
        let varEnv = VarEnv.ofSeq [
            initialA, initialAv
            initialB, initialBv
            seqA, seqAv
            constAExt, constAv
        ]
        
        expr, varEnv

    [<Fact>]
    let ``Trace compare: Complicated loop 1`` () =   
        requireEqualTraces output ``Complicated loop 1 Expr``

    let ``Derivative of complicated loop 1`` (ctx: Context) =   
        let resultA, resultB, initialA, initialB, seqA, constAExt = ``Build complicated loop 1`` ctx

        let result = Expr.sum resultA + Expr.sum resultB
        printfn "result:\n%A" result

        let dResult = Deriv.compute result
        let dInitialA = dResult.Wrt initialA
        let dInitialB = dResult.Wrt initialB
        let dSeqA = dResult.Wrt seqA
        let dConstAExt = dResult.Wrt constAExt

        printfn "result:\n%A" result
        printfn "dresult / dInitialA:\n%A" dInitialA
        printfn "dresult / dInitialB:\n%A" dInitialB
        printfn "dresult / dSeqA:\n%A" dSeqA
        printfn "dresult / dConstAExt:\n%A" dConstAExt

        let resultFn = 
            ExprFunc.make (result, dInitialA, dInitialB, dSeqA, dConstAExt)
            |> ExprFunc.arg4 initialA initialB seqA constAExt

        let initialAv, initialBv, seqAv, constAv = ``Values for complicated loop 1`` ctx.Dev
        let resultV, dInitialAV, dInitialBV, dSeqAV, dConstAExtV = 
            resultFn initialAv initialBv seqAv constAv

        printfn "resultV=\n%A" resultV
        printfn "dInitialAV=\n%A" dInitialAV.Full
        printfn "dInitialBV=\n%A" dInitialBV.Full
        printfn "dSeqA=\n%A" dSeqAV.Full
        printfn "dConstAExt=\n%A" dConstAExtV.Full

    let ``Derivative of complicated loop 1 Expr`` (ctx: Context) =   
        let resultA, resultB, initialA, initialB, seqA, constAExt = ``Build complicated loop 1`` ctx

        let result = Expr.sum resultA + Expr.sum resultB

        let dResult = Deriv.compute result
        let dInitialA = dResult.Wrt initialA
        let dInitialB = dResult.Wrt initialB
        let dSeqA = dResult.Wrt seqA
        let dConstAExt = dResult.Wrt constAExt

        let expr = UExpr.discard [
            result.Untyped
            dInitialA.Untyped
            dInitialB.Untyped
            dSeqA.Untyped
            dConstAExt.Untyped
        ]

        let initialAv, initialBv, seqAv, constAv = ``Values for complicated loop 1`` ctx.Dev
        let varEnv = VarEnv.ofSeq [
            initialA, initialAv
            initialB, initialBv
            seqA, seqAv
            constAExt, constAv
        ]      
        expr, varEnv

    [<Fact>]
    let ``Evaluate derivative of complicated loop 1`` () =   
        runOnAllDevs output ``Derivative of complicated loop 1`` 

    [<Fact>]
    let ``Trace compare: Derivative of complicated loop 1`` () =   
        requireEqualTraces output ``Derivative of complicated loop 1 Expr``


#if false
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



#endif
