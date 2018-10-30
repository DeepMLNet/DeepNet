namespace SymTensor

open System
open System.IO
open System.Collections.Generic
open System.Threading

open Tensor
open DeepNet.Utils

open UExprTypes



module Trace =

    let mutable WithMessage = false

    type LoopIter = {
        LoopExpr:       UExprT
        Iter:           int64
    }

    type LoopStack = LoopIter list
    
    type EvalEvent = 
        | ExprEvaled of UExprT * LoopStack * ITensor * string
        | EnteringLoop of UExprT
        | LeavingLoop of UExprT
        | LoopIteration of LoopIter

    type ExprEvaluation = {
        Id:             int
        Exprs:          UExprT list
        Compiler:       string
        Start:          DateTime
        mutable End:    DateTime option
        Trace:          ResizeArray<EvalEvent>
    }

    type TraceSession = {
        Name:                   string
        Start:                  DateTime
        mutable End:            DateTime option
        ExprEvals:              ResizeArray<ExprEvaluation>
        StoreResultEventRng:    int option * int option
    }

    let private activeTraceSession = new ThreadLocal<TraceSession option>()
    let private activeExprEval = new ThreadLocal<ExprEvaluation option>()
    let private activeLoopStack = new ThreadLocal<Stack<LoopIter>>()

    let inline private getActiveTraceSession () = 
        match activeTraceSession.Value with
        | Some ts -> ts
        | None -> failwith "no trace has been started"

    let isActive () =
        match activeTraceSession.Value with
        | Some ts -> true
        | None -> false

    let private endSession () =
        let ts = 
            match activeTraceSession.Value with
            | Some ts -> ts
            | None -> failwith "no trace has been started"
        if activeExprEval.Value.IsSome then
            failwith "trace session cannot end while expression is being evaluated"

        activeTraceSession.Value <- None
        ts.End <- Some DateTime.Now
        ts

    let private abortSession () =
        activeExprEval.Value <- None
        activeTraceSession.Value <- None

    type TraceSessionHandle internal (ts: TraceSession) =       
        let mutable ended = false

        let abort () =
            if not ended then
                ended <- true
                abortSession ()

        member this.End () =
            if ended then
                failwithf "Trace session %s already finished" ts.Name
            ended <- true
            endSession ()

        interface IDisposable with
            member this.Dispose () = abort ()
        override this.Finalize() = abort ()

    let startSessionWithRng name rng =
        match activeTraceSession.Value with
        | Some ts -> failwithf "trace session %s already active" ts.Name
        | None -> ()

        let ts = {
            Name                = name
            Start               = DateTime.Now
            End                 = None
            ExprEvals           = ResizeArray<ExprEvaluation>()
            StoreResultEventRng = rng
        }
        activeTraceSession.Value <- Some ts
        new TraceSessionHandle(ts)

    let startSession name = 
        startSessionWithRng name (None, None)

    let inline private getActiveExpr () = 
        match activeExprEval.Value with
        | Some ses -> ses
        | None -> failwith "no ExprEvaluation is currently being traced"
      
    let startExprEval uexprs compiler =
        match activeTraceSession.Value with
        | Some ts ->
            match activeExprEval.Value with
            | Some ee -> failwithf "already tracing ExprEvaluation %A" ee
            | None -> ()

            let ses = {
                Id = ts.ExprEvals.Count;
                Exprs = uexprs;
                Compiler = compiler;
                Start = DateTime.Now;
                End = None;
                Trace = ResizeArray()
            }
            ts.ExprEvals.Add ses
            activeExprEval.Value <- Some ses
            activeLoopStack.Value <- Stack<_> ()
        | None -> ()

    let endExprEval () =
        if isActive () then
            let ses = getActiveExpr ()
            ses.End <- Some DateTime.Now
            activeExprEval.Value <- None

    let enteringLoop uexpr =
        if isActive () then
            match uexpr with
            | UExpr (UExtraOp (Loop _), _, _) -> ()
            | _ -> failwithf "not a loop expression: %A" uexpr
            activeLoopStack.Value.Push {LoopExpr=uexpr; Iter=0L}
            let ee = getActiveExpr ()
            ee.Trace.Add (EnteringLoop uexpr)

    let leavingLoop uexpr =
        if isActive () then
            if activeLoopStack.Value.Count = 0 then failwith "no loop active"
            let loop = activeLoopStack.Value.Pop ()
            if loop.LoopExpr <> uexpr then
                failwithf "loop %A must not end before loop %A" uexpr loop.LoopExpr
            let ee = getActiveExpr ()
            ee.Trace.Add (LeavingLoop uexpr)

    let setLoopIter iter =
        if isActive () then
            if activeLoopStack.Value.Count = 0 then failwith "no loop active"
            let loopIter = {activeLoopStack.Value.Pop() with Iter=iter}
            activeLoopStack.Value.Push loopIter
            let ee = getActiveExpr ()
            ee.Trace.Add (LoopIteration loopIter)

    let loopStack () : LoopStack =
        activeLoopStack.Value.ToArray() |> List.ofArray |> List.rev

    let private empty = HostTensor.zeros<int> [0L] :> ITensor

    let exprEvaledWithMsg uexpr (res: Lazy<ITensor>) msg =
        if isActive () then
            let ee, es = getActiveExpr (), getActiveTraceSession ()
            let id = ee.Trace.Count
            let first, last = es.StoreResultEventRng
            let first, last = first |? 0, last |? Int32.MaxValue
            let resVal =
                if (first <= id && id <= last) then ITensor.copy (res.Force())
                else empty
            ee.Trace.Add (ExprEvaled (uexpr, loopStack(), resVal, msg))
            
    let exprEvaled uexpr res =
        exprEvaledWithMsg uexpr res ""

    let maxSimilar (a: ITensor) (b: ITensor) =
        match a.DataType, b.DataType with
        | ta, tb when ta <> tb -> false
        | t, _ when t = typeof<float> ->
            let a = a :?> Tensor<float>
            let b = b :?> Tensor<float>
            Tensor.almostEqual (a, b, absTol=1e-5, relTol=1e-5) 
        | t, _ when t = typeof<single> ->
            let a = a :?> Tensor<single>
            let b = b :?> Tensor<single>
            Tensor.almostEqual (a, b, absTol=1e-5f, relTol=1e-5f) 
        | t, _ when t = typeof<bool> ->
            let a = a :?> Tensor<bool>
            let b = b :?> Tensor<bool>
            Tensor.all (a ==== b) 
        | t, _ when t = typeof<int> ->
            let a = a :?> Tensor<int>
            let b = b :?> Tensor<int>
            Tensor.all (a ==== b) 
        | t, _ when t = typeof<int64> ->
            let a = a :?> Tensor<int64>
            let b = b :?> Tensor<int64>
            Tensor.all (a ==== b) 
        | t -> failwithf "unsupported trace data type %A" t

    let compareCustom isSimilar a b =
        let maxDiffs = 3
        let mutable diffs = 0

        printfn "Comparing trace sessions %s and %s:" a.Name b.Name

        if a.ExprEvals.Count <> b.ExprEvals.Count then
            printfn "Different number of expression evaluations: %d vs %d" 
                a.ExprEvals.Count b.ExprEvals.Count
        else
            for e, (ae, be) in Seq.indexed (Seq.zip a.ExprEvals b.ExprEvals) do
                printfn ""
                printfn "Evaluation %d using evaluator %s vs %s:" e ae.Compiler be.Compiler

                for aEvent in ae.Trace do
                    match aEvent with
                    | ExprEvaled (uexpr, aLs, aRes, aMsg) ->
                        match Seq.tryFind (function 
                                            | ExprEvaled (oexpr, bLs, _, _) -> oexpr = uexpr && 
                                                                               aLs = bLs 
                                            | _ -> false) be.Trace with
                        | Some (ExprEvaled (_, _, bRes, bMsg) as bEvent) ->
                            if not (isSimilar aRes bRes) then
                                if diffs < maxDiffs then
                                    printfn ""
                                    printfn "Difference in expression:\n%A" uexpr
                                    printfn "Loop stack: %A" (aLs |> List.map (fun l -> l.Iter))
                                    printfn "%s index: %d    %s index: %d" 
                                        a.Name (Seq.findIndex ((=) aEvent) ae.Trace)
                                        b.Name (Seq.findIndex ((=) bEvent) be.Trace)
                                    printfn ""
                                    if WithMessage && aMsg.Length > 0 then printfn "%s message: %s" a.Name aMsg
                                    if WithMessage && bMsg.Length > 0 then printfn "%s message: %s" b.Name bMsg
                                    printfn ""
                                    printfn "%s result:\n%A\n" a.Name aRes
                                    printfn "%s result:\n%A\n" b.Name bRes
                                elif diffs = maxDiffs then
                                    printfn ""
                                    printfn "(more differences not shown)"
                                diffs <- diffs + 1
                        | _ -> ()
                    | _ -> ()
        printfn ""
        printfn "Total number of differences: %d" diffs
        diffs

    let compare = compareCustom maxSimilar

    let dump txtFile (hdfFile: HDF5) trace =
        let exprMaxLength = 300
        let out fmt = fprintfn txtFile fmt
        out "Trace session %s" trace.Name
        out "Start: %A" trace.Start
        let endStr =
            match trace.End with
            | Some e -> sprintf "%A" e
            | None -> "in progress"
        out "End:   %s" endStr
        out ""

        for exprEval in trace.ExprEvals do
            let exprsStr = 
                exprEval.Exprs 
                |> List.map UExpr.toExpr
                |> List.map (fun e -> e.ToString exprMaxLength)
                |> String.concat "\n"
            out "Evaluation of expression(s):" 
            out "%s" exprsStr
            out ""
            out "Id:       %d" exprEval.Id
            out "Compiler: %s" exprEval.Compiler
            out "Start:    %A" exprEval.Start
            let endStr =
                match exprEval.End with
                | Some e -> sprintf "%A" e
                | None -> "in progress"
            out "End:      %s" endStr

            let hdfGroup = sprintf "%05d" exprEval.Id
            hdfFile.CreateGroups(hdfGroup)
            hdfFile.SetAttribute(hdfGroup, "Expressions", exprsStr)
            hdfFile.SetAttribute(hdfGroup, "Compiler", exprEval.Compiler)
            hdfFile.SetAttribute(hdfGroup, "Start", sprintf "%A" exprEval.Start)
            hdfFile.SetAttribute(hdfGroup, "End", sprintf "%A" exprEval.End)

            out ""
            out "==== Begin of trace ===="
            out ""

            for idx, evnt in Seq.indexed exprEval.Trace do                
                out "Event index: %d" idx
                match evnt with
                | EnteringLoop uexpr -> out "Entering loop:\n%A" uexpr
                | LeavingLoop uexpr -> out "Leaving loop:\n%A" uexpr
                | LoopIteration li -> out "Performing loop iteration %d" li.Iter
                | ExprEvaled (uexpr, ls, res, msg) ->
                    let exprStr =
                        match UExpr.tryToExpr uexpr with
                        | Some expr -> expr.ToString exprMaxLength
                        | None -> sprintf "%A" uexpr
                    let loopStackStr =
                        sprintf "%A" (ls |> List.map (fun l -> l.Iter))

                    out "(Unified) expression: %s" exprStr
                    out "Loop stack: %s" loopStackStr
                    out "Result:\n%A" res
                    if WithMessage then out "Message: %s" msg

                    let hdfPath = sprintf "%s/%05d" hdfGroup idx
                    HostTensor.write hdfFile hdfPath res
                    hdfFile.SetAttribute(hdfPath, "Expression", exprStr)
                    hdfFile.SetAttribute(hdfPath, "LoopStack", loopStackStr)
                out ""

            out "==== End of trace ===="
            out ""

    let dumpToFile txtPath hdfPath trace =
        use txtFile = File.CreateText txtPath
        use hdfFile = HDF5.OpenWrite hdfPath
        dump txtFile hdfFile trace

    let dumpActiveTrace txtFile hdfFile =
        getActiveTraceSession () |> dump txtFile hdfFile

    let extractLoop uexpr =
        match uexpr with
        | UExpr (UExtraOp (Channel _), [UExpr (UExtraOp (Loop _), _, _) as uLoop], _) -> uLoop
        | _ -> failwith "not a loop channel expression"


