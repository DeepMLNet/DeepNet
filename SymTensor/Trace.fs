namespace SymTensor

open System
open System.IO
open System.Collections.Generic
open System.Threading

open ArrayNDNS
open Basics
open UExprTypes


module Trace =

    type LoopIter = {
        LoopExpr:       UExprT
        Iter:           int
    }

    type LoopStack = LoopIter list
    
    type EvalEvent = 
        | ExprEvaled of UExprT * LoopStack * IArrayNDT * string
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
        Name:           string
        Start:          DateTime
        mutable End:    DateTime option
        ExprEvals:      ResizeArray<ExprEvaluation>
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

    let startSession name = 
        match activeTraceSession.Value with
        | Some ts -> failwithf "trace session %s already active" ts.Name
        | None -> ()

        let ts = {
            Name        = name
            Start       = DateTime.Now
            End         = None
            ExprEvals   = ResizeArray<ExprEvaluation>()
        }
        activeTraceSession.Value <- Some ts
        new TraceSessionHandle(ts)

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
            activeLoopStack.Value.Push {LoopExpr=uexpr; Iter=0}
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

    let exprEvaledWithMsg uexpr res msg =
        if isActive () then
            let ee = getActiveExpr ()
            ee.Trace.Add (ExprEvaled (uexpr, loopStack(), ArrayND.copyUntyped res, msg))
            
    let exprEvaled uexpr res =
        exprEvaledWithMsg uexpr res ""

    let maxSimilar (a: IArrayNDT) (b: IArrayNDT) =
        let epsilon = 1e-4f
        match a.DataType, b.DataType with
        | ta, tb when ta <> tb -> false
        | t, _ when t = typeof<single> ->
            let a = a :?> ArrayNDT<single>
            let b = b :?> ArrayNDT<single>
            let diff = abs (a - b)
            if ArrayND.nElems diff > 0 then
                let maxDiff = ArrayND.max diff |> ArrayND.value
                maxDiff <= epsilon
            else true
        | t, _ when t = typeof<bool> ->
            let a = a :?> ArrayNDT<bool>
            let b = b :?> ArrayNDT<bool>
            ArrayND.all (a ==== b) |> ArrayND.value
        | t, _ when t = typeof<int> ->
            let a = a :?> ArrayNDT<int>
            let b = b :?> ArrayNDT<int>
            ArrayND.all (a ==== b) |> ArrayND.value
        | t -> failwithf "unsupported data type %A" t

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
                                    if aMsg.Length > 0 then printfn "%s message: %s" a.Name aMsg
                                    if bMsg.Length > 0 then printfn "%s message: %s" b.Name bMsg
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

    let dump file trace =
        let out fmt = fprintfn file fmt
        out "Trace session %s" trace.Name
        out "Start: %A" trace.Start
        let endStr =
            match trace.End with
            | Some e -> sprintf "%A" e
            | None -> "in progress"
        out "End:   %s" endStr
        out ""

        for exprEval in trace.ExprEvals do
            out "Evaluation of expression(s) %A" 
                (exprEval.Exprs |> List.map UExpr.toExpr)
            out "Id:       %d" exprEval.Id
            out "Compiler: %s" exprEval.Compiler
            out "Start:    %A" exprEval.Start
            let endStr =
                match exprEval.End with
                | Some e -> sprintf "%A" e
                | None -> "in progress"
            out "End:      %s" endStr
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
                    match UExpr.tryToExpr uexpr with
                    | Some expr -> out "Expression: %A" expr
                    | None -> out "Unified expression: %A" uexpr
                    out "Loop stack: %A" (ls |> List.map (fun l -> l.Iter))
                    out "Result:\n%A" res
                    out "Message: %s" msg
                out ""

            out "==== End of trace ===="
            out ""

    let dumpToFile path trace =
        use file = File.CreateText path
        dump file trace

    let dumpActiveTrace file =
        getActiveTraceSession () |> dump file

    let extractLoop uexpr =
        match uexpr with
        | UExpr (UExtraOp (Channel _), [UExpr (UExtraOp (Loop _), _, _) as uLoop], _) -> uLoop
        | _ -> failwith "not a loop channel expression"

