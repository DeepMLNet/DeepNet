namespace SymTensor

open System
open System.Collections.Generic
open System.Threading

open ArrayNDNS
open Basics


module Trace =
    
    type EvalEvent = 
        | ExprEvaled of UExprT * IArrayNDT

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

    let inline private getActiveTraceSession () = 
        match activeTraceSession.Value with
        | Some ts -> ts
        | None -> failwith "no trace has been started"

    let isActive () =
        match activeTraceSession.Value with
        | Some ts -> true
        | None -> false

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

    let endSession () =
        let ts = 
            match activeTraceSession.Value with
            | Some ts -> ts
            | None -> failwith "no trace has been started"
        if activeExprEval.Value.IsSome then
            failwith "trace session cannot end while expression is being evaluated"

        activeTraceSession.Value <- None
        ts.End <- Some DateTime.Now
        ts

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
        | None -> ()

    let endExprEval () =
        if isActive () then
            let ses = getActiveExpr ()
            ses.End <- Some DateTime.Now
            activeExprEval.Value <- None

    let exprEvaled uexpr res =
        if isActive () then
            let ee = getActiveExpr ()
            ee.Trace.Add (ExprEvaled (uexpr, ArrayND.copyUntyped res))
            

    let maxSimilar (a: IArrayNDT) (b: IArrayNDT) =
        let epsilon = 1e-4f
        let a = a :?> ArrayNDT<single>
        let b = b :?> ArrayNDT<single>
        let diff = abs (a - b)
        let maxDiff = ArrayND.max diff |> ArrayND.value
        maxDiff <= epsilon

    let compareCustom isSimilar a b =
        let maxDiffs = 5
        let mutable diffs = 0

        printfn "Comparing trace sessions %s and %s:" a.Name b.Name

        if a.ExprEvals.Count <> b.ExprEvals.Count then
            printfn "Different number of expression evaluations: %d vs %d" 
                a.ExprEvals.Count b.ExprEvals.Count
        else
            for e, (ae, be) in Seq.indexed (Seq.zip a.ExprEvals b.ExprEvals) do
                printfn ""
                printfn "Evaluation %d using evaluator %s vs %s:" e ae.Compiler be.Compiler

                for ExprEvaled (uexpr, aRes) in ae.Trace do
                    match Seq.tryPick (fun (ExprEvaled (oexpr, bRes)) -> 
                                            if oexpr = uexpr then Some bRes else None) be.Trace with
                    | Some bRes ->
                        if not (isSimilar aRes bRes) then
                            if diffs < maxDiffs then
                                printfn "Difference in %A: %A vs %A" uexpr aRes bRes
                            elif diffs = maxDiffs then
                                printfn "(more differences not shown)"
                            diffs <- diffs + 1
                    | None -> ()
        printfn ""
        printfn "Total number of differences: %d" diffs

    let compare = compareCustom maxSimilar

        


