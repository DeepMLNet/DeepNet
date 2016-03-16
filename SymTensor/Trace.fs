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
            





