namespace SymTensor

open System
open System.Collections.Generic
open System.Threading

open ArrayNDNS
open Basics


module Trace =
    

    [<Literal>]
    let Enabled = true

    type EvalEvent = 
        | ExprEvaled of UExprT * IArrayNDT

    type EvalSession = {
        Id:            int;
        Exprs:         UExprT list;
        Compiler:      string;
        Start:         DateTime;
        mutable End:   DateTime option;
        Trace:         ResizeArray<EvalEvent>;
    }

    let private sessions = ResizeArray<EvalSession>()

    let private activeSession = new ThreadLocal<EvalSession option>()


    let start uexprs compiler =
        if Enabled then
            match activeSession.Value with
            | Some ses -> failwithf "already tracing EvalSession %A" ses
            | None -> ()

            let ses = {
                Id = sessions.Count;
                Exprs = uexprs;
                Compiler = compiler;
                Start = DateTime.Now;
                End = None;
                Trace = ResizeArray()
            }
            sessions.Add ses
            activeSession.Value <- Some ses

    let inline private getActive () = 
        match activeSession.Value with
        | Some ses -> ses
        | None -> failwith "no EvalSession is currently being traced"

    let stop () =
        if Enabled then
            let ses = getActive ()
            ses.End <- Some DateTime.Now
            activeSession.Value <- None

    let exprEvaled uexpr res =
        if Enabled then
            let ses = getActive ()
            ses.Trace.Add (ExprEvaled (uexpr, ArrayND.copyUntyped res))
            





