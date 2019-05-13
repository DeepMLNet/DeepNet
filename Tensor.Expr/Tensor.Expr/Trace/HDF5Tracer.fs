namespace Tensor.Expr

open MBrace.FsPickler.Json

open DeepNet.Utils
open Tensor.Expr.Ops
open Tensor

 

/// Traces an expression evaluation to a HDF5 file.
type HDF5Tracer (hdf: HDF5, ?prefix: string) =
    let prefix = defaultArg prefix "/"

    let exprId = Dictionary<BaseExpr, int> ()
    let mutable nextExprId = 0

    let getExprId (expr: BaseExpr) =
        exprId.IGetOrAdd expr (fun _ ->
            nextExprId <- nextExprId + 1
            nextExprId)

    let writeExprs () =
        let exprMap = 
            exprId
            |> Seq.map (fun item -> item.Value, item.Key)
            |> Map.ofSeq
        let serializer = FsPickler.CreateJsonSerializer()
        let exprMapJson = serializer.PickleToString exprMap
        hdf.SetAttribute (prefix, "ExprMap", exprMapJson)
        
    let pathMsgIdx = Dictionary<string, int> ()
    let getPathMsgIdx path =
        let idx = pathMsgIdx.GetOrDefault path 0
        pathMsgIdx.[path] <- idx + 1
        idx

    let mutable nextSubtraceId = 0

    let rec writeEvent (path: string) (event: TraceEvent) =
        hdf.CreateGroups path

        match event with
        | TraceEvent.Expr expr -> 
            hdf.SetAttribute (path, "Expr", getExprId expr)
        | TraceEvent.ParentExpr expr -> 
            hdf.SetAttribute (path, "ParentExpr", getExprId expr)
        | TraceEvent.Msg msg -> 
            let attr = sprintf "Msg%d" (getPathMsgIdx path)
            hdf.SetAttribute (path, attr, msg)
        | TraceEvent.EvalValue vals -> 
            for KeyValue(ch, value) in vals do
                let devId = value.Dev.Id
                let value =
                    if value.Dev = HostTensor.Dev then value
                    else ITensor.transfer HostTensor.Dev value
                let valPath = sprintf "%s/Ch/%s" path (ch.ToString())
                value |> HostTensor.write hdf valPath
                hdf.SetAttribute (valPath, "DevId", devId)
        | TraceEvent.EvalStart startTime -> 
            hdf.SetAttribute (path, "EvalStart", startTime)
        | TraceEvent.EvalEnd endTime -> 
            hdf.SetAttribute (path, "EvalEnd", endTime)
            if path = prefix then
                writeExprs()
        | TraceEvent.LoopIter iter -> 
            hdf.SetAttribute (path, "LoopIter", iter)
        | TraceEvent.Custom (key, data) -> 
            hdf.SetAttribute (path, sprintf "Custom:%s" key, data.Value)
        | TraceEvent.ForExpr (expr, exprEvent) ->
            let id = getExprId expr
            let exprPath = sprintf "%s/%d" path id
            writeEvent exprPath exprEvent
        | TraceEvent.Subtrace _ -> 
            failwith "Subtrace event not expected."

              
    interface ITracer with

        member this.Log event =
            writeEvent prefix event

        member this.GetSubTracer () =
            let subPrefix = sprintf "%s/Subtrace/%d" prefix nextSubtraceId
            let subtracer = HDF5Tracer (hdf, subPrefix)
            nextSubtraceId <- nextSubtraceId + 1
            subtracer :> _
               


/// Trace data for an expression.
type ExprTraceData = {
    /// Expression.
    Expr: BaseExpr
    /// Evaluation start time.
    EvalStart: DateTime
    /// Evaluation end time.
    EvalEnd: DateTime
    /// Messages.
    Msg: string list
    /// Custom trace data.
    Custom: Map<string, TraceCustomData>
    /// Channel value reader functions.
    Ch: Map<Ch, unit -> ITensor>
} with

    /// Evaluation duration. 
    member this.EvalDuration = this.EvalEnd - this.EvalStart

    /// Channel values.
    member this.ChVals = this.Ch |> Map.map (fun _ v -> v())


/// A difference between two traces.
type TraceDiff = {
    /// Expression from trace A that has a value that differs.
    AExpr: BaseExpr
    /// Expression from trace B that has a value that differs.
    BExpr: BaseExpr
    /// The channel that has a value that differs.
    Ch: Ch
    /// The value from trace A.
    AValue: ITensor
    /// The value from trace B.
    BValue: ITensor
}


/// Reads a trace captured by `HDF5Tracer` from a HDF5 file.              
type HDF5Trace (hdf: HDF5, ?prefix: string) =
    let prefix = defaultArg prefix "/"

    let exprMap : Map<int, BaseExpr> =
        let serializer = FsPickler.CreateJsonSerializer()
        let exprMapJson = hdf.GetAttribute (prefix, "ExprMap")
        serializer.UnPickleOfString exprMapJson

    let idMap: Map<BaseExpr, int> =
        exprMap
        |> Map.toSeq
        |> Seq.map (fun (id, expr) -> expr, id)
        |> Map.ofSeq

    let getId expr =
        match idMap |> Map.tryFind expr with
        | Some id -> id
        | None -> failwith "The specified expression is not contained in the trace."

    /// The expression that was traced.
    member this.Root =
        let id = hdf.GetAttribute (prefix, "Expr")
        exprMap.[id]
   
    /// Gets the trace data for the specified (sub-)expression.
    member this.TryGet (expr: BaseExpr) =
        match idMap |> Map.tryFind expr with
        | Some id ->
            let exprPath = sprintf "%s/%d" prefix id
            let atrs = hdf.Attributes exprPath

            // get messages
            let msgs: string list =
                atrs 
                |> Map.toList
                |> List.choose (fun (name, typ) ->
                    match name with
                    | String.Prefixed "Msg" (String.Int n) when typ = typeof<string> ->
                        Some (n, hdf.GetAttribute (exprPath, name))
                    | _ -> None)
                |> List.sortBy fst
                |> List.map snd

            // get custom attributes
            let custom =
                atrs
                |> Map.toSeq
                |> Seq.choose (fun (name, typ) ->
                    match name with
                    | String.Prefixed "Custom:" key ->
                        let inline get () = hdf.GetAttribute (exprPath, name)
                        if typ = typeof<bool> then Some (key, TraceCustomData.Bool (get()))
                        elif typ = typeof<int> then Some (key, TraceCustomData.Int (get()))
                        elif typ = typeof<int64> then Some (key, TraceCustomData.Int64 (get()))
                        elif typ = typeof<single> then Some (key, TraceCustomData.Single (get()))
                        elif typ = typeof<double> then Some (key, TraceCustomData.Double (get()))
                        elif typ = typeof<string> then Some (key, TraceCustomData.String (get()))
                        elif typ = typeof<DateTime> then Some (key, TraceCustomData.DateTime (get()))
                        else None
                    | _ -> None)
                |> Map.ofSeq

            // get channels
            let chs =
                hdf.Entries (sprintf "%s/Ch" exprPath)
                |> Seq.choose (function
                               | HDF5Entry.Dataset chStr ->
                                   let chPath = sprintf "%s/Ch/%s" exprPath chStr
                                   match Ch.tryParse chStr with
                                   | Some ch -> Some (ch, fun () -> HostTensor.readUntyped hdf chPath)
                                   | None -> None
                               | _ -> None)
                |> Map.ofSeq

            Some {
                Expr = expr
                EvalStart = hdf.GetAttribute (exprPath, "EvalStart")
                EvalEnd = hdf.GetAttribute (exprPath, "EvalEnd")
                Msg = msgs
                Custom = custom
                Ch = chs
            }
        | None -> None  
             
    /// Gets the trace data for the specified (sub-)expression.
    member this.Item
        with get (expr: BaseExpr) =
            match this.TryGet expr with
            | Some data -> data
            | None ->
                failwith "The specified expression is not contained in the trace."         

    /// The subtraces contained in this trace.
    member this.Subtraces =
        failwith "TODO"

    /// Compares two traces and returns all subexpressions that have different values.
    /// Subexpressions are matched by their position within the expression tree.
    static member diff (a: HDF5Trace, b: HDF5Trace, ?isEqual: ITensor -> ITensor -> bool) =
        let isEqual = defaultArg isEqual (fun a b -> a.AlmostEqual b)

        // enumerate expression trees
        let aIds = BaseExpr.enumerate a.Root
        let bIds = BaseExpr.enumerate b.Root
        let bExprs = 
            bIds 
            |> Map.toSeq 
            |> Seq.map (fun (expr, id) -> id, expr) 
            |> Map.ofSeq
        if Map.count aIds <> Map.count bIds then
            failwithf "Expression tree of this trace has %d expressions but \
                       expression tree of other trace has %d expressions."
                (Map.count aIds) (Map.count bIds)

        // compare each subexpression value
        seq {
            for KeyValue(aExpr, id) in aIds do
                let aData = a.[aExpr]
                let bExpr = bExprs.[id]
                let bData = b.[bExpr]

                // compare each channel
                for KeyValue(ch, aVal) in aData.Ch do
                    let aVal = aVal()
                    let bVal =
                        match bData.Ch |> Map.tryFind ch with
                        | Some bVal -> bVal()
                        | None -> 
                            failwithf "Channel %A is missing for expression %A \
                                       (this trace) respectively %A (other trace)."
                                ch aExpr bExpr
                    if not (isEqual aVal bVal) then
                        yield {
                            AExpr = aExpr
                            BExpr = bExpr
                            Ch = ch
                            AValue = aVal
                            BValue = bVal
                        }
        }                
