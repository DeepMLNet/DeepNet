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
        exprId.GetOrAdd expr (fun _ ->
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
            hdf.SetAttribute (path, key, data.Value)
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
               

