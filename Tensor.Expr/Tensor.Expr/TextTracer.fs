namespace Tensor.Expr

open Tensor.Expr.Ops
open DeepNet.Utils
open Tensor


/// Limits for TextTracer output.
type TextTracerLimits = {
    /// Maximum expression length.
    MaxExprLength: int
    /// Maximum tensor elements per dimension.
    MaxTensorElems: int64
} with
    /// Default TextTracer limits.
    static member standard = {
        MaxExprLength = 100
        MaxTensorElems = 10L
    }
        


type private TextTracerState = {
    ActiveSubExpr: BaseExpr option
    EvalStart: DateTime option
    ForExprState: Map<BaseExpr, TextTracerState>
} with 
    static member initial = {
        ActiveSubExpr = None
        EvalStart = None
        ForExprState = Map.empty
    }
        


/// Translates trace information into text messages and outputs them using the specified write funciton.
type TextTracer private (writeFn: string -> unit, limits: TextTracerLimits, level: int) =

    let exprStr (expr: BaseExpr) = expr.ToString limits.MaxExprLength
    let tensorStr (value: ITensor) = value.ToString limits.MaxTensorElems

    let indent (indenter: string) (str: string) =
        indenter + str.Replace ("\n", "\n" + indenter)

    let mutable globalState = TextTracerState.initial

    let rec eventMsg (state: TextTracerState) (event: TraceEvent) =
        let newState = {state with ActiveSubExpr=None}
        match event with
        | TraceEvent.Expr expr ->
            newState, Some (sprintf "Evaluating expression: %s" (exprStr expr))
        | TraceEvent.ParentExpr expr ->
            newState, Some (sprintf "Associated parent expression: %s" (exprStr expr))
        | TraceEvent.Msg msg ->
            newState, Some msg
        | TraceEvent.EvalValue vals ->
            let msg = 
                if Map.keys vals = Set [Ch.Default] then
                    tensorStr vals.[Ch.Default]
                else
                    seq {
                        for KeyValue(ch, value) in vals do
                            yield sprintf "%A=" ch
                            yield indent "    " (tensorStr value)
                    } |> String.concat "\n"
            newState, Some msg
        | TraceEvent.EvalStart startTime ->
            {newState with EvalStart=Some startTime}, None
        | TraceEvent.EvalEnd endTime ->
            match state.EvalStart with
            | Some startTime ->
                let duration = endTime - startTime
                state, Some (sprintf "Duration: %.3f s" duration.TotalSeconds)
            | None -> state, None
        | TraceEvent.Custom (key, data) ->
            newState, Some (sprintf "%s: %A" key data)
        | TraceEvent.ForExpr (expr, exprEvent) ->
            let subState = 
                match state.ForExprState |> Map.tryFind expr with
                | Some ss -> ss
                | None -> TextTracerState.initial
            let newSubState, exprMsg = eventMsg subState exprEvent
            let msgs = seq {
                if state.ActiveSubExpr <> Some expr then
                    yield sprintf "\nSubexpression: %s" (exprStr expr) 
                match exprMsg with
                | Some exprMsg -> yield indent "    " exprMsg   
                | None -> ()
            }
            let msg = 
                if Seq.isEmpty msgs then None
                else Some (msgs |> String.concat "\n")
            let newState =
                {state with ActiveSubExpr = Some expr
                            ForExprState = state.ForExprState |> Map.add expr newSubState}
            newState, msg
        | TraceEvent.Subtrace _ -> 
            newState, None

    new (writeFn, limits) =
        TextTracer (writeFn, limits, 0)

    new (writeFn) =
        TextTracer (writeFn, TextTracerLimits.standard)
              
    interface ITracer with

        member this.Log event =
            let newState, msg = eventMsg globalState event
            globalState <- newState
            match msg with
            | Some msg ->
                let subMsg =
                    (msg, [0 .. level-2])
                    ||> List.fold (fun msg _ -> indent "====" msg)
                writeFn subMsg
            | None -> ()

        member this.GetSubTracer () =
            TextTracer (writeFn, limits, level + 1) :> _
               

