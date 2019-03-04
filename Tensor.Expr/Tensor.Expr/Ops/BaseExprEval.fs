namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor


/// Evaluates a BaseExpr.
module BaseExprEval =

    /// Evaluates a BaseExpr using the specified evaluation environment.
    let eval (env: EvalEnv) (rootExpr: BaseExpr) = 

        env.Tracer.Log (TraceEvent.Expr rootExpr)
        env.Tracer.Log (TraceEvent.EvalStart DateTime.Now)

        /// Expression info.
        let info = BaseExprGroup [rootExpr]
        /// Evaluated values for each BaseExpr channel.
        let exprChValues = Dictionary<BaseExprCh, ITensor> ()
        /// BaseExprs that have all their arguments evaluated and thus can be evaluated themselves.
        let evalQueue = Queue<BaseExpr> ()
        /// Values that are still missing but needed for evaluation of a BaseExpr.
        let missingValuesForExpr = Dictionary<BaseExpr, HashSet<BaseExprCh>> ()
        /// Dependants of an expression channel that are not yet evaluated.
        let unevaledDeps = Dictionary<BaseExprCh, HashSet<BaseExpr>> ()
                  
        // Build missing values for each expression and enqueue leafs.
        for expr in info.AllExprs do
            let dependingOn = HashSet<_> ()
            for KeyValue(_, arg) in expr.Args do
                dependingOn.Add arg |> ignore
            missingValuesForExpr.[expr] <- dependingOn

            if dependingOn.Count = 0 then
                evalQueue.Enqueue expr

            for ch in expr.Channels do
                unevaledDeps.[expr.[ch]] <- HashSet (info.Dependants expr.[ch])

        // Loop until all expressions are evaluated.
        while evalQueue.Count > 0 do
            // Get expression to evaluate and its argument values.
            let expr = evalQueue.Dequeue ()
            let argVals = expr.Args |> Map.map (fun _ argExpr -> exprChValues.[argExpr])

            // Evaluate.
            env.Tracer.Log (TraceEvent.ForExpr (expr, TraceEvent.EvalStart DateTime.Now))
            let chVals = expr.Op.Eval env argVals
            env.Tracer.Log (TraceEvent.ForExpr (expr, TraceEvent.EvalEnd DateTime.Now))
            env.Tracer.Log (TraceEvent.ForExpr (expr, TraceEvent.EvalValue chVals))

            // Store the result value of the expression and update its dependants.
            for KeyValue(ch, value) in chVals do
                // Store channel value.
                exprChValues.[expr.[ch]] <- value

                // Update dependants.
                for dep in info.Dependants expr.[ch] do
                    let mv = missingValuesForExpr.[dep]
                    mv.Remove expr.[ch] |> ignore

                    // Enqueue, if all arguments have evaluated values.
                    if mv.Count = 0 then
                        evalQueue.Enqueue dep

                // Update unevaled deps.
                for KeyValue(_, arg) in expr.Args do    
                    let ueDeps = unevaledDeps.[arg]
                    ueDeps.Remove expr |> ignore

                    // Remove value, if all dependants are evaluated.
                    if ueDeps.Count = 0 then
                        exprChValues.Remove arg |> ignore

        /// Evaluated channels values of root expression.
        let rootVals =
            rootExpr.Channels 
            |> Set.toSeq 
            |> Seq.map (fun ch -> ch, exprChValues.[rootExpr.[ch]]) 
            |> Map.ofSeq

        env.Tracer.Log (TraceEvent.EvalEnd DateTime.Now)
        env.Tracer.Log (TraceEvent.EvalValue rootVals)

        rootVals

