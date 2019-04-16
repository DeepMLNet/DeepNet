namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor


/// Evaluates a BaseExpr.
module BaseExprEval =

    /// Evaluates a BaseExpr using the specified evaluation environment.
    let eval (env: EvalEnv) (rootExpr: BaseExpr) = 

        env.Tracer.Log (TraceEvent.Expr rootExpr)
        env.Tracer.Log (TraceEvent.EvalStart DateTime.Now)

        /// Evaluation function for an expression.
        let evalFn (expr: BaseExpr) (argVals: Map<Arg, ITensor>) =
            env.Tracer.Log (TraceEvent.ForExpr (expr, TraceEvent.EvalStart DateTime.Now))
            let chVals = expr.Op.Eval env argVals
            env.Tracer.Log (TraceEvent.ForExpr (expr, TraceEvent.EvalEnd DateTime.Now))
            env.Tracer.Log (TraceEvent.ForExpr (expr, TraceEvent.EvalValue chVals))
            chVals

        // Perform evaluation of expression tree.
        let group = BaseExprGroup [rootExpr]
        let exprChValues = group |> BaseExprGroup.eval evalFn true

        /// Evaluated channel values of root expression.
        let rootVals =
            rootExpr.Channels 
            |> Set.toSeq 
            |> Seq.map (fun ch -> ch, exprChValues.[rootExpr.[ch]]) 
            |> Map.ofSeq

        env.Tracer.Log (TraceEvent.EvalEnd DateTime.Now)
        env.Tracer.Log (TraceEvent.EvalValue rootVals)

        rootVals

