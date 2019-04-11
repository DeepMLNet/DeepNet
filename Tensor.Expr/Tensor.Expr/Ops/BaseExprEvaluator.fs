namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor


type EvaluatorEnv = {
    /// List of optimizers to apply.
    /// If None, all registered optimizers are applied.
    Optimizers: string list option
}


type BaseExprEvaluator (expr: BaseExpr, cfg: EvaluatorEnv) =

    /// optimized expression
    let optExpr =
        let opts =
            match cfg.Optimizers with
            | Some optNames -> BaseExprOpt.getOptimizers optNames
            | None -> BaseExprOpt.allOptimizers ()
        BaseExprOpt.optimize opts expr

    member this.Expr = expr

    member this.Optimized = optExpr

    member this.Eval (evalEnv: EvalEnv) : Map<Ch, ITensor> =
        failwith "TODO"

    interface System.IDisposable with
        member this.Dispose() =
            ()

    // how to specifiy the EvalEnv?
    // well, so far it only contains the tracer and varenv
    // so it might make sense to just specifiy the evalenv during evaluation
    // well, varenv should be flexible, but tracer could be fixed -> have to think about that
    // currently the determination of size symbols from input args is done in
    // UExpr and MultiChannelExpr -> this is a bit difficult.
    // Could we take a different approach where optimized expression
    // is cached in background? 
    // => Yes, but that to do about GC?? 
    //   => workspace would never be released.

    // so actual way of doing that is to become more explicit
    // this means, that the BaseExprEvaluator will have to deal with shape inference
    // also how will it interact with the EvalUpdateBundle?
    // Will we be able to create an evaluator for an EvalUpdateBundle?
    // Well, it could allocate the evaluator but then it would own it and
    //   that might be a problem.
    // There would need to be a way to instantiate an EvalUpdateBundle.
    // This should be similar to instanciating an expression for evaluation.
    
    // However, this shouldn't be bad since we need a common base type
    // for evaluation anyway. The EvalUpdateBundle should probably
    // just wrap it.

    // Question is how to move the shape inference?
    // Should not be a big problem? => No, probably not.

    // So, first step:
    // Add API in UExpr to create BaseExprEvaluator,
    // - Expr<'T> also needs API, but how to handle type of return value, i.e. Tensor<'T>?
    // - will probably need custom wrapper
    // - then wrapper could also do the shape inference?
    // - is there any reason that we need to handle the size substitution in here?
    // - probably not, we could just fixate on one, immutable expression
    // - good, is also cleaner.
    // - then another wrapper will handle the size substitutions.

    // things we do here:
    // 1. optimize expression
    // 2. precompile expression according to old method
    // 2a. will need TensorManikins?
    // 2b. Will need the allocation/stride mechanism interfaces.
    // 2c. What about CudaStreams? => probably ignore for now.
    // 2d. also need to thing about dynamic strides, perhaps not everything can
    //     be precomputed, should allow fallback to simple eval now.
