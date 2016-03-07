namespace SymTensor

open System
open System.Reflection

open Basics
open VarSpec
open SizeSymbolTypes
open ArrayNDNS
open ArrayNDNS.ArrayND


module HostEval =
    open Expr

    /// if true, intermediate results are printed during evaluation.
    let mutable debug = false

    /// evaluate expression to numeric array 
    let rec eval (evalEnv: EvalEnvT) (expr: ExprT<'T>) =
        let varEval vs = VarEnv.getVarSpecT vs evalEnv.VarEnv
        let shapeEval symShape = ShapeSpec.eval symShape
        let sizeEval symSize = SizeSpec.eval symSize
        let rngEval = SimpleRangesSpec.eval (fun expr -> evalInt evalEnv expr |> ArrayND.value)

        let rec doEval (expr: ExprT<'T>) =
            let subEval subExpr = 
                let subVal = doEval subExpr
                if debug then printfn "Evaluated %A to %A." subExpr subVal
                subVal

            match expr with
            | Leaf(op) ->
                match op with
                | Identity ss -> ArrayNDHost.identity (sizeEval ss) 
                | Zeros ss -> ArrayNDHost.zeros (shapeEval ss)
                | ScalarConst f -> ArrayNDHost.scalar f
                | Var(vs) -> varEval vs 
            | Unary(op, a) ->
                let av = subEval a
                match op with
                | Negate -> -av
                | Abs -> abs av
                | SignT -> ArrayND.signt av
                | Log -> log av
                | Log10 -> log10 av
                | Exp -> exp av
                | Sin -> sin av
                | Cos -> cos av
                | Tan -> tan av
                | Asin -> asin av
                | Acos -> acos av
                | Atan -> atan av
                | Sinh -> sinh av
                | Cosh -> cosh av
                | Tanh -> tanh av
                | Sqrt -> sqrt av
                | Ceil -> ceil av
                | Floor -> floor av
                | Round -> round av
                | Truncate -> truncate av
                | Sum -> ArrayND.sum av
                | SumAxis ax -> ArrayND.sumAxis ax av
                | Reshape ss -> ArrayND.reshape (shapeEval ss) av
                | DoBroadcast ss -> ArrayND.broadcastToShape (shapeEval ss) av
                | SwapDim (ax1, ax2) -> ArrayND.swapDim ax1 ax2 av
                | Subtensor sr -> av.[rngEval sr]
                | StoreToVar vs -> 
                    // TODO: stage variable write to avoid overwrite of used variables
                    ArrayND.copyTo av (VarEnv.getVarSpecT vs evalEnv.VarEnv)
                    ArrayND.relayout ArrayNDLayout.emptyVector av
                | Annotated _-> av                
            | Binary(op, a, b) ->
                let av, bv = subEval a, subEval b  
                match op with
                | Add -> av + bv
                | Substract -> av - bv
                | Multiply -> av * bv
                | Divide -> av / bv
                | Modulo -> av % bv
                | Power -> av ** bv
                | Dot -> av .* bv
                | TensorProduct -> av %* bv
                | SetSubtensor sr -> 
                    let v = ArrayND.copy av
                    v.[sr] <- bv
                    v

            | Nary(op, es) ->
                let esv = List.map subEval es
                match op with 
                | Discard -> ArrayNDHost.zeros [0]
                | ExtensionOp eop -> failwith "not implemented"
            
        doEval expr

    and private evalInt (evalEnv: EvalEnvT) (expr: ExprT<int>) =
        eval evalEnv expr

    /// helper type for dynamic method invocation
    type private EvalT =
        static member Eval<'T> (evalEnv: EvalEnvT, expr: ExprT<'T>) : IArrayNDT =
            eval evalEnv expr :> IArrayNDT

    /// evaluates a unified expression
    let evalUExpr (evalEnv: EvalEnvT) (UExpr (_, tn, _, _) as uexpr) =
        let expr = UExpr.toExpr uexpr
        let gm = 
            typeof<EvalT>.GetMethod ("Eval", 
                                     BindingFlags.NonPublic ||| 
                                     BindingFlags.Public ||| 
                                     BindingFlags.Static)
        let m = gm.MakeGenericMethod ([| TypeName.getType tn |])
        m.Invoke(null, [| evalEnv; expr |]) :?> IArrayNDT
        
    /// evaluates all unified expressions
    let evalUExprs (evalEnv: EvalEnvT) (uexprs: UExprT list) =
        List.map (evalUExpr evalEnv) uexprs


[<AutoOpen>]
module HostEvalTypes =
    /// evaluates expression on host using interpreter
    let onHost (compileEnv: CompileEnvT) (uexprs: UExprT list) = 
        if compileEnv.ResultLoc <> LocHost then
            failwith "host evaluator needs host result location"
        for KeyValue(vs, loc) in compileEnv.VarLocs do
            if loc <> LocHost then
                failwithf "host evaluator cannot evaluate expression with variable %A located in %A" vs loc

        fun evalEnv -> HostEval.evalUExprs evalEnv uexprs


        
