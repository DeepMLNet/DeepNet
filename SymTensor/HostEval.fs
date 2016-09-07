namespace SymTensor

open System
open System.Reflection

open Basics
open VarSpec
open SizeSymbolTypes
open UExprTypes
open ArrayNDNS
open ArrayNDNS.ArrayND


module HostEval =
    open Expr

    /// if true, intermediate results are printed during evaluation.
    let mutable debug = false

    let private doInterpolate (ip: InterpolatorT<'T>) (es: ArrayNDHostT<'T> list) : ArrayNDHostT<'T> =
        let tbl = Expr.getInterpolatorTable ip

        /// returns interpolation in dimensions to the right of leftIdxs
        let rec interpolateInDim (leftIdxs: int list) (x: float list) =
            let d = leftIdxs.Length
            if d = ip.NDims then
                conv<float> tbl.[leftIdxs]
            else 
                let pos = (conv<float> x.[d] - conv<float> ip.MinArg.[d]) / ip.Resolution.[d]
                let posLeft = floor pos 
                let fac = pos - posLeft
                let idx = int posLeft 
                match idx, ip.Outside.[d], ip.Mode with
                | _, Nearest, _ when idx < 0                 -> interpolateInDim (leftIdxs @ [0]) x
                | _, Zero,    _ when idx < 0                 -> 0.0
                | _, Nearest, _ when idx > tbl.Shape.[d] - 2 -> interpolateInDim (leftIdxs @ [tbl.Shape.[d] - 1]) x
                | _, Zero,    _ when idx > tbl.Shape.[d] - 2 -> 0.0
                | _, _, InterpolateLinearaly -> 
                    let left = interpolateInDim (leftIdxs @ [idx]) x
                    let right = interpolateInDim (leftIdxs @ [idx+1]) x
                    (1.0 - fac) * left + fac * right
                | _, _, InterpolateToLeft -> 
                    interpolateInDim (leftIdxs @ [idx]) x

        let res = ArrayND.zerosLike es.Head
        for idx in ArrayND.allIdx res do
            let x = es |> List.map (fun src -> conv<float> src.[idx])
            res.[idx] <- interpolateInDim [] x |> conv<'T>
        res

    /// evaluate expression to numeric array 
    let rec eval (evalEnv: EvalEnvT) (expr: ExprT<'T>) =
        let varEval vs = VarEnv.getVarSpecT vs evalEnv.VarEnv :?> ArrayNDHostT<_>
        let shapeEval symShape = ShapeSpec.eval symShape
        let sizeEval symSize = SizeSpec.eval symSize
        let rngEval = SimpleRangesSpec.eval (fun expr -> evalInt evalEnv expr |> ArrayND.value)

        let rec doEval (expr: ExprT<'T>) =
            let res = 
                match expr with
                | Leaf(op) ->
                    match op with
                    | Identity ss -> ArrayNDHost.identity (sizeEval ss) 
                    | Zeros ss -> ArrayNDHost.zeros (shapeEval ss)
                    | SizeValue sv -> sizeEval sv |> conv<'T> |> ArrayNDHost.scalar
                    | ScalarConst f -> ArrayNDHost.scalar f
                    | Var(vs) -> varEval vs 
                | Unary(op, a) ->
                    let av = doEval a
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
                    | Diag(ax1, ax2) -> ArrayND.diagAxis ax1 ax2 av
                    | DiagMat(ax1, ax2) -> ArrayND.diagMatAxis ax1 ax2 av
                    | Invert -> ArrayND.invert av
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
                    | Print msg ->
                        printfn "%s=\n%A\n" msg av
                        av
                    | Annotated _-> av                
                | Binary(op, a, b) ->
                    let av, bv = doEval a, doEval b  
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
                        v.[rngEval sr] <- bv
                        v

                | Nary(op, es) ->
                    let esv = List.map doEval es
                    match op with 
                    | Discard -> ArrayNDHost.zeros [0]
                    | Elements (resShape, elemExpr) -> 
                        let esv = esv |> List.map (fun v -> v :> ArrayNDT<'T>)
                        let nResShape = shapeEval resShape
                        ElemExpr.eval elemExpr esv nResShape    
                    | Interpolate ip -> doInterpolate ip esv
                    | ExtensionOp eop -> eop.EvalSimple esv

            if Trace.isActive () then
                Trace.exprEvaled (expr |> UExpr.toUExpr) res
            res
            
        doEval expr

    and private evalInt (evalEnv: EvalEnvT) (expr: ExprT<int>) =
        eval evalEnv expr

    /// helper type for dynamic method invocation
    type private EvalT =
        static member Eval<'T> (evalEnv: EvalEnvT, expr: ExprT<'T>) : IArrayNDT =
            eval evalEnv expr :> IArrayNDT

    /// Evaluates a unified expression.
    /// This is done by evaluating the generating expression.
    let evalUExpr (evalEnv: EvalEnvT) (UExpr (_, _, {TargetType=tn}) as uexpr) =
        let expr = UExpr.toExpr uexpr
        let gm = 
            typeof<EvalT>.GetMethod ("Eval", 
                                     BindingFlags.NonPublic ||| 
                                     BindingFlags.Public ||| 
                                     BindingFlags.Static)
        let m = gm.MakeGenericMethod ([| TypeName.getType tn |])
        m.Invoke(null, [| evalEnv; expr |]) :?> IArrayNDT
        
    /// Evaluates the specified unified expressions.
    /// This is done by evaluating the generating expressions.
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


        
