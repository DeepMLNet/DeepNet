namespace ExprNS

open System

open Util
open VarSpec
open SizeSymbolTypes
open ArrayNDNS
open ArrayNDNS.ArrayND
open EvalEnv
open Expr


module EvalHost =

    [<Literal>]
    let DebugEval = false

    /// evaluate expression to numeric array 
    let inline evalWithEvalEnv (evalEnv: EvalEnvT) (expr: ExprT<'T>) =
        let varEval vs = VarEnv.getVarSpecT vs evalEnv.VarEnv
        let shapeEval symShape = ShapeSpec.eval evalEnv.SizeSymbolEnv symShape
        let sizeEval symSize = SizeSpec.eval evalEnv.SizeSymbolEnv symSize

        let rec doEval (expr: ExprT<'T>) =
            let subEval subExpr = 
                let subVal = doEval subExpr
                if DebugEval then printfn "Evaluated %A to %A." subExpr subVal
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
                | Log -> log av
                | Exp -> exp av
                | Sum -> ArrayND.sum av
                | SumAxis ax -> ArrayND.sumAxis ax av
                | Reshape ss -> ArrayND.reshape (shapeEval ss) av
                | Broadcast ss -> ArrayND.broadcastToShape (shapeEval ss) av
                | SwapDim (ax1, ax2) -> ArrayND.swapDim ax1 ax2 av
                | StoreToVar vs -> ArrayND.copyTo av (VarEnv.getVarSpecT vs evalEnv.VarEnv); av
                | Annotated _-> av                
            | Binary(op, a, b) ->
                let av, bv = subEval a, subEval b  
                match op with
                | Add -> av + bv
                //| Substract -> av - bv
                //| Multiply -> av * bv
                //| Divide -> av / bv
                //| Power -> av ** bv
                //| Dot -> av .* bv
                //| TensorProduct -> failwith "not impl"
                //| TensorProduct -> av %* bv
            | Nary(op, es) ->
                let esv = List.map subEval es
                match op with 
                | Discard -> ArrayNDHost.zeros [0]
                | ExtensionOp eop -> failwith "not implemented"
            
        doEval expr

    /// Evaluates an expression on the host using the given variable values.
    let inline eval (varEnv: VarEnvT) expr = 
        let evalEnv = EvalEnv.create varEnv (Seq.singleton expr)
        evalWithEvalEnv evalEnv expr

    let toFun expr =
        fun evalEnv -> eval evalEnv expr

    let addArg (var: ExprT<'T>) f =
        fun (evalEnv: EvalEnvT) value -> 
            f {evalEnv with VarEnv = evalEnv.VarEnv |> VarEnv.add var value}

    let usingEvalEnv (evalEnv: EvalEnvT) f =
        fun value -> f evalEnv value


