namespace SymTensor

open System

open Util
open VarSpec
open SizeSymbolTypes
open ArrayNDNS
open ArrayNDNS.ArrayND


[<AutoOpen>]
module VarEnvTypes = 
    /// variable environment
    type VarEnvT = Map<IVarSpec, IArrayNDT>

module VarEnv = 
    /// add variable value to environment
    let add (var: Expr.ExprT<'T>) (value: ArrayNDT<'T>) (varEnv: VarEnvT) =
        let vs = Expr.extractVar var
        Map.add (vs :> IVarSpec) (value :> IArrayNDT) varEnv

    /// get variable value from environment
    let getVarSpecT (vs: VarSpecT<'T>) (varEnv: VarEnvT) : ArrayNDT<'T> =
        varEnv.[vs] :?> ArrayNDT<'T>

    /// get variable value from environment
    let get (var: Expr.ExprT<'T>) (varEnv: VarEnvT) : ArrayNDT<'T> =
        getVarSpecT (Expr.extractVar var) varEnv

    /// empty variable environment
    let (empty: VarEnvT) =
        Map.empty

    /// joins two variable environments
    let join (a: VarEnvT) (b: VarEnvT) =
        Map.join a b
   
    /// enhances an existing size symbol environment from the variables occuring in the expression  
    let enhanceSizeSymbolEnvUsingExprs sizeSymbolEnv exprs (varEnv: VarEnvT) =
        let varSymShapes = 
            exprs
            |> Seq.map Expr.extractVars
            //|> Seq.map (fun vss -> Set.map (fun vs -> vs :> IVarSpec) vss)
            |> Set.unionMany
            |> Set.toSeq 
            |> Seq.map (fun vs -> VarSpec.name vs, VarSpec.shape vs)
            |> Map.ofSeq
        let varValShapes = 
            varEnv 
            |> Map.toSeq 
            |> Seq.map (fun (vs, ary) -> VarSpec.name vs, ArrayND.shape ary) 
            |> Map.ofSeq
        ShapeSpec.enhanceSizeSymbolEnv sizeSymbolEnv varValShapes varSymShapes

    /// builds a size symbol environment from the variables occuring in the expression
    let inferSizeSymbolEnvUsingExprs exprs (varEnv: VarEnvT) =
        enhanceSizeSymbolEnvUsingExprs SizeSymbolEnv.empty exprs varEnv


[<AutoOpen>]
module EvalEnvTypes =
    /// Information neccessary to evaluate an expression.
    /// Contains numeric values for variables and size symbols.
    type EvalEnvT = {
        VarEnv:             VarEnvT; 
        SizeSymbolEnv:      SizeSymbolEnvT
    }


module EvalEnv = 
    open VarEnv

    /// empty EvalEnvT
    let empty = 
        {VarEnv = Map.empty; SizeSymbolEnv = Map.empty}

    /// Create an EvalEnvT using the specified VarEnvT and infers the size symbol values
    /// from the given expressions.
    let create varEnv exprs = 
        {VarEnv = varEnv; SizeSymbolEnv = inferSizeSymbolEnvUsingExprs exprs varEnv;}

    /// Enhances an existing EvalEnvT using the specified VarEnvT and infers the size symbol values
    /// from the given expressions.
    let enhance varEnv exprs evalEnv =
        let joinedVarEnv = VarEnv.join evalEnv.VarEnv varEnv
        {VarEnv = joinedVarEnv;
         SizeSymbolEnv = enhanceSizeSymbolEnvUsingExprs evalEnv.SizeSymbolEnv exprs joinedVarEnv;}

    /// get variable value from environment
    let getVarSpecT vs (evalEnv: EvalEnvT)  =
        VarEnv.getVarSpecT vs evalEnv.VarEnv

    /// get variable value from environment
    let get var (evalEnv: EvalEnvT) =
        VarEnv.get var evalEnv.VarEnv



module Eval =
    open Expr

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
                | DoBroadcast ss -> ArrayND.broadcastToShape (shapeEval ss) av
                | SwapDim (ax1, ax2) -> ArrayND.swapDim ax1 ax2 av
                | StoreToVar vs -> ArrayND.copyTo av (VarEnv.getVarSpecT vs evalEnv.VarEnv); av
                | Annotated _-> av                
            | Binary(op, a, b) ->
                let av, bv = subEval a, subEval b  
                match op with
                | Add -> av + bv
                | Substract -> av - bv
                | Multiply -> av * bv
                | Divide -> av / bv
                | Power -> av ** bv
                | Dot -> av .* bv
                | TensorProduct -> av %* bv
            | Nary(op, es) ->
                let esv = List.map subEval es
                match op with 
                | Discard -> ArrayNDHost.zeros [0]
                | ExtensionOp eop -> failwith "not implemented"
            
        doEval expr

    /// Evaluates an expression on the host using the given variable values.
    let inline eval (varEnv: VarEnvT) (expr: ExprT<'T>) : ArrayNDT<'T> = 
        let evalEnv = EvalEnv.create varEnv (Seq.singleton expr)
        evalWithEvalEnv evalEnv expr

    let inline toFun expr =
        fun evalEnv -> evalWithEvalEnv evalEnv expr

    let inline addArg (var: ExprT<'T>) f =
        fun (evalEnv: EvalEnvT) value -> 
            f {evalEnv with VarEnv = evalEnv.VarEnv |> VarEnv.add var value}

    let inline usingEvalEnv (evalEnv: EvalEnvT) f =
        fun value -> f evalEnv value


