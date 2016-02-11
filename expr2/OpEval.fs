module OpEval

open Util
open Shape
open Op
open NDArray

[<Literal>]
let DebugEval = false

/// variable environment
type VarEnvT = Map<VarSpecT, NDArray.NDArray>

module VarEnv =
    /// add variable value to environment
    let add var value varEnv =
        let vs = extractVar var
        Map.add vs value varEnv

    /// get variable value from environment
    let get var (varEnv: VarEnvT) =
        let vs = extractVar var
        varEnv.[vs]

    /// empty variable environment
    let (empty: VarEnvT) =
        Map.empty

/// evaluation environment
type EvalEnvT = {VarEnv: VarEnvT; SizeSymbolEnv: SymbolEnvT}
let DefaultEvalEnv = {VarEnv = Map.empty; SizeSymbolEnv = Map.empty}

/// builds a size symbol environment from the variables occuring in the expression
let buildSizeSymbolEnvFromVarEnv varEnv expr =
    let varSymShapes = extractVars expr |> Set.toSeq |> Map.ofSeq
    let varValShapes = 
        varEnv 
        |> Map.toSeq 
        |> Seq.map (fun (vs, ary) -> VarSpec.name vs, NDArray.shape ary) 
        |> Map.ofSeq
    SymbolEnv.fromShapeValues varSymShapes varValShapes

module EvalEnv =
    let fromVarEnvAndExpr varEnv expr = 
        {DefaultEvalEnv with VarEnv = varEnv; SizeSymbolEnv = buildSizeSymbolEnvFromVarEnv varEnv expr;}

    let fromVarEnv varEnv =
        {DefaultEvalEnv with VarEnv = varEnv;}

    let addSizeSymbolsFromExpr expr evalEnv =
        fromVarEnvAndExpr evalEnv.VarEnv expr

/// evaluate expression to numeric array 
let eval (evalEnv: EvalEnvT) expr =
    let varEnv = evalEnv.VarEnv
    let sizeSymbolsFromVars = buildSizeSymbolEnvFromVarEnv varEnv expr
    let sizeSymEnv = Map.join evalEnv.SizeSymbolEnv sizeSymbolsFromVars
    let shapeEval symShape = ShapeSpec.eval sizeSymEnv symShape

    let rec doEval (varEnv: VarEnvT) expr =
        let subEval subExpr = 
            let subVal = doEval varEnv subExpr
            if DebugEval then printfn "Evaluated %A to %A." subExpr subVal
            subVal

        match expr with
            | Leaf(op) ->
                match op with
                | DiagonalOne ss -> identity (shapeEval ss)
                | Zeros ss -> zeros (shapeEval ss)
                | ScalarConst f -> scalar f
                | TensorConst(f, ss) -> scalar f |> broadcastToShape (shapeEval ss) 
                | Var(v) -> varEnv.[v]
            | Unary(op, a) ->
                let av = subEval a
                match op with
                | Negate -> -av
                | Log -> log av
                | Exp -> exp av
                | Sum -> sum av
                | SumAxis ax -> sumAxis ax av
                | Reshape ss -> reshape (shapeEval ss) av
                | Broadcast ss -> broadcastToShape (shapeEval ss) av
                | SwapDim (ax1, ax2) -> swapDim ax1 ax2 av
                | StoreToVar v -> NDArray.copyTo av varEnv.[v]; av
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
                | Discard -> NDArray.zeros [0]
            
    doEval varEnv expr

let toFun expr =
    fun evalEnv -> eval evalEnv expr

let addArg (var: ExprT) f =
    fun (evalEnv: EvalEnvT) value -> 
        f {evalEnv with VarEnv = evalEnv.VarEnv |> VarEnv.add var value}

let usingEvalEnv (evalEnv: EvalEnvT) f =
    fun value -> f evalEnv value

