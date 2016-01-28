module OpEval

open Shape
open Op
open NDArray

[<Literal>]
let DebugEval = false

/// variable environment
type VarEnvT = Map<string, NDArray.NDArray>

module VarEnv =
    /// add variable value to environment
    let add var value varEnv =
        let varName, _ = extractVar var
        Map.add varName value varEnv

    /// empty variable environment
    let (empty: VarEnvT) =
        Map.empty

let buildSizeSymbolEnv varEnv expr =
    let varSymShapes = extractVars expr |> Set.toSeq |> Map.ofSeq
    let varValShapes = varEnv |> Map.map (fun _ ary -> NDArray.shape ary) 
    SymbolEnv.fromShapeValues varSymShapes varValShapes

/// evaluate expression to numeric array
let eval (varEnv: VarEnvT) expr =
    let sizeSymEnv = buildSizeSymbolEnv varEnv expr
    let shapeEval symShape = ShapeSpec.eval sizeSymEnv symShape

    let rec doEval (varEnv: VarEnvT) expr =
        let subEval subExpr = 
            let subVal = doEval varEnv subExpr
            if DebugEval then printfn "Evaluated %A to %A." subExpr subVal
            subVal

        match expr with
            | Leaf(op) ->
                match op with
                | Identity ss ->  identity (shapeEval ss)
                | Zeros ss -> zeros (shapeEval ss)
                | ScalarConst f -> scalar f
                | TensorConst(f, ss) -> scalar f |> broadcastToShape (shapeEval ss) 
                | Var(v) -> varEnv.[VarSpec.name v]
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
            
    doEval varEnv expr


