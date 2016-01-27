module OpEval

open Shape
open Op
open NDArray

/// variable environment
type Environment = Map<string, NDArray.NDArray>

let debugEval = false

let rec eval (env: Environment) expr =
    let subeval subExpr = 
        let subVal = eval env subExpr
        if debugEval then printfn "Evaluated %A to %A." subExpr subVal
        subVal

    // TODO: create symbol environment
    let symEnv = SymbolEnv []

    let shapeEval symShape = ShapeSpec.eval symEnv symShape

    match expr with
        | Leaf(op) ->
            match op with
            | Identity ss ->  identity (shapeEval ss)
            | Zeros ss -> zeros (shapeEval ss)
            | ScalarConst f -> scalar f
            | TensorConst(f, ss) -> scalar f |> broadcastToShape (shapeEval ss) 
            | Var(v) -> env.[VarSpec.name v]
        | Unary(op, a) ->
            let av = subeval a
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
            let av, bv = subeval a, subeval b  
            match op with
            | Add -> av + bv
            | Substract -> av - bv
            | Multiply -> av * bv
            | Divide -> av / bv
            | Power -> av ** bv
            | Dot -> av .* bv
            | TensorProduct -> av %* bv
            

