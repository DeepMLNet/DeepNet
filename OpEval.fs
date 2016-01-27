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

    let symEnv = SymbolEnv {}

    let shapeEval symShape = ShapeSpec.eval symEnv symShape

    match expr with
        | Leaf(op) ->
            match op with
            | Identity ss ->  identity (shapeEval ss)
            | Zeros ss -> zeros (shapeEval ss)
            | ScalarConst f -> scalar f
            | TensorConst(f, ss) -> scalar f |> broadcastToShape (shapeEval ss) 
            | Var(v) -> env.[v]
        | Unary(op, a) ->
            let av = subeval a
            match op with
            | Negate -> -av
            | Log -> log av
            | Exp -> exp av
            | Sum -> sum av
            | SumAxis ax -> sumAxis ax av
            | Reshape ss -> reshape
            | Annotated _-> av


        | Binary(op, a, b) ->
            let av, bv = subeval a, subeval b
   
            match op with
            | Add -> av + bv


        | Add (a, b) -> NDArray.add (subeval a) (subeval b)
        | Substract (a, b) -> NDArray.substract (subeval a) (subeval b)
        | Multiply (a, b) -> NDArray.multiply (subeval a) (subeval b)
        | Divide (a, b) -> NDArray.divide (subeval a) (subeval b)
        | Power (a, b) -> NDArray.power (subeval a) (subeval b)
        | Dot (a, b) -> NDArray.dot (subeval a) (subeval b)


