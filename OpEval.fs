module OpEval

open Op


type SubEvalException(op: Op, inner: System.Exception) =
    inherit System.Exception((sprintf "while evaluating op %A" op), inner)
    member x.Op = op

type AnnotatedEvalException(anoOp: Op, inner: System.Exception) = 
    inherit System.Exception(null, inner)
    let op_, ano =
        match anoOp with
        | Annotated(op, ano) -> op, ano
        | _ -> failwith "op must be Annotated"
    member x.Op = op_
    member x.Annotation = ano
    override x.Message = sprintf "inside %A (which is %A)" op_ ano

let debugEval = false

let rec eval (env: Environment) op =
    let subeval subop = 
        let subval = eval env subop
        if debugEval then printfn "Evaluated %A to %A." subop subval
        subval
    try 
        match op with
            | Add (a, b) -> NDArray.add (subeval a) (subeval b)
            | Substract (a, b) -> NDArray.substract (subeval a) (subeval b)
            | Multiply (a, b) -> NDArray.multiply (subeval a) (subeval b)
            | Divide (a, b) -> NDArray.divide (subeval a) (subeval b)
            | Power (a, b) -> NDArray.power (subeval a) (subeval b)
            | Negate a -> NDArray.negate (subeval a)
            | Log a -> NDArray.log (subeval a)
            | Exp a -> NDArray.exp (subeval a)
            | Dot (a, b) -> NDArray.dot (subeval a) (subeval b)
            | Sum a -> NDArray.sum (subeval a)
            | SumAxis (ax, a) -> NDArray.sumAxis ax (subeval a)
            | Var v -> env.[v]
            | ScalarConst s -> NDArray.scalar s
            | TensorConst (a, s) -> NDArray.scalarBroadcastedTo (subeval a) s
            | Annotated(a, _) -> subeval a
    with
        | :? SubEvalException as ex ->
            match op with
                | Annotated(_) -> raise (AnnotatedEvalException(op, ex))
                | _ -> reraise()
        | :? AnnotatedEvalException -> reraise()
        | ex -> raise (SubEvalException(op, ex))



