module Op

type op =
    | Add of op * op
    | Substract of op * op
    | Multiply of op * op
    | Divide of op * op
    | Power of op * op
    | Negate of op
    | Log of op
    | Exp of op
    | Dot of op * op
    | Sum of op 
    | SumAxis of int * op
    | Const of float
    | ConstLike of op * float
    | Var of string
    | Annotated of op * Annotation

and Annotation =
    | GradOf of op
    | Text of string
    
let rec grad op wrt =    
    let g = 
        match op with        
        | Add(a, b) -> Add(grad a wrt, grad b wrt)
        | Substract(a, b) -> Substract(grad a wrt, grad b wrt)
        | Multiply(a, b) -> Add(Multiply(a, grad b wrt), Multiply(b, grad a wrt))
        | Divide(a, b) -> grad (Multiply(a, Power(b, Const -1.0))) wrt
        | Power(a, b) -> Add(Multiply(Multiply(Power(a, Substract(b, Const 1.0)), b), grad a wrt),
                             Multiply(Multiply(Power(a, b), Log a), grad b wrt))
        | Negate a -> Negate (grad a wrt)
        | Log a -> Multiply(Power(a, Const -1.0), grad a wrt)
        | Exp a -> Multiply(Exp a, grad a wrt)
        | Dot(a, b) -> Add(Dot(grad a wrt, b), Dot(a, grad b wrt)) 
        | Sum a -> Sum(grad a wrt)
        | SumAxis (ax, a) -> SumAxis(ax, grad a wrt) // TODO: verify
        | Const _ -> Const 0.0
        | ConstLike (a, _) -> ConstLike (a, 0.0)
        | Var v -> if v = wrt then ConstLike (op, 1.0) else ConstLike (op, 0.0)
        | Annotated(a, ano) -> Annotated(grad a wrt, ano)
    Annotated(g, GradOf op)

    
//exception SubEvalException of System.Exception * op

type SubEvalException(op: op, inner: System.Exception) =
    inherit System.Exception((sprintf "while evaluating op %A" op), inner)
    member x.op = op

type AnnotatedEvalException(anoOp: op, inner: System.Exception) = 
    inherit System.Exception(null, inner)
    let op_, ano =
        match anoOp with
        | Annotated(op, ano) -> op, ano
        | _ -> failwith "op must be Annotated"
    member x.op = op_
    member x.annotation = ano
    override x.Message = sprintf "inside %A (which is %A)" op_ ano

let debugEval = false

let rec eval (env: Map<string, NDArray.ndarray>) op =
    //let subeval = eval env
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
            | Const s -> NDArray.scalar s
            | ConstLike (a, s) -> NDArray.scalarBroadcastedTo (subeval a) s
            | Annotated(a, _) -> subeval a
    with
        | :? SubEvalException as ex ->
            match op with
                | Annotated(_, _) -> raise (AnnotatedEvalException(op, ex))
                | _ -> reraise()
        | :? AnnotatedEvalException -> reraise()
        | ex -> raise (SubEvalException(op, ex))


let a = AnnotatedEvalException(Const 1.0, new System.Exception())

