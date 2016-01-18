module Op

open Util

/// variable environment
type Environment = Map<string, NDArray.ndarray>

type SizeSpec =
    | SizeSymbol of string
    | SizeConst of int
    | SizeOne

type ShapeSpecT = SizeSpec list

module ShapeSpec =
    let withoutAxis ax sa =
        List.without ax sa

    let ndim sa =
        List.length sa

    let concat sa sb =
        sa @ sb

    let transpose sa =
        List.rev sa

    let scalar = []

    let vector (ss: SizeSpec) = [ss]

    let matrix (sr: SizeSpec) (sc: SizeSpec) = [sr; sc]

    let padLeft sa =
        (SizeOne)::sa

    let padRight sa =
        sa @ [SizeOne]

    let broadcast (sa: ShapeSpecT) dim size =
        match sa.[dim] with
            | SizeOne -> List.set sa dim size
            | _ -> failwithf "dimension %d of shape %A is not broadcastable (must be SizeOne)" dim sa

    let broadcastToSame saIn sbIn =
        let mutable sa = saIn
        let mutable sb = sbIn 
        while ndim sa < ndim sb do
            sa <- padLeft sa
        while ndim sb < ndim sa do
            sb <- padLeft sb
        for d = 0 to (ndim sa) - 1 do
            match sa.[d], sb.[d] with
                | al, bl when al = bl -> ()
                | al, bl when al = SizeOne -> sa <- broadcast sa d bl
                | al, bl when bl = SizeOne -> sb <- broadcast sb d al
                | _ -> failwithf "cannot broadcast shapes %A and %A to same size in dimension %d" sa sb d
        sa, sb

type VarSpecT = string * ShapeSpecT


type Op =
    | Add of Op * Op
    | Substract of Op * Op
    | Multiply of Op * Op
    | Divide of Op * Op
    | Power of Op * Op
    | Negate of Op
    | Transpose of Op
    | Log of Op
    | Exp of Op
    | Dot of Op * Op
    | TensorProduct of Op * Op
    | Sum of Op 
    | SumAxis of int * Op
    | ScalarConst of float
    | TensorConst of float * ShapeSpecT
    | Identity of ShapeSpecT
    | Var of VarSpecT
    | Annotated of Op * Annotation

and Annotation =
    | GradOf of Op
    | Text of string
    
let rec shapeOf op =
    match op with
    | Add(a, b) 
    | Substract(a, b)
    | Multiply(a, b) 
    | Divide(a, b)
    | Power(a, b)
        -> ShapeSpec.broadcastToSame (shapeOf a) (shapeOf b) |> fst
    | Negate a
    | Log a
    | Exp a
        -> shapeOf a
    | Transpose a ->
        let sa = shapeOf a
        match ShapeSpec.ndim sa with
            | 0 -> sa
            | 1 -> sa
            | 2 -> ShapeSpec.transpose sa
            | d -> failwithf "cannot transpose array of rank %d" d
    | Dot(a, b) -> 
        let sa = shapeOf a
        let sb = shapeOf b
        match ShapeSpec.ndim sa, ShapeSpec.ndim sb with
            | 0, _ -> sb
            | _, 0 -> sa
            | 1, 1 -> ShapeSpec.scalar
            | 2, 1 -> ShapeSpec.vector sa.[0]
            | 2, 2 when sa.[1] = sb.[0] -> ShapeSpec.matrix sa.[0] sb.[1]
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb
    | TensorProduct(a, b) -> ShapeSpec.concat (shapeOf a) (shapeOf b)
    | Sum a -> ShapeSpec.scalar
    | SumAxis(ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax
    | ScalarConst _ -> ShapeSpec.scalar
    | TensorConst(_, ss) -> ss
    | Identity ss -> ss
    | Var (_, ss) -> ss
    | Annotated (a, _) -> shapeOf a
       

let rec grad op wrt =    
    let g = 
        let subgrad x = grad x wrt
        match op with        
        | Add(a, b) -> Add(subgrad a, subgrad b)
        | Substract(a, b) -> Substract(subgrad a, subgrad b)
        | Multiply(a, b) -> Add(Multiply(a, subgrad b), Multiply(b, subgrad a))
        | Divide(a, b) -> subgrad (Multiply(a, Power(b, ScalarConst -1.0))) 
        | Power(a, b) -> Add(Multiply(Multiply(Power(a, Substract(b, ScalarConst 1.0)), b), subgrad a),
                             Multiply(Multiply(Power(a, b), Log a), subgrad b))
        | Negate a -> Negate (subgrad a)
        | Log a -> Multiply(Power(a, ScalarConst -1.0), subgrad a)
        | Exp a -> Multiply(Exp a, subgrad a)
        | Dot(a, b) -> 
            let sa = shapeOf a
            let sb = shapeOf b
            match ShapeSpec.ndim sa, ShapeSpec.ndim sb with
                | 0, _ | _, 0 -> subgrad (Multiply(a, b))
                | 1, 1 -> subgrad (Sum(Multiply(a, b)))
                | 2, 1 -> 
                    Add(Dot(TensorProduct(b, Identity (ShapeSpec.matrix sa.[0] sa.[0])), // wrt a
                            subgrad a), 
                        Dot(a, subgrad b)) // wrt b
                | 2, 2 when sa.[1] = sb.[0] -> 
                    Add(Dot(TensorProduct(Transpose(b), Identity (ShapeSpec.matrix sa.[0] sa.[0])), // wrt a
                            subgrad a),
                        Dot(TensorProduct(Identity (ShapeSpec.matrix sb.[1] sb.[1]), a), // wrt b
                            subgrad b))
                | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        | Transpose a -> failwith "not implemented"          
        | Sum a -> Sum(subgrad a)
        | SumAxis (ax, a) -> SumAxis(ax, subgrad a) // TODO: verify
        | ScalarConst _ -> ScalarConst 0.0
        | TensorConst (a, _) -> TensorConst (a, 0.0)
        | Var v -> if v = wrt then TensorConst (op, 1.0) else TensorConst (op, 0.0)
        | Annotated(a, ano) -> Annotated(subgrad a, ano)
    Annotated(g, GradOf op)

    
//exception SubEvalException of System.Exception * op

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


let a = AnnotatedEvalException(ScalarConst 1.0, new System.Exception())

