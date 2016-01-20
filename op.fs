module Op

open Util

/// variable environment
type Environment = Map<string, NDArray.ndarray>

type SizeSpec =
    | SizeSymbol of string
    | SizeConst of int
    | SizeOne
    | SizeProduct of SizeSpec list

type ShapeSpecT = SizeSpec list

module ShapeSpec =
    let withoutAxis ax sa =
        List.without ax sa

    let ndim sa =
        List.length sa

    let flat sa =
        match ndim sa with
        | 0 -> SizeOne
        | 1 -> sa.[0]
        | _ -> SizeProduct(sa)

    let concat sa sb =
        sa @ sb

    let transpose sa =
        List.rev sa

    let swap (ax1: int) (ax2: int) (sa: ShapeSpecT) =
        sa  |> List.set ax1 sa.[ax2]
            |> List.set ax2 sa.[ax1]

    let scalar = []

    let vector (ss: SizeSpec) = [ss]

    let matrix (sr: SizeSpec) (sc: SizeSpec) = [sr; sc]

    let padLeft sa =
        (SizeOne)::sa

    let padRight sa =
        sa @ [SizeOne]

    let broadcast (sa: ShapeSpecT) dim size =
        match sa.[dim] with
            | SizeOne -> List.set dim size sa
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
    // binary elementwise
    | Add of Op * Op
    | Substract of Op * Op
    | Multiply of Op * Op
    | Divide of Op * Op
    | Power of Op * Op
    // unary elementwise
    | Negate of Op
    | Log of Op
    | Exp of Op
    // matrix operations
    | Transpose of Op
    | Dot of Op * Op
    // tensor operations
    | TensorProduct of Op * Op
    // reductions
    | Sum of Op 
    | SumAxis of int * Op
    // tensor creation
    | Identity of ShapeSpecT
    | Zeros of ShapeSpecT
    | ScalarConst of float
    | TensorConst of float * ShapeSpecT
    // shape operations
    | Reshape of ShapeSpecT * Op
    | SwapDim of int * int * Op
    // varible access
    | Var of VarSpecT
    // misc
    | Annotated of Op * Annotation

and Annotation =
    | GradOf of Op
    | Text of string
    
let (|ElemwiseOp|_|) op =
    match op with
    | Add _ | Substract _ | Multiply _ | Divide _ | Power _ | Negate _ | Log _ | Exp _ 
        -> Some ()
    | _ -> None

let (|BinaryOp|_|) op =
    match op with
    | Add(a, b) 
    | Substract(a, b)
    | Multiply(a, b) 
    | Divide(a, b)  
    | Power(a, b) 
    | Dot(a, b) 
    | TensorProduct (a, b)
        -> Some (a, b)
    | _ -> None

let (|UnaryOp|_|) op =
    match op with
    | Negate a 
    | Log a 
    | Exp a 
    | Transpose a 
    | Sum a 
    | SumAxis (_, a) 
    | Reshape (_, a) 
    | SwapDim (_, _, a)
    | Annotated (a, _)
        -> Some (a)
    | _ -> None



let rec mapOperands unaryMap binaryMap op =
    let subMap = mapOperands unaryMap binaryMap
    let um a = unaryMap op (subMap a)
    let bm a b = binaryMap op (subMap a) (subMap b)
    match op with
    | Add(a, b) -> Add(bm a b)
    | Substract(a, b) -> Substract(bm a b)
    | Multiply(a, b) -> Multiply(bm a b)
    | Divide(a, b) -> Divide(bm a b)
    | Power(a, b) -> Power(bm a b)
    | Negate a -> Negate(um a)
    | Log a -> Log(um a)
    | Exp a -> Exp(um a)
    | Transpose a -> Transpose(um a)
    | Dot(a, b) -> Dot(bm a b)
    | TensorProduct(a, b) -> TensorProduct(bm a b)
    | Sum a -> Sum(um a)
    | SumAxis(ax, a) -> SumAxis(ax, um a)
    | Reshape(ss, a) -> Reshape(ss, um a)
    | SwapDim(ax1, ax2, a) -> SwapDim(ax1, ax2, um a)
    | Annotated(a, ano) -> Annotated(um a, ano)
    | _ -> op



let rec shapeOf op =
    match op with
    | Add(a, b) 
    | Substract(a, b)
    | Multiply(a, b) 
    | Divide(a, b)
    | Power(a, b)
        -> shapeOf a
    | Negate a
    | Log a
    | Exp a
        -> shapeOf a
    | Transpose a -> ShapeSpec.transpose (shapeOf a)
    | Dot(a, b) -> 
        let sa = shapeOf a
        let sb = shapeOf b
        match ShapeSpec.ndim sa, ShapeSpec.ndim sb with
            | 1, 1 -> ShapeSpec.scalar
            | 2, 1 -> ShapeSpec.vector sa.[0]
            | 2, 2 when sa.[1] = sb.[0] -> ShapeSpec.matrix sa.[0] sb.[1]
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb
    | TensorProduct(a, b) -> ShapeSpec.concat (shapeOf a) (shapeOf b)
    | Sum a -> ShapeSpec.scalar
    | SumAxis(ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax
    | Reshape(ss, a) -> ss
    | SwapDim(ax1, ax2, a) -> shapeOf a |> ShapeSpec.swap ax1 ax2
    | ScalarConst _ -> ShapeSpec.scalar
    | TensorConst(_, ss) -> ss
    | Identity ss -> ss
    | Var (_, ss) -> ss
    | Annotated (a, _) -> shapeOf a
       

let broadcastForElementwise opa opb =
    let sa = shapeOf opa
    let sb = shapeOf opb
    let bsa, bsb = ShapeSpec.broadcastToSame sa sb

    let bopa = if bsa = sa then opa else Reshape(bsa, opa)
    let bopb = if bsb = sb then opb else Reshape(bsb, opb)
    bopa, bopb
  

let checkAndAdaptShapes =
    let mapUnaryOp op a =
        match op with
        | Transpose _ ->
            let sa = shapeOf a
            match ShapeSpec.ndim sa with
            | 2 -> a
            | _ -> failwithf "cannot transpose array of shape %A" sa
        | SwapDim(ax1, ax2, _) ->
            let sa = shapeOf a
            if 0 <= ax1 && ax1 < ShapeSpec.ndim sa && 
                0 <= ax2 && ax2 < ShapeSpec.ndim sa then
                a
            else
                failwithf "cannot swap axis %d with axis %d of array with shape %A" ax1 ax2 sa
        | _ -> a

    let mapBinaryOp op a b =
        match op with
        | ElemwiseOp -> broadcastForElementwise a b
        | Dot(_) -> 
            let sa, sb = shapeOf a, shapeOf b
            match ShapeSpec.ndim sa, ShapeSpec.ndim sb with
            | 1, 1 when sa = sb -> a, b
            | 2, 1 when sa.[1] = sb.[0] -> a, b
            | 2, 2 when sa.[1] = sb.[0] -> a, b
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        | _ -> a, b

    mapOperands mapUnaryOp mapBinaryOp


let rec grad op wrt =    
    let g =
        // We assume that all operands have compatible size. 
        // For elementwise operations we assume that a and b are already broadcasted
        // to have the *same* size.
        let subgrad x = grad x wrt
        let constGrad ss = Zeros [ShapeSpec.flat ss; ShapeSpec.flat (shapeOf (Var wrt))]
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
            let sa, sb = shapeOf a, shapeOf b
            match ShapeSpec.ndim sa, ShapeSpec.ndim sb with
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
        | TensorProduct(a, b) ->
            let ga, gb = subgrad a, subgrad b
            let sa, sb = shapeOf a, shapeOf b
            let sga, sgb = shapeOf ga, shapeOf gb
            let g = Add(TensorProduct(Reshape(sa @ [sga.[1]], ga), b),
                        TensorProduct(a, Reshape(sb @ [sgb.[1]], gb)))
            let sg = shapeOf g            
            Reshape(sg.[0 .. (ShapeSpec.ndim sg) - 1] @ [sga.[1]], g)            
        | Transpose a -> subgrad (SwapDim(0, 1, a))
        | SwapDim (ax1, ax2, a) ->
            let g = subgrad a
            let sg, sa = shapeOf g, shapeOf a
            Reshape(sg, SwapDim(ax1, ax2, Reshape(sa @ [sg.[1]], g)))
        | Reshape (ss, a) ->
            let g = subgrad a
            let sg, sa = shapeOf g, shapeOf a
            Reshape([ShapeSpec.flat ss; sg.[1]], Reshape(ss @ [sg.[1]], Reshape(sa @ [sg.[1]], g)))
        | Sum a -> 
            let ga = subgrad a
            let sga = shapeOf ga
            Reshape([SizeOne; sga.[1]], SumAxis(0, ga))
        | SumAxis (ax, a) -> 
            let ga = subgrad a
            let sa = shapeOf a
            let sga = shapeOf ga
            let g = SumAxis(ax, Reshape(sa @ [sga.[1]], ga)) 
            let sg = shapeOf g
            Reshape(sg.[0 .. (ShapeSpec.ndim sg) - 1] @ [sga.[1]], g)            
        | Zeros ss -> constGrad ss
        | ScalarConst _ -> constGrad ShapeSpec.scalar
        | TensorConst (_, ss) -> constGrad ss
        | Identity ss -> constGrad ss
        | Var v -> 
            let sv = shapeOf (Var v)
            if v = wrt then                 
                Identity [ShapeSpec.flat sv; ShapeSpec.flat sv]
            else 
                constGrad sv
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

