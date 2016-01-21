module Op

open Util
open Shape

/// variable environment
type Environment = Map<string, NDArray.ndarray>

/// variable specification
type VarSpecT = string * ShapeSpecT

/// an op(eration)
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
    // matrix/tensor operations
    | Transpose of Op
    | Dot of Op * Op
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
    | Broadcast of int * ShapeSpecT * Op
    | SwapDim of int * int * Op
    // varible access
    | Var of VarSpecT
    // misc
    | Annotated of Op * Annotation

and Annotation =
    | GradOf of Op
    | Text of string
    
/// matches all ops that work elementwise on their argument(s)
let (|ElemwiseOp|_|) op =
    match op with
    | Add _ | Substract _ | Multiply _ | Divide _ | Power _ | Negate _ | Log _ | Exp _ 
        -> Some ()
    | _ -> None

/// matches all ops that take one input
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
    | Broadcast(_, _, a)
        -> Some (a)
    | _ -> None

/// matches all ops that take two inputs
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



/// Traverses the op tree and for each op calls a function on its arguments and replaces 
/// them by the function's return value(s).
let rec mapOperands unaryMapping binaryMapping op =
    let subMap = mapOperands unaryMapping binaryMapping
    let um a = unaryMapping op (subMap a)
    let bm a b = binaryMapping op (subMap a) (subMap b)
    match op with
    // binary elementwise
    | Add(a, b) -> Add(bm a b)
    | Substract(a, b) -> Substract(bm a b)
    | Multiply(a, b) -> Multiply(bm a b)
    | Divide(a, b) -> Divide(bm a b)
    | Power(a, b) -> Power(bm a b)
    // unary elementwise
    | Negate a -> Negate(um a)
    | Log a -> Log(um a)
    | Exp a -> Exp(um a)
    // matrix/tensor operations
    | Transpose a -> Transpose(um a)
    | Dot(a, b) -> Dot(bm a b)
    | TensorProduct(a, b) -> TensorProduct(bm a b)
    // reductions
    | Sum a -> Sum(um a)
    | SumAxis(ax, a) -> SumAxis(ax, um a)
    // shape operations
    | Reshape(ss, a) -> Reshape(ss, um a)
    | SwapDim(ax1, ax2, a) -> SwapDim(ax1, ax2, um a)
    | Broadcast(axp, ss, a) -> Broadcast(axp, ss, um a)
    // misc
    | Annotated(a, ano) -> Annotated(um a, ano)
    | _ -> op


/// Produces an error message about incompatible shapes.
let failshape op sa sb =
    failwithf "op %A was provided with arrays of incompatible shapes %A and %A" op sa sb


/// Returns the shape of the given op.
let rec shapeOf op =
    // We assume that all operands have compatible size. 
    // For elementwise operations we assume that a and b are already broadcasted
    // to have the *same* size.
    match op with
    // binary elementwise
    | Add(a, b) 
    | Substract(a, b)
    | Multiply(a, b) 
    | Divide(a, b)
    | Power(a, b)
        -> shapeOf a
    // unary elementwise
    | Negate a
    | Log a
    | Exp a
        -> shapeOf a
    // matrix/tensor operations
    | Transpose a -> ShapeSpec.transpose (shapeOf a)
    | Dot(a, b) -> 
        let sa, sb = shapeOf a, shapeOf b
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 1, 1 -> ShapeSpec.scalar
            | 2, 1 -> ShapeSpec.vector sa.[0]
            | 2, 2 when sa.[1] = sb.[0] -> ShapeSpec.matrix sa.[0] sb.[1]
            | _ -> failshape op sa sb
    | TensorProduct(a, b) -> 
        let sa, sb = shapeOf a, shapeOf b
        List.map2 SizeSpec.multiply sa sb
    // reductions
    | Sum a -> ShapeSpec.scalar
    | SumAxis(ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax
    // tensor creation
    | Identity ss -> ss
    | Zeros ss -> ss
    | ScalarConst _ -> ShapeSpec.scalar
    | TensorConst(_, ss) -> ss
    // shape operations
    | Reshape(ss, _) -> ss
    | Broadcast(_, ss, _) -> ss
    | SwapDim(ax1, ax2, a) -> shapeOf a |> ShapeSpec.swap ax1 ax2
    // variable access
    | Var (_, ss) -> ss
    // misc
    | Annotated (a, _) -> shapeOf a
       

/// Wraps the given op in a reshape-op if its shape does not match ss.
let reshapeIfNecessary ss op =
    if ss = shapeOf op then op else Reshape(ss, op)

/// Traverses the op tree and checks ops' arguments for compatible shapes and inserts reshape
/// ops if necessary.
let checkAndAdaptShapes =
    let mapUnaryOp op a =
        let sa = shapeOf a
        match op with
        | Transpose _ when ShapeSpec.nDim sa <> 2 ->
            failwithf "cannot transpose array of shape %A" sa
        | SumAxis(ax, _) when not (0 <= ax && ax < ShapeSpec.nDim sa) ->
            failwithf "cannot sum over non-existant axis %d of array with shape %A" ax sa
        | Reshape(ss, _) when not (SizeSpec.equal (ShapeSpec.nElem sa) (ShapeSpec.nElem ss)) ->
            failwithf "cannot reshape array of shape %A with %A elements into shape %A with %A elements"
                sa (ShapeSpec.nElem sa) ss (ShapeSpec.nElem ss)
        | Broadcast(axp, ss, _) -> 
            let psa = iterate ShapeSpec.padLeft axp ss
            if ShapeSpec.nDim ss <> ShapeSpec.nDim psa then
                failwithf "array of shape %A does not have same number of dimesions as %A after padding %d dimensions"
                    sa ss axp
            for dim in 0 .. (ShapeSpec.nDim ss) - 1 do
                match psa.[dim], ss.[dim] with
                | SizeBroadcast, _ -> ()
                | ssa, ssb when SizeSpec.equal ssa ssb -> ()
                | _ -> failwithf "dimension %d of array with shape %A is not broadcastable to shape %A" dim psa ss
            a
        | SwapDim(ax1, ax2, _) when 
                not (0 <= ax1 && ax1 < ShapeSpec.nDim sa && 0 <= ax2 && ax2 < ShapeSpec.nDim sa) ->
            failwithf "cannot swap axis %d with axis %d of array with shape %A" ax1 ax2 sa
        | _ -> a

    let mapBinaryOp op a b =
        let sa, sb = shapeOf a, shapeOf b
        match op with
        | ElemwiseOp -> 
            // TODO: change to broadcast
            let bsa, bsb = ShapeSpec.broadcastToSame sa sb
            reshapeIfNecessary bsa a, reshapeIfNecessary bsb b
        | Dot(_) -> 
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 1, 1 when sa = sb -> a, b
            | 2, 1 when sa.[1] = sb.[0] -> a, b
            | 2, 2 when sa.[1] = sb.[0] -> a, b
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        | TensorProduct(_) ->
            let psa, psb = ShapeSpec.padToSame sa sb
            reshapeIfNecessary psa a, reshapeIfNecessary psb b
        | _ -> a, b

    mapOperands mapUnaryOp mapBinaryOp


    