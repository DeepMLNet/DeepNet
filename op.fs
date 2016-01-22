module Op

open Util
open Shape

/// variable specification: has a name and shape specificaiton
type VarSpecT = string * ShapeSpecT

/// an op(eration)
type Op =
    // binary elementwise
    | Add of Op * Op                /// elementwise addition
    | Substract of Op * Op          /// elementwise substraction
    | Multiply of Op * Op           /// elementwise multiplication
    | Divide of Op * Op             /// elementwise division
    | Power of Op * Op              /// elementwise power
    // unary elementwise
    | Negate of Op                  /// elementwise negation
    | Log of Op                     /// elementwise logarithm
    | Exp of Op                     /// elementwise exponential funcion
    // matrix/tensor operations
    | Dot of Op * Op                /// vector*vector, matrix*vector or matrix*matrix dot product
    | TensorProduct of Op * Op      /// tensor product 
    // reductions
    | Sum of Op                     /// summation of all elements
    | SumAxis of int * Op           /// summation over given dimension
    // tensor creation
    | Identity of ShapeSpecT        /// tensor with 1 on diagonal of given shape
    | Zeros of ShapeSpecT           /// zero tensor of given shape
    | ScalarConst of float          /// scalar of given value
    | TensorConst of float * ShapeSpecT     /// tensor of given shape filled with given value
    // shape operations
    | Reshape of ShapeSpecT * Op    /// reshape (assuming C-continguous order) tensor; element count does not change
    | Broadcast of ShapeSpecT * Op  /// broadcast of SizeBroadcast dimensions
    | SwapDim of int * int * Op     /// swaps two dimensions of a tensor
    // varible access
    | Var of VarSpecT               /// variable read
    // misc
    | Annotated of Op * Annotation  /// annotation (no influence on value)

/// annotation of an op
and Annotation =
    | GradOf of Op                  /// op is gradient of given op
    | Text of string                /// text label
    
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
    | Sum a 
    | SumAxis (_, a) 
    | Reshape (_, a) 
    | SwapDim (_, _, a)
    | Broadcast(_, a)
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
    | Dot(a, b) -> Dot(bm a b)
    | TensorProduct(a, b) -> TensorProduct(bm a b)
    // reductions
    | Sum a -> Sum(um a)
    | SumAxis(ax, a) -> SumAxis(ax, um a)
    // shape operations
    | Reshape(ss, a) -> Reshape(ss, um a)
    | SwapDim(ax1, ax2, a) -> SwapDim(ax1, ax2, um a)
    | Broadcast(ss, a) -> Broadcast(ss, um a)
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
    | Broadcast(ss, _) -> ss
    | SwapDim(ax1, ax2, a) -> shapeOf a |> ShapeSpec.swap ax1 ax2
    // variable access
    | Var (_, ss) -> ss
    // misc
    | Annotated (a, _) -> shapeOf a
       

/// Wraps the given op in a Reshape op if its shape does not match ss.
let reshapeIfNecessary ss op =
    if ss = shapeOf op then op else Reshape(ss, op)

/// Wraps the given op in a Broadcast op if its shape does not match ss.
let broadcastIfNecessary ss op =
    if ss = shapeOf op then op else Broadcast(ss, op)

/// Traverses the op tree and checks ops' arguments for compatible shapes and inserts reshape
/// ops if necessary.
let checkAndAdaptShapes =
    let mapUnaryOp op a =
        let sa = shapeOf a
        match op with
        | SumAxis(ax, _) when not (0 <= ax && ax < ShapeSpec.nDim sa) ->
            failwithf "cannot sum over non-existant axis %d of array with shape %A" ax sa
        | Reshape(ss, _) when not (SizeSpec.equal (ShapeSpec.nElem sa) (ShapeSpec.nElem ss)) ->
            failwithf "cannot reshape array of shape %A with %A elements into shape %A with %A elements"
                sa (ShapeSpec.nElem sa) ss (ShapeSpec.nElem ss)
        | Broadcast(ss, _) -> 
            if ShapeSpec.nDim ss <> ShapeSpec.nDim sa then
                failwithf "array of shape %A does not have same number of dimesions as broadcast shape %A"
                    sa ss
            for dim in 0 .. (ShapeSpec.nDim ss) - 1 do
                match sa.[dim], ss.[dim] with
                | SizeBroadcast, _ -> ()
                | ssa, ssb when SizeSpec.equal ssa ssb -> ()
                | _ -> failwithf "dimension %d of array with shape %A is not broadcastable to shape %A" dim sa ss
            a
        | SwapDim(ax1, ax2, _) when 
                not (0 <= ax1 && ax1 < ShapeSpec.nDim sa && 0 <= ax2 && ax2 < ShapeSpec.nDim sa) ->
            failwithf "cannot swap axis %d with axis %d of array with shape %A" ax1 ax2 sa
        | _ -> a

    let mapBinaryOp op a b =
        let sa, sb = shapeOf a, shapeOf b
        match op with
        | ElemwiseOp -> 
            let psa, psb = ShapeSpec.padToSame sa sb
            let bsa, bsb = ShapeSpec.broadcastToSame psa psb
            let ba = a |> reshapeIfNecessary psa |> broadcastIfNecessary bsa
            let bb = b |> reshapeIfNecessary psb |> broadcastIfNecessary bsb
            ba, bb
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


   

let check = checkAndAdaptShapes

/// scalar of given value
let scalar f = ScalarConst(f) |> check

/// swaps two dimensions of a tensor
let swapDim ax1 ax2 a = SwapDim(ax1, ax2, a) |> check

/// transpose matrix
let transpose a =
    if shapeOf a |> ShapeSpec.nDim <> 2 then
        failwithf "cannot transpose array of shape %A" (shapeOf a)
    swapDim 0 1 a

/// transpose 
type T = T

// operators
type Op with

    // elementwise binary
    static member (+) (a: Op, b: Op) = Add(a, b) |> check
    static member (-) (a: Op, b: Op) = Substract(a, b) |> check
    static member (*) (a: Op, b: Op) = Multiply(a, b) |> check
    static member (/) (a: Op, b: Op) = Divide(a, b) |> check
    static member Pow (a: Op, b: Op) = Power(a, b) |> check 

    // elementwise binary with float
    static member (+) (a: Op, b: float) = a + (scalar b)
    static member (-) (a: Op, b: float) = a - (scalar b)
    static member (*) (a: Op, b: float) = a * (scalar b)
    static member (/) (a: Op, b: float) = a / (scalar b)
    static member Pow (a: Op, b: float) = a ** (scalar b)

    static member (+) (a: float, b: Op) = (scalar a) + b
    static member (-) (a: float, b: Op) = (scalar a) - b
    static member (*) (a: float, b: Op) = (scalar a) * b
    static member (/) (a: float, b: Op) = (scalar a) / b
    static member Pow (a: float, b: Op) = (scalar a) ** b

    // transposition
    static member Pow (a: Op, b: T) = transpose a 
  
    // tensor binary
    static member (.*) (a: Op, b: Op) = Dot(a, b) |> check
    static member (%*) (a: Op, b: Op) = TensorProduct(a, b) |> check

    // unary
    static member (~-) (a: Op) = Negate(a) |> check 

/// elementwise logarithm
let log a = Log(a) |> check

/// elementwise exponential function
let exp a = Exp(a) |> check

/// summaiton of all elements
let sum a = Sum(a) |> check

/// summation over given dimension
let sumAxis ax a = SumAxis(ax, a) |> check

/// tensor of given shape with 1s on the diagonal
let id ss = Identity(ss) |> check
let eye = id

/// zero tensor of given shape
let zeros ss = Zeros(ss) |> check

/// zero tensor with same shape as given tensor
let zerosLike a = Zeros(shapeOf a) |> check

/// reshape (assuming C-continguous order) tensor; element count does not change
let reshape ss a = Reshape(ss, a) |> check

/// broadcast of SizeBroadcast dimensions
let broadcast ss a = Broadcast(ss, a) |> check

/// variable of given name and shape
let var name (ss: ShapeSpecT) = Var(name, ss) 




