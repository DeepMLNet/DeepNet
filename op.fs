module Op

open Util
open Shape

/// variable specification: has a name and shape specificaiton
type VarSpecT = string * ShapeSpecT

module VarSpec =
    let name (vs: VarSpecT) = 
        let name, _ = vs
        name

    let shape (vs: VarSpecT) =
        let _, shape = vs
        shape


/// ops with no exprs as arguments
type LeafOp =
    // tensor creation
    | Identity of ShapeSpecT                /// tensor with 1 on diagonal of given shape
    | Zeros of ShapeSpecT                   /// zero tensor of given shape
    | ScalarConst of float                  /// scalar of given value
    | TensorConst of float * ShapeSpecT     /// tensor of given shape filled with given value
    // varible access
    | Var of VarSpecT                       /// variable read

/// ops with one expr as argument
and UnaryOp =
    // unary elementwise
    | Negate                        /// elementwise negation
    | Log                           /// elementwise logarithm
    | Exp                           /// elementwise exponential funcion
    // reductions
    | Sum                           /// summation of all elements
    | SumAxis of int                /// summation over given dimension
    // shape operations
    | Reshape of ShapeSpecT         /// reshape (assuming C-continguous order) tensor; element count does not change
    | Broadcast of ShapeSpecT       /// broadcast of SizeBroadcast dimensions
    | SwapDim of int * int          /// swaps two dimensions of a tensor
    // misc
    | Annotated of Annotation       /// annotation (no influence on value)

/// annotation of an op
and Annotation =
    | GradOf of Expr                /// expr is gradient of given expr
    | Text of string                /// text label

/// ops with two exprs as arguments
and BinaryOp =
    // binary elementwise
    | Add                           /// elementwise addition
    | Substract                     /// elementwise substraction
    | Multiply                      /// elementwise multiplication
    | Divide                        /// elementwise division
    | Power                         /// elementwise power
    // matrix/tensor operations
    | Dot                           /// vector*vector, matrix*vector or matrix*matrix dot product
    | TensorProduct                 /// tensor product 

  
/// an expression
and Expr =
    | Leaf of LeafOp
    | Unary of UnaryOp * Expr
    | Binary of BinaryOp * Expr * Expr

    
/// matches all exprs that work elementwise on their argument(s)
let (|ElemwiseOp|_|) (op: obj) =
    match op with
    | :? UnaryOp as uop ->
        match uop with
        | Negate
        | Log
        | Exp
            -> Some ()
        | _ -> None
    | :? BinaryOp as bop ->
        match bop with
        | Add
        | Substract
        | Multiply
        | Divide
        | Power
            -> Some ()
        | _ -> None
    | _ -> None


/// Traverses the op tree and for each op calls a function on its arguments and replaces 
/// them by the function's return value(s).
let rec mapOperands unaryMapping binaryMapping expr =
    let subMap = mapOperands unaryMapping binaryMapping
    match expr with
    | Unary(op, a) -> Unary(op, unaryMapping op (subMap a))
    | Binary(op, a, b) -> 
        let ma, mb = binaryMapping op (subMap a) (subMap b)
        Binary(op, ma, mb)
    | _ -> expr

/// returns true if subExpr is contained in expr
let rec contains subExpr expr =
    if expr = subExpr then
        true
    else
        match expr with
        | Unary(_, a) -> contains subExpr a
        | Binary(_, a, b) -> contains subExpr a || contains subExpr b
        | _ -> false

/// Produces an error message about incompatible shapes.
let failshape op sa sb =
    failwithf "op %A was provided with arrays of incompatible shapes %A and %A" op sa sb

/// Returns the shape of the given op.
let rec shapeOf expr =
    // We assume that all operands have compatible size. 
    // For elementwise operations we assume that a and b are already broadcasted
    // to have the *same* size.
    match expr with
    // binary elementwise
    | Unary (ElemwiseOp, a) 
    | Binary (ElemwiseOp, a, _) 
        -> shapeOf a
    // matrix/tensor operations
    | Binary (Dot, a, b) -> 
        let sa, sb = shapeOf a, shapeOf b
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 1, 1 -> ShapeSpec.scalar
            | 2, 1 -> ShapeSpec.vector sa.[0]
            | 2, 2 when sa.[1] .= sb.[0] -> ShapeSpec.matrix sa.[0] sb.[1]
            | _ -> failshape expr sa sb
    | Binary (TensorProduct, a, b) -> 
        let sa, sb = shapeOf a, shapeOf b
        List.map2 (*) sa sb
    // reductions
    | Unary(Sum, a) -> ShapeSpec.scalar
    | Unary(SumAxis(ax), a) -> shapeOf a |> ShapeSpec.withoutAxis ax
    // tensor creation
    | Leaf(Identity(ss)) -> ss
    | Leaf(Zeros(ss)) -> ss
    | Leaf(ScalarConst(_)) -> ShapeSpec.scalar
    | Leaf(TensorConst(_, ss)) -> ss
    // shape operations
    | Unary(Reshape(ss), _) -> ss
    | Unary(Broadcast(ss), _) -> ss
    | Unary(SwapDim(ax1, ax2), a) -> shapeOf a |> ShapeSpec.swap ax1 ax2
    // variable access
    | Leaf(Var(_, ss)) -> ss
    // misc
    | Unary(Annotated(_), a) -> shapeOf a
    | _ -> failwithf "unknown expr: %A" expr

/// Wraps the given op in a Reshape op if its shape does not match ss.
let reshapeIfNecessary ss expr =
    if ss = shapeOf expr then expr else Unary(Reshape(ss), expr)

/// Wraps the given op in a Broadcast op if its shape does not match ss.
let broadcastIfNecessary ss expr =
    if ss = shapeOf expr then expr else Unary(Broadcast(ss), expr)

/// Traverses the op tree and checks ops' arguments for compatible shapes and inserts reshape
/// ops if necessary.
let checkAndAdaptShapes =
    let mapUnaryOp op a =
        let sa = shapeOf a
        match op with
        | SumAxis(ax) when not (0 <= ax && ax < ShapeSpec.nDim sa) ->
            failwithf "cannot sum over non-existant axis %d of array with shape %A" ax sa
        | Reshape(ss) when not ((ShapeSpec.nElem sa) .= (ShapeSpec.nElem ss)) ->
            failwithf "cannot reshape array of shape %A with %A elements into shape %A with %A elements"
                sa (ShapeSpec.nElem sa) ss (ShapeSpec.nElem ss)
        | Broadcast(ss) -> 
            if ShapeSpec.nDim ss <> ShapeSpec.nDim sa then
                failwithf "array of shape %A does not have same number of dimesions as broadcast shape %A"
                    sa ss
            for dim in 0 .. (ShapeSpec.nDim ss) - 1 do
                match sa.[dim], ss.[dim] with
                | Shape.Broadcast, _ -> ()
                | ssa, ssb when ssa .= ssb -> ()
                | _ -> failwithf "dimension %d of array with shape %A is not broadcastable to shape %A" dim sa ss
            a
        | SwapDim(ax1, ax2) when 
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
        | Dot -> 
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 1, 1 when sa.[0] .= sb.[0] -> ()
            | 2, 1 when sa.[1] .= sb.[0] -> ()
            | 2, 2 when sa.[1] .= sb.[0] -> ()
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
            let dsa, dsb = ShapeSpec.disableAllBroadcasts sa, ShapeSpec.disableAllBroadcasts sb
            reshapeIfNecessary dsa a, reshapeIfNecessary dsb b
        | TensorProduct ->
            let psa, psb = ShapeSpec.padToSame sa sb
            reshapeIfNecessary psa a, reshapeIfNecessary psb b
        | _ -> a, b

    mapOperands mapUnaryOp mapBinaryOp

let check = checkAndAdaptShapes

/// scalar of given value
let scalar f = Leaf(ScalarConst(f)) |> check

/// swaps two dimensions of a tensor
let swapDim ax1 ax2 a = Unary(SwapDim(ax1, ax2), a) |> check

/// transpose matrix
let transpose a =
    if shapeOf a |> ShapeSpec.nDim <> 2 then
        failwithf "cannot transpose array of shape %A" (shapeOf a)
    swapDim 0 1 a

/// transpose 
type T = T

// operators
type Expr with

    // elementwise binary
    static member (+) (a: Expr, b: Expr) = Binary(Add, a, b) |> check
    static member (-) (a: Expr, b: Expr) = Binary(Substract, a, b) |> check
    static member (*) (a: Expr, b: Expr) = Binary(Multiply, a, b) |> check
    static member (/) (a: Expr, b: Expr) = Binary(Divide, a, b) |> check
    static member Pow (a: Expr, b: Expr) = Binary(Power, a, b) |> check 

    // elementwise binary with float
    static member (+) (a: Expr, b: float) = a + (scalar b)
    static member (-) (a: Expr, b: float) = a - (scalar b)
    static member (*) (a: Expr, b: float) = a * (scalar b)
    static member (/) (a: Expr, b: float) = a / (scalar b)
    static member Pow (a: Expr, b: float) = a ** (scalar b)

    static member (+) (a: float, b: Expr) = (scalar a) + b
    static member (-) (a: float, b: Expr) = (scalar a) - b
    static member (*) (a: float, b: Expr) = (scalar a) * b
    static member (/) (a: float, b: Expr) = (scalar a) / b
    static member Pow (a: float, b: Expr) = (scalar a) ** b

    // transposition
    static member Pow (a: Expr, b: T) = transpose a 
  
    // tensor binary
    static member (.*) (a: Expr, b: Expr) = Binary(Dot, a, b) |> check
    static member (%*) (a: Expr, b: Expr) = Binary(TensorProduct, a, b) |> check

    // unary
    static member (~-) (a: Expr) = Unary(Negate, a) |> check 

/// elementwise logarithm
let log a = Unary(Log, a) |> check

/// elementwise exponential function
let exp a = Unary(Exp, a) |> check

/// summaiton of all elements
let sum a = Unary(Sum, a) |> check

/// summation over given dimension
let sumAxis ax a = Unary(SumAxis(ax), a) |> check

/// tensor of given shape with 1s on the diagonal
let identity ss = Leaf(Identity(ss)) |> check
let eye = identity
let idMatrix rows cols = identity (ShapeSpec.matrix rows cols)

/// zero tensor of given shape
let zeros ss = Leaf(Zeros(ss)) |> check
let zeroMatrix rows cols = zeros (ShapeSpec.matrix rows cols)

/// zero tensor with same shape as given tensor
let zerosLike a = Leaf(Zeros(shapeOf a)) |> check

/// reshape (assuming C-continguous order) tensor; element count does not change
let reshape ss a = Unary(Reshape(ss), a) |> check

/// broadcast of SizeBroadcast dimensions
let broadcast ss a = Unary(Broadcast(ss), a) |> check

/// variable of given name and shape
let var name (ss: ShapeSpecT) = Leaf(Var(name, ss)) 

/// annotated expression
let annotate ano a = Unary(Annotated(ano), a) |> check

/// adds one broadcastable dimension to the left
let padLeft a =
    let sa = shapeOf a
    reshape (ShapeSpec.padLeft sa) a

/// adds one broadcastable dimension to the right
let padRight a =
    let sa = shapeOf a
    reshape (ShapeSpec.padRight sa) a


/// extract all variables from an expression
let rec extractVars expr =
    match expr with
    | Leaf(Var(v)) -> Set.singleton v
    | Leaf _ -> Set.empty
    | Unary(_, a) -> extractVars a
    | Binary(_, a, b) -> Set.union (extractVars a) (extractVars b)

/// extract VarSpec from variable expression
let extractVar expr = 
    match expr with
    | Leaf(Var(v)) -> v
    | _ -> invalidArg "expr" "not a expr consisting solely of a variable"
