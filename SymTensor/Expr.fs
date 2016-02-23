namespace SymTensor

open Util
open ArrayNDNS
open ShapeSpec
open VarSpec


module Expr =

    open ArrayND

    /// arity of an op
    type ArityT =
        | FixedArity of int
        | DynamicArity

    /// annotation of an op
    type Annotation =
        /// text label
        | Text of string      

     
    /// ops with no exprs as arguments
    type LeafOpT<'T> =

        // ==== tensor creation ====
        /// tensor with 1 on diagonal of given shape
        | Identity of SizeSpecT
        /// zero tensor of given shape       
        | Zeros of ShapeSpecT                   
        /// scalar of given value
        | ScalarConst of 'T

        // ==== variable access ====
        /// variable read
        | Var of VarSpecT<'T>       
        

    /// ops with one expr as argument
    and UnaryOpT<'T> =

        // ==== unary elementwise ==== 
        /// elementwise negation
        | Negate                        
        /// elementwise logarithm
        | Log                           
        /// elementwise exponential funcion
        | Exp                           

        // ==== reductions ====
        /// summation of all elements
        | Sum                           
        /// summation over given dimension
        | SumAxis of int                

        // ==== shape operations ====
        /// reshape tensor; element count does not change
        | Reshape of ShapeSpecT         
        /// broadcast tensor; element count may change
        | DoBroadcast of ShapeSpecT       
        /// swaps two dimensions of a tensor
        | SwapDim of int * int          

        // ==== variable storage ====
        /// variable write
        | StoreToVar of VarSpecT<'T>

        // ==== misc ====
        /// annotation (no influence on value)
        | Annotated of Annotation       


    /// ops with two exprs as arguments
    and BinaryOpT<'T> =
        // ==== binary elementwise ====
        /// elementwise addition
        | Add                           
        /// elementwise substraction
        | Substract                     
        /// elementwise multiplication
        | Multiply                      
        /// elementwise division
        | Divide                        
        /// elementwise power
        | Power                         
    
        // ==== matrix/tensor operations ====
        /// matrix*matrix => matrix dot product
        | Dot                           
        /// tensor product 
        | TensorProduct                 


    /// ops with an arbitrary exprs as arguments
    and NaryOpT<'T> =
        /// evaluate all subexpressions but discard them
        | Discard        
        /// extension op
        | ExtensionOp of IExtensionOp<'T>
   
     
    /// an extension op
    and IExtensionOp<'T> =
        /// the arity 
        abstract Arity: ArityT with get                   


    /// an expression
    and ExprT<'T> =
        | Leaf of LeafOpT<'T>
        | Unary of UnaryOpT<'T> * ExprT<'T>
        | Binary of BinaryOpT<'T> * ExprT<'T> * ExprT<'T>
        | Nary of NaryOpT<'T> * (ExprT<'T> list)


    /// matches all unary ops that work elementwise
    let (|UnaryElemwiseOp|_|) uop =
        match uop with
        | Negate
        | Log
        | Exp
            -> Some ()
        | _ -> None

    /// matches all binary ops that work elementwise
    let (|BinaryElemwiseOp|_|) bop =
        match bop with
        | Add
        | Substract
        | Multiply
        | Divide
        | Power
            -> Some ()
        | _ -> None


    /// Traverses the op tree and for each op calls a function on its arguments and replaces 
    /// them by the function's return value(s).
    let rec mapOperands unaryMapping binaryMapping naryMapping expr =
        let subMap = mapOperands unaryMapping binaryMapping naryMapping
        match expr with
        | Unary(op, a) -> Unary(op, unaryMapping op (subMap a))
        | Binary(op, a, b) -> 
            let ma, mb = binaryMapping op (subMap a) (subMap b)
            Binary(op, ma, mb)
        | Nary(op, es) ->
            let mes = naryMapping op (es |> List.map subMap)
            Nary(op, mes)
        | _ -> expr

    /// returns true if subExpr is contained in expr
    let rec contains subExpr expr =
        if expr = subExpr then true
        else
            match expr with
            | Unary(_, a) -> contains subExpr a
            | Binary(_, a, b) -> contains subExpr a || contains subExpr b
            | Nary(_, es) -> List.exists (contains subExpr) es
            | _ -> false

    /// Produces an error message about incompatible shapes.
    let failshape op sa sb =
        failwithf "op %A was provided with arrays of incompatible shapes %A and %A" op sa sb

    /// Returns the shape of the given expression.
    let rec shapeOf expr =
        // We assume that all operands have compatible size. 
        // For elementwise operations we assume that a and b are already broadcasted
        // to have the *same* size.
        match expr with
        // binary elementwise
        | Unary (UnaryElemwiseOp, a) 
        | Binary (BinaryElemwiseOp, a, _) 
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
        | Unary(Sum, _) -> ShapeSpec.scalar
        | Unary(SumAxis(ax), a) -> shapeOf a |> ShapeSpec.withoutAxis ax
        // tensor creation
        | Leaf(Identity(ss)) -> ShapeSpec.matrix ss ss
        | Leaf(Zeros(ss)) -> ss
        | Leaf(ScalarConst(_)) -> ShapeSpec.scalar
        // shape operations
        | Unary(Reshape(ss), _) -> ss
        | Unary(DoBroadcast(ss), _) -> ss
        | Unary(SwapDim(ax1, ax2), a) -> shapeOf a |> ShapeSpec.swap ax1 ax2
        // variable access
        | Leaf(Var vs) -> VarSpec.shape vs
        | Unary(StoreToVar _, a) -> shapeOf a
        // misc
        | Nary(Discard, _) -> ShapeSpec.emptyVector 
        | Unary(Annotated(_), a) -> shapeOf a
        | _ -> failwithf "unknown expr: %A" expr

    /// Wraps the given op in a Reshape op if its shape does not match ss.
    let reshapeIfNecessary ss expr =
        if ss = shapeOf expr then expr else Unary(Reshape(ss), expr)

    /// Wraps the given op in a Broadcast op if its shape does not match ss.
    let broadcastIfNecessary ss expr =
        if ss = shapeOf expr then expr else Unary(DoBroadcast(ss), expr)

    /// Traverses the expression and checks ops' arguments for compatible shapes.
    let check (expr: ExprT<'T>) : ExprT<'T> =
        let mapUnaryOp op a =
            let sa = shapeOf a
            match op with
            | SumAxis(ax) when not (0 <= ax && ax < ShapeSpec.nDim sa) ->
                failwithf "cannot sum over non-existant axis %d of array with shape %A" ax sa
            | Reshape(ss) when not ((ShapeSpec.nElem sa) .= (ShapeSpec.nElem ss)) ->
                failwithf "cannot reshape array of shape %A with %A elements into shape %A with %A elements"
                    sa (ShapeSpec.nElem sa) ss (ShapeSpec.nElem ss)
            | DoBroadcast(ss) -> 
                if ShapeSpec.nDim ss <> ShapeSpec.nDim sa then
                    failwithf "array of shape %A does not have same number of dimesions as broadcast shape %A"
                        sa ss
                for dim in 0 .. (ShapeSpec.nDim ss) - 1 do
                    match sa.[dim], ss.[dim] with
                    | SizeSpecT.Broadcast, _ -> ()
                    | ssa, ssb when ssa .= ssb -> ()
                    | _ -> failwithf "dimension %d of array with shape %A is not broadcastable to shape %A" dim sa ss
                a
            | SwapDim(ax1, ax2) when 
                    not (0 <= ax1 && ax1 < ShapeSpec.nDim sa && 0 <= ax2 && ax2 < ShapeSpec.nDim sa) ->
                failwithf "cannot swap axis %d with axis %d of array with shape %A" ax1 ax2 sa
            | StoreToVar vs when not (ShapeSpec.equalWithoutBroadcastability (VarSpec.shape vs) sa) -> 
                failwithf "cannot store resulst of shape %A into variable %A" sa vs
            | _ -> a

        let mapBinaryOp op a b =
            let sa, sb = shapeOf a, shapeOf b
            match op with
            | BinaryElemwiseOp when not (ShapeSpec.equalWithoutBroadcastability sa sb) -> 
                failwithf "cannot perform elementwise operation %A on arrays of shapes %A and %A" op sa sb
            | Dot -> 
                match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                | 2, 2 when sa.[1] .= sb.[0] -> a, b
                | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
            | TensorProduct ->
                let psa, psb = ShapeSpec.padToSame sa sb
                reshapeIfNecessary psa a, reshapeIfNecessary psb b
            | _ -> a, b

        let mapNaryOp op es =
            match op with
            | _ -> es

        mapOperands mapUnaryOp mapBinaryOp mapNaryOp expr

    /// scalar of given value
    let inline scalar<'T> (f: 'T) = Leaf(ScalarConst(f)) 

    /// scalar 0 of appropriate type
    let inline zero<'T> () = scalar (ArrayNDT<'T>.Zero)

    /// scalar 1 of appropriate type
    let inline one<'T> () = scalar (ArrayNDT<'T>.One)

    /// swaps two dimensions of a tensor
    let swapDim ax1 ax2 a = Unary(SwapDim(ax1, ax2), a) |> check

    /// transpose matrix
    let transpose a =
        if shapeOf a |> ShapeSpec.nDim <> 2 then invalidArg "a" "need matrix to transpose"
        swapDim 0 1 a

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    let constructElementwise op a b =
        let sa, sb = shapeOf a, shapeOf b
        let psa, psb = ShapeSpec.padToSame sa sb
        let bsa, bsb = ShapeSpec.broadcastToSame psa psb
        let ba = a |> reshapeIfNecessary psa |> broadcastIfNecessary bsa
        let bb = b |> reshapeIfNecessary psb |> broadcastIfNecessary bsb    
        Binary(op, ba, bb) |> check

    // elementwise operators
    type ExprT<'T> with

        // elementwise unary
        static member (~-) (a: ExprT<'T>) = Unary(Negate, a) |> check 
        static member Exp (a: ExprT<'T>) = Unary(Exp, a) |> check
        static member Log (a: ExprT<'T>) = Unary(Log, a) |> check

        // elementwise binary
        static member (+) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Add a b
        static member (-) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Substract a b
        static member (*) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Multiply a b
        static member (/) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Divide a b
        static member Pow (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Power a b

        // elementwise binary with basetype
        static member (+) (a: ExprT<'T>, b: 'T) = a + (scalar b)
        static member (-) (a: ExprT<'T>, b: 'T) = a - (scalar b)
        static member (*) (a: ExprT<'T>, b: 'T) = a * (scalar b)
        static member (/) (a: ExprT<'T>, b: 'T) = a / (scalar b)
        static member Pow (a: ExprT<'T>, b: 'T) = a ** (scalar b)

        static member (+) (a: 'T, b: ExprT<'T>) = (scalar a) + b
        static member (-) (a: 'T, b: ExprT<'T>) = (scalar a) - b
        static member (*) (a: 'T, b: ExprT<'T>) = (scalar a) * b
        static member (/) (a: 'T, b: ExprT<'T>) = (scalar a) / b
        static member Pow (a: 'T, b: ExprT<'T>) = (scalar a) ** b

        // transposition
        member this.T = transpose this

    /// reshape (assuming C-continguous order) tensor; element count does not change
    let reshape ss a = Unary(Reshape(ss), a) |> check

    /// broadcast of SizeBroadcast dimensions
    let broadcast ss a = Unary(DoBroadcast(ss), a) |> check

    /// enables broadcasting in the given dimension, it must be of size one
    let enableBroadcast dim a = 
        a |> reshape (shapeOf a |> ShapeSpec.enableBroadcast dim)

    /// disables broadcasting in the given dimension
    let disableBroadcast dim a =
        a |> reshape (shapeOf a |> ShapeSpec.disableBroadcast dim)
  
    /// inserts a broadcast axis at the given dimension
    let insertBroadcastAxis dim a =
        a |> reshape (shapeOf a |> ShapeSpec.insertBroadcastAxis dim)

    /// summaiton of all elements
    let sum a = Unary(Sum, a) |> check

    /// summation over given dimension
    let sumAxis ax a = Unary(SumAxis(ax), a) |> check

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    let sumKeepingAxis ax a =
        a |> sumAxis ax |> insertBroadcastAxis ax

    /// identity matrix of given size
    let identity size = Leaf(Identity(size)) |> check

    /// zero tensor of given shape
    let zeros ss = Leaf(Zeros(ss)) |> check

    /// zero matrix of given size
    let zeroMatrix rows cols = zeros (ShapeSpec.matrix rows cols)

    /// zero tensor with same shape as given tensor
    let zerosLike a = Leaf(Zeros(shapeOf a)) |> check

    /// variable of given name and shape
    let var name (ss: ShapeSpecT) = Leaf(Var(VarSpec.ofNameAndShape name ss)) 

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

    /// dot product
    let dot (a: ExprT<'T>) (b: ExprT<'T>) =
        let sa, sb = shapeOf a, shapeOf b
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 1, 1 when sa.[0] .= sb.[0] -> sum (a * b)
            | 2, 1 when sa.[1] .= sb.[0] -> 
                let bm = b |> reshape (ShapeSpec.padRight sb)
                Binary(Dot, a, bm) |> reshape [sa.[0]]
            | 2, 2 when sa.[1] .= sb.[0] -> Binary(Dot, a, b)
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        |> check

    type ExprT with
        // tensor binary
        static member (.*) (a: ExprT<'T>, b: ExprT<'T>) = dot a b
        static member (%*) (a: ExprT<'T>, b: ExprT<'T>) = Binary(TensorProduct, a, b) |> check

    /// extract all variables from an expression
    let rec extractVars expr =
        match expr with
        | Leaf(Var(v)) -> Set.singleton v
        | Leaf _ -> Set.empty
        | Unary(_, a) -> extractVars a
        | Binary(_, a, b) -> Set.union (extractVars a) (extractVars b)
        | Nary(_, es) -> Set.unionMany (es |> List.map extractVars)

    /// extract VarSpec from variable expression
    let extractVar expr = 
        match expr with
        | Leaf(Var(v)) -> v
        | _ -> invalidArg "expr" "not a expr consisting solely of a variable"

    /// make variable expression from VarSpec
    let makeVar vs =
        Leaf(Var(vs))

    /// store to variable
    let storeToVar ve a =
        let vs = extractVar ve
        Unary(StoreToVar(vs), a) |> check

    /// computes specified expressions, but discards the result
    let discard es =
        Nary(Discard, es) |> check


[<AutoOpen>]
module ExprTypes2 =
    type ArityT = Expr.ArityT
    type Annotation = Expr.Annotation
    type LeafOpT<'T> = Expr.LeafOpT<'T>
    type UnaryOpT<'T> = Expr.UnaryOpT<'T>
    type BinaryOpT<'T> = Expr.BinaryOpT<'T>
    type NaryOpT<'T> = Expr.NaryOpT<'T>
    type IExtensionOp<'T> = Expr.IExtensionOp<'T>
    type ExprT<'T> = Expr.ExprT<'T>



