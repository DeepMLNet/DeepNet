namespace SymTensor

open Basics
open ArrayNDNS
open ShapeSpec
open VarSpec


module Expr =

    open ArrayND

    let inline boxOption (oo: #obj option) = 
        match oo with
        | Some o -> Some (o :> obj)
        | None -> None

    /// start plus the specified number of (symbolic elements)
    type PlusElems (elems: SizeSpecT) =
        new (intElems: int) = PlusElems (SizeSpec.fix intElems)
        member this.Elems = elems

    /// arity of an op
    type ArityT =
        | FixedArity of int
        | DynamicArity

    /// annotation of an op
    type Annotation =
        /// text label
        | Text of string      

     
    /// ops with no exprs as arguments
    [<StructuralComparison; StructuralEquality>]
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
    and [<StructuralComparison; StructuralEquality>] 
        UnaryOpT<'T> =

        // ==== unary elementwise ==== 
        | Negate                        
        | Abs
        | SignT
        | Log
        | Log10                           
        | Exp                           
        | Sin
        | Cos
        | Tan
        | Asin
        | Acos
        | Atan
        | Sinh
        | Cosh
        | Tanh
        | Sqrt
        | Ceil
        | Floor
        | Round
        | Truncate

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
        // subtensor 
        | Subtensor of ExprRngsSpecT

        // ==== variable storage ====
        /// variable write
        | StoreToVar of VarSpecT<'T>

        // ==== misc ====
        /// annotation (no influence on value)
        | Annotated of Annotation       

    and ExprRngSpecT = RangeSpecT<ExprT<int>>
    and ExprRngsSpecT = RangesSpecT<ExprT<int>>

    /// ops with two exprs as arguments
    and [<StructuralComparison; StructuralEquality>] 
        BinaryOpT<'T> =

        // ==== binary elementwise ====
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power                         
    
        // ==== matrix/tensor operations ====
        /// matrix*matrix => matrix dot product
        | Dot                           
        /// tensor product 
        | TensorProduct        
        
        // ==== shape operations ====
        /// replace subtensor
        | SetSubtensor of ExprRngsSpecT                 


    /// ops with an arbitrary exprs as arguments
    and [<StructuralComparison; StructuralEquality>] 
        NaryOpT<'T> =

        /// evaluate all subexpressions but discard them
        | Discard        
        /// extension op
        | ExtensionOp of IExtensionOp<'T>
   
     
    /// an extension op
    and IExtensionOp<'T> =
        inherit System.IComparable
        inherit System.IComparable<'T>
        inherit System.IEquatable<'T>

        /// the arity 
        abstract Arity: ArityT with get                   

    /// a type conversion op
    and [<StructuralComparison; StructuralEquality>] 
        ConvertOpT<'T> =

        | FromInt of ExprT<int>
        | FromSingle of ExprT<single>
        | FromDouble of ExprT<double>

    /// an expression
    and [<StructuralComparison; StructuralEquality>] 
        ExprT<'T> =
        | Leaf of LeafOpT<'T>
        | Unary of UnaryOpT<'T> * ExprT<'T>
        | Binary of BinaryOpT<'T> * ExprT<'T> * ExprT<'T>
        | Nary of NaryOpT<'T> * (ExprT<'T> list)
        //| Convert of ConvertOpT<'T>

    /// matches all unary ops that work elementwise
    let (|UnaryElemwiseOp|_|) uop =
        match uop with
        | Negate                        
        | Abs
        | SignT
        | Log
        | Log10                           
        | Exp                           
        | Sin
        | Cos
        | Tan
        | Asin
        | Acos
        | Atan
        | Sinh
        | Cosh
        | Tanh
        | Sqrt
        | Ceil
        | Floor
        | Round
        | Truncate
            -> Some ()
        | _ -> None

    /// matches all binary ops that work elementwise
    let (|BinaryElemwiseOp|_|) bop =
        match bop with
        | Add
        | Substract
        | Multiply
        | Divide
        | Modulo
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

        // tensor creation
        | Leaf(Identity(ss)) -> ShapeSpec.matrix ss ss
        | Leaf(Zeros(ss)) -> ss
        | Leaf(ScalarConst(_)) -> ShapeSpec.scalar

        // variable access
        | Leaf(Var vs) -> VarSpec.shape vs

        // unary elementwise
        | Unary (Negate, a)                       
        | Unary (Abs, a)
        | Unary (SignT, a)
        | Unary (Log, a)
        | Unary (Log10, a)                           
        | Unary (Exp, a)                           
        | Unary (Sin, a)
        | Unary (Cos, a)
        | Unary (Tan, a)
        | Unary (Asin, a)
        | Unary (Acos, a)
        | Unary (Atan, a)
        | Unary (Sinh, a)
        | Unary (Cosh, a)
        | Unary (Tanh, a)
        | Unary (Sqrt, a)
        | Unary (Ceil, a)
        | Unary (Floor, a)
        | Unary (Round, a)
        | Unary (Truncate, a)
            -> shapeOf a

        // reductions
        | Unary(Sum, _) -> ShapeSpec.scalar
        | Unary(SumAxis(ax), a) -> shapeOf a |> ShapeSpec.withoutAxis ax

        // shape operations
        | Unary(Reshape(ss), _) -> ss
        | Unary(DoBroadcast(ss), _) -> ss
        | Unary(SwapDim(ax1, ax2), a) -> shapeOf a |> ShapeSpec.swap ax1 ax2
        | Unary(Subtensor(sr), a) ->
            let rec subtensorShape rng baseShp =
                let ss = subtensorShape 
                match rng, baseShp with
                | RSSymElem _                  ::rngs, _::shps   -> ss rngs shps
                | RSDynElem _                  ::rngs, _::shps   -> ss rngs shps
                | RSSymStartSymEnd (s, f)      ::rngs, _::shps   -> (f + 1 - s) :: ss rngs shps
                | RSSymStartToEnd s            ::rngs, shp::shps -> (shp - s) :: ss rngs shps 
                | RSStartToSymEnd f            ::rngs, _::shps   -> (f + 1) :: ss rngs shps
                | RSDynStartSymSize (_, size)  ::rngs, _::shps   -> size :: ss rngs shps
                | RSNewAxis                    ::rngs, _::shps   -> SizeSpec.broadcastable :: ss rngs shps
                | RSAll                        ::rngs, _::shps   -> (List.head baseShp) :: ss rngs shps
                | RSAllFill                    ::_   , _ when List.length rng <= List.length baseShp 
                                                                 -> ss (RSAll :: rng) baseShp 
                | RSAllFill                    ::rngs, _         -> ss rngs baseShp 
                | []                                 , []        -> []
                | [], _  | _, []                                 -> failwith "incompatible subtensor range specification"                    
            subtensorShape sr (shapeOf a)
                    
        // misc
        | Unary(StoreToVar _, a) -> shapeOf a
        | Unary(Annotated(_), a) -> shapeOf a

        // binary elementwise
        | Binary (Add, a, _)                         
        | Binary (Substract, a, _)                     
        | Binary (Multiply, a, _)                      
        | Binary (Divide, a, _)                        
        | Binary (Modulo, a, _)
        | Binary (Power, a, _)                         
            -> shapeOf a
            
        // matrix/tensor operations
        | Binary (Dot, a, b) -> 
            let sa, sb = shapeOf a, shapeOf b
            match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                | 1, 1 -> ShapeSpec.scalar
                | 2, 1 -> ShapeSpec.vector sa.[0]
                | 2, 2 -> ShapeSpec.matrix sa.[0] sb.[1]
                | _ -> failwith "invalid expression"
        | Binary (TensorProduct, a, b) -> 
            let sa, sb = shapeOf a, shapeOf b
            List.map2 (*) sa sb

        // shape operations
        | Binary (SetSubtensor ss, a, b) ->
            shapeOf a

        // misc
        | Nary(Discard, _) -> ShapeSpec.emptyVector 
        | Nary(ExtensionOp _, _) -> failwith "not implemented"


    /// Wraps the given op in a Reshape op if its shape does not match ss.
    let reshapeIfNecessary ss expr =
        if ss = shapeOf expr then expr else Unary(Reshape(ss), expr)

    /// Wraps the given op in a Broadcast op if its shape does not match ss.
    let broadcastIfNecessary ss expr =
        if ss = shapeOf expr then expr else Unary(DoBroadcast(ss), expr)

    /// Infers symbol sizes from the given expression and checks ops' arguments for compatible shapes.
    let rec inferSymSizes (expr: ExprT<'T>) =
        match expr with
        | Leaf op ->
            SymSizeEnv.empty

        | Unary (op, a) ->
            let env = inferSymSizes a
            let sa = shapeOf a

            match op with
            | SumAxis(ax) when not (0 <= ax && ax < ShapeSpec.nDim sa) ->
                failwithf "cannot sum over non-existant axis %d of array with shape %A" ax sa
            | Reshape(ss) ->
                SymSizeEnv.needEqual (ShapeSpec.nElem sa) (ShapeSpec.nElem ss) env
            | DoBroadcast(ss) -> 
                if ShapeSpec.nDim ss <> ShapeSpec.nDim sa then
                    failwithf "array of shape %A does not have same number of dimesions as broadcast shape %A"
                        sa ss

                let mutable env = env
                for dim in 0 .. (ShapeSpec.nDim ss) - 1 do
                    match sa.[dim], ss.[dim] with
                    | SizeSpecT.Broadcast, _ -> ()
                    | ssa, ssb -> env <- SymSizeEnv.needEqual ssa ssb env
                env

            | SwapDim(ax1, ax2) when 
                    not (0 <= ax1 && ax1 < ShapeSpec.nDim sa && 0 <= ax2 && ax2 < ShapeSpec.nDim sa) ->
                failwithf "cannot swap axis %d with axis %d of array with shape %A" ax1 ax2 sa
            | StoreToVar vs ->
                SymSizeEnv.needEqualShape sa (VarSpec.shape vs) env
            | _ -> env

        | Binary (op, a, b) ->
            let env = SymSizeEnv.merge (inferSymSizes a) (inferSymSizes b)
            let sa, sb = shapeOf a, shapeOf b

            match op with
            | BinaryElemwiseOp ->
                SymSizeEnv.needEqualShape sa sb env
            | Dot -> 
                match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                | 2, 2 -> SymSizeEnv.needEqual sa.[1] sb.[0] env
                | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
            | TensorProduct when ShapeSpec.nDim sa <> ShapeSpec.nDim sb ->
                failwithf "cannot compute tensor product between arrays of shapes %A and %A" sa sb
            | _ -> env

        | Nary (op, es) ->
            let env = 
                es |> List.fold (fun env a -> SymSizeEnv.merge env (inferSymSizes a)) SymSizeEnv.empty 
            let ss =
                es |> List.map shapeOf

            match op with
            | _ -> env

    /// substitues the given symbol sizes into the expression
    let rec substSymSizes symSizes (expr: ExprT<'T>) =
        let sSub = substSymSizes symSizes
        let sSize = SymSizeEnv.subst symSizes
        let sShp = SymSizeEnv.substShape symSizes

        match expr with
        | Leaf (Identity ss) -> Leaf (Identity (sSize ss))
        | Leaf (Zeros ss) -> Leaf (Zeros (sShp ss))
        | Leaf (Var vs) -> Leaf (Var {vs with Shape = sShp vs.Shape})
        | Leaf _ -> expr

        | Unary (Reshape ss, a) -> Unary (Reshape (sShp ss), sSub a)
        | Unary (DoBroadcast ss, a) -> Unary (DoBroadcast (sShp ss), sSub a)
        | Unary (StoreToVar vs, a) -> Unary (StoreToVar {vs with Shape = sShp vs.Shape}, sSub a)
        | Unary (op, a) -> Unary (op, sSub a)

        | Binary (op, a, b) -> Binary (op, sSub a, sSub b)

        | Nary (op, es) -> Nary (op, List.map sSub es)

    /// Infers symbol sizes from the given expression and substitues them into the expression.
    let inferAndSubstSymSizes (expr: ExprT<'T>) =
        substSymSizes (inferSymSizes expr) expr

    /// true if all shapes in the expression can be evaluated to numeric shapes
    let rec canEvalAllSymSizes (expr: ExprT<'T>) =
        match expr with
        | Leaf (Identity ss) -> SizeSpec.canEval ss
        | Leaf (Zeros ss) -> ShapeSpec.canEval ss
        | Leaf (Var vs) -> ShapeSpec.canEval (VarSpec.shape vs)
        | Leaf _ -> true

        | Unary (Reshape ss, a) -> ShapeSpec.canEval ss && canEvalAllSymSizes a
        | Unary (DoBroadcast ss, a) -> ShapeSpec.canEval ss && canEvalAllSymSizes a
        | Unary (StoreToVar vs, a) -> ShapeSpec.canEval (VarSpec.shape vs) && canEvalAllSymSizes a
        | Unary (op, a) -> canEvalAllSymSizes a

        | Binary (op, a, b) -> canEvalAllSymSizes a && canEvalAllSymSizes b

        | Nary (op, es) -> List.forall canEvalAllSymSizes es

    /// Traverses the expression and checks ops' arguments for compatible shapes.
    let check (expr: ExprT<'T>) : ExprT<'T> =
        inferAndSubstSymSizes expr

    /// scalar of given value
    let inline scalar<'T> (f: 'T) = Leaf(ScalarConst(f)) 

    /// scalar of given value and type
    let inline scalart<'T> f =
        scalar (conv<'T> f)

    /// scalar 0 of appropriate type
    let inline zero<'T> () = scalar (ArrayNDT<'T>.Zero)

    /// scalar 1 of appropriate type
    let inline one<'T> () = scalar (ArrayNDT<'T>.One)

    /// scalar 2 of appropriate type
    let inline two<'T> () = scalart<'T> 2

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
        let bsa, bsb = ShapeSpec.broadcastToSame false psa psb
        let ba = a |> reshapeIfNecessary psa |> broadcastIfNecessary bsa
        let bb = b |> reshapeIfNecessary psb |> broadcastIfNecessary bsb    
        Binary(op, ba, bb) |> check


    // elementwise operators
    type ExprT<'T> with

        // elementwise unary
        static member (~+) (a: ExprT<'T>) = a |> check
        static member (~-) (a: ExprT<'T>) = Unary(Negate, a) |> check 
        static member Abs (a: ExprT<'T>) = Unary(Abs, a) |> check
        static member SignT (a: ExprT<'T>) = Unary(SignT, a) |> check
        static member Log (a: ExprT<'T>) = Unary(Log, a) |> check
        static member Log10 (a: ExprT<'T>) = Unary(Log10, a) |> check
        static member Exp (a: ExprT<'T>) = Unary(Exp, a) |> check
        static member Sin (a: ExprT<'T>) = Unary(Sin, a) |> check
        static member Cos (a: ExprT<'T>) = Unary(Cos, a) |> check
        static member Tan (a: ExprT<'T>) = Unary(Tan, a) |> check
        static member Asin (a: ExprT<'T>) = Unary(Asin, a) |> check
        static member Acos (a: ExprT<'T>) = Unary(Acos, a) |> check
        static member Atan (a: ExprT<'T>) = Unary(Atan, a) |> check
        static member Sinh (a: ExprT<'T>) = Unary(Sinh, a) |> check
        static member Cosh (a: ExprT<'T>) = Unary(Cosh, a) |> check
        static member Tanh (a: ExprT<'T>) = Unary(Tanh, a) |> check
        static member Sqrt (a: ExprT<'T>) = Unary(Sqrt, a) |> check
        static member Ceiling (a: ExprT<'T>) = Unary(Ceil, a) |> check
        static member Floor (a: ExprT<'T>) = Unary(Floor, a) |> check
        static member Round (a: ExprT<'T>) = Unary(Round, a) |> check
        static member Truncate (a: ExprT<'T>) = Unary(Truncate, a) |> check

        // elementwise binary
        static member (+) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Add a b
        static member (-) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Substract a b
        static member (*) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Multiply a b
        static member (/) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Divide a b
        static member (%) (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Modulo a b
        static member Pow (a: ExprT<'T>, b: ExprT<'T>) = constructElementwise Power a b

        // elementwise binary with basetype
        static member (+) (a: ExprT<'T>, b: 'T) = a + (scalar b)
        static member (-) (a: ExprT<'T>, b: 'T) = a - (scalar b)
        static member (*) (a: ExprT<'T>, b: 'T) = a * (scalar b)
        static member (/) (a: ExprT<'T>, b: 'T) = a / (scalar b)
        static member (%) (a: ExprT<'T>, b: 'T) = a % (scalar b)
        static member Pow (a: ExprT<'T>, b: 'T) = a ** (scalar b)

        static member (+) (a: 'T, b: ExprT<'T>) = (scalar a) + b
        static member (-) (a: 'T, b: ExprT<'T>) = (scalar a) - b
        static member (*) (a: 'T, b: ExprT<'T>) = (scalar a) * b
        static member (/) (a: 'T, b: ExprT<'T>) = (scalar a) / b
        static member (%) (a: 'T, b: ExprT<'T>) = (scalar a) % b
        static member Pow (a: 'T, b: ExprT<'T>) = (scalar a) ** b

        // transposition
        member this.T = transpose this

    /// sign keeping type
    let signt (a: ExprT<'T>) =
        ExprT<'T>.SignT a 

    let sqrtt (a: ExprT<'T>) =
        ExprT<'T>.Sqrt a

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
            | 1, 1 -> sum (a * b)
            | 2, 1 -> 
                let bm = b |> reshape (ShapeSpec.padRight sb)
                Binary(Dot, a, bm) |> reshape [sa.[0]]
            | 2, 2 -> Binary(Dot, a, b)
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        |> check

    /// tensor product
    let tensorProduct (a: ExprT<'T>) (b: ExprT<'T>) =
        let sa, sb = shapeOf a, shapeOf b
        let psa, psb = ShapeSpec.padToSame sa sb
        let a, b = reshapeIfNecessary psa a, reshapeIfNecessary psb b
        Binary(TensorProduct, a, b) |> check

    type ExprT with
        // tensor binary
        static member (.*) (a: ExprT<'T>, b: ExprT<'T>) = dot a b
        static member (%*) (a: ExprT<'T>, b: ExprT<'T>) = tensorProduct a b

    /// extract all variables from an expression
    let rec extractVars expr =
        match expr with
        | Leaf(Var vs) -> Set.singleton vs
        | Leaf _ -> Set.empty
        | Unary(StoreToVar vs, a) -> extractVars a |> Set.add vs
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

    /// expression a with the specified subtensor replaced with b
    let setSubtensor a b =
        match a with
        | Unary (Subtensor sr, t) ->
            Binary (SetSubtensor sr, t, b) |> check
        | _ ->
            failwith "the first argument of setSubtensor must be an item or slice of an expression, i.e. a.[...]"

    type ExprT with
        // item / slicing
        member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =
            /// converts argument list to range specification
            let rec toRSV (args: obj list) =
                match args with
                // direct range specification
                | [:? ExprRngsSpecT as rngs] -> rngs

                // slices
                | (:? (SizeSpecT option) as so)  :: (:? (SizeSpecT option) as fo)    :: rest ->
                    RSSymStartSymEnd (so.Value, fo.Value) :: toRSV rest
                | (:? (SizeSpecT option) as so)  :: null                             :: rest ->
                    RSSymStartToEnd so.Value :: toRSV rest
                | null                           :: (:? (SizeSpecT option) as fo)    :: rest ->
                    RSStartToSymEnd fo.Value :: toRSV rest
                | (:? (ExprT<int> option) as so) :: (:? (PlusElems option) as fo)    :: rest ->
                    RSDynStartSymSize (so.Value, fo.Value.Elems) :: toRSV rest
                | null                           :: null                             :: rest ->
                    RSAll :: toRSV rest

                // items
                | (:? SizeSpecT as s)     :: rest -> RSSymElem s :: toRSV rest
                | (:? SpecialAxisT as s)  :: rest -> match s with
                                                     | NewAxis -> RSNewAxis :: toRSV rest
                                                     | Fill    -> RSAllFill :: toRSV rest
                | (:? ExprT<int> as e)    :: rest -> RSDynElem e :: toRSV rest

                | []                              -> []
                | _                               -> failwithf "invalid item/slice specification: %A" allArgs

            /// converts ints to SizeSpecTs
            let intToSizeSpec (arg: obj) =
                match arg with
                | :? int as f -> SizeSpec.fix f :> obj
                | :? (int option) as fo -> 
                    match fo with
                    | Some f -> Some (SizeSpec.fix f) :> obj
                    | None -> None :> obj
                | _ -> arg

            allArgs
            |> Array.toList
            |> List.map intToSizeSpec
            |> toRSV
            |> fun ss -> Unary (Subtensor ss, this)  
            |> check

        member this.Item 
            with get ([<System.ParamArray>] allArgs: obj []) = 
                this.GetSlice (allArgs)
                      


    let testme () =
        let a : ExprT<float> = Leaf (Identity (SizeSpec.one))
        let b : ExprT<int> = Leaf (Identity (SizeSpec.one))
        let c : ExprT<int> = Leaf (Identity (SizeSpec.one))
        let aitm1 = a.[SizeSpec.one]
        let aitm1b = a.[SizeSpec.one, 12, 3 .. 6]

        let asl1 = a.[SizeSpec.fix 5 .. SizeSpec.fix 10]
        //let asl2 = a.[b .. c]
        let asl2b = a.[b ..]
        let asl3c = a.[*]

        //let d = b + (SizeSpec.one)
        let asl3 = a.[b .. PlusElems 3, b .. b, 33]
        //let asl3 = a.[b .. b]
        ()


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




