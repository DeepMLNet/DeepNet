namespace SymTensor

open System.Collections.Generic
open Microsoft.FSharp.Reflection

open Basics
open Tensor
open ShapeSpec
open VarSpec


/// expression module
module Expr =
    /// cache for ExprT hashes by reference
    let private exprHashCache = Dictionary<obj, int> (HashIdentity.Reference)

    /// boxes the contents of an option
    let inline boxOption (oo: #obj option) = 
        match oo with
        | Some o -> Some (o :> obj)
        | None -> None

    /// start plus the specified number of (symbolic elements)
    type PlusElems (elems: SizeSpecT) =
        new (intElems: int64) = PlusElems (SizeSpec.fix intElems)
        member this.Elems = elems

    /// arity of an op
    type ArityT =
        | FixedArity of int
        | DynamicArity

    /// ops with no exprs as arguments
    [<StructuralComparison; StructuralEquality>]
    type LeafOpT =

        // ==== scalars ============
        /// scalar of given value
        | ScalarConst of value:ConstSpecT
        /// scalar of the given size
        | SizeValue of value:SizeSpecT * typ:TypeNameT

        // ==== tensor creation ====
        /// tensor with 1 on diagonal of given shape
        | Identity of shape:SizeSpecT * typ:TypeNameT
        /// vector counting from zero to given size minus one
        | Arange of size:SizeSpecT * typ:TypeNameT

        // ==== variable access ====
        /// variable read
        | Var of VarSpecT      
        

    /// ops with one expr as argument
    and [<StructuralComparison; StructuralEquality>] 
        UnaryOpT =

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

        // ==== element-wise unary logic ====
        | Not

        // ==== tensor operations ====
        /// extract diagonal along given axes
        | Diag of int * int
        /// build diagonal matrix along given axes
        | DiagMat of int * int
        /// matrix inverse
        | Invert

        // ==== reductions ====
        /// summation of all elements
        | Sum                           
        /// summation over given dimension
        | SumAxis of int
        /// product of all elements
        | Product                          
        /// product over given dimension
        | ProductAxis of int
        /// maximum over given dimension                
        | MaxAxis of int
        /// minimum over given dimension
        | MinAxis of int

        // ==== index reductions ====
        /// inidices of maximums over given dimension
        | ArgMaxAxis of int
        /// inidices of minimums over given dimension
        | ArgMinAxis of int

        // ==== shape operations ====
        /// reshape tensor; element count does not change
        | Reshape of ShapeSpecT         
        /// broadcast tensor; element count may change
        | DoBroadcast of ShapeSpecT       
        /// permutes the axes of the tensor
        | PermuteAxes of perm:int list
        /// subtensor 
        | Subtensor of ExprRngsSpecT
        /// reverses the tensor in the given dimension 
        | ReverseAxis of dim:int
        /// select elements according to the specified index arrays
        | Gather of indices:ExprT option list
        /// disperses elements according to the specified index arrays
        | Scatter of indices:ExprT option list * shp:ShapeSpecT

        // ==== variable storage ====
        /// variable write
        | StoreToVar of VarSpecT

        // ==== misc ====
        /// nullifies the Jacobian of its argument when calculating derivatives
        | NullifyJacobian
        /// assumes the specified Jacobian for its argument when calculating derivatives
        | AssumeJacobian of ExprT
        /// prints the value together with the given string
        | Print of string
        /// dumps the value into the given dataset in the active HDF5 dump file
        | Dump of string
        /// checks the value for NaNs and infinities, outputs their location and 
        /// stops the computation
        | CheckFinite of string
        /// annotation (no influence on value)
        | Annotated of string       
        /// an op that will expand into an expression once symbolic sizes have
        /// been substituted
        | Held of derivsShp:ShapeSpecT list * op:UnaryHeldOpT

    /// an op that will expand into an expression once symbolic sizes have been substituted
    and UnaryHeldOpT =
        /// replicates the axes to the specified size
        | ReplicateTo of dim:int * size:SizeSpecT

    /// a simplified range specification of one dimension
    and ExprRngSpecT = SimpleRangeSpecT<ExprT>

    /// a simplified range specification of all dimensions
    and ExprRngsSpecT = SimpleRangesSpecT<ExprT>

    /// ops with two exprs as arguments
    and [<StructuralComparison; StructuralEquality>] 
        BinaryOpT =

        // ==== binary elementwise ====
        | Add                           
        | Substract                     
        | Multiply                      
        | Divide                        
        | Modulo
        | Power            
        | MaxElemwise
        | MinElemwise        
           
        // ==== element-wise binary comparison ====
        | Equal
        | Less
        | LessEqual
        | Greater
        | GreaterEqual
        | NotEqual     

        // ==== element-wise binary logic ====
        | And
        | Or

        // ==== element-wise conditional ====
        | IfThenElse of ExprT

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
        NaryOpT =

        /// evaluate all subexpressions but discard them
        | Discard        
        /// build tensor using numeric ranges
        | BuildTensor of shp:ShapeSpecT * rngs:BaseRangesSpecT list
        /// elementwise calculated tensor
        | Elements of shape:ShapeSpecT * elemExpr:ElemExpr.ElemExprT
        /// elementwise interpolation
        | Interpolate of InterpolatorT
        /// use specified channel of a multi-channel op
        | Channel of channelOp:MultiChannelOpT * channel:ChannelT
        /// extension op
        | ExtensionOp of IOp

    /// a channel of a multi-channel op or loop
    and ChannelT = string

    /// an n-ary op with multiple output channels
    and MultiChannelOpT =
        /// iterative evaluation of one or multiple expresisons
        | Loop of spec:LoopSpecT    
     
    /// a slice of an argument to the loop
    and SequenceArgSliceT = {
        /// the index of the argument
        ArgIdx:     int
        /// the dimension the loop is performed over
        SliceDim:   int
    }

    /// references a loop channel of a previous iteration
    and PreviousChannelT = {
        /// the channel to use
        Channel:       ChannelT
        /// the delay, must be at least one
        Delay:         SizeSpecT
        /// the index of the argument specifying the initial values
        InitialArg:    int
    }

    /// a loop variable value specification
    and LoopInputT = 
        /// provides the loop argument to all loop iterations
        | ConstArg of argIdx:int
        /// provides a slice of the loop argument to each loop iteration
        | SequenceArgSlice of SequenceArgSliceT
        /// provides the value of a loop channel from a previous loop iteration
        | PreviousChannel of PreviousChannelT
        /// provides the index of the current loop iteration (zero-based)
        | IterationIndex
        /// provides the number of remaining loop iterations after this iteration
        | IterationsRemaining

    /// the value of a loop channel
    and LoopValueT = {
        /// the expression to compute the loop channel;
        /// it may only use variables defined in LoopSpecT.Vars
        Expr:       ExprT
        /// the dimension to concatenate the results along to produce the loop output
        SliceDim:   int
    }

    /// A loop specification.
    /// A loop provides iterative evaluation of one or multiple expresisons.
    /// A loop can slice over its arguments and reference values computed in previous
    /// loop iterations.
    /// A loop can compute multiple values at once. Each computed values is referred to
    /// as a channel.
    and LoopSpecT = {
        /// number of loop iterations
        Length:     SizeSpecT
        /// specifies the values of the variables used in the channel value expressions,
        /// i.e. LoopValueT.Expr
        Vars:       Map<VarSpecT, LoopInputT>   
        /// specifies the values of the loop channels
        Channels:   Map<ChannelT, LoopValueT>
    }

    /// A mathematical operation in an expression.
    /// This models a mathematical function or operator that takes one or more tensors
    /// and returns one tensor.
    and IOp =
        inherit System.IComparable
      
        /// Should return the shape of the result, given the shape of the arguments.
        abstract Shape: argShapes: ShapeSpecT list -> ShapeSpecT      
        
        /// Should check if the shapes of the arguments are acceptable and,
        /// if not, raise an exception.
        abstract CheckArgs: argShapes: ShapeSpecT list -> unit      

        /// Should return the op with all symbolic sizes substituted using the specified
        /// substitution table.
        /// Return a *new* op with substitution applied. Do not apply the mapping in-place.
        abstract SubstSymSizes: symSizes: SymSizeEnvT -> IOp

        /// Should be true, if all symbolic sizes can be evaluated to numeric sizes.
        /// This is the case if the function ShapeSpec.canEval or SizeSpec.canEval respectively
        /// return true on all sizes used in this op.
        abstract CanEvalAllSymSizes: bool

        /// Should compute the derivative w.r.t. each argument given the derivative w.r.t. the op.
        /// The derivative is always an NxM matrix where N is the number of elements of the function
        /// the derivative of which is being taken and M is the number of elements of the argument
        /// w.r.t. which the derivative is being taken. 
        /// Thus, if dOp is an NxK matrix and an argument has M elements, the derivative matrix
        /// you return w.r.t. that argument must have NxM elements.
        abstract Deriv: dOp:ExprT -> args:ExprT list -> ExprT list

        /// Should evaluate the numerical value of this op given the numerical values of its arguments.
        /// This evaluation should be done on the host using the simplest means possible and is used
        /// as a reference implementation for verifying the correctness of optimized (e.g. CUDA) 
        /// implementations. This method may be omitted when no verification will be done.
        abstract EvalSimple: args:Tensor<'T> list -> Tensor<'T>

        /// Should return the set of variables that this op instance depends on.
        abstract ContainedVars: Set<VarSpecT>

    and [<StructuralComparison; StructuralEqualityAttribute>]
        private ExprProxyT = 
        | ProxyLeaf of LeafOpT
        | ProxyUnary of UnaryOpT * ExprT
        | ProxyBinary of BinaryOpT * ExprT * ExprT
        | ProxyNary of NaryOpT * (ExprT list)

    /// an expression
    and [<CustomComparison; CustomEqualityAttribute; StructuredFormatDisplay("{Pretty}")>] 
        ExprT =
        | Leaf of LeafOpT
        | Unary of UnaryOpT * ExprT
        | Binary of BinaryOpT * ExprT * ExprT
        | Nary of NaryOpT * (ExprT list)

        member inline private this.Proxy = 
            match this with
            | Leaf op -> ProxyLeaf op
            | Unary (op, a) -> ProxyUnary (op, a)
            | Binary (op, a, b) -> ProxyBinary (op, a, b)
            | Nary (op, es) -> ProxyNary (op, es)

        // cache hash code using object reference
        override this.Equals other =
            match other with
            | :? ExprT as other -> (this :> System.IEquatable<_>).Equals other
            | _ -> false
        interface System.IEquatable<ExprT> with
            member this.Equals other = 
                if obj.ReferenceEquals (this, other) then true
                elif this.GetHashCode() <> other.GetHashCode() then false
                else 
                    let knownEqualTo = Dictionary<ExprT, HashSet<ExprT>> (HashIdentity.Reference)
                    let rec treeCompare t o =
                        match knownEqualTo.TryFind t with
                        | Some k when k.Contains o -> true
                        | _ ->
                            let eq = 
                                match t, o with
                                | Leaf tOp, Leaf oOp -> 
                                    tOp = oOp
                                | Unary (tOp, ta), Unary (oOp, oa) -> 
                                    tOp = oOp && treeCompare ta oa
                                | Binary (tOp, ta, tb), Binary (oOp, oa, ob) -> 
                                    tOp = oOp && treeCompare ta oa && treeCompare tb ob
                                | Nary (tOp, tes), Nary(oOp, oes) ->
                                    tOp = oOp && List.forall2 (fun te oe -> treeCompare te oe) tes oes
                                | _ -> false
                            if eq then
                                if not (knownEqualTo.ContainsKey t) then
                                    knownEqualTo.[t] <- HashSet<ExprT> (HashIdentity.Reference)
                                knownEqualTo.[t].Add o |> ignore
                            eq
                    treeCompare this other
        override this.GetHashCode() =
            match exprHashCache.TryFind this with
            | Some h -> h
            | None ->
                let h = hash this.Proxy
                exprHashCache.[this] <- h
                h
        interface System.IComparable<ExprT> with
            member this.CompareTo other =
                compare this.Proxy other.Proxy
        interface System.IComparable with
            member this.CompareTo other =
                match other with
                | :? ExprT as other -> (this :> System.IComparable<_>).CompareTo other
                | _ -> failwithf "cannot compare ExprT to type %A" (other.GetType())

        /// pretty string
        member this.Pretty =
            match this with
            | Leaf op -> sprintf "{%A}" op 
            | Unary (op, a) -> sprintf "{%A} (%A)" op a
            | Binary (op, a, b) -> sprintf "{%A} (%A, %A)" op a b
            | Nary (op, es) -> sprintf "{%A} (%A)" op es

    type FullExprRngSpecT = RangeSpecT<ExprT>
    type FullExprRngsSpecT = RangesSpecT<ExprT>
   
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
        | Not
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
        | MaxElemwise
        | MinElemwise
        | Equal
        | Less
        | LessEqual
        | Greater
        | GreaterEqual
        | NotEqual     
        | And
        | Or
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
        let subCon = contains subExpr
        if expr = subExpr then true
        else
            match expr with
            | Unary (Gather indices, a) ->
                subCon a || 
                indices |> List.exists (function | Some idx -> subCon idx | None -> false)
            | Unary (Scatter (indices, _), a) ->
                subCon a || 
                indices |> List.exists (function | Some idx -> subCon idx | None -> false)
            | Unary (AssumeJacobian jac, a) ->
                subCon a || subCon jac
            | Binary (IfThenElse cond, a, b) ->
                subCon a || subCon b || subCon cond

            | Unary (_, a) -> subCon a
            | Binary (_, a, b) -> subCon a || subCon b
            | Nary (_, es) -> List.exists subCon es
            | _ -> false

    /// Produces an error message about incompatible shapes.
    let failshape op sa sb =
        failwithf "op %A was provided with arrays of incompatible shapes %A and %A" op sa sb

    /// Returns the type of the given expression.
    let rec typename expr =
        match expr with
        | Leaf (Identity (_, tn)) -> tn
        | Leaf (ScalarConst cs) -> cs.TypeName
        | Leaf (SizeValue (_, tn)) -> tn
        | Leaf (Arange (_, tn)) -> tn
        | Leaf (Var vs) -> vs.TypeName

        | Unary (ArgMinAxis _, _)
        | Unary (ArgMaxAxis _, _)
            -> TypeName.ofType<int64>

        | Binary (Equal, _, _)
        | Binary (Less, _, _)
        | Binary (LessEqual, _, _)
        | Binary (Greater, _, _)
        | Binary (GreaterEqual, _, _)
        | Binary (NotEqual, _, _)
            -> TypeName.ofType<bool>

        | Nary (Elements (_, elemExpr), _) 
            -> ElemExpr.typeName elemExpr
        | Nary (Channel (Loop spec, channel), _)
            -> loopOutputTypeNames spec |> Map.find channel 

        | Unary (_, a) -> typename a
        | Binary (_, a, b) -> typename a
        | Nary (_, es) -> typename (List.head es)

    /// data type of loop otuput
    and internal loopOutputTypeNames (spec: LoopSpecT) =
        spec.Channels |> Map.map (fun ch lv -> typename lv.Expr)

    let private shapeCache = ConcurrentDictionary<ExprT, ShapeSpecT> (HashIdentity.Reference)

    /// Returns the shape of the given expression.
    let rec shapeOf expr =
        // We assume that all operands have compatible size. 
        // For elementwise operations we assume that a and b are already broadcasted
        // to have the *same* size.

        match shapeCache.TryFind expr with
        | Some shp -> shp
        | None ->
            let shp =
                match expr with

                // tensor creation
                | Leaf(Identity (ss, _)) -> ShapeSpec.matrix ss ss
                | Leaf(ScalarConst _) -> ShapeSpec.scalar
                | Leaf(SizeValue _) -> ShapeSpec.scalar
                | Leaf(Arange (size, _)) -> ShapeSpec.vector size

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
                | Unary (Not, a)
                | Unary (NullifyJacobian, a)
                | Unary (AssumeJacobian _, a)
                    -> shapeOf a

                // tensor operations
                | Unary(Diag(ax1, ax2), a) -> shapeOf a |> ShapeSpec.withoutAxis ax2
                | Unary(DiagMat(ax1, ax2), a) ->  shapeOf a |> List.insert ax2 (shapeOf a).[ax1]
                | Unary(Invert, a) -> shapeOf a

                // reductions
                | Unary(Sum, _) -> ShapeSpec.scalar
                | Unary(Product, _) -> ShapeSpec.scalar
                | Unary(SumAxis ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax
                | Unary(ProductAxis ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax
                | Unary(MaxAxis ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax
                | Unary(MinAxis ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax

                // index reductions
                | Unary(ArgMaxAxis ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax
                | Unary(ArgMinAxis ax, a) -> shapeOf a |> ShapeSpec.withoutAxis ax

                // shape operations
                | Unary(Reshape(ss), _) -> ss
                | Unary(DoBroadcast(ss), _) -> ss
                | Unary(PermuteAxes perm, a) -> shapeOf a |> ShapeSpec.permuteAxes perm
                | Unary(Subtensor(srs), a) ->
                    (srs, shapeOf a)
                    ||> List.map2 (fun sr shp ->
                         match sr with
                         | SRSSymStartSymEnd (s, fo)    -> (fo |? (shp - SizeSpec.one)) + 1L - s
                         | SRSDynStartSymSize (_, size) -> size)
                | Unary(ReverseAxis _, a) -> shapeOf a
                | Unary(Held ([], ReplicateTo (dim, s)), a) -> shapeOf a |> ShapeSpec.set dim s
                | Unary(Gather indices, a) -> indices |> List.pick id |> shapeOf
                | Unary (Scatter (indices, shp), a) -> shp

                // misc
                | Unary(StoreToVar _, a) -> ShapeSpec.emptyVector
                | Unary(Print _, a) -> shapeOf a
                | Unary(Dump _, a) -> shapeOf a
                | Unary(CheckFinite _, a) -> shapeOf a
                | Unary(Annotated(_), a) -> shapeOf a
                | Unary(Held (derivShp :: _, heldOp), a) -> [(shapeOf a).[0]; ShapeSpec.nElem derivShp]

                // binary elementwise
                | Binary (Add, a, _)                         
                | Binary (Substract, a, _)                     
                | Binary (Multiply, a, _)                      
                | Binary (Divide, a, _)                        
                | Binary (Modulo, a, _)
                | Binary (Power, a, _)     
                | Binary (MaxElemwise, a, _)                    
                | Binary (MinElemwise, a, _)       
                | Binary (Equal, a, _)             
                | Binary (Less, a, _)
                | Binary (LessEqual, a, _)
                | Binary (Greater, a, _)
                | Binary (GreaterEqual, a, _)
                | Binary (NotEqual, a, _)
                | Binary (IfThenElse _, a, _)
                | Binary (And, a, _)
                | Binary (Or, a, _)
                    -> shapeOf a
            
                // matrix/tensor operations
                | Binary (Dot, a, b) -> 
                    let sa, sb = shapeOf a, shapeOf b
                    match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                    | 2, 2 -> ShapeSpec.matrix sa.[0] sb.[1]
                    | na, nb when na=nb -> sa.[0 .. na-2] @ [sb.[nb-1]]
                    | _ -> failwithf "invalid dot product shapes: %A, %A" sa sb
                | Binary (TensorProduct, a, b) -> 
                    let sa, sb = shapeOf a, shapeOf b
                    List.map2 (*) sa sb

                // shape operations
                | Binary (SetSubtensor ss, a, b) ->
                    shapeOf a

                // misc
                | Nary(Discard, _) -> ShapeSpec.emptyVector 
                | Nary(BuildTensor (shp, _), _) -> shp
                | Nary(Elements (resShape, elemExpr), _) -> resShape
                | Nary(Interpolate _, es) -> shapeOf es.Head
                | Nary(Channel (Loop spec, channel), es) -> loopOutputShapes spec |> Map.find channel
                | Nary(ExtensionOp eop, es) -> eop.Shape (es |> List.map shapeOf)

            shapeCache.[expr] <- shp
            shp
    

    /// Returns the shapes of the outputs of the loop channels.
    and internal loopOutputShapes (spec: LoopSpecT) =
        spec.Channels
        |> Map.map (fun ch lv ->
            shapeOf lv.Expr |> ShapeSpec.insertAxis lv.SliceDim spec.Length)

    /// number of elements of given expression
    let nElems expr =
        expr |> shapeOf |> ShapeSpec.nElem

    /// number of dimensions of given expression
    let nDims expr =
        expr |> shapeOf |> ShapeSpec.nDim

    /// Wraps the given op in a Reshape op if its shape does not match ss.
    let reshapeIfNecessary ss expr =
        if ss = shapeOf expr then expr else Unary(Reshape(ss), expr)

    /// Wraps the given op in a Broadcast op if its shape does not match ss.
    let broadcastIfNecessary ss expr =
        if ss = shapeOf expr then expr else Unary(DoBroadcast(ss), expr)

    /// Caches for extracted variables.
    let private extractedVars = Dictionary<ExprT, Set<VarSpecT>> () 

    /// extract all variables from an expression
    let rec extractVars expr =
        match extractedVars.LockedTryFind expr with
        | Some evs -> evs
        | None ->
            let evs =
                match expr with
                | Leaf (Var vs) -> Set.singleton vs
                | Unary (StoreToVar vs, a) -> extractVars a |> Set.add vs
                | Unary (Gather indices, a) ->
                    let indicesVars = indices |> List.choose (Option.map extractVars)
                    Set.unionMany (extractVars a :: indicesVars)
                | Unary (Scatter (indices, _), a) ->
                    let indicesVars = indices |> List.choose (Option.map extractVars)
                    Set.unionMany (extractVars a :: indicesVars)
                | Unary (AssumeJacobian jac, a) -> 
                    Set.union (extractVars jac) (extractVars a)
                | Binary (IfThenElse cond, a, b) -> 
                    Set.unionMany [extractVars cond; extractVars a; extractVars b]
                | Nary (ExtensionOp eop, es) ->
                    Set.unionMany (eop.ContainedVars :: (es |> List.map extractVars))

                | Leaf _ -> Set.empty
                | Unary (_, a) -> extractVars a
                | Binary (_, a, b) -> Set.union (extractVars a) (extractVars b)
                | Nary (_, es) -> Set.unionMany (es |> List.map extractVars)

            extractedVars.[expr] <- evs
            evs

    type ExprT with
        /// symbolic shape
        member this.Shape = shapeOf this

        /// number of dimensions
        member this.NDims = ShapeSpec.nDim this.Shape

        /// symbolic number of elements
        member this.NElems = nElems this

        /// type name of this expression
        member this.TypeName = typename this  

        /// type of this expression
        member this.Type = this.TypeName |> TypeName.getType 

    /// checks that given axis is valid for specified expression
    let checkAxis ax expr =
        if not (0 <= ax && ax < nDims expr) then
            failwithf "invalid axis %d for expression of shape %A" ax (shapeOf expr)

    /// expressions that were already checked for correctness
    let checkedExprs = HashSet<ExprT> (HashIdentity.Reference)

    /// Checks ops' arguments for compatible shapes.
    let rec checkExpr (expr: ExprT) =
        if not (checkedExprs.LockedContains expr) then

            if typename expr = TypeName.ofType<obj> then
                failwith "Expression type cannot be object."

            let (..=) (sa: ShapeSpecT) (sb: ShapeSpecT) =
                if sa.Length = sb.Length then List.forall2 (.=) sa sb
                else false
            let (..<>) sa sb = not (sa ..= sb)
            let reqBool op a =
                if typename a <> TypeName.ofType<bool> then
                    failwithf "logical operation %A requires data type bool but got %A" 
                        op (typename a).Type

            match expr with 
            | Leaf op -> ()           

            | Unary (op, a) ->
                checkExpr a
                let sa = shapeOf a
                let nda = ShapeSpec.nDim sa

                match op with
                | Not -> reqBool op a
                | SumAxis(ax)
                | ProductAxis(ax)
                | MaxAxis(ax) 
                | MinAxis(ax) 
                | ArgMaxAxis(ax) 
                | ArgMinAxis(ax) when not (0 <= ax && ax < nda) ->
                    failwithf "cannot recude over non-existant axis %d of array with shape %A" ax sa
                | Reshape(ss) ->
                    if ShapeSpec.nElem sa .<> ShapeSpec.nElem ss then
                        failwithf "reshape cannot change number of elements while reshaping from %A to %A" sa ss
                | DoBroadcast(ss) -> 
                    if ShapeSpec.nDim ss <> nda then
                        failwithf "array of shape %A does not have same number of dimesions as broadcast shape %A"
                            sa ss
                    for dim in 0 .. (ShapeSpec.nDim ss) - 1 do
                        match sa.[dim], ss.[dim] with
                        | SizeSpecT.Broadcast, _ -> ()
                        | ssa, ssb when ssa .<> ssb -> 
                            failwithf "cannot broadcast from %A to %A because non-broadcast dimensions must not change" sa ss
                        | _ -> ()
                | PermuteAxes perm -> 
                    if nda <> List.length perm then
                        failwithf "permutation %A must have same rank as shape %A" perm sa
                    if not (Permutation.is perm) then
                        failwithf "%A is not a valid permutation of an %d-dimensional tensor" perm nda
                | StoreToVar vs ->
                    if sa ..<> vs.Shape then
                        failwithf "cannot store expression of shape %A into variable of shape %A" 
                            sa vs.Shape
                | Diag(ax1, ax2) ->
                    if not (0 <= ax1 && ax1 < nda && 0 <= ax2 && ax2 < nda) then
                        failwithf "cannot extract diagonal from non-existant axis %d or %d of array with shape %A" 
                            ax1 ax2 sa
                    if not (ax1 < ax2) then 
                        failwith "first axis for extracting diagonal must come before second axis"
                    if sa.[ax1] .<> sa.[ax2] then
                        failwithf "cannot extract diagonal along axes %d and %d from non-square matrix %A" ax1 ax2 sa
                | DiagMat(ax1, ax2) ->
                    if not (0 <= ax1 && ax1 < nda && 0 <= ax2 && ax2 <= nda) then
                        failwithf "cannot build diagonal over non-existant axis %d or %d of array with shape %A" 
                            ax1 ax2 sa
                    if not (ax1 < ax2) then 
                        failwith "first axis for building diagonal matrix must come before second axis"
                | Invert ->
                    if nda < 2 then
                        failwithf "need at least a matrix to invert but got shape %A" sa
                    if sa.[nda-2] .<> sa.[nda-1] then
                        failwithf "cannot invert non-square matrix %A along last two axes" sa 
                | AssumeJacobian jac ->
                    checkExpr jac
                    if typename jac <> typename expr then
                        failwithf "Jacobian type %A does not match expression type %A."
                            (typename jac).Type (typename expr).Type
                    if nDims jac <> 2 then
                        failwithf "Jacobian shape %A must be two-dimensional" (shapeOf jac)
                    if (shapeOf jac).[1] <> nElems expr then
                        failwithf "Jacobian shape %A must have %A elements in second dimension" 
                            (shapeOf jac) (nElems expr)
                | ReverseAxis ax when not (0 <= ax && ax < nda) ->
                    failwithf "cannot reverse non-existant axis %d of array with shape %A" ax sa
                | Held ([], ReplicateTo (dim, s)) -> 
                    a |> checkAxis dim
                | Gather indices ->
                    if nda <> indices.Length then
                        failwithf "gather argument has %d dimensions but %d index arrays were specified" 
                            nda indices.Length
                    let trgtShape =
                        match indices |> List.tryPick id with
                        | Some idx -> idx.Shape
                        | None -> failwith "gather needs at least one specified index expression"  
                    for dim, idx in List.indexed indices do
                        match idx with
                        | Some idx when idx.Type <> typeof<int64> ->
                            failwithf "all index arrays for gather must be of type int64, but got type %A" idx.Type
                        | Some idx when idx.Shape <> trgtShape ->
                            failwithf "all gather indices must have equal shape, but got %A"
                                (indices |> List.map (Option.map shapeOf))
                        | None when dim >= ShapeSpec.nDim trgtShape ->
                            failwithf "gather index dimensions beyond the number of target dimensions \
                                       must not be None"
                        | _ -> ()
                | Scatter (indices, shp) ->
                    for dim, idx in List.indexed indices do
                        match idx with
                        | Some idx when idx.Type <> typeof<int64> ->
                            failwithf "all index arrays for scatter must be of type int64, but got type %A" idx.Type
                        | Some idx when idx.Shape <> a.Shape ->
                            failwithf "all scatter indices must have shape of source %A, but got %A" a.Shape
                                (indices |> List.map (Option.map shapeOf))
                        | None when dim >= a.NDims ->
                            failwithf "scatter index dimensions beyond the number of source dimensions \
                                       must not be None"
                        | _ -> ()
                | _ -> ()

            | Binary (op, a, b) ->
                checkExpr a
                checkExpr b
                let sa, sb = shapeOf a, shapeOf b
                let nda, ndb = ShapeSpec.nDim sa, ShapeSpec.nDim sb

                let ta, tb = typename a, typename b
                if ta <> tb then
                    failwithf "cannot apply binary operation %A to expressions of \
                               different types %A and %A" op ta.Type tb.Type

                match op with
                | And
                | Or ->
                    reqBool op a
                    reqBool op b
                | IfThenElse c ->
                    checkExpr c
                    if c.Type <> typeof<bool> then
                        failwith "condition of IfThenElse must be expression of type bool"
                    if c.Shape ..<> sa || sa ..<> sb then
                        failwithf "shape of condition %A and both argument shapes %A and %A must be equal"
                            c.Shape sa sb                    
                | BinaryElemwiseOp ->
                    if sa ..<> sb then
                        failwithf "cannot apply element-wise operation %A to unequal shapes %A and %A" op sa sb
                | Dot -> 
                    match nda, ndb with
                    | 2, 2 -> 
                        if sa.[1] .<> sb.[0] then
                            failwithf "incompatible shapes for dot product: %A and %A" sa sb
                    | na, nb when na = nb -> 
                        if sa.[na-1] .<> sb.[nb-2] || 
                           [0 .. na-3] |> List.exists (fun n -> sa.[n] .<> sb.[n]) then
                            failwithf "incompatible shapes for batched dot product: %A and %A" sa sb
                    | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
                | TensorProduct when nda <> ndb ->
                    failwithf "cannot compute tensor product between arrays of shapes %A and %A" sa sb
                | _ -> ()

            | Nary (op, es) ->
                es |> List.iter checkExpr
                let ss = es |> List.map shapeOf

                let checkEqualTypes() =
                    if es |> List.exists (fun e -> typename e <> typename es.Head) then
                        failwithf "cannot apply n-ary operation %A to expressions of different types %A"
                            op (es |> List.map (typename >> TypeName.getType))
                let checkArg idx =
                    if not (0 <= idx && idx < es.Length) then
                        failwithf "the zero-based index %d does not exist for %d specified arguments" idx es.Length

                match op with
                | Elements (trgtShp, elemExpr) -> 
                    checkEqualTypes()
                    let tns = es |> List.map typename
                    ElemExpr.check elemExpr |> ignore
                    ElemExpr.checkCompatibility elemExpr ss tns trgtShp
                | BuildTensor (shp, rngs) ->
                    if List.length rngs <> List.length es then
                        failwithf "BuildTensor ranges must match arguments, but got %d ranges and %d arguments"
                                  rngs.Length es.Length
                    match ShapeSpec.tryEval shp with
                    | Some shp ->
                        for rng, arg in List.zip rngs es do
                            if rng.Length <> shp.Length then
                                failwithf "BuildTensor range %A has wrong dimensionality for shape %A" rng shp
                            for (start, stop), size, argSize in List.zip3 rng shp (shapeOf arg) do
                                if argSize <> stop - start + 1L then
                                    failwithf "BuildTensor range %A is invalid for argument of shape %A" rng (shapeOf arg)
                                match SizeSpec.tryEval start, SizeSpec.tryEval stop with
                                | Some start, Some stop when not (0L <= start && start < size && 0L <= stop && 
                                                                  stop < size && start <= stop) ->
                                    failwithf "BuildTensor range %A is invalid for shape %A" rng shp
                                | _, _ -> ()
                    | None -> ()                                          
                | Interpolate ip ->
                    checkEqualTypes()
                    let nDims = ip.MinArg.Length
                    if nDims < 1 then
                        failwith "interpolator must be at least one-dimensional"
                    if ip.MaxArg.Length <> nDims || ip.Outside.Length <> nDims || ip.Resolution.Length <> nDims then
                        failwith "MinArg, MaxArg, Resolution and Outside have inconsistent lengths"
                    if es.Length <> nDims then
                        failwith "number of arguments does not match dimensionality of interpolator"
                    if not ((ip.MinArg, ip.MaxArg) ||> List.forall2 (fun mi ma -> conv<float> mi < conv<float> ma)) then
                        failwith "MinArg of interpolator must be smaller than MaxArg"
                    if ip.Resolution |> List.exists ((>) 0.0) then
                        failwith "Resolution of interpolator must be positive"
                    for s in ss do 
                        if s ..<> ss.Head then
                            failwithf "all arguments to interpolator must have equal shape but got: %A" ss
                | Channel (Loop spec, channel) ->
                    // check that the referenced loop channel exists
                    if not (spec.Channels |> Map.containsKey channel) then
                        failwithf "specified loop channel %A does not exist" channel
                    // check that all variables are defined
                    let usedVars =
                        Map.toSeq spec.Channels
                        |> Seq.map (fun (_, lv) -> extractVars lv.Expr)
                        |> Set.unionMany
                    let specifiedVars = 
                        Map.toSeq spec.Vars
                        |> Seq.map (fun (var, _) -> var)
                        |> Set.ofSeq
                    if not (Set.isEmpty (usedVars - specifiedVars)) then
                        failwithf "the variables %A were used in the loop but not defined"
                            (usedVars - specifiedVars)
                    // check that shapes of loop variables are correct and referenced arguments exist
                    for KeyValue(vs, li) in spec.Vars do
                        match li with
                        | ConstArg idx -> 
                            checkArg idx
                            if es.[idx].TypeName <> vs.TypeName then
                                failwithf "constant argument variable %A was given argument of type %A" vs es.[idx].Type
                            if vs.Shape ..<> ss.[idx] then
                                failwithf "constant argument variable %A was given argument of shape %A" vs ss.[idx]
                        | SequenceArgSlice {ArgIdx=idx; SliceDim=dim} ->
                            checkArg idx
                            if es.[idx].TypeName <> vs.TypeName then
                                failwithf "sequence argument variable %A was given argument of type %A" vs es.[idx].Type
                            let reqShp = vs.Shape |> ShapeSpec.insertAxis dim spec.Length
                            if reqShp ..<> ss.[idx] then
                                failwithf "sequence argument variable %A requires argument shape %A but was given %A" vs reqShp ss.[idx]
                        | PreviousChannel {Channel=prvCh; Delay=delay; InitialArg=ivIdx} ->
                            // check previous channel
                            match spec.Channels |> Map.tryFind prvCh with
                            | Some chVal -> 
                                if vs.TypeName <> chVal.Expr.TypeName then
                                    failwithf "previous channel variable %A was given channel of type %A" vs chVal.Expr.Type
                                if chVal.Expr.Shape ..<> vs.Shape then
                                    failwithf "previous channel variable %A was given channel of shape %A" vs chVal.Expr.Shape                                
                            | None -> 
                                failwithf "previous channel %A for variable %A does not exist" prvCh vs
                            
                            // check initial value arg
                            checkArg ivIdx
                            if es.[ivIdx].TypeName <> vs.TypeName then
                                failwithf "previous channel variable %A was given initial value of type %A" vs es.[ivIdx].Type
                            let sliceDim = spec.Channels.[prvCh].SliceDim
                            let reqShp = vs.Shape |> ShapeSpec.insertAxis sliceDim delay
                            if reqShp ..<> ss.[ivIdx] then
                                failwithf "previous channel variable %A needs initial value of shape %A but was given %A" vs reqShp ss.[ivIdx]                                
                        | IterationIndex 
                        | IterationsRemaining -> 
                            if vs.TypeName <> TypeName.ofType<int> then
                                failwithf "iteration index variable %A must be of type int" vs
                            if vs.Shape ..<> [] then
                                failwithf "iteration index variable %A must be scalar" vs
                | ExtensionOp eop -> eop.CheckArgs ss
                | _ -> ()

            checkedExprs.LockedAdd expr |> ignore

    /// substitues the given symbol sizes into the expression
    let rec substSymSizes symSizes (expr: ExprT) =
        let substituted = Dictionary<ExprT, ExprT> ()
        let sSize = SymSizeEnv.subst symSizes
        let sShp = SymSizeEnv.substShape symSizes
        let sSrs = SymSizeEnv.substRange symSizes

        let rec sSub expr = 
            match substituted.TryFind expr with
            | Some subst -> subst
            | None ->
                let subst = 
                    match expr with
                    | Leaf (Identity (ss, tn)) -> Leaf (Identity (sSize ss, tn))
                    | Leaf (SizeValue (sc, tn)) -> Leaf (SizeValue (sSize sc, tn))
                    | Leaf (Var vs) -> Leaf (Var {vs with Shape = sShp vs.Shape})
                    | Leaf (Arange (size, tn)) -> Leaf (Arange (sSize size, tn))
                    | Leaf _ -> expr

                    | Unary (Reshape ss, a) -> Unary (Reshape (sShp ss), sSub a)
                    | Unary (DoBroadcast ss, a) -> Unary (DoBroadcast (sShp ss), sSub a)
                    | Unary (StoreToVar vs, a) -> Unary (StoreToVar {vs with Shape = sShp vs.Shape}, sSub a)
                    | Unary (Subtensor srs, a) -> Unary (Subtensor (sSrs srs), sSub a)
                    | Unary (Held (derivsShp, heldOp), a) ->
                        let substOp =
                            match heldOp with
                            | ReplicateTo (dim, s) -> ReplicateTo (dim, sSize s)
                        Unary (Held (derivsShp |> List.map sShp, substOp), sSub a)
                    | Unary (Gather indices, a) ->
                        let indices = indices |> List.map (Option.map sSub)
                        Unary (Gather indices, sSub a)
                    | Unary (Scatter (indices, shp), a) ->
                        let indices = indices |> List.map (Option.map sSub)
                        Unary (Scatter (indices, sShp shp), sSub a)
                    | Unary (AssumeJacobian jac, a) -> Unary (AssumeJacobian (sSub jac), sSub a)
                    | Unary (op, a) -> Unary (op, sSub a)

                    | Binary (IfThenElse c, a, b) -> Binary (IfThenElse (sSub c), sSub a, sSub b)
                    | Binary (SetSubtensor srs, a, b) -> Binary (SetSubtensor (sSrs srs), sSub a, sSub b)
                    | Binary (op, a, b) -> Binary (op, sSub a, sSub b)

                    | Nary (BuildTensor (shp, rngs), es) ->
                        Nary (BuildTensor (sShp shp, rngs |> List.map (List.map (fun (f,l) -> sSize f, sSize l))), 
                              List.map sSub es)
                    | Nary (Elements (trgtShp, elemExpr), es) -> 
                        Nary (Elements (sShp trgtShp, ElemExpr.substSymSizes symSizes elemExpr), List.map sSub es)
                    | Nary (Channel (Loop spec, channel), es) ->
                        let substSpec = {
                            Length = sSize spec.Length
                            Vars = spec.Vars
                                   |> Map.toSeq
                                   |> Seq.map (fun (vs, li) ->
                                       let vs = {vs with Shape = sShp vs.Shape}
                                       let li = match li with
                                                | PreviousChannel pc -> 
                                                    PreviousChannel {pc with Delay = sSize pc.Delay}
                                                | _ -> li
                                       vs, li)
                                   |> Map.ofSeq
                            Channels = spec.Channels
                                       |> Map.map (fun ch lv -> {lv with Expr = sSub lv.Expr})
                        }
                        Nary (Channel (Loop substSpec, channel), es |> List.map sSub)
                    | Nary (ExtensionOp eop, es) -> Nary (ExtensionOp (eop.SubstSymSizes symSizes), List.map sSub es)
                    | Nary (op, es) -> Nary (op, List.map sSub es)
                
                substituted.[expr] <- subst
                subst
        sSub expr

    let private exprsWithEvalableSymSizes = HashSet<ExprT> ()

    /// tests if all symbolic sizes can be evaluated
    let rec private testEvalAllSymSizes (failIfNot: bool) (expr: ExprT) =
        let subTest = testEvalAllSymSizes failIfNot
        let tSize = SizeSpec.canEval
        let tShp = ShapeSpec.canEval
        let tSrs = SimpleRangesSpec.canEvalSymbols
        let evalable =
            if exprsWithEvalableSymSizes.LockedContains expr then true
            else 
                match expr with
                | Leaf (Identity (ss, tn)) -> tSize ss
                | Leaf (SizeValue (sc, tn)) -> tSize sc
                | Leaf (Var vs) -> tShp (VarSpec.shape vs)
                | Leaf (Arange (size, tn)) -> tSize size
                | Leaf _ -> true

                | Unary (Reshape ss, a) -> tShp ss && subTest a
                | Unary (DoBroadcast ss, a) -> tShp ss && subTest a
                | Unary (StoreToVar vs, a) -> tShp (VarSpec.shape vs) && subTest a
                | Unary (Subtensor srs, a) -> tSrs srs && subTest a
                | Unary (Held (derivsShp, heldOp), a) ->
                    let canEvalOp =
                        match heldOp with 
                        | ReplicateTo (dim, s) -> tSize s
                    List.forall tShp derivsShp && canEvalOp && subTest a
                | Unary (Gather indices, a) ->
                    let someIndices = indices |> List.choose id
                    List.forall subTest someIndices && subTest a
                | Unary (Scatter (indices, shp), a) ->
                    let someIndices = indices |> List.choose id
                    List.forall subTest someIndices && tShp shp && subTest a
                | Unary (AssumeJacobian jac, a) -> subTest jac && subTest a
                | Unary (op, a) -> subTest a

                | Binary (SetSubtensor srs, a, b) -> 
                    tSrs srs && subTest a && subTest b
                | Binary (IfThenElse c, a, b) ->
                    subTest c && subTest a && subTest b 
                | Binary (op, a, b) -> subTest a && subTest b

                | Nary (BuildTensor (shp, rngs), es) -> 
                    tShp shp && 
                    List.forall BaseRangesSpec.canEval rngs &&
                    List.forall subTest es
                | Nary (Elements (trgtShp, elemExpr), es) -> 
                    tShp trgtShp && 
                    ElemExpr.canEvalAllSymSizes elemExpr && 
                    List.forall subTest es
                | Nary (Channel (Loop spec, channel), es) ->
                    (tSize spec.Length) 
                    &&
                    (spec.Vars |> Map.toSeq |> Seq.forall (fun (vs, li) ->
                        tShp vs.Shape &&
                        match li with
                        | PreviousChannel pc -> tSize pc.Delay
                        | _ -> true)) 
                    &&
                    (spec.Channels |> Map.toSeq |> Seq.forall (fun (ch, lv) -> subTest lv.Expr))                
                | Nary (ExtensionOp eop, es) -> eop.CanEvalAllSymSizes && List.forall subTest es
                | Nary (op, es) -> List.forall subTest es
        
        if evalable then exprsWithEvalableSymSizes.LockedAdd expr |> ignore
        if failIfNot && not evalable then
            failwithf "expression %A contains a symbolic size that cannot be evaluated to \
                       a numeric value" expr
        evalable

    /// true if all shapes in the expression can be evaluated to numeric shapes
    let canEvalAllSymSizes (expr: ExprT) =
        testEvalAllSymSizes false expr

    /// fails if the expression contains a shape that cannot be evaluated to a numeric shape
    let failOnNotEvalableSymSize (expr: ExprT) =
        testEvalAllSymSizes true expr |> ignore

    /// Traverses the expression and checks ops' arguments for compatible shapes.
    let check (expr: ExprT) : ExprT =
        checkExpr expr |> ignore
        expr

    /// Replaces all occurences of the map key with its value in the specified expression.
    /// Does not replace subexpressions within loop channel value expressions.
    let subst (replacements: Map<ExprT, ExprT>) expr =
        let substituted = Dictionary<ExprT, ExprT> ()

        // TODO: currently does not substitues into Subtensor and SetSubtensor dyanmic range expression.
        let rec subSubst expr =       
            match substituted.TryFind expr with
            | Some subst -> subst
            | None ->
                let subst = 
                    match replacements.TryFind expr with
                    | Some replacement -> replacement
                    | None ->
                        match expr with
                        // substitute into ops containing expressions
                        | Unary (AssumeJacobian jac, a) ->
                            Unary (AssumeJacobian (subSubst jac), subSubst a)
                        | Unary (Gather indices, a) ->
                            let indices = indices |> List.map (Option.map subSubst)
                            Unary (Gather indices, subSubst a)
                        | Unary (Scatter (indices, shp), a) ->
                            let indices = indices |> List.map (Option.map subSubst)
                            Unary (Scatter (indices, shp), subSubst a)
                        | Binary (IfThenElse c, a, b) -> 
                            Binary (IfThenElse (subSubst c), subSubst a, subSubst b)

                        // apply recursively
                        | Leaf _ -> expr
                        | Unary (op, a) -> Unary (op, subSubst a)
                        | Binary (op, a, b) -> Binary (op, subSubst a, subSubst b)
                        | Nary (op, es) -> Nary (op, es |> List.map subSubst)

                substituted.[expr] <- subst
                subst
        subSubst expr |> check

    /// True if expression is zero.
    /// False does not indicate that expression is non-zero.
    let rec isZero expr =
        match expr with
        | Leaf (ScalarConst ConstZero) -> true
        | Unary (Reshape _, a) -> isZero a
        | Unary (DoBroadcast _, a) -> isZero a
        | Unary (PermuteAxes _, a) -> isZero a
        | _ -> false

    /// Matches expressions with value zero.
    let (|ZeroExpr|_|) expr =
        if isZero expr then Some () else None

    /// counts operators, not counting repeating subexpressions
    let countUniqueOps expr  =
        let visited = HashSet<ExprT> (HashIdentity.Structural)
        let rec doCount expr =
            if visited.Contains expr then 0
            else
                visited.Add expr |> ignore
                match expr with
                | Leaf _ -> 1
                | Unary (_, a) -> 1 + doCount a
                | Binary (_, a, b) -> 1 + doCount a + doCount b
                | Nary (_, es) -> 1 + List.sumBy doCount es
        doCount expr

    /// counts operators, including repeating subexpressions
    let rec countOps expr  =
        match expr with
        | Leaf _ -> 1
        | Unary (_, a) -> 1 + countOps a
        | Binary (_, a, b) -> 1 + countOps a + countOps b
        | Nary (_, es) -> 1 + List.sumBy countOps es

    /// scalar constant of given value
    let scalar f = 
        Leaf (ScalarConst (ConstSpec.ofValue f)) |> check

    /// scalar of given value converted to same type as given expression
    let scalarOfSameType expr f = 
        let tn = typename expr
        let v = System.Convert.ChangeType (box f, TypeName.getType tn)
        scalar v

    /// scalar 0 of the same type as given expression
    let inline zeroOfSameType expr = scalarOfSameType expr 0

    /// scalar 1 of the same type as given expression
    let inline oneOfSameType expr = scalarOfSameType expr 1

    /// scalar 2 of the same type as given expression
    let inline twoOfSameType expr = scalarOfSameType expr 2

    /// scalar with value of given size converted to the given type
    [<RequiresExplicitTypeArguments>]
    let sizeValue<'T> size = 
        Leaf (SizeValue (size, TypeName.ofType<'T>)) |> check

    /// scalar with value of given size converted to the same type as given expression
    let sizeValueOfSameType expr size = 
        Leaf (SizeValue (size, typename expr)) |> check

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    let permuteAxes perm a =
        Unary (PermuteAxes perm, a) |> check

    /// swaps two dimensions of a tensor
    let swapDim ax1 ax2 a = 
        a |> checkAxis ax1
        a |> checkAxis ax2
        if ax1 = ax2 then a
        else
            let perm = 
                [0 .. nDims a - 1]
                |> List.map (function
                             | d when d=ax1 -> ax2
                             | d when d=ax2 -> ax1
                             | d -> d)
            a |> permuteAxes perm

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    let transpose a =
        let nd = shapeOf a |> ShapeSpec.nDim
        if nd < 2 then invalidArg "a" "need at least a matrix to transpose"
        swapDim (nd-2) (nd-1) a

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    let constructElementwise op a b =
        let sa, sb = shapeOf a, shapeOf b
        let psa, psb = ShapeSpec.padToSame sa sb
        let bsa, bsb = ShapeSpec.broadcastToSame false psa psb
        let ba = a |> reshapeIfNecessary psa |> broadcastIfNecessary bsa
        let bb = b |> reshapeIfNecessary psb |> broadcastIfNecessary bsb    
        Binary (op, ba, bb) |> check

    /// pads from the left and broadcasts the argument to the given shape if possible
    let broadcastToShape shp a =
        let sa = shapeOf a
        let psa = sa |> ShapeSpec.padTo (nDim shp)
        let bsa = psa |> ShapeSpec.broadcastToShape shp
        a |> reshapeIfNecessary psa |> broadcastIfNecessary bsa        

    /// pads and broadcasts all arguments to same shape if possible
    let broadcastToSameMany es =
        let ss = es |> List.map shapeOf
        let ps = ShapeSpec.padToSameMany ss
        let bs = ShapeSpec.broadcastToSameMany false ps
        List.zip3 es ps bs
        |> List.map (fun (e, p, b) -> e |> reshapeIfNecessary p |> broadcastIfNecessary b)

    /// pads and broadcasts `a` and `b` to same shape if possible
    let broadcastToSame a b =
        match broadcastToSameMany [a; b] with
        | [bcA; bcB] -> bcA, bcB
        | _ -> failwith "impossible"

    /// select elements according to the specified index arrays
    let gather indices a =
        let someIndices = indices |> List.choose id
        if List.isEmpty someIndices then
            failwith "need to specify at least one index array"
        let bcSomeIndices = broadcastToSameMany someIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs ->
                Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSomeIndices
        Unary (Gather bcIndices, a) |> check

    /// select elements according to the specified index arrays
    let scatter indices trgtShp a =
        let aShp = shapeOf a
        let indices = indices |> List.map (Option.map (broadcastToShape aShp))
        Unary (Scatter (indices, trgtShp), a) |> check


    // elementwise operators
    type ExprT with

        // elementwise unary
        static member (~+) (a: ExprT) = a |> check
        static member (~-) (a: ExprT) = Unary(Negate, a) |> check 
        static member Abs (a: ExprT) = Unary(Abs, a) |> check
        static member SignT (a: ExprT) = Unary(SignT, a) |> check
        static member Log (a: ExprT) = Unary(Log, a) |> check
        static member Log10 (a: ExprT) = Unary(Log10, a) |> check
        static member Exp (a: ExprT) = Unary(Exp, a) |> check
        static member Sin (a: ExprT) = Unary(Sin, a) |> check
        static member Cos (a: ExprT) = Unary(Cos, a) |> check
        static member Tan (a: ExprT) = Unary(Tan, a) |> check
        static member Asin (a: ExprT) = Unary(Asin, a) |> check
        static member Acos (a: ExprT) = Unary(Acos, a) |> check
        static member Atan (a: ExprT) = Unary(Atan, a) |> check
        static member Sinh (a: ExprT) = Unary(Sinh, a) |> check
        static member Cosh (a: ExprT) = Unary(Cosh, a) |> check
        static member Tanh (a: ExprT) = Unary(Tanh, a) |> check
        static member Sqrt (a: ExprT) = Unary(Sqrt, a) |> check
        static member Ceiling (a: ExprT) = Unary(Ceil, a) |> check
        static member Floor (a: ExprT) = Unary(Floor, a) |> check
        static member Round (a: ExprT) = Unary(Round, a) |> check
        static member Truncate (a: ExprT) = Unary(Truncate, a) |> check

        // element-wise unary logic
        static member (~~~~) (a: ExprT) = Unary(Not, a) |> check

        // elementwise binary
        static member (+) (a: ExprT, b: ExprT) = constructElementwise Add a b
        static member (-) (a: ExprT, b: ExprT) = constructElementwise Substract a b
        static member (*) (a: ExprT, b: ExprT) = constructElementwise Multiply a b
        static member (/) (a: ExprT, b: ExprT) = constructElementwise Divide a b
        static member (%) (a: ExprT, b: ExprT) = constructElementwise Modulo a b
        static member Pow (a: ExprT, b: ExprT) = constructElementwise Power a b    
        static member ( *** ) (a: ExprT, b: ExprT) = a ** b

        // element-wise binary logic
        static member (&&&&) (a: ExprT, b: ExprT) = constructElementwise And a b
        static member (||||) (a: ExprT, b: ExprT) = constructElementwise Or a b

        // element-wise binary comparison
        static member (====) (a: ExprT, b: ExprT) = constructElementwise Equal a b
        static member (<<<<) (a: ExprT, b: ExprT) = constructElementwise Less a b
        static member (<<==) (a: ExprT, b: ExprT) = constructElementwise LessEqual a b
        static member (>>>>) (a: ExprT, b: ExprT) = constructElementwise Greater a b
        static member (>>==) (a: ExprT, b: ExprT) = constructElementwise GreaterEqual a b
        static member (<<>>) (a: ExprT, b: ExprT) = constructElementwise NotEqual a b

        // elementwise binary with basetype
        static member (+) (a: ExprT, b: System.IComparable) = a + (scalar b)
        static member (-) (a: ExprT, b: System.IComparable) = a - (scalar b)
        static member (*) (a: ExprT, b: System.IComparable) = a * (scalar b)
        static member (/) (a: ExprT, b: System.IComparable) = a / (scalar b)
        static member (%) (a: ExprT, b: System.IComparable) = a % (scalar b)
        static member Pow (a: ExprT, b: System.IComparable) = a ** (scalar b)
        static member ( *** ) (a: ExprT, b: System.IComparable) = a ** (scalar b)
        static member (====) (a: ExprT, b: System.IComparable) = constructElementwise Equal a (scalar b)
        static member (<<<<) (a: ExprT, b: System.IComparable) = constructElementwise Less a (scalar b)
        static member (<<==) (a: ExprT, b: System.IComparable) = constructElementwise LessEqual a (scalar b)
        static member (>>>>) (a: ExprT, b: System.IComparable) = constructElementwise Greater a (scalar b)
        static member (>>==) (a: ExprT, b: System.IComparable) = constructElementwise GreaterEqual a (scalar b)
        static member (<<>>) (a: ExprT, b: System.IComparable) = constructElementwise NotEqual a (scalar b)

        static member (+) (a: System.IComparable, b: ExprT) = (scalar a) + b
        static member (-) (a: System.IComparable, b: ExprT) = (scalar a) - b
        static member (*) (a: System.IComparable, b: ExprT) = (scalar a) * b
        static member (/) (a: System.IComparable, b: ExprT) = (scalar a) / b
        static member (%) (a: System.IComparable, b: ExprT) = (scalar a) % b
        static member Pow (a: System.IComparable, b: ExprT) = (scalar a) ** b
        static member ( *** ) (a: System.IComparable, b: ExprT) = (scalar a) ** b
        static member (====) (a: System.IComparable, b: ExprT) = constructElementwise Equal (scalar a) b
        static member (<<<<) (a: System.IComparable, b: ExprT) = constructElementwise Less (scalar a) b
        static member (<<==) (a: System.IComparable, b: ExprT) = constructElementwise LessEqual (scalar a) b
        static member (>>>>) (a: System.IComparable, b: ExprT) = constructElementwise Greater (scalar a) b
        static member (>>==) (a: System.IComparable, b: ExprT) = constructElementwise GreaterEqual (scalar a) b
        static member (<<>>) (a: System.IComparable, b: ExprT) = constructElementwise NotEqual (scalar a) b

        // transposition
        member this.T = transpose this
    

    /// sign keeping type
    let signt (a: ExprT) =
        ExprT.SignT a 

    /// square root
    let sqrtt (a: ExprT) =
        ExprT.Sqrt a

    /// elementwise uses elements from ifTrue if cond is true, 
    /// otherwise elements from ifFalse
    let ifThenElse cond ifTrue ifFalse =
        let shps = [shapeOf cond; shapeOf ifTrue; shapeOf ifFalse]
        let pShps = ShapeSpec.padToSameMany shps
        let bcShps = ShapeSpec.broadcastToSameMany false pShps           
        match pShps, bcShps with
        | [condPShp; ifTruePShp; ifFalsePShp], [condBcShp; ifTrueBcShp; ifFalseBcShp] -> 
            let condBc = cond |> reshapeIfNecessary condPShp |> broadcastIfNecessary condBcShp
            let ifTrueBc = ifTrue |> reshapeIfNecessary ifTruePShp |> broadcastIfNecessary ifTrueBcShp
            let ifFalseBc = ifFalse |> reshapeIfNecessary ifFalsePShp |> broadcastIfNecessary ifFalseBcShp
            Binary (IfThenElse condBc, ifTrueBc, ifFalseBc) |> check
        | _ -> failwith "impossible"

    /// elementwise maximum
    let maxElemwise a b =
        constructElementwise MaxElemwise a b

    /// elementwise minimum
    let minElemwise a b =
        constructElementwise MinElemwise a b

    /// Ensures that all elements are between Some minVal and Some maxVal.
    let cage (minVal, maxVal) a =
        let a =
            match minVal with
            | Some mv -> maxElemwise (scalar mv) a
            | None -> a
        let a =
            match maxVal with
            | Some mv -> minElemwise (scalar mv) a
            | None -> a
        a

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

    /// Replicates the tensor the given number of repetitions along the given axis.
    let replicate dim reps a =
        a |> checkAxis dim

        // 1. insert axis of size one left to repetition axis
        // 2. broadcast along the new axis to number of repetitions
        // 3. reshape to result shape
        a 
        |> insertBroadcastAxis dim
        |> broadcast (a.Shape |> ShapeSpec.insertAxis dim reps)
        |> reshape (a.Shape |> List.set dim (reps * a.Shape.[dim]))

    /// Replicates the tensor along the given axis, so that after replication it has
    /// the specified `size`. If `size` is not a multiple of the current size of the
    /// tensor along the specified axis, the last replication is truncated appropriately.
    let replicateTo dim size a =
        Unary (Held ([], ReplicateTo (dim, size)), a) |> check

    /// summaiton of all elements
    let sum a = Unary(Sum, a) |> check

    /// summation over given dimension
    let sumAxis ax a = Unary(SumAxis(ax), a) |> check

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    let sumKeepingAxis ax a =
        a |> sumAxis ax |> insertBroadcastAxis ax

    /// product of all elements
    let product a = Unary(Product, a) |> check

    /// product over given dimension
    let productAxis ax a = Unary(ProductAxis(ax), a) |> check

    /// product over given dimension, while keeping the axis with one (broadcastable) element
    let productKeepingAxis ax a =
        a |> productAxis ax |> insertBroadcastAxis ax

    /// maximum over given dimension
    let maxAxis ax a = Unary(MaxAxis(ax), a) |> check

    /// maximum over given dimension, while keeping the axis with one (broadcastable) element
    let maxKeepingAxis ax a =
        a |> maxAxis ax |> insertBroadcastAxis ax

    /// maximum over given dimension
    let minAxis ax a = Unary(MinAxis(ax), a) |> check

    /// maximum over given dimension, while keeping the axis with one (broadcastable) element
    let minKeepingAxis ax a =
        a |> minAxis ax |> insertBroadcastAxis ax

    /// index of maximum over given dimension
    let argMaxAxis ax a = Unary(ArgMaxAxis(ax), a) |> check

    /// index of maximum over given dimension, while keeping the axis with one (broadcastable) element
    let argMaxKeepingAxis ax a =
        a |> argMaxAxis ax |> insertBroadcastAxis ax

    /// index of maximum over given dimension
    let argMinAxis ax a = Unary(ArgMinAxis(ax), a) |> check

    /// index of maximum over given dimension, while keeping the axis with one (broadcastable) element
    let argMinKeepingAxis ax a =
        a |> argMinAxis ax |> insertBroadcastAxis ax

    /// mean over all elements
    let mean (a: ExprT) = 
        sum a / sizeValueOfSameType a a.NElems

    /// mean over given dimension
    let meanAxis ax (a: ExprT) =
        sumAxis ax a / sizeValueOfSameType a a.Shape.[ax]

    /// mean over given dimension, while keeping the axis with one (broadcastable) element
    let meanKeepingAxis ax a =
        a |> meanAxis ax |> insertBroadcastAxis ax

    /// identity matrix of given size
    [<RequiresExplicitTypeArguments>]
    let identity<'T> size = 
        Leaf(Identity(size, TypeName.ofType<'T>)) |> check

    /// identity matrix of given size and same type as given expression
    let identityOfSameType expr size =
        Leaf(Identity(size, typename expr)) |> check

    /// tensor of given shape filled with specified value
    let filled (shp: ShapeSpecT) value =
        let bcShp = shp |> List.map (fun _ -> SizeSpec.broadcastable)
        scalar value
        |> reshape bcShp
        |> broadcast shp

    /// zero tensor of given shape
    [<RequiresExplicitTypeArguments>]
    let zeros<'T> (shp: ShapeSpecT) =
        filled shp (conv<'T> 0)

    /// zero tensor of given type and shape
    let zerosOfType typ shp =
        filled shp (convTo typ 0)

    /// zero tensor of given shape and same type as given expression
    let zerosOfSameType expr shp =
        let zero = System.Convert.ChangeType (box 0, (typename expr).Type)
        filled shp zero

    /// zero tensor with same shape and type as given tensor
    let zerosLike expr = 
        zerosOfSameType expr expr.Shape

    /// variable of given name and shape
    [<RequiresExplicitTypeArguments>]
    let var<'T> name (ss: ShapeSpecT) = 
        Leaf(Var({Name=name; Shape=ss; TypeName=TypeName.ofType<'T>})) |> check

    /// variable of given name, type and shape
    let varOfType name typ (ss: ShapeSpecT) = 
        Leaf(Var({Name=name; Shape=ss; TypeName=TypeName.ofTypeInst typ})) |> check

    /// Vector counting from zero to given size minus one.
    [<RequiresExplicitTypeArguments>]
    let arange<'T> size =
        Leaf(Arange(size, TypeName.ofType<'T>)) |> check

    /// Vector counting from zero to given size minus one.
    let arangeOfType shp typ =
        Leaf(Arange(shp, TypeName.ofTypeInst typ)) |> check

    /// annotated expression
    let annotate ano a = 
        Unary(Annotated(ano), a) |> check

    /// adds one broadcastable dimension to the left
    let padLeft a =
        let sa = shapeOf a
        reshape (ShapeSpec.padLeft sa) a

    /// adds one broadcastable dimension to the right
    let padRight a =
        let sa = shapeOf a
        reshape (ShapeSpec.padRight sa) a

    /// Dot product.
    /// Behavior depends on the dimensionality of the arguments.
    /// Cases: 
    /// (1, 1) -> vector-vector dot product resulting in a scalar
    /// (2, 1) -> matrix-vector dot product resulting in a vector
    /// (2, 2) -> matrix-matrix dot product resulting in a matrix
    /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
    /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
    let dot (a: ExprT) (b: ExprT) =
        let sa, sb = shapeOf a, shapeOf b
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 1, 1 -> 
                // vector-vector dot product
                sum (a * b)
            | 2, 1 -> 
                // matrix-vector dot product
                let bm = b |> reshape (ShapeSpec.padRight sb)
                Binary(Dot, a, bm) |> reshape [sa.[0]]
            | 2, 2 -> 
                // matrix-matrix dot product
                Binary(Dot, a, b)
            | na, nb when na = nb -> 
                // batched matrix-matrix dot product
                let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa sb
                let ba = a |> broadcastIfNecessary bsa
                let bb = b |> broadcastIfNecessary bsb    
                Binary(Dot, ba, bb)
            | na, nb when na = nb + 1 ->
                // batched matrix-vector dot product
                let psb = ShapeSpec.padRight sb
                let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa psb
                let ba = a |> broadcastIfNecessary bsa
                let bb = b |> reshapeIfNecessary psb |> broadcastIfNecessary bsb    
                Binary(Dot, ba, bb) |> reshape bsa.[0 .. na-2]
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        |> check

    /// tensor product
    let tensorProduct (a: ExprT) (b: ExprT) =
        let sa, sb = shapeOf a, shapeOf b
        let psa, psb = ShapeSpec.padToSame sa sb
        let a, b = reshapeIfNecessary psa a, reshapeIfNecessary psb b
        Binary(TensorProduct, a, b) |> check

    type ExprT with
        // tensor binary

        /// Dot product.
        /// Behavior depends on the dimensionality of the arguments.
        /// Cases: 
        /// (1, 1) -> vector-vector dot product resulting in a scalar
        /// (2, 1) -> matrix-vector dot product resulting in a vector
        /// (2, 2) -> matrix-matrix dot product resulting in a matrix
        /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
        /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
        static member (.*) (a: ExprT, b: ExprT) = dot a b
        static member (%*) (a: ExprT, b: ExprT) = tensorProduct a b

    /// extract VarSpec from variable expression
    let extractVar expr = 
        match expr with
        | Leaf(Var(v)) -> v
        | _ -> invalidArg "expr" "not a expr consisting solely of a variable"

    /// make variable expression from VarSpec
    let makeVar vs =
        Leaf(Var(vs)) |> check

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
        | Unary (Reshape _, (Unary (Subtensor srs, t) as st)) ->
            let stShp = shapeOf st
            Binary (SetSubtensor srs, t, Unary (Reshape stShp, b)) |> check
        | _ ->
            invalidArg "a" "the first argument of setSubtensor must be an item or slice of an expression, i.e. a.[...]"

    type ExprT with
        // item / slicing
        member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =

            /// converts ints to SizeSpecTs
            let intToSizeSpec (arg: obj) =
                match arg with
                | :? int64 as f -> SizeSpec.fix f :> obj
                | :? (int64 option) as fo -> 
                    match fo with
                    | Some f -> Some (SizeSpec.fix f) :> obj
                    | None -> None :> obj
                | _ -> arg

            /// converts argument list to range specification
            let rec parseArgs (args: obj list) : FullExprRngsSpecT =
                match args with
                // direct range specification
                | [:? FullExprRngsSpecT as rngs] -> rngs

                // slices
                | (:? (SizeSpecT option) as so)  :: (:? (SizeSpecT option) as fo)    :: rest ->
                    RSSymStartSymEnd (so, fo) :: parseArgs rest
                | (:? (SizeSpecT option) as so)  :: null                             :: rest ->
                    RSSymStartSymEnd (so, None) :: parseArgs rest
                | null                           :: (:? (SizeSpecT option) as fo)    :: rest ->
                    RSSymStartSymEnd (None, fo) :: parseArgs rest
                | (:? (ExprT option) as so)      :: (:? (PlusElems option) as fo)    :: rest ->
                    if typename so.Value <> TypeName.ofType<int> then
                        failwith "need int expression for range start"
                    RSDynStartSymSize (so.Value, fo.Value.Elems) :: parseArgs rest
                | null                           :: null                             :: rest ->
                    RSSymStartSymEnd (None, None) :: parseArgs rest

                // items
                | (:? SizeSpecT as s)     :: rest -> RSSymElem s :: parseArgs rest
                | (:? int64 as s)         :: rest when s = NewAxis -> RSNewAxis :: parseArgs rest
                | (:? int64 as s)         :: rest when s = Fill ->    RSAllFill :: parseArgs rest
                | (:? ExprT as e)         :: rest -> if typename e <> TypeName.ofType<int> then
                                                         failwith "need int expression for element"               
                                                     RSDynElem e :: parseArgs rest
                | []                              -> []
                | _                               -> failwithf "invalid item/slice specification: %A" allArgs

            /// converts a full range specification into a simple range specification
            let rec splitFRS (rngs: FullExprRngsSpecT) (shps: ShapeSpecT) (simpleRs: ExprRngsSpecT) (newShape: ShapeSpecT) =
                match rngs, shps with
                | RSSymElem e :: rngs, _::shps -> splitFRS rngs shps (SRSSymStartSymEnd (e, Some e)::simpleRs) newShape
                | RSDynElem e :: rngs, _::shps -> splitFRS rngs shps (SRSDynStartSymSize (e, SizeSpec.one)::simpleRs) newShape
                | RSSymStartSymEnd (so, fo) :: rngs, size::shps -> 
                    let size = (fo |? (size-1L)) - (so |? SizeSpec.zero) + 1L
                    splitFRS rngs shps (SRSSymStartSymEnd (so |? SizeSpec.zero, fo)::simpleRs) (size::newShape)
                | RSDynStartSymSize (s, size) :: rngs, _::shps ->
                    splitFRS rngs shps (SRSDynStartSymSize (s, size)::simpleRs) (size::newShape)
                | RSNewAxis :: rngs, _ ->
                    splitFRS rngs shps simpleRs (SizeSpec.broadcastable::newShape)
                | RSAllFill :: rrngs, _ ->
                    if List.length rngs <= List.length shps then splitFRS (RSAll::rngs) shps simpleRs newShape
                    else splitFRS rrngs shps simpleRs newShape
                | [], [] -> List.rev simpleRs, List.rev newShape
                | _ -> failwith "item/slice processing error"

            // build full range specificaton
            let argList = allArgs |> Array.toList |> List.map intToSizeSpec

            let srs, reshp = 
                match argList with
                | [:? ExprRngsSpecT as srs] -> 
                    // simplified range specification was specified, use directly
                    srs, shapeOf (Unary (Subtensor srs, this))
                | [:? FullExprRngsSpecT as frs] ->
                    // split into simplified range specification and reshape operation
                    splitFRS frs (shapeOf this) [] []
                | _ ->
                    // parse, then split into simplified range specification and reshape operation
                    splitFRS (argList |> parseArgs) (shapeOf this) [] []

            // emit expression
            Unary (Reshape reshp, Unary (Subtensor srs, this))  
            |> check

        member this.Item 
            with get ([<System.ParamArray>] allArgs: obj []) = 
                this.GetSlice (allArgs)
                   
    /// Extracts the diagonal along the given axes.
    let diagAxis ax1 ax2 a = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        Unary(Diag (ax1, ax2), a) |> check
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    let diag a = 
        let nd = shapeOf a |> ShapeSpec.nDim
        if nd < 2 then failwith "need at least a matrix to extract diagonal"
        diagAxis (nd-2) (nd-1) a

    /// Creates a diagonal matrix by duplicating the given dimension.
    let diagMatAxis ax1 ax2 a = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        Unary(DiagMat (ax1, ax2), a) |> check

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    let diagMat a =
        let nd = shapeOf a |> ShapeSpec.nDim
        if nd < 1 then failwith "need at least a vector to create diagonal matrix"
        diagMatAxis (nd-1) nd a

    /// Computes the traces along the given axes.
    let traceAxis ax1 ax2 a =
        let tax = if ax1 < ax2 then ax1 else ax1 + 1
        a |> diagAxis ax1 ax2 |> sumAxis tax

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    let trace a =
        let nd = shapeOf a |> ShapeSpec.nDim
        if nd < 2 then
            failwith "need at least a two dimensional array for trace"      
        traceAxis (nd-2) (nd-1) a

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    let invert a =
        Unary(Invert, a) |> check

    /// calculates a tensor elementwise using the given element expression and
    /// result shape
    let elements trgtShp elemExpr args =
        Nary (Elements (trgtShp, elemExpr), args) |> check

    /// nullifies the Jacobian when calculating derivatives
    let assumeZeroDerivative expr =
        Unary (NullifyJacobian, expr) |> check

    /// assumes the specified Jacobian when calculating derivatives
    let assumeJacobian jac expr =
        Unary (AssumeJacobian jac, expr) |> check

    /// print the result with the given message when evaluated
    let print msg a =
        Unary (Print msg, a) |> check

    /// dumps the result into the active dump session HDF5 file
    let dump name a =
        Unary (Dump name, a) |> check

    /// checks the value for NaNs and infinities, outputs their location and stops the computation
    let checkFinite name a =
        if Debug.EnableCheckFinite then
            Unary (CheckFinite name, a) |> check
        else a |> check

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    let interpolate interpolator e =
        let e = broadcastToSameMany e
        Nary (Interpolate interpolator, e) |> check

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    let interpolate1D interpolator a =
        interpolate interpolator [a]

    /// Element-wise two-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    let interpolate2D interpolator a b =
        interpolate interpolator [a; b]

    /// Element-wise three-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    let interpolate3D interpolator a b c =
        interpolate interpolator [a; b; c]
   
    /// A loop provides iterative evaluation of one or multiple expresisons.
    /// All variables occurs in the loop channel expressions must be defined as loop variables.
    /// The function `loop` performs automatic lifting of constants and thus allows for easy
    /// usage of variables external to the loop.
    let loopNoLift spec channel args =
        Nary (Channel (Loop spec, channel), args) |> check

    /// A loop provides iterative evaluation of one or multiple expresisons.
    let loop spec channel args =       
        let mutable args = args
        let mutable vars = spec.Vars

        /// adds an argument and returns its index
        let addArg (expr: ExprT) =
            match args |> List.tryFindIndex ((=) expr) with
            | Some argIdx -> argIdx
            | None ->
                let argIdx = args.Length
                args <- args @ [expr]
                argIdx

        /// adds a constant variable, its required argument and returns the associated VarSpecT
        let addConstVar (expr: ExprT) =
            match vars |> Map.tryFindKey (fun vs lv ->
                                           match lv with
                                           | ConstArg argIdx when args.[argIdx] = expr -> true
                                           | _ -> false) with
            | Some vs -> vs
            | None ->
                let rec genName i =
                    let name = sprintf "CONST%d" i
                    match vars |> Map.tryFindKey (fun vs _ -> vs.Name = name) with
                    | Some _ -> genName (i + 1)
                    | None -> name
                let vs = VarSpec.ofNameShapeAndTypeName (genName 0) expr.Shape expr.TypeName
                let lv = ConstArg (addArg expr)
                vars <- vars |> Map.add vs lv
                vs

        let loopVarSet = vars |> Map.toSeq |> Seq.map (fun (vs, _) -> vs) |> Set.ofSeq
        let lifted = Dictionary<ExprT, ExprT> ()

        let rec lift expr =
            match lifted.TryFind expr with
            | Some rep -> rep
            | None ->
                let exprVars = extractVars expr
                let dependsOnVars = not (Set.isEmpty exprVars)
                let dependsOnLoopVars = Set.intersect exprVars loopVarSet |> Set.isEmpty |> not
                let rep =
                    if dependsOnVars && not dependsOnLoopVars then
                        //if not (dependsOnLoopVars expr) then
                        let vs = addConstVar expr
                        makeVar vs
                    else
                        match expr with                   
                        | Unary (Gather indices, a) ->
                            Unary (Gather (indices |> List.map (Option.map lift)), lift a)
                        | Unary (Scatter (indices, trgtShp), a) ->
                            Unary (Scatter (indices |> List.map (Option.map lift), trgtShp), lift a)
                        | Binary (IfThenElse cond, a, b) ->
                            Binary (IfThenElse (lift cond), lift a, lift b)

                        | Leaf _ -> expr
                        | Unary (op, a) -> Unary (op, lift a)
                        | Binary (op, a, b) -> Binary (op, lift a, lift b)
                        | Nary (op, es) -> Nary (op, es |> List.map lift)
                lifted.[expr] <- rep
                rep
                
        // lift constants out of loop
        let liftedChannels = spec.Channels |> Map.map (fun ch lv -> {lv with Expr = lift lv.Expr})
        let spec = {spec with Channels = liftedChannels; Vars = vars}            

        loopNoLift spec channel args

    /// reverses the tensor in the given dimension 
    let reverseAxis dim (a: ExprT) : ExprT =
        Unary (ReverseAxis dim, a) |> check

    /// concatenates the sequence of tensors in the specified dimension
    let concat dim (es: ExprT seq) =
        // check that arguments are correctly sized
        let es = List.ofSeq es
        let shps = es |> List.map shapeOf
        match es with
        | [] -> failwithf "need at least one tensor to concatenate"
        | h :: ts ->
            if not (0 <= dim && dim < h.NDims) then
                failwithf "cannot concatenate over non-existant dimension %d given shapes %A" dim shps
            for t in ts do
                if t.Type <> h.Type then
                    failwithf "all arguments must have same type but got types %A" (es |> List.map (fun e -> e.Type))
                if t.NDims <> h.NDims then
                    failwithf "all arguments must have same number of dimensions but shapes %A were specifed" shps                        
                for i, (sa, sb) in List.indexed (List.zip h.Shape t.Shape) do
                    if i <> dim && sa .<> sb then
                        failwithf "all arguments must have same shape expect in concatenation dimension %d but \
                                   shapes %A were specified" dim shps
                    
        // calculate shape of concatenation
        let totalElems = es |> Seq.sumBy (fun e -> e.Shape.[dim])
        let shp = es.Head.Shape |> ShapeSpec.set dim totalElems

        // build concatenation using iterative subtensor replacement
        let concatenated, _ =
            ((zerosOfSameType es.Head shp, SizeSpec.zero), es)
            ||> List.fold (fun (concatSoFar, pos) e ->
                let len = e.Shape.[dim]
                let slice : FullExprRngsSpecT = 
                    List.replicate e.NDims RSAll
                    |> List.set dim (RSSymStartSymEnd (Some pos, Some (pos + len - 1L)))
                setSubtensor concatSoFar.[slice] e, pos + len)
        concatenated

    /// Build tensor from numeric ranges.
    let internal buildTensor shp rngs srcs =
        Nary (BuildTensor (shp, rngs), srcs) |> check


[<AutoOpen>]
module ExprTypes =
    type ArityT = Expr.ArityT
    type LeafOpT = Expr.LeafOpT
    type UnaryOpT = Expr.UnaryOpT
    type BinaryOpT = Expr.BinaryOpT
    type NaryOpT = Expr.NaryOpT
    type IOp = Expr.IOp
    type ExprT = Expr.ExprT
    type MultiChannelOpT = Expr.MultiChannelOpT
    type ChannelT = Expr.ChannelT
    type UnaryHeldOpT = Expr.UnaryHeldOpT
    type SequenceArgSliceT = Expr.SequenceArgSliceT
    type PreviousChannelT = Expr.PreviousChannelT
    type LoopInputT = Expr.LoopInputT
    type LoopValueT = Expr.LoopValueT
    type LoopSpecT = Expr.LoopSpecT

    




