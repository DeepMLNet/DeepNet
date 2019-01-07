namespace rec SymTensor

(**

open DeepNet.Utils


module private Cache =
    /// cache for ExprT hashes by reference
    let hash = Dictionary<obj, int> (HashIdentity.Reference)

    let shape = ConcurrentDictionary<Expr, ShapeSpec> (HashIdentity.Reference)

    /// expressions that were already checked for correctness
    let checkedExprs = HashSet<Expr> (HashIdentity.Reference)

    let exprsWithEvalableSymSizes = HashSet<Expr> ()

    /// Caches for extracted variables.
    let extractedVars = Dictionary<Expr, Set<Var>> () 


/// start plus the specified number of (symbolic elements)
type internal PlusElems (elems: SizeSpec) =
    new (intElems: int64) = PlusElems (SizeSpec.fix intElems)
    member this.Elems = elems


/// A mathematical operation in an expression.
/// This models a mathematical function or operator that takes one or more tensors
/// and returns one tensor.
type IOp =
    inherit System.IComparable
      
    /// Should return the type of the result, given the types of the arguments.
    abstract TypeName: argTypes: TypeName list -> TypeName

    /// Should return the shape of the result, given the shape of the arguments.
    abstract Shape: argShapes: ShapeSpec list -> ShapeSpec      
        
    /// Should check if the shapes of the arguments are acceptable and,
    /// if not, raise an exception.
    abstract CheckArgs: argShapes: ShapeSpec list -> unit      

    /// Should return the op with all symbolic sizes substituted using the specified
    /// substitution table.
    /// Return a *new* op with substitution applied. Do not apply the mapping in-place.
    abstract SubstSymSizes: symSizes: SymSizeEnv -> IOp

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
    abstract Deriv: dOp:Expr -> args:Expr list -> Expr list

    /// Should evaluate the numerical value of this op given the numerical values of its arguments.
    /// This evaluation should be done on the host using the simplest means possible and is used
    /// as a reference implementation for verifying the correctness of optimized (e.g. CUDA) 
    /// implementations. This method may be omitted when no verification will be done.
    abstract EvalSimple: args:Tensor.Tensor<'T> list -> Tensor.Tensor<'T>

    /// Should return the set of variables that this op instance depends on.
    abstract ContainedVars: Set<Var>

/// ops with no exprs as arguments
[<StructuralComparison; StructuralEquality>]
type LeafOp =

    // ==== scalars ============
    /// scalar of given value
    | ScalarConst of value:Const // DONE
    /// scalar of the given size
    | SizeValue of value:SizeSpec * typ:TypeName // DONE

    // ==== tensor creation ====
    /// tensor with 1 on diagonal of given shape
    | Identity of shape:SizeSpec * typ:TypeName // DONE
    /// vector counting from zero to given size minus one
    | Arange of size:SizeSpec * typ:TypeName // DONE

    // ==== variable access ====
    /// variable read
    | Var of Var       // DONE
        

/// ops with one expr as argument
[<StructuralComparison; StructuralEquality>] 
type UnaryOp =

    // ==== unary elementwise ==== 
    | Negate  // DONE                        
    | Abs     // DONE
    | SignT   // DONE
    | Log     // DONE
    | Log10   // DONE                       
    | Exp     // DONE               
    | Sin     // DONE
    | Cos     // DONE
    | Tan     // DONE
    | Asin    // DONE
    | Acos    // DONE
    | Atan    // DONE
    | Sinh    // DONE
    | Cosh    // DONE
    | Tanh    // DONE
    | Sqrt    // DONE
    | Ceil    // DONE
    | Floor   // DONE
    | Round   // DONE
    | Truncate// DONE

    // ==== element-wise unary logic ====
    | Not     // DONE

    // ==== tensor operations ====
    /// extract diagonal along given axes
    | Diag of int * int             // DONE
    /// build diagonal matrix along given axes
    | DiagMat of int * int          // DONE
    /// matrix inverse
    | Invert                        // DONE

    // ==== reductions ====
    /// summation of all elements
    | Sum                           // removed
    /// summation over given dimension
    | SumAxis of int                // DONE
    /// product of all elements
    | Product                       // removed
    /// product over given dimension
    | ProductAxis of int            // DONE
    /// maximum over given dimension                
    | MaxAxis of int                // DONE
    /// minimum over given dimension
    | MinAxis of int                // DONE

    // ==== index reductions ====
    /// inidices of maximums over given dimension
    | ArgMaxAxis of int             // DONE
    /// inidices of minimums over given dimension
    | ArgMinAxis of int             // DONE

    // ==== shape operations ====
    /// reshape tensor; element count does not change
    | Reshape of ShapeSpec          // DONE
    /// broadcast tensor; element count may change
    | DoBroadcast of ShapeSpec      // DONE
    /// permutes the axes of the tensor
    | PermuteAxes of perm:int list  // DONE
    /// subtensor 
    | Subtensor of ExprRngsSpec     // DONE 
    /// reverses the tensor in the given dimension 
    | ReverseAxis of dim:int        // DONE
    /// select elements according to the specified index arrays
    | Gather of indices:Expr option list  // DONE
    /// disperses elements according to the specified index arrays
    | Scatter of indices:Expr option list * shp:ShapeSpec // DONE

    // ==== variable storage ====
    /// variable write
    | StoreToVar of Var             // DONE

    // ==== misc ====
    /// nullifies the Jacobian of its argument when calculating derivatives
    | NullifyJacobian               // DONE
    /// assumes the specified Jacobian for its argument when calculating derivatives
    | AssumeJacobian of Expr        // DONE
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
    | Held of derivsShp:ShapeSpec list * op:UnaryHeldOp

/// an op that will expand into an expression once symbolic sizes have been substituted
type UnaryHeldOp =
    /// replicates the axes to the specified size
    | ReplicateTo of dim:int * size:SizeSpec

/// a simplified range specification of one dimension
type ExprRngSpec = SimpleRangeSpec<Expr>

/// a simplified range specification of all dimensions
type ExprRngsSpec = SimpleRangesSpec<Expr>

type FullExprRngSpecT = RangeSpec<Expr>
type FullExprRngsSpecT = RangesSpec<Expr>


/// ops with two exprs as arguments
[<StructuralComparison; StructuralEquality>] 
type BinaryOp =

    // ==== binary elementwise ====
    | Add               // DONE         
    | Substract         // DONE            
    | Multiply          // DONE     
    | Divide            // DONE     
    | Modulo            // DONE
    | Power             // DONE
    | MaxElemwise       // DONE
    | MinElemwise       // DONE
           
    // ==== element-wise binary comparison ====
    | Equal             // DONE
    | Less              // DONE
    | LessEqual         // DONE
    | Greater           // DONE
    | GreaterEqual      // DONE
    | NotEqual          // DONE

    // ==== element-wise binary logic ====
    | And               // DONE
    | Or                // DONE

    // ==== element-wise conditional ====
    | IfThenElse of Expr  // DONE

    // ==== matrix/tensor operations ====
    /// matrix*matrix => matrix dot product
    | Dot              // DONE             
    /// tensor product 
    | TensorProduct    // DONE
        
    // ==== shape operations ====
    /// replace subtensor
    | SetSubtensor of ExprRngsSpec      // DONE?           

/// ops with an arbitrary exprs as arguments
[<StructuralComparison; StructuralEquality>] 
type NaryOp =

    /// evaluate all subexpressions but discard them
    | Discard        
    /// build tensor using numeric ranges
    | BuildTensor of shp:ShapeSpec * rngs:BaseRangesSpec list
    /// elementwise calculated tensor
    | Elements of shape:ShapeSpec * elemExpr:Elem.Expr
    /// elementwise interpolation
    | Interpolate of Interpolator
    /// use specified channel of a multi-channel op
    | Channel of channelOp:MultiChannelOp * channel:Channel
    /// extension op
    | ExtensionOp of IOp

/// a channel of a multi-channel op or loop
type Channel = string

/// an n-ary op with multiple output channels
type MultiChannelOp =
    /// iterative evaluation of one or multiple expresisons
    | Loop of spec:LoopSpec    
     
/// a slice of an argument to the loop
type SequenceArgSlice = {
    /// the index of the argument
    ArgIdx:     int
    /// the dimension the loop is performed over
    SliceDim:   int
}

/// references a loop channel of a previous iteration
type PreviousChannel = {
    /// the channel to use
    Channel:       Channel
    /// the delay, must be at least one
    Delay:         SizeSpec
    /// the index of the argument specifying the initial values
    InitialArg:    int
}

/// a loop variable value specification
type LoopInput = 
    /// provides the loop argument to all loop iterations
    | ConstArg of argIdx:int
    /// provides a slice of the loop argument to each loop iteration
    | SequenceArgSlice of SequenceArgSlice
    /// provides the value of a loop channel from a previous loop iteration
    | PreviousChannel of PreviousChannel
    /// provides the index of the current loop iteration (zero-based)
    | IterationIndex
    /// provides the number of remaining loop iterations after this iteration
    | IterationsRemaining

/// the value of a loop channel
type LoopValue = {
    /// the expression to compute the loop channel;
    /// it may only use variables defined in LoopSpecT.Vars
    Expr:       Expr
    /// the dimension to concatenate the results along to produce the loop output
    SliceDim:   int
}

/// A loop specification.
/// A loop provides iterative evaluation of one or multiple expresisons.
/// A loop can slice over its arguments and reference values computed in previous
/// loop iterations.
/// A loop can compute multiple values at once. Each computed values is referred to
/// as a channel.
type LoopSpec = {
    /// number of loop iterations
    Length:     SizeSpec
    /// specifies the values of the variables used in the channel value expressions,
    /// i.e. LoopValueT.Expr
    Vars:       Map<Var, LoopInput>   
    /// specifies the values of the loop channels
    Channels:   Map<Channel, LoopValue>
}



[<StructuralComparison; StructuralEqualityAttribute>] 
type private ExprProxy = 
    | ProxyLeaf of LeafOp
    | ProxyUnary of UnaryOp * Expr
    | ProxyBinary of BinaryOp * Expr * Expr
    | ProxyNary of NaryOp * (Expr list)

/// an expression
[<CustomComparison; CustomEqualityAttribute; StructuredFormatDisplay("{Pretty}")>] 
type Expr =
    | Leaf of LeafOp
    | Unary of UnaryOp * Expr
    | Binary of BinaryOp * Expr * Expr
    | Nary of NaryOp * (Expr list)

    member inline private this.Proxy = 
        match this with
        | Leaf op -> ProxyLeaf op
        | Unary (op, a) -> ProxyUnary (op, a)
        | Binary (op, a, b) -> ProxyBinary (op, a, b)
        | Nary (op, es) -> ProxyNary (op, es)

    // cache hash code using object reference
    override this.Equals other =
        match other with
        | :? Expr as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false
    interface System.IEquatable<Expr> with
        member this.Equals other = 
            if obj.ReferenceEquals (this, other) then true
            elif this.GetHashCode() <> other.GetHashCode() then false
            else 
                let knownEqualTo = Dictionary<Expr, HashSet<Expr>> (HashIdentity.Reference)
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
                                knownEqualTo.[t] <- HashSet<Expr> (HashIdentity.Reference)
                            knownEqualTo.[t].Add o |> ignore
                        eq
                treeCompare this other
    override this.GetHashCode() =
        match Cache.hash.TryFind this with
        | Some h -> h
        | None ->
            let h = hash this.Proxy
            Cache.hash.[this] <- h
            h
    interface System.IComparable<Expr> with
        member this.CompareTo other =
            compare this.Proxy other.Proxy
    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? Expr as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "cannot compare Expr to type %A" (other.GetType())

    /// converts expression to string with approximate maximum length
    member this.ToString maxLength =
        match this with
        | Leaf op -> sprintf "{%A}" op
        | Unary (op, a) -> 
            String.limited maxLength [String.Formatter (fun _ -> sprintf "{%A}" op)
                                      String.Delim " ("
                                      String.Formatter (fun ml -> a.ToString ml)
                                      String.Delim ")"]
        | Binary (op, a, b) -> 
            String.limited maxLength [String.Formatter (fun _ -> sprintf "{%A}" op)
                                      String.Delim " ("
                                      String.Formatter (fun ml -> a.ToString ml)
                                      String.Delim ", "
                                      String.Formatter (fun ml -> b.ToString ml)
                                      String.Delim ")"]
        | Nary (op, es) -> 
            String.limited maxLength [yield String.Formatter (fun _ -> sprintf "{%A}" op)
                                      yield String.Delim " ("
                                      for p, e in List.indexed es do
                                          yield String.Formatter (fun ml -> e.ToString ml)
                                          if p < List.length es - 1 then yield String.Delim ", "
                                      yield String.Delim ")"]               

    /// converts expression to string with unlimited length
    override this.ToString () = this.ToString System.Int32.MaxValue

    /// pretty string
    member this.Pretty = this.ToString 80

    /// symbolic shape
    member this.Shape = Expr.shapeOf this

    /// number of dimensions
    member this.NDims = ShapeSpec.nDim this.Shape

    /// symbolic number of elements
    member this.NElems = Expr.nElems this

    /// type name of this expression
    member this.TypeName = Expr.typename this  

    /// type of this expression
    member this.Type = this.TypeName |> TypeName.getType 

    // elementwise unary
    static member (~+) (a: Expr) = a |> Expr.check
    static member (~-) (a: Expr) = Unary(Negate, a) |> Expr.check 
    static member Abs (a: Expr) = Unary(Abs, a) |> Expr.check
    static member SignT (a: Expr) = Unary(SignT, a) |> Expr.check
    static member Log (a: Expr) = Unary(Log, a) |> Expr.check
    static member Log10 (a: Expr) = Unary(Log10, a) |> Expr.check
    static member Exp (a: Expr) = Unary(Exp, a) |> Expr.check
    static member Sin (a: Expr) = Unary(Sin, a) |> Expr.check
    static member Cos (a: Expr) = Unary(Cos, a) |> Expr.check
    static member Tan (a: Expr) = Unary(Tan, a) |> Expr.check
    static member Asin (a: Expr) = Unary(Asin, a) |> Expr.check
    static member Acos (a: Expr) = Unary(Acos, a) |> Expr.check
    static member Atan (a: Expr) = Unary(Atan, a) |> Expr.check
    static member Sinh (a: Expr) = Unary(Sinh, a) |> Expr.check
    static member Cosh (a: Expr) = Unary(Cosh, a) |> Expr.check
    static member Tanh (a: Expr) = Unary(Tanh, a) |> Expr.check
    static member Sqrt (a: Expr) = Unary(Sqrt, a) |> Expr.check
    static member Ceiling (a: Expr) = Unary(Ceil, a) |> Expr.check
    static member Floor (a: Expr) = Unary(Floor, a) |> Expr.check
    static member Round (a: Expr) = Unary(Round, a) |> Expr.check
    static member Truncate (a: Expr) = Unary(Truncate, a) |> Expr.check

    // element-wise unary logic
    static member (~~~~) (a: Expr) = Unary(Not, a) |> Expr.check

    // elementwise binary
    static member (+) (a: Expr, b: Expr) = Expr.constructElementwise Add a b
    static member (-) (a: Expr, b: Expr) = Expr.constructElementwise Substract a b
    static member (*) (a: Expr, b: Expr) = Expr.constructElementwise Multiply a b
    static member (/) (a: Expr, b: Expr) = Expr.constructElementwise Divide a b
    static member (%) (a: Expr, b: Expr) = Expr.constructElementwise Modulo a b
    static member Pow (a: Expr, b: Expr) = Expr.constructElementwise Power a b    
    static member ( *** ) (a: Expr, b: Expr) = a ** b

    // element-wise binary logic
    static member (&&&&) (a: Expr, b: Expr) = Expr.constructElementwise And a b
    static member (||||) (a: Expr, b: Expr) = Expr.constructElementwise Or a b

    // element-wise binary comparison
    static member (====) (a: Expr, b: Expr) = Expr.constructElementwise Equal a b
    static member (<<<<) (a: Expr, b: Expr) = Expr.constructElementwise Less a b
    static member (<<==) (a: Expr, b: Expr) = Expr.constructElementwise LessEqual a b
    static member (>>>>) (a: Expr, b: Expr) = Expr.constructElementwise Greater a b
    static member (>>==) (a: Expr, b: Expr) = Expr.constructElementwise GreaterEqual a b
    static member (<<>>) (a: Expr, b: Expr) = Expr.constructElementwise NotEqual a b

    // elementwise binary with basetype
    static member (+) (a: Expr, b: System.IComparable) = a + (Expr.scalar b)
    static member (-) (a: Expr, b: System.IComparable) = a - (Expr.scalar b)
    static member (*) (a: Expr, b: System.IComparable) = a * (Expr.scalar b)
    static member (/) (a: Expr, b: System.IComparable) = a / (Expr.scalar b)
    static member (%) (a: Expr, b: System.IComparable) = a % (Expr.scalar b)
    static member Pow (a: Expr, b: System.IComparable) = a ** (Expr.scalar b)
    static member ( *** ) (a: Expr, b: System.IComparable) = a ** (Expr.scalar b)
    static member (====) (a: Expr, b: System.IComparable) = Expr.constructElementwise Equal a (Expr.scalar b)
    static member (<<<<) (a: Expr, b: System.IComparable) = Expr.constructElementwise Less a (Expr.scalar b)
    static member (<<==) (a: Expr, b: System.IComparable) = Expr.constructElementwise LessEqual a (Expr.scalar b)
    static member (>>>>) (a: Expr, b: System.IComparable) = Expr.constructElementwise Greater a (Expr.scalar b)
    static member (>>==) (a: Expr, b: System.IComparable) = Expr.constructElementwise GreaterEqual a (Expr.scalar b)
    static member (<<>>) (a: Expr, b: System.IComparable) = Expr.constructElementwise NotEqual a (Expr.scalar b)

    static member (+) (a: System.IComparable, b: Expr) = (Expr.scalar a) + b
    static member (-) (a: System.IComparable, b: Expr) = (Expr.scalar a) - b
    static member (*) (a: System.IComparable, b: Expr) = (Expr.scalar a) * b
    static member (/) (a: System.IComparable, b: Expr) = (Expr.scalar a) / b
    static member (%) (a: System.IComparable, b: Expr) = (Expr.scalar a) % b
    static member Pow (a: System.IComparable, b: Expr) = (Expr.scalar a) ** b
    static member ( *** ) (a: System.IComparable, b: Expr) = (Expr.scalar a) ** b
    static member (====) (a: System.IComparable, b: Expr) = Expr.constructElementwise Equal (Expr.scalar a) b
    static member (<<<<) (a: System.IComparable, b: Expr) = Expr.constructElementwise Less (Expr.scalar a) b
    static member (<<==) (a: System.IComparable, b: Expr) = Expr.constructElementwise LessEqual (Expr.scalar a) b
    static member (>>>>) (a: System.IComparable, b: Expr) = Expr.constructElementwise Greater (Expr.scalar a) b
    static member (>>==) (a: System.IComparable, b: Expr) = Expr.constructElementwise GreaterEqual (Expr.scalar a) b
    static member (<<>>) (a: System.IComparable, b: Expr) = Expr.constructElementwise NotEqual (Expr.scalar a) b

    /// transposition
    member this.T = Expr.transpose this

    /// Dot product.
    /// Behavior depends on the dimensionality of the arguments.
    /// Cases: 
    /// (1, 1) -> vector-vector dot product resulting in a scalar
    /// (2, 1) -> matrix-vector dot product resulting in a vector
    /// (2, 2) -> matrix-matrix dot product resulting in a matrix
    /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
    /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
    static member ( .* ) (a: Expr, b: Expr) = Expr.dot a b
    static member ( %* ) (a: Expr, b: Expr) = Expr.tensorProduct a b

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
            | (:? (SizeSpec option) as so)  :: (:? (SizeSpec option) as fo)    :: rest ->
                RangeSpec.SymStartSymEnd (so, fo) :: parseArgs rest
            | (:? (SizeSpec option) as so)  :: null                             :: rest ->
                RangeSpec.SymStartSymEnd (so, None) :: parseArgs rest
            | null                           :: (:? (SizeSpec option) as fo)    :: rest ->
                RangeSpec.SymStartSymEnd (None, fo) :: parseArgs rest
            | (:? (Expr option) as so)      :: (:? (PlusElems option) as fo)    :: rest ->
                if Expr.typename so.Value <> TypeName.ofType<int> then
                    failwith "need int expression for range start"
                RangeSpec.DynStartSymSize (so.Value, fo.Value.Elems) :: parseArgs rest
            | null                           :: null                             :: rest ->
                RangeSpec.SymStartSymEnd (None, None) :: parseArgs rest

            // items
            | (:? SizeSpec as s)     :: rest -> RangeSpec.SymElem s :: parseArgs rest
            | (:? int64 as s)        :: rest when s = Tensor.TensorVal.NewAxis -> RangeSpec.NewAxis :: parseArgs rest
            | (:? int64 as s)        :: rest when s = Tensor.TensorVal.Fill ->    RangeSpec.AllFill :: parseArgs rest
            | (:? Expr as e)         :: rest -> if Expr.typename e <> TypeName.ofType<int> then
                                                    failwith "need int expression for element"               
                                                RangeSpec.DynElem e :: parseArgs rest
            | []                              -> []
            | _                               -> failwithf "invalid item/slice specification: %A" allArgs

        /// converts a full range specification into a simple range specification
        let rec splitFRS (rngs: FullExprRngsSpecT) (shps: ShapeSpec) (simpleRs: ExprRngsSpec) (newShape: ShapeSpec) =
            match rngs, shps with
            | RangeSpec.SymElem e :: rngs, _::shps -> splitFRS rngs shps (SimpleRangeSpec.SymStartSymEnd (e, Some e)::simpleRs) newShape
            | RangeSpec.DynElem e :: rngs, _::shps -> splitFRS rngs shps (SimpleRangeSpec.DynStartSymSize (e, SizeSpec.one)::simpleRs) newShape
            | RangeSpec.SymStartSymEnd (so, fo) :: rngs, size::shps -> 
                let size = (fo |? (size-1L)) - (so |? SizeSpec.zero) + 1L
                splitFRS rngs shps (SimpleRangeSpec.SymStartSymEnd (so |? SizeSpec.zero, fo)::simpleRs) (size::newShape)
            | RangeSpec.DynStartSymSize (s, size) :: rngs, _::shps ->
                splitFRS rngs shps (SimpleRangeSpec.DynStartSymSize (s, size)::simpleRs) (size::newShape)
            | RangeSpec.NewAxis :: rngs, _ ->
                splitFRS rngs shps simpleRs (SizeSpec.broadcastable::newShape)
            | RangeSpec.AllFill :: rrngs, _ ->
                if List.length rngs <= List.length shps then splitFRS (RangeSpec.All::rngs) shps simpleRs newShape
                else splitFRS rrngs shps simpleRs newShape
            | [], [] -> List.rev simpleRs, List.rev newShape
            | _ -> failwith "item/slice processing error"

        // build full range specificaton
        let argList = allArgs |> Array.toList |> List.map intToSizeSpec

        let srs, reshp = 
            match argList with
            | [:? ExprRngsSpec as srs] -> 
                // simplified range specification was specified, use directly
                srs, Expr.shapeOf (Unary (Subtensor srs, this))
            | [:? FullExprRngsSpecT as frs] ->
                // split into simplified range specification and reshape operation
                splitFRS frs (Expr.shapeOf this) [] []
            | _ ->
                // parse, then split into simplified range specification and reshape operation
                splitFRS (argList |> parseArgs) (Expr.shapeOf this) [] []

        // emit expression
        Unary (Reshape reshp, Unary (Subtensor srs, this))  
        |> Expr.check

    member this.Item 
        with get ([<System.ParamArray>] allArgs: obj []) = 
            this.GetSlice (allArgs)


    /// Traverses the op tree and for each op calls a function on its arguments and replaces 
    /// them by the function's return value(s).
    static member mapOperands unaryMapping binaryMapping naryMapping expr =
        let subMap = Expr.mapOperands unaryMapping binaryMapping naryMapping
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
    static member contains subExpr expr =
        let subCon = Expr.contains subExpr
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
    static member internal failshape op sa sb =
        failwithf "op %A was provided with arrays of incompatible shapes %A and %A" op sa sb

    /// Returns the type of the given expression.
    static member typename expr =
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
            -> Elem.Expr.typeName elemExpr
        | Nary (Channel (Loop spec, channel), _)
            -> Expr.loopOutputTypeNames spec |> Map.find channel 

        | Unary (_, a) -> Expr.typename a
        | Binary (_, a, b) -> Expr.typename a
        | Nary (_, es) -> Expr.typename (List.head es)

    /// data type of loop output
    static member loopOutputTypeNames (spec: LoopSpec) =
        spec.Channels |> Map.map (fun ch lv -> Expr.typename lv.Expr)

    /// Returns the shape of the given expression.
    static member shapeOf expr =
        // We assume that all operands have compatible size. 
        // For elementwise operations we assume that a and b are already broadcasted
        // to have the *same* size.

        match Cache.shape.TryFind expr with
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
                | Leaf(Var vs) -> Var.shape vs

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
                    -> Expr.shapeOf a

                // tensor operations
                | Unary(Diag(ax1, ax2), a) -> Expr.shapeOf a |> ShapeSpec.withoutAxis ax2
                | Unary(DiagMat(ax1, ax2), a) ->  Expr.shapeOf a |> List.insert ax2 (Expr.shapeOf a).[ax1]
                | Unary(Invert, a) -> Expr.shapeOf a

                // reductions
                | Unary(Sum, _) -> ShapeSpec.scalar
                | Unary(Product, _) -> ShapeSpec.scalar
                | Unary(SumAxis ax, a) -> Expr.shapeOf a |> ShapeSpec.withoutAxis ax
                | Unary(ProductAxis ax, a) -> Expr.shapeOf a |> ShapeSpec.withoutAxis ax
                | Unary(MaxAxis ax, a) -> Expr.shapeOf a |> ShapeSpec.withoutAxis ax
                | Unary(MinAxis ax, a) -> Expr.shapeOf a |> ShapeSpec.withoutAxis ax

                // index reductions
                | Unary(ArgMaxAxis ax, a) -> Expr.shapeOf a |> ShapeSpec.withoutAxis ax
                | Unary(ArgMinAxis ax, a) -> Expr.shapeOf a |> ShapeSpec.withoutAxis ax

                // shape operations
                | Unary(Reshape(ss), _) -> ss
                | Unary(DoBroadcast(ss), _) -> ss
                | Unary(PermuteAxes perm, a) -> Expr.shapeOf a |> ShapeSpec.permuteAxes perm
                | Unary(Subtensor(srs), a) ->
                    (srs, Expr.shapeOf a)
                    ||> List.map2 (fun sr shp ->
                         match sr with
                         | SimpleRangeSpec.SymStartSymEnd (s, fo)    -> (fo |? (shp - SizeSpec.one)) + 1L - s
                         | SimpleRangeSpec.DynStartSymSize (_, size) -> size)
                | Unary(ReverseAxis _, a) -> Expr.shapeOf a
                | Unary(Held ([], ReplicateTo (dim, s)), a) -> Expr.shapeOf a |> ShapeSpec.set dim s
                | Unary(Gather indices, a) -> indices |> List.pick id |> Expr.shapeOf
                | Unary (Scatter (indices, shp), a) -> shp

                // misc
                | Unary(StoreToVar _, a) -> ShapeSpec.emptyVector
                | Unary(Print _, a) -> Expr.shapeOf a
                | Unary(Dump _, a) -> Expr.shapeOf a
                | Unary(CheckFinite _, a) -> Expr.shapeOf a
                | Unary(Annotated(_), a) -> Expr.shapeOf a
                | Unary(Held (derivShp :: _, heldOp), a) -> [(Expr.shapeOf a).[0]; ShapeSpec.nElem derivShp]

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
                    -> Expr.shapeOf a
            
                // matrix/tensor operations
                | Binary (Dot, a, b) -> 
                    let sa, sb = Expr.shapeOf a, Expr.shapeOf b
                    match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
                    | 2, 2 -> ShapeSpec.matrix sa.[0] sb.[1]
                    | na, nb when na=nb -> sa.[0 .. na-2] @ [sb.[nb-1]]
                    | _ -> failwithf "invalid dot product shapes: %A, %A" sa sb
                | Binary (TensorProduct, a, b) -> 
                    let sa, sb = Expr.shapeOf a, Expr.shapeOf b
                    List.map2 (*) sa sb

                // shape operations
                | Binary (SetSubtensor ss, a, b) ->
                    Expr.shapeOf a

                // misc
                | Nary(Discard, _) -> ShapeSpec.emptyVector 
                | Nary(BuildTensor (shp, _), _) -> shp
                | Nary(Elements (resShape, elemExpr), _) -> resShape
                | Nary(Interpolate _, es) -> Expr.shapeOf es.Head
                | Nary(Channel (Loop spec, channel), es) -> Expr.loopOutputShapes spec |> Map.find channel
                | Nary(ExtensionOp eop, es) -> eop.Shape (es |> List.map Expr.shapeOf)

            Cache.shape.[expr] <- shp
            shp
    
    /// Returns the shapes of the outputs of the loop channels.
    static member internal loopOutputShapes (spec: LoopSpec) =
        spec.Channels
        |> Map.map (fun ch lv ->
            Expr.shapeOf lv.Expr |> ShapeSpec.insertAxis lv.SliceDim spec.Length)

    /// number of elements of given expression
    static member nElems expr =
        expr |> Expr.shapeOf |> ShapeSpec.nElem

    /// number of dimensions of given expression
    static member nDims expr =
        expr |> Expr.shapeOf |> ShapeSpec.nDim

    /// Wraps the given op in a Reshape op if its shape does not match ss.
    static member reshapeIfNecessary ss expr =
        if ss = Expr.shapeOf expr then expr else Unary(Reshape(ss), expr)

    /// Wraps the given op in a Broadcast op if its shape does not match ss.
    static member broadcastIfNecessary ss expr =
        if ss = Expr.shapeOf expr then expr else Unary(DoBroadcast(ss), expr)

    /// extract all variables from an expression
    static member extractVars expr =
        match Cache.extractedVars.LockedTryFind expr with
        | Some evs -> evs
        | None ->
            let evs =
                match expr with
                | Leaf (Var vs) -> Set.singleton vs
                | Unary (StoreToVar vs, a) -> Expr.extractVars a |> Set.add vs
                | Unary (Gather indices, a) ->
                    let indicesVars = indices |> List.choose (Option.map Expr.extractVars)
                    Set.unionMany (Expr.extractVars a :: indicesVars)
                | Unary (Scatter (indices, _), a) ->
                    let indicesVars = indices |> List.choose (Option.map Expr.extractVars)
                    Set.unionMany (Expr.extractVars a :: indicesVars)
                | Unary (AssumeJacobian jac, a) -> 
                    Set.union (Expr.extractVars jac) (Expr.extractVars a)
                | Binary (IfThenElse cond, a, b) -> 
                    Set.unionMany [Expr.extractVars cond; Expr.extractVars a; Expr.extractVars b]
                | Nary (ExtensionOp eop, es) ->
                    Set.unionMany (eop.ContainedVars :: (es |> List.map Expr.extractVars))

                | Leaf _ -> Set.empty
                | Unary (_, a) -> Expr.extractVars a
                | Binary (_, a, b) -> Set.union (Expr.extractVars a) (Expr.extractVars b)
                | Nary (_, es) -> Set.unionMany (es |> List.map Expr.extractVars)

            Cache.extractedVars.[expr] <- evs
            evs

    /// checks that given axis is valid for specified expression
    static member checkAxis ax expr =
        if not (0 <= ax && ax < Expr.nDims expr) then
            failwithf "invalid axis %d for expression of shape %A" ax (Expr.shapeOf expr)

    /// Checks ops' arguments for compatible shapes.
    static member checkExpr (expr: Expr) =
        if not (Cache.checkedExprs.LockedContains expr) then

            if Expr.typename expr = TypeName.ofType<obj> then
                failwith "Expression type cannot be object."

            let (..=) (sa: ShapeSpec) (sb: ShapeSpec) =
                if sa.Length = sb.Length then List.forall2 (.=) sa sb
                else false
            let (..<>) sa sb = not (sa ..= sb)
            let reqBool op a =
                if Expr.typename a <> TypeName.ofType<bool> then
                    failwithf "logical operation %A requires data type bool but got %A" 
                        op (Expr.typename a).Type

            match expr with 
            | Leaf op -> ()           

            | Unary (op, a) ->
                Expr.checkExpr a
                let sa = Expr.shapeOf a
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
                        | SizeSpec.Broadcast, _ -> ()
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
                    Expr.checkExpr jac
                    if Expr.typename jac <> Expr.typename expr then
                        failwithf "Jacobian type %A does not match expression type %A."
                            (Expr.typename jac).Type (Expr.typename expr).Type
                    if Expr.nDims jac <> 2 then
                        failwithf "Jacobian shape %A must be two-dimensional" (Expr.shapeOf jac)
                    if (Expr.shapeOf jac).[1] <> Expr.nElems expr then
                        failwithf "Jacobian shape %A must have %A elements in second dimension" 
                            (Expr.shapeOf jac) (Expr.nElems expr)
                | ReverseAxis ax when not (0 <= ax && ax < nda) ->
                    failwithf "cannot reverse non-existant axis %d of array with shape %A" ax sa
                | Held ([], ReplicateTo (dim, s)) -> 
                    a |> Expr.checkAxis dim
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
                                (indices |> List.map (Option.map Expr.shapeOf))
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
                                (indices |> List.map (Option.map Expr.shapeOf))
                        | None when dim >= a.NDims ->
                            failwithf "scatter index dimensions beyond the number of source dimensions \
                                       must not be None"
                        | _ -> ()
                | _ -> ()

            | Binary (op, a, b) ->
                Expr.checkExpr a
                Expr.checkExpr b
                let sa, sb = Expr.shapeOf a, Expr.shapeOf b
                let nda, ndb = ShapeSpec.nDim sa, ShapeSpec.nDim sb

                let ta, tb = Expr.typename a, Expr.typename b
                if ta <> tb then
                    failwithf "cannot apply binary operation %A to expressions of \
                               different types %A and %A" op ta.Type tb.Type

                match op with
                | And
                | Or ->
                    reqBool op a
                    reqBool op b
                | IfThenElse c ->
                    Expr.checkExpr c
                    if c.Type <> typeof<bool> then
                        failwith "condition of IfThenElse must be expression of type bool"
                    if c.Shape ..<> sa || sa ..<> sb then
                        failwithf "shape of condition %A and both argument shapes %A and %A must be equal"
                            c.Shape sa sb                    
                | Expr.BinaryElemwiseOp ->
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
                es |> List.iter Expr.checkExpr
                let ss = es |> List.map Expr.shapeOf

                let checkEqualTypes() =
                    if es |> List.exists (fun e -> Expr.typename e <> Expr.typename es.Head) then
                        failwithf "cannot apply n-ary operation %A to expressions of different types %A"
                            op (es |> List.map (Expr.typename >> TypeName.getType))
                let checkArg idx =
                    if not (0 <= idx && idx < es.Length) then
                        failwithf "the zero-based index %d does not exist for %d specified arguments" idx es.Length

                match op with
                | Elements (trgtShp, elemExpr) -> 
                    checkEqualTypes()
                    let tns = es |> List.map Expr.typename
                    Elem.Expr.check elemExpr |> ignore
                    Elem.Expr.checkCompatibility elemExpr ss tns trgtShp
                | BuildTensor (shp, rngs) ->
                    if List.length rngs <> List.length es then
                        failwithf "BuildTensor ranges must match arguments, but got %d ranges and %d arguments"
                                  rngs.Length es.Length
                    match ShapeSpec.tryEval shp with
                    | Some shp ->
                        for rng, arg in List.zip rngs es do
                            if rng.Length <> shp.Length then
                                failwithf "BuildTensor range %A has wrong dimensionality for shape %A" rng shp
                            for (start, stop), size, argSize in List.zip3 rng shp (Expr.shapeOf arg) do
                                if argSize <> stop - start + 1L then
                                    failwithf "BuildTensor range %A is invalid for argument of shape %A" rng (Expr.shapeOf arg)
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
                        |> Seq.map (fun (_, lv) -> Expr.extractVars lv.Expr)
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

            Cache.checkedExprs.LockedAdd expr |> ignore

    /// substitues the given symbol sizes into the expression
    static member substSymSizes symSizes (expr: Expr) =
        let substituted = Dictionary<Expr, Expr> ()
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
                        Nary (Elements (sShp trgtShp, Elem.Expr.substSymSizes symSizes elemExpr), List.map sSub es)
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
                        Nary (NaryOp.Channel (Loop substSpec, channel), es |> List.map sSub)
                    | Nary (ExtensionOp eop, es) -> Nary (ExtensionOp (eop.SubstSymSizes symSizes), List.map sSub es)
                    | Nary (op, es) -> Nary (op, List.map sSub es)
                
                substituted.[expr] <- subst
                subst
        sSub expr
  
    /// tests if all symbolic sizes can be evaluated
    static member private testEvalAllSymSizes (failIfNot: bool) (expr: Expr) =
        let subTest = Expr.testEvalAllSymSizes failIfNot
        let tSize = SizeSpec.canEval
        let tShp = ShapeSpec.canEval
        let tSrs = SimpleRangesSpec.canEvalSymbols
        let evalable =
            if Cache.exprsWithEvalableSymSizes.LockedContains expr then true
            else 
                match expr with
                | Leaf (Identity (ss, tn)) -> tSize ss
                | Leaf (SizeValue (sc, tn)) -> tSize sc
                | Leaf (Var vs) -> tShp (Var.shape vs)
                | Leaf (Arange (size, tn)) -> tSize size
                | Leaf _ -> true

                | Unary (Reshape ss, a) -> tShp ss && subTest a
                | Unary (DoBroadcast ss, a) -> tShp ss && subTest a
                | Unary (StoreToVar vs, a) -> tShp (Var.shape vs) && subTest a
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
                    Elem.Expr.canEvalAllSymSizes elemExpr && 
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
        
        if evalable then Cache.exprsWithEvalableSymSizes.LockedAdd expr |> ignore
        if failIfNot && not evalable then
            failwithf "expression %A contains a symbolic size that cannot be evaluated to \
                       a numeric value" expr
        evalable

    /// true if all shapes in the expression can be evaluated to numeric shapes
    static member canEvalAllSymSizes (expr: Expr) =
        Expr.testEvalAllSymSizes false expr

    /// fails if the expression contains a shape that cannot be evaluated to a numeric shape
    static member failOnNotEvalableSymSize (expr: Expr) =
        Expr.testEvalAllSymSizes true expr |> ignore

    /// Traverses the expression and checks ops' arguments for compatible shapes.
    static member check (expr: Expr) : Expr =
        Expr.checkExpr expr |> ignore
        expr

    /// Replaces all occurences of the map key with its value in the specified expression.
    /// Does not replace subexpressions within loop channel value expressions.
    static member subst (replacements: Map<Expr, Expr>) expr =
        let substituted = Dictionary<Expr, Expr> ()

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
        subSubst expr |> Expr.check

    /// True if expression is zero.
    /// False does not indicate that expression is non-zero.
    static member isZero expr =
        match expr with
        | Leaf (ScalarConst Const.Zero) -> true
        | Unary (Reshape _, a) -> Expr.isZero a
        | Unary (DoBroadcast _, a) -> Expr.isZero a
        | Unary (PermuteAxes _, a) -> Expr.isZero a
        | _ -> false

    /// counts operators, not counting repeating subexpressions
    static member countUniqueOps expr  =
        let visited = HashSet<Expr> (HashIdentity.Structural)
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
    static member countOps expr  =
        match expr with
        | Leaf _ -> 1
        | Unary (_, a) -> 1 + Expr.countOps a
        | Binary (_, a, b) -> 1 + Expr.countOps a + Expr.countOps b
        | Nary (_, es) -> 1 + List.sumBy Expr.countOps es

    /// scalar constant of given value
    static member scalar (f: obj) = 
        Leaf (ScalarConst (Const.ofValue f)) |> Expr.check

    /// scalar of given value converted to same type as given expression
    static member scalarOfSameType expr f = 
        let tn = Expr.typename expr
        let v = System.Convert.ChangeType (box f, TypeName.getType tn)
        Expr.scalar v

    /// scalar 0 of the same type as given expression
    static member inline zeroOfSameType expr = Expr.scalarOfSameType expr 0

    /// scalar 1 of the same type as given expression
    static member inline oneOfSameType expr = Expr.scalarOfSameType expr 1

    /// scalar 2 of the same type as given expression
    static member inline twoOfSameType expr = Expr.scalarOfSameType expr 2

    /// scalar with value of given size converted to the given type
    [<RequiresExplicitTypeArguments>]
    static member sizeValue<'T> size = 
        Leaf (SizeValue (size, TypeName.ofType<'T>)) |> Expr.check

    /// scalar with value of given size converted to the same type as given expression
    static member sizeValueOfSameType expr size = 
        Leaf (SizeValue (size, Expr.typename expr)) |> Expr.check

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    static member permuteAxes perm a =
        Unary (PermuteAxes perm, a) |> Expr.check

    /// swaps two dimensions of a tensor
    static member swapDim ax1 ax2 a = 
        a |> Expr.checkAxis ax1
        a |> Expr.checkAxis ax2
        if ax1 = ax2 then a
        else
            let perm = 
                [0 .. Expr.nDims a - 1]
                |> List.map (function
                             | d when d=ax1 -> ax2
                             | d when d=ax2 -> ax1
                             | d -> d)
            a |> Expr.permuteAxes perm

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    static member transpose a =
        let nd = Expr.shapeOf a |> ShapeSpec.nDim
        if nd < 2 then invalidArg "a" "need at least a matrix to transpose"
        Expr.swapDim (nd-2) (nd-1) a

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    static member constructElementwise op a b =
        let sa, sb = Expr.shapeOf a, Expr.shapeOf b
        let psa, psb = ShapeSpec.padToSame sa sb
        let bsa, bsb = ShapeSpec.broadcastToSame false psa psb
        let ba = a |> Expr.reshapeIfNecessary psa |> Expr.broadcastIfNecessary bsa
        let bb = b |> Expr.reshapeIfNecessary psb |> Expr.broadcastIfNecessary bsb    
        Binary (op, ba, bb) |> Expr.check

    /// pads from the left and broadcasts the argument to the given shape if possible
    static member broadcastToShape shp a =
        let sa = Expr.shapeOf a
        let psa = sa |> ShapeSpec.padTo (ShapeSpec.nDim shp)
        let bsa = psa |> ShapeSpec.broadcastToShape shp
        a |> Expr.reshapeIfNecessary psa |> Expr.broadcastIfNecessary bsa        

    /// pads and broadcasts all arguments to same shape if possible
    static member broadcastToSameMany es =
        let ss = es |> List.map Expr.shapeOf
        let ps = ShapeSpec.padToSameMany ss
        let bs = ShapeSpec.broadcastToSameMany false ps
        List.zip3 es ps bs
        |> List.map (fun (e, p, b) -> e |> Expr.reshapeIfNecessary p |> Expr.broadcastIfNecessary b)

    /// pads and broadcasts `a` and `b` to same shape if possible
    static member broadcastToSame a b =
        match Expr.broadcastToSameMany [a; b] with
        | [bcA; bcB] -> bcA, bcB
        | _ -> failwith "impossible"

    /// select elements according to the specified index arrays
    static member gather indices a =
        let someIndices = indices |> List.choose id
        if List.isEmpty someIndices then
            failwith "need to specify at least one index array"
        let bcSomeIndices = Expr.broadcastToSameMany someIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs ->
                Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSomeIndices
        Unary (Gather bcIndices, a) |> Expr.check

    /// select elements according to the specified index arrays
    static member scatter indices trgtShp a =
        let aShp = Expr.shapeOf a
        let indices = indices |> List.map (Option.map (Expr.broadcastToShape aShp))
        Unary (Scatter (indices, trgtShp), a) |> Expr.check
 
    /// sign keeping type
    static member signt (a: Expr) =
        Expr.SignT a 

    /// square root
    static member sqrtt (a: Expr) =
        Expr.Sqrt a

    /// elementwise uses elements from ifTrue if cond is true, 
    /// otherwise elements from ifFalse
    static member ifThenElse cond ifTrue ifFalse =
        let shps = [Expr.shapeOf cond; Expr.shapeOf ifTrue; Expr.shapeOf ifFalse]
        let pShps = ShapeSpec.padToSameMany shps
        let bcShps = ShapeSpec.broadcastToSameMany false pShps           
        match pShps, bcShps with
        | [condPShp; ifTruePShp; ifFalsePShp], [condBcShp; ifTrueBcShp; ifFalseBcShp] -> 
            let condBc = cond |> Expr.reshapeIfNecessary condPShp |> Expr.broadcastIfNecessary condBcShp
            let ifTrueBc = ifTrue |> Expr.reshapeIfNecessary ifTruePShp |> Expr.broadcastIfNecessary ifTrueBcShp
            let ifFalseBc = ifFalse |> Expr.reshapeIfNecessary ifFalsePShp |> Expr.broadcastIfNecessary ifFalseBcShp
            Binary (IfThenElse condBc, ifTrueBc, ifFalseBc) |> Expr.check
        | _ -> failwith "impossible"

    /// elementwise maximum
    static member maxElemwise a b =
        Expr.constructElementwise MaxElemwise a b

    /// elementwise minimum
    static member minElemwise a b =
        Expr.constructElementwise MinElemwise a b

    /// Ensures that all elements are between Some minVal and Some maxVal.
    static member cage (minVal, maxVal) a =
        let a =
            match minVal with
            | Some mv -> Expr.maxElemwise (Expr.scalar mv) a
            | None -> a
        let a =
            match maxVal with
            | Some mv -> Expr.minElemwise (Expr.scalar mv) a
            | None -> a
        a

    /// reshape (assuming C-continguous order) tensor; element count does not change // DONE
    static member reshape ss a = Unary(Reshape(ss), a) |> Expr.check

    /// broadcast of SizeBroadcast dimensions // DONE
    static member broadcast ss a = Unary(DoBroadcast(ss), a) |> Expr.check

    /// enables broadcasting in the given dimension, it must be of size one
    static member enableBroadcast dim a = 
        a |> Expr.reshape (Expr.shapeOf a |> ShapeSpec.enableBroadcast dim)

    /// disables broadcasting in the given dimension
    static member disableBroadcast dim a =
        a |> Expr.reshape (Expr.shapeOf a |> ShapeSpec.disableBroadcast dim)
  
    /// inserts a broadcast axis at the given dimension // DONE
    static member insertBroadcastAxis dim a =
        a |> Expr.reshape (Expr.shapeOf a |> ShapeSpec.insertBroadcastAxis dim)

    /// Replicates the tensor the given number of repetitions along the given axis.
    static member replicate dim reps a =
        a |> Expr.checkAxis dim

        // 1. insert axis of size one left to repetition axis
        // 2. broadcast along the new axis to number of repetitions
        // 3. reshape to result shape
        a 
        |> Expr.insertBroadcastAxis dim
        |> Expr.broadcast (a.Shape |> ShapeSpec.insertAxis dim reps)
        |> Expr.reshape (a.Shape |> List.set dim (reps * a.Shape.[dim]))

    /// Replicates the tensor along the given axis, so that after replication it has
    /// the specified `size`. If `size` is not a multiple of the current size of the
    /// tensor along the specified axis, the last replication is truncated appropriately.
    static member replicateTo dim size a =
        Unary (Held ([], ReplicateTo (dim, size)), a) |> Expr.check

    /// summaiton of all elements
    static member sum a = Unary(Sum, a) |> Expr.check

    /// summation over given dimension
    static member sumAxis ax a = Unary(SumAxis(ax), a) |> Expr.check

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    static member sumKeepingAxis ax a =
        a |> Expr.sumAxis ax |> Expr.insertBroadcastAxis ax

    /// product of all elements
    static member product a = Unary(Product, a) |> Expr.check

    /// product over given dimension
    static member productAxis ax a = Unary(ProductAxis(ax), a) |> Expr.check

    /// product over given dimension, while keeping the axis with one (broadcastable) element
    static member productKeepingAxis ax a =
        a |> Expr.productAxis ax |> Expr.insertBroadcastAxis ax

    /// maximum over given dimension
    static member maxAxis ax a = Unary(MaxAxis(ax), a) |> Expr.check

    /// maximum over given dimension, while keeping the axis with one (broadcastable) element
    static member maxKeepingAxis ax a =
        a |> Expr.maxAxis ax |> Expr.insertBroadcastAxis ax

    /// maximum over given dimension
    static member minAxis ax a = Unary(MinAxis(ax), a) |> Expr.check

    /// maximum over given dimension, while keeping the axis with one (broadcastable) element
    static member minKeepingAxis ax a =
        a |> Expr.minAxis ax |> Expr.insertBroadcastAxis ax

    /// index of maximum over given dimension
    static member argMaxAxis ax a = Unary(ArgMaxAxis(ax), a) |> Expr.check

    /// index of maximum over given dimension, while keeping the axis with one (broadcastable) element
    static member argMaxKeepingAxis ax a =
        a |> Expr.argMaxAxis ax |> Expr.insertBroadcastAxis ax

    /// index of maximum over given dimension
    static member argMinAxis ax a = Unary(ArgMinAxis(ax), a) |> Expr.check

    /// index of maximum over given dimension, while keeping the axis with one (broadcastable) element
    static member argMinKeepingAxis ax a =
        a |> Expr.argMinAxis ax |> Expr.insertBroadcastAxis ax

    /// mean over all elements
    static member mean (a: Expr) = 
        Expr.sum a / Expr.sizeValueOfSameType a a.NElems

    /// mean over given dimension
    static member meanAxis ax (a: Expr) =
        Expr.sumAxis ax a / Expr.sizeValueOfSameType a a.Shape.[ax]

    /// mean over given dimension, while keeping the axis with one (broadcastable) element
    static member meanKeepingAxis ax a =
        a |> Expr.meanAxis ax |> Expr.insertBroadcastAxis ax

    /// identity matrix of given size
    [<RequiresExplicitTypeArguments>]
    static member identity<'T> size = 
        Leaf(Identity(size, TypeName.ofType<'T>)) |> Expr.check

    /// identity matrix of given size and same type as given expression
    static member identityOfSameType expr size =
        Leaf(Identity(size, Expr.typename expr)) |> Expr.check

    /// tensor of given shape filled with specified value
    static member filled (shp: ShapeSpec) value =
        let bcShp = shp |> List.map (fun _ -> SizeSpec.broadcastable)
        Expr.scalar value
        |> Expr.reshape bcShp
        |> Expr.broadcast shp

    /// zero tensor of given shape
    [<RequiresExplicitTypeArguments>]
    static member zeros<'T> (shp: ShapeSpec) =
        Expr.filled shp (conv<'T> 0)

    /// zero tensor of given type and shape
    static member zerosOfType typ shp =
        Expr.filled shp (convTo typ 0)

    /// zero tensor of given shape and same type as given expression
    static member zerosOfSameType expr shp =
        let zero = System.Convert.ChangeType (box 0, (Expr.typename expr).Type)
        Expr.filled shp zero

    /// zero tensor with same shape and type as given tensor
    static member zerosLike expr = 
        Expr.zerosOfSameType expr expr.Shape

    /// variable of given name and shape
    [<RequiresExplicitTypeArguments>]
    static member var<'T> name (ss: ShapeSpec) = 
        Leaf(Var({Name=name; Shape=ss; TypeName=TypeName.ofType<'T>})) |> Expr.check

    /// variable of given name, type and shape
    static member varOfType name typ (ss: ShapeSpec) = 
        Leaf(Var({Name=name; Shape=ss; TypeName=TypeName.ofTypeInst typ})) |> Expr.check

    /// Vector counting from zero to given size minus one.
    [<RequiresExplicitTypeArguments>]
    static member arange<'T> size =
        Leaf(Arange(size, TypeName.ofType<'T>)) |> Expr.check

    /// Vector counting from zero to given size minus one.
    static member arangeOfType shp typ =
        Leaf(Arange(shp, TypeName.ofTypeInst typ)) |> Expr.check

    /// annotated expression
    static member annotate ano a = 
        Unary(Annotated(ano), a) |> Expr.check

    /// adds one broadcastable dimension to the left
    static member padLeft a =
        let sa = Expr.shapeOf a
        Expr.reshape (ShapeSpec.padLeft sa) a

    /// adds one broadcastable dimension to the right
    static member padRight a =
        let sa = Expr.shapeOf a
        Expr.reshape (ShapeSpec.padRight sa) a

    /// Dot product.
    /// Behavior depends on the dimensionality of the arguments.
    /// Cases: 
    /// (1, 1) -> vector-vector dot product resulting in a scalar
    /// (2, 1) -> matrix-vector dot product resulting in a vector
    /// (2, 2) -> matrix-matrix dot product resulting in a matrix
    /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
    /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
    static member dot (a: Expr) (b: Expr) =
        let sa, sb = Expr.shapeOf a, Expr.shapeOf b
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
            | 1, 1 -> 
                // vector-vector dot product
                Expr.sum (a * b)
            | 2, 1 -> 
                // matrix-vector dot product
                let bm = b |> Expr.reshape (ShapeSpec.padRight sb)
                Binary(Dot, a, bm) |> Expr.reshape [sa.[0]]
            | 2, 2 -> 
                // matrix-matrix dot product
                Binary(Dot, a, b)
            | na, nb when na = nb -> 
                // batched matrix-matrix dot product
                let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa sb
                let ba = a |> Expr.broadcastIfNecessary bsa
                let bb = b |> Expr.broadcastIfNecessary bsb    
                Binary(Dot, ba, bb)
            | na, nb when na = nb + 1 ->
                // batched matrix-vector dot product
                let psb = ShapeSpec.padRight sb
                let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa psb
                let ba = a |> Expr.broadcastIfNecessary bsa
                let bb = b |> Expr.reshapeIfNecessary psb |> Expr.broadcastIfNecessary bsb    
                Binary(Dot, ba, bb) |> Expr.reshape bsa.[0 .. na-2]
            | _ -> failwithf "cannot compute dot product between arrays of shapes %A and %A" sa sb  
        |> Expr.check

    /// tensor product
    static member tensorProduct (a: Expr) (b: Expr) =
        let sa, sb = Expr.shapeOf a, Expr.shapeOf b
        let psa, psb = ShapeSpec.padToSame sa sb
        let a, b = Expr.reshapeIfNecessary psa a, Expr.reshapeIfNecessary psb b
        Binary(TensorProduct, a, b) |> Expr.check

    /// extract VarSpec from variable expression
    static member extractVar expr = 
        match expr with
        | Leaf(Var(v)) -> v
        | _ -> invalidArg "expr" "not a expr consisting solely of a variable"

    /// make variable expression from VarSpec
    static member makeVar vs =
        Leaf(Var(vs)) |> Expr.check

    /// store to variable
    static member storeToVar ve a =
        let vs = Expr.extractVar ve
        Unary(StoreToVar(vs), a) |> Expr.check

    /// computes specified expressions, but discards the result
    static member discard es =
        Nary(Discard, es) |> Expr.check

    /// expression a with the specified subtensor replaced with b
    static member setSubtensor a b =
        match a with
        | Unary (Reshape _, (Unary (Subtensor srs, t) as st)) ->
            let stShp = Expr.shapeOf st
            Binary (SetSubtensor srs, t, Unary (Reshape stShp, b)) |> Expr.check
        | _ ->
            invalidArg "a" "the first argument of setSubtensor must be an item or slice of an expression, i.e. a.[...]"
                  
    /// Extracts the diagonal along the given axes.
    static member diagAxis ax1 ax2 a = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        Unary(Diag (ax1, ax2), a) |> Expr.check
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    static member diag a = 
        let nd = Expr.shapeOf a |> ShapeSpec.nDim
        if nd < 2 then failwith "need at least a matrix to extract diagonal"
        Expr.diagAxis (nd-2) (nd-1) a

    /// Creates a diagonal matrix by duplicating the given dimension.
    static member diagMatAxis ax1 ax2 a = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        Unary(DiagMat (ax1, ax2), a) |> Expr.check

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    static member diagMat a =
        let nd = Expr.shapeOf a |> ShapeSpec.nDim
        if nd < 1 then failwith "need at least a vector to create diagonal matrix"
        Expr.diagMatAxis (nd-1) nd a

    /// Computes the traces along the given axes.
    static member traceAxis ax1 ax2 a =
        let tax = if ax1 < ax2 then ax1 else ax1 + 1
        a |> Expr.diagAxis ax1 ax2 |> Expr.sumAxis tax

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    static member trace a =
        let nd = Expr.shapeOf a |> ShapeSpec.nDim
        if nd < 2 then
            failwith "need at least a two dimensional array for trace"      
        Expr.traceAxis (nd-2) (nd-1) a

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    static member invert a =
        Unary(Invert, a) |> Expr.check

    /// calculates a tensor elementwise using the given element expression and
    /// result shape
    static member elements trgtShp elemExpr args =
        Nary (Elements (trgtShp, elemExpr), args) |> Expr.check

    /// nullifies the Jacobian when calculating derivatives
    static member assumeZeroDerivative expr =
        Unary (NullifyJacobian, expr) |> Expr.check

    /// assumes the specified Jacobian when calculating derivatives
    static member assumeJacobian jac expr =
        Unary (AssumeJacobian jac, expr) |> Expr.check

    /// print the result with the given message when evaluated
    static member print msg a =
        Unary (Print msg, a) |> Expr.check

    /// dumps the result into the active dump session HDF5 file
    static member dump name a =
        Unary (Dump name, a) |> Expr.check

    /// checks the value for NaNs and infinities, outputs their location and stops the computation
    static member checkFinite name a =
        if Debug.EnableCheckFinite then
            Unary (CheckFinite name, a) |> Expr.check
        else a |> Expr.check

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate interpolator e =
        let e = Expr.broadcastToSameMany e
        Nary (Interpolate interpolator, e) |> Expr.check

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate1D interpolator a =
        Expr.interpolate interpolator [a]

    /// Element-wise two-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate2D interpolator a b =
        Expr.interpolate interpolator [a; b]

    /// Element-wise three-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate3D interpolator a b c =
        Expr.interpolate interpolator [a; b; c]
   
    /// A loop provides iterative evaluation of one or multiple expresisons.
    /// All variables occurs in the loop channel expressions must be defined as loop variables.
    /// The function `loop` performs automatic lifting of constants and thus allows for easy
    /// usage of variables external to the loop.
    static member loopNoLift spec channel args =
        Nary (NaryOp.Channel (Loop spec, channel), args) |> Expr.check

    /// A loop provides iterative evaluation of one or multiple expresisons.
    static member loop spec channel args =       
        let mutable args = args
        let mutable vars = spec.Vars

        /// adds an argument and returns its index
        let addArg (expr: Expr) =
            match args |> List.tryFindIndex ((=) expr) with
            | Some argIdx -> argIdx
            | None ->
                let argIdx = args.Length
                args <- args @ [expr]
                argIdx

        /// adds a constant variable, its required argument and returns the associated VarSpecT
        let addConstVar (expr: Expr) =
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
                let vs = Var.ofNameShapeAndTypeName (genName 0) expr.Shape expr.TypeName
                let lv = ConstArg (addArg expr)
                vars <- vars |> Map.add vs lv
                vs

        let loopVarSet = vars |> Map.toSeq |> Seq.map (fun (vs, _) -> vs) |> Set.ofSeq
        let lifted = Dictionary<Expr, Expr> ()

        let rec lift expr =
            match lifted.TryFind expr with
            | Some rep -> rep
            | None ->
                let exprVars = Expr.extractVars expr
                let dependsOnVars = not (Set.isEmpty exprVars)
                let dependsOnLoopVars = Set.intersect exprVars loopVarSet |> Set.isEmpty |> not
                let rep =
                    if dependsOnVars && not dependsOnLoopVars then
                        //if not (dependsOnLoopVars expr) then
                        let vs = addConstVar expr
                        Expr.makeVar vs
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

        Expr.loopNoLift spec channel args

    /// reverses the tensor in the given dimension 
    static member reverseAxis dim (a: Expr) : Expr =
        Unary (ReverseAxis dim, a) |> Expr.check

    /// concatenates the sequence of tensors in the specified dimension
    static member concat dim (es: Expr seq) =
        // check that arguments are correctly sized
        let es = List.ofSeq es
        let shps = es |> List.map Expr.shapeOf
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
            ((Expr.zerosOfSameType es.Head shp, SizeSpec.zero), es)
            ||> List.fold (fun (concatSoFar, pos) e ->
                let len = e.Shape.[dim]
                let slice : FullExprRngsSpecT = 
                    List.replicate e.NDims RangeSpec.All
                    |> List.set dim (RangeSpec.SymStartSymEnd (Some pos, Some (pos + len - 1L)))
                Expr.setSubtensor concatSoFar.[slice] e, pos + len)
        concatenated

    /// Build tensor from numeric ranges.
    static member internal buildTensor shp rngs srcs =
        Nary (BuildTensor (shp, rngs), srcs) |> Expr.check


module Expr = 
   
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

    /// Matches expressions with value zero.
    let (|ZeroExpr|_|) expr =
        if Expr.isZero expr then Some () else None

**)
