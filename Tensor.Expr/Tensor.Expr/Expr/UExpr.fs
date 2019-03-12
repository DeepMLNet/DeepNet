namespace rec Tensor.Expr

open System

open DeepNet.Utils
open Tensor.Expr.Ops
open Tensor.Backend



/// start plus the specified number of (symbolic elements)
type internal PlusElems (elems: Size) =
    new (intElems: int64) = PlusElems (Size.fix intElems)
    member this.Elems = elems



module internal ExprHelpers =

    let (|SubtensorExpr|_|) (expr: BaseExpr) =
        match expr.Op with
        | :? Reshape as reshp ->
            let subtensorExpr = reshp.X.Expr
            match subtensorExpr.Op with
            | :? Subtensor as subtensor ->
                let trgtExpr = subtensor.X
                Some (subtensor.Range, subtensorExpr, trgtExpr)
            | _ -> None
        | _ -> None


/// An tensor-valued expression with a single output channel.
[<StructuredFormatDisplay("{Pretty}")>]
type UExpr (baseExpr: BaseExpr) =    
    do 
        if not (baseExpr.IsSingleChannel) then
            failwithf "Expr is for single-channel expressions only, but got %A." baseExpr
    
    /// Create expression from specified single-channel op.
    new (op: IOp) =
        UExpr (BaseExpr.ofOp op)

    /// Create expression by accessing the specified channel of the BaseExpr.
    new (exprCh: BaseExprCh) =
        match exprCh with
        | BaseExprCh (Ch.Default, baseExpr) -> UExpr baseExpr
        | BaseExprCh (ch, baseExpr) ->
            UExpr {Channel.X=baseExpr.[ch]}

    /// Expression having the value of the specified variable.
    new (baseVar: Var) = 
        UExpr {VarArg.Var=baseVar}

    /// Expression having the value of the specified tensor.
    /// A reference to that tensor is stored.
    new (tensor: Tensor.ITensor) =
        UExpr {DataArg.Data=OrdRef tensor}

    member this.BaseExpr = baseExpr
    static member baseExpr (expr: UExpr) = expr.BaseExpr

    member this.BaseExprCh = baseExpr.[Ch.Default]
    static member baseExprCh (expr: UExpr) = expr.BaseExprCh

    member this.Op = baseExpr.Op
    static member op (expr: UExpr) = expr.Op

    member this.TypeName = baseExpr.[Ch.Default].TypeName
    static member typeName (expr: UExpr) = expr.TypeName

    member this.DataType = baseExpr.[Ch.Default].DataType
    static member dataType (expr: UExpr) = expr.DataType

    member this.Shape = baseExpr.[Ch.Default].Shape
    static member shape (expr: UExpr) = expr.Shape

    member this.NDims = baseExpr.[Ch.Default].NDims
    static member nDims (expr: UExpr) = expr.NDims

    member this.NElems = baseExpr.[Ch.Default].NElems
    static member nElems (expr: UExpr) = expr.NElems

    member this.Dev = baseExpr.[Ch.Default].Dev
    static member dev (expr: UExpr) = expr.Dev

    member this.Args = baseExpr.Args |> Map.map (fun _ arg -> UExpr arg)
    static member args (expr: UExpr) = expr.Args

    member this.VarMap = baseExpr.VarMap
    static member varMap (expr: UExpr) = expr.VarMap

    member this.Vars = baseExpr.Vars
    static member vars (expr: UExpr) = expr.Vars

    member this.CanEvalAllSymSizes = baseExpr.CanEvalAllSymSizes
    static member canEvalAllSymSizes (expr: UExpr) = expr.CanEvalAllSymSizes

    static member substSymSizes (env: SymSizeEnv) (expr: UExpr) : UExpr =
        expr.BaseExpr |> BaseExpr.substSymSizes env |> UExpr

    interface System.IEquatable<UExpr> with
        member this.Equals other = this.BaseExpr = other.BaseExpr

    override this.Equals other =
        match other with
        | :? UExpr as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<UExpr> with
        member this.CompareTo other = compare this.BaseExpr other.BaseExpr

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? UExpr as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare Expr to type %A." (other.GetType())

    override this.GetHashCode() = hash this.BaseExpr

    /// Converts expression to string with specified approximate maximum length.
    member this.ToString maxLength =     
        let opStr =
            match this.Op with
            | :? IOpFormat as opFormat -> opFormat.Text
            | _ -> this.Op.GetType().Name
        let args = this.Args
        let argSet = args |> Map.keys
        let argList, withLabel =
            match argSet with
            | _ when argSet = Set [Arg.Only] -> [Arg.Only], false
            | _ when argSet = Set [Arg.X; Arg.Y] -> [Arg.X; Arg.Y], false
            | _ when argSet |> Set.toSeq |> Seq.forall (function | Arg.N _ -> true | _ -> false) ->
                argSet |> Set.toList |> List.sortBy (function | Arg.N n -> n | _ -> 0), false
            | _ ->
                argSet |> Set.toList |> List.sortBy (sprintf "%A"), true
        String.limited maxLength [
            yield String.Formatter (fun _ -> opStr)
            if not argList.IsEmpty then
                yield String.Delim " ("
                for i, arg in List.indexed argList do
                    if i > 0 then
                        yield String.Delim ", "
                    if withLabel then
                        yield String.Formatter (fun _ -> sprintf "%A=" arg)
                    yield String.Formatter (fun ml -> args.[arg].ToString ml)
                yield String.Delim ")"
        ]

    /// Converts expression to string with unlimited length.
    override this.ToString () = this.ToString System.Int32.MaxValue

    /// Pretty string.
    member this.Pretty = this.ToString 80

    /// Checks that given axis is valid for specified expression
    static member internal checkAxis ax (expr: UExpr) =
        if not (0 <= ax && ax < expr.NDims) then
            failwithf "Specified axis %d is invalid for expression of shape %A." ax expr.Shape

    /// Reshapes the expression into the given shape.
    /// The element count must not change.
    static member reshape ss (expr: UExpr) : UExpr =
        if ss = expr.Shape then expr 
        else UExpr {Reshape.Shape=ss; X=expr.BaseExprCh}

    /// Broadcasts the expression into the given shape.
    static member broadcast ss (expr: UExpr) : UExpr =
        if ss = expr.Shape then expr 
        else UExpr {DoBroadcast.Shape=ss; X=expr.BaseExprCh}

    /// Inserts a broadcast axis at the given dimension.
    static member insertBroadcastAxis dim (expr: UExpr) : UExpr =
        expr |> UExpr.reshape (expr.Shape |> Shape.insertBroadcastAxis dim)

    /// adds one broadcastable dimension to the left
    static member padLeft (a: UExpr) : UExpr =
        a |> UExpr.reshape (Shape.padLeft a.Shape)

    /// adds one broadcastable dimension to the right
    static member padRight (a: UExpr) : UExpr =
        a |> UExpr.reshape (Shape.padRight a.Shape)

    /// Reshapes the expression so that a single dimension remains.
    static member flatten (expr: UExpr) =
        expr |> UExpr.reshape (Shape.flatten expr.Shape)

    /// pads from the left and broadcasts the argument to the given shape if possible
    static member broadcastToShape shp (a: UExpr) =
        let psa = a.Shape |> Shape.padTo (Shape.nDim shp)
        let bsa = psa |> Shape.broadcastToShape shp
        a |> UExpr.reshape psa |> UExpr.broadcast bsa        

    /// pads and broadcasts all arguments to same shape if possible
    static member broadcastToSameMany (es: UExpr list) =
        let ss = es |> List.map UExpr.shape
        let ps = Shape.padToSameMany ss
        let bs = Shape.broadcastToSameMany false ps
        List.zip3 es ps bs
        |> List.map (fun (e, p, b) -> e |> UExpr.reshape p |> UExpr.broadcast b)

    /// pads and broadcasts `a` and `b` to same shape if possible
    static member broadcastToSame (a: UExpr) (b: UExpr) =
        match UExpr.broadcastToSameMany [a; b] with
        | [bcA; bcB] -> bcA, bcB
        | _ -> failwith "impossible"

    /// enables broadcasting in the given dimension, it must be of size one
    static member enableBroadcast dim (a: UExpr) = 
        a |> UExpr.reshape (a.Shape |> Shape.enableBroadcast dim)

    /// disables broadcasting in the given dimension
    static member disableBroadcast dim (a: UExpr) =
        a |> UExpr.reshape (a.Shape |> Shape.disableBroadcast dim)
  
    /// scalar constant of given value
    static member scalar dev (value: obj) : UExpr = 
        UExpr {Scalar.Value=Const value; Dev=dev} 

    /// scalar of given value converted to same type as given expression
    static member scalarLike (expr: UExpr) (value: obj) = 
        let v = System.Convert.ChangeType (value, expr.DataType) 
        UExpr.scalar expr.Dev v

    /// Converts the data to the specified type.
    static member convert (dataType: Type) (expr: UExpr) =
        let typeName = TypeName.ofTypeInst dataType
        if expr.TypeName <> typeName then
            UExpr {Convert.ToType=TypeName.ofTypeInst dataType; X=expr.BaseExprCh}
        else
            expr

    /// Transfers the data to the specified device.
    static member transfer (dev: ITensorDevice) (expr: UExpr) =
        if expr.Dev <> dev then
            UExpr {Transfer.ToDev=dev; X=expr.BaseExprCh}
        else
            expr

    /// Scalar with value of given size and type int64.
    static member size dev (size: Size) = 
        UExpr {SizeValue.Value=size; Dev=dev} 

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    static member permuteAxes permutation (expr: UExpr) =
        UExpr {PermuteAxes.Permutation=permutation; X=expr.BaseExprCh}

    /// Swaps two dimensions of a tensor.
    static member swapDim ax1 ax2 (expr: UExpr) = 
        expr |> UExpr.checkAxis ax1
        expr |> UExpr.checkAxis ax2
        if ax1 = ax2 then expr
        else
            let perm = 
                [0 .. expr.NDims - 1]
                |> List.map (function
                             | d when d=ax1 -> ax2
                             | d when d=ax2 -> ax1
                             | d -> d)
            expr |> UExpr.permuteAxes perm

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    static member transpose (expr: UExpr) =
        if expr.NDims < 2 then invalidArg "expr" "Need at least a matrix to transpose."
        expr |> UExpr.swapDim (expr.NDims - 2) (expr.NDims - 1)

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    member this.T = UExpr.transpose this

    // item / slicing
    member this.GetSlice ([<System.ParamArray>] allArgs: obj []) : UExpr =

        /// converts ints to SizeSpecTs
        let intToSizeSpec (arg: obj) =
            match arg with
            | :? int64 as f -> Size.fix f :> obj
            | :? (int64 option) as fo -> 
                match fo with
                | Some f -> Some (Size.fix f) :> obj
                | None -> None :> obj
            | _ -> arg

        /// converts argument list to range specification
        let rec parseArgs (args: obj list) : RangesSpec =
            match args with
            // direct range specification
            | [:? RangesSpec as rngs] -> rngs

            // slices
            | (:? (Size option) as so)  :: (:? (Size option) as fo)    :: rest ->
                RangeSpec.SymStartSymEnd (so, fo) :: parseArgs rest
            | (:? (Size option) as so)  :: null                            :: rest ->
                RangeSpec.SymStartSymEnd (so, None) :: parseArgs rest
            | null                          :: (:? (Size option) as fo)    :: rest ->
                RangeSpec.SymStartSymEnd (None, fo) :: parseArgs rest
            | (:? (UExpr option) as so)      :: (:? (PlusElems option) as fo)   :: rest ->
                if so.Value.TypeName <> TypeName.ofType<int64> then
                    failwith "Need expression of type int64 for range start."
                RangeSpec.DynStartSymSize (so.Value.BaseExprCh, fo.Value.Elems) :: parseArgs rest
            | null                           :: null                           :: rest ->
                RangeSpec.SymStartSymEnd (None, None) :: parseArgs rest

            // items
            | (:? Size as s)     :: rest -> RangeSpec.SymElem s :: parseArgs rest
            | (:? int64 as s)        :: rest when s = Tensor.TensorVal.NewAxis -> RangeSpec.NewAxis :: parseArgs rest
            | (:? int64 as s)        :: rest when s = Tensor.TensorVal.Fill    -> RangeSpec.AllFill :: parseArgs rest
            | (:? UExpr as e)        :: rest  -> if e.TypeName <> TypeName.ofType<int64> then
                                                     failwith "Need expression of type int64 for element index." 
                                                 RangeSpec.DynElem e.BaseExprCh :: parseArgs rest                                                             
            | []                              -> []
            | _                               -> failwithf "Invalid item/slice specification: %A" allArgs

        /// converts a full range specification into a simple range specification
        let rec splitFRS (rngs: RangesSpec) (shps: Shape) (simpleRs: SimpleRangesSpec) (newShape: Shape) =
            match rngs, shps with
            | RangeSpec.SymElem e :: rngs, _::shps -> splitFRS rngs shps (SimpleRangeSpec.SymStartSymEnd (e, Some e)::simpleRs) newShape
            | RangeSpec.DynElem e :: rngs, _::shps -> splitFRS rngs shps (SimpleRangeSpec.DynStartSymSize (e, Size.one)::simpleRs) newShape
            | RangeSpec.SymStartSymEnd (so, fo) :: rngs, size::shps -> 
                let size = (fo |? (size-1L)) - (so |? Size.zero) + 1L
                splitFRS rngs shps (SimpleRangeSpec.SymStartSymEnd (so |? Size.zero, fo)::simpleRs) (size::newShape)
            | RangeSpec.DynStartSymSize (s, size) :: rngs, _::shps ->
                splitFRS rngs shps (SimpleRangeSpec.DynStartSymSize (s, size)::simpleRs) (size::newShape)
            | RangeSpec.NewAxis :: rngs, _ ->
                splitFRS rngs shps simpleRs (Size.broadcastable::newShape)
            | RangeSpec.AllFill :: rrngs, _ ->
                if List.length rngs <= List.length shps then splitFRS (RangeSpec.All::rngs) shps simpleRs newShape
                else splitFRS rrngs shps simpleRs newShape
            | [], [] -> List.rev simpleRs, List.rev newShape
            | _ -> failwith "Item/slice processing error."

        // build full range specificaton
        let argList = allArgs |> Array.toList |> List.map intToSizeSpec

        let srs, reshp = 
            match argList with
            | [:? SimpleRangesSpec as srs] -> 
                // simplified range specification was specified, use directly
                srs, (UExpr {Subtensor.Range=srs; X=this.BaseExprCh}).Shape
            | [:? RangesSpec as frs] ->
                // split into simplified range specification and reshape operation
                splitFRS frs this.Shape [] []
            | _ ->
                // parse, then split into simplified range specification and reshape operation
                splitFRS (argList |> parseArgs) this.Shape [] []

        // emit expression
        let sub = {Subtensor.Range=srs; X=this.BaseExprCh} |> UExpr
        let reshaped = {Reshape.Shape=reshp; X=sub.BaseExprCh} |> UExpr
        reshaped

    member this.Item 
        with get ([<System.ParamArray>] allArgs: obj []) = 
            this.GetSlice (allArgs)

    /// Expression a with the specified subtensor replaced with b.
    static member setSubtensor (trgt: UExpr) (src: UExpr) =
        match trgt.BaseExpr with
        | ExprHelpers.SubtensorExpr (range, subtensorExpr, trgtExpr) ->
            let srcReshaped = UExpr {Reshape.Shape=subtensorExpr.[Ch.Default].Shape; X=src.BaseExprCh}
            UExpr {SetSubtensor.Range=range; X=trgtExpr; Y=srcReshaped.BaseExprCh}
        | _ ->
            invalidArg "trgt" "The first argument of setSubtensor must be an item or slice of an expression, i.e. a.[...]."                 

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    static member constructElementwise op (a: UExpr) (b: UExpr) =
        let psa, psb = Shape.padToSame a.Shape b.Shape
        let bsa, bsb = Shape.broadcastToSame false psa psb
        let ba = a |> UExpr.reshape psa |> UExpr.broadcast bsa
        let bb = b |> UExpr.reshape psb |> UExpr.broadcast bsb 
        let opInst: IOp = op ba bb
        UExpr opInst

    // elementwise unary arithmetic
    static member (~+) (x: UExpr) = UExpr {UnaryPlus.X=x.BaseExprCh}
    static member (~-) (x: UExpr) = UExpr {Negate.X=x.BaseExprCh}
    static member Abs (x: UExpr) = UExpr {Abs.X=x.BaseExprCh}
    static member SignT (x: UExpr) = UExpr {SignT.X=x.BaseExprCh}
    static member Log (x: UExpr) = UExpr {Log.X=x.BaseExprCh}
    static member Log10 (x: UExpr) = UExpr {Log10.X=x.BaseExprCh}
    static member Exp (x: UExpr) = UExpr {Exp.X=x.BaseExprCh}
    static member Sin (x: UExpr) = UExpr {Sin.X=x.BaseExprCh}
    static member Cos (x: UExpr) = UExpr {Cos.X=x.BaseExprCh}
    static member Tan (x: UExpr) = UExpr {Tan.X=x.BaseExprCh}
    static member Asin (x: UExpr) = UExpr {Asin.X=x.BaseExprCh}
    static member Acos (x: UExpr) = UExpr {Acos.X=x.BaseExprCh}
    static member Atan (x: UExpr) = UExpr {Atan.X=x.BaseExprCh}
    static member Sinh (x: UExpr) = UExpr {Sinh.X=x.BaseExprCh}
    static member Cosh (x: UExpr) = UExpr {Cosh.X=x.BaseExprCh}
    static member Tanh (x: UExpr) = UExpr {Tanh.X=x.BaseExprCh}
    static member Sqrt (x: UExpr) = UExpr {Sqrt.X=x.BaseExprCh}
    static member Ceiling (x: UExpr) = UExpr {Ceiling.X=x.BaseExprCh}
    static member Floor (x: UExpr) = UExpr {Floor.X=x.BaseExprCh}
    static member Round (x: UExpr) = UExpr {Round.X=x.BaseExprCh}
    static member Truncate (x: UExpr) = UExpr {Truncate.X=x.BaseExprCh}

    // element-wise unary logic
    static member (~~~~) (x: UExpr) = UExpr {Not.X=x.BaseExprCh}

    // elementwise binary arithmetic
    static member (+) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Add.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (-) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Subtract.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (*) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Multiply.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (/) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Divide.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (%) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Modulo.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member Pow (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Pow.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y   
    static member ( *** ) (x: UExpr, y: UExpr) = x ** y

    // element-wise binary logic
    static member (&&&&) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {And.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (||||) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Or.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y

    // element-wise binary comparison
    static member (====) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Equal.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (<<<<) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Less.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (<<==) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {LessOrEqual.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (>>>>) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {Greater.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (>>==) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {GreaterOrEqual.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (<<>>) (x: UExpr, y: UExpr) = UExpr.constructElementwise (fun x y -> {NotEqual.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y

    // element-wise binary logic with basetype
    //static member (&&&&) (x: UExpr, y: bool) = x &&&& (UExpr.scalar x.Dev y)
    //static member (||||) (x: UExpr, y: bool) = x |||| (UExpr.scalar x.Dev y)

    //static member (&&&&) (x: bool, y: UExpr) = (UExpr.scalar y.Dev x) &&&& y
    //static member (||||) (x: bool, y: UExpr) = (UExpr.scalar y.Dev x) |||| y

    // elementwise binary arithmetic with basetype
    //static member (+) (x: UExpr, y: obj) = x + (UExpr.scalar x.Dev y)
    //static member (-) (x: UExpr, y: obj) = x - (UExpr.scalar x.Dev y)
    //static member (*) (x: UExpr, y: obj) = x * (UExpr.scalar x.Dev y)
    //static member (/) (x: UExpr, y: obj) = x / (UExpr.scalar x.Dev y)
    //static member (%) (x: UExpr, y: obj) = x % (UExpr.scalar x.Dev y)
    //static member Pow (x: UExpr, y: obj) = x ** (UExpr.scalar x.Dev y)
    //static member ( *** ) (x: UExpr, y: obj) = x ** (UExpr.scalar x.Dev y)   

    //static member (+) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) + y
    //static member (-) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) - y
    //static member (*) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) * y
    //static member (/) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) / y
    //static member (%) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) % y
    //static member Pow (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) ** y
    //static member ( *** ) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) ** y

    // element-wise binary comparison with basetype
    //static member (====) (x: UExpr, y: obj) = x ==== (UExpr.scalar x.Dev y)
    //static member (<<<<) (x: UExpr, y: obj) = x <<<< (UExpr.scalar x.Dev y)
    //static member (<<==) (x: UExpr, y: obj) = x <<== (UExpr.scalar x.Dev y)
    //static member (>>>>) (x: UExpr, y: obj) = x >>>> (UExpr.scalar x.Dev y)
    //static member (>>==) (x: UExpr, y: obj) = x >>== (UExpr.scalar x.Dev y)
    //static member (<<>>) (x: UExpr, y: obj) = x <<>> (UExpr.scalar x.Dev y)

    //static member (====) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) ==== y
    //static member (<<<<) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) <<<< y
    //static member (<<==) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) <<== y
    //static member (>>>>) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) >>>> y
    //static member (>>==) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) >>== y
    //static member (<<>>) (x: obj, y: UExpr) = (UExpr.scalar y.Dev x) <<>> y

    /// Dot product.
    /// Behavior depends on the dimensionality of the arguments.
    /// Cases: 
    /// (1, 1) -> vector-vector dot product resulting in a scalar
    /// (2, 1) -> matrix-vector dot product resulting in a vector
    /// (2, 2) -> matrix-matrix dot product resulting in a matrix
    /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
    /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
    static member ( .* ) (a: UExpr, b: UExpr) = 
        let sa, sb = a.Shape, b.Shape
        match Shape.nDim sa, Shape.nDim sb with
        | 1, 1 -> 
            // vector-vector dot product
            UExpr.sum (a * b)
        | 2, 1 -> 
            // matrix-vector dot product
            let bm = b |> UExpr.reshape (Shape.padRight sb)
            UExpr {Dot.X=a.BaseExprCh; Y=bm.BaseExprCh} |> UExpr.reshape [sa.[0]]
        | 2, 2 -> 
            // matrix-matrix dot product
            UExpr {Dot.X=a.BaseExprCh; Y=b.BaseExprCh} 
        | na, nb when na = nb -> 
            // batched matrix-matrix dot product
            let bsa, bsb = Shape.broadcastToSameInDims [0 .. na-3] false sa sb
            let ba = a |> UExpr.broadcast bsa
            let bb = b |> UExpr.broadcast bsb    
            UExpr {Dot.X=ba.BaseExprCh; Y=bb.BaseExprCh} 
        | na, nb when na = nb + 1 ->
            // batched matrix-vector dot product
            let psb = Shape.padRight sb
            let bsa, bsb = Shape.broadcastToSameInDims [0 .. na-3] false sa psb
            let ba = a |> UExpr.broadcast bsa
            let bb = b |> UExpr.reshape psb |> UExpr.broadcast bsb    
            UExpr {Dot.X=ba.BaseExprCh; Y=bb.BaseExprCh} |> UExpr.reshape bsa.[0 .. na-2]
        | _ -> failwithf "Cannot compute dot product between tensors of shapes %A and %A." sa sb  

    /// Sign keeping type.
    static member signt (expr: UExpr) =
        UExpr.SignT expr

    /// Square root.
    static member sqrtt (expr: UExpr) =
        UExpr.Sqrt expr

    /// Elementwise maximum.
    static member maxElemwise (x: UExpr) (y: UExpr) =
        UExpr.constructElementwise (fun x y -> {MaxElemwise.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y

    /// Elementwise minimum.
    static member minElemwise (x: UExpr) (y: UExpr) =
        UExpr.constructElementwise (fun x y -> {MinElemwise.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y

    /// Ensures that all elements are between minVal and maxVal.
    /// Values outside these limits are capped to the limits.
    static member limit (x: UExpr, ?minVal: obj, ?maxVal: obj) =
        let checkType (value: obj) =
            if value.GetType() <> x.DataType then
                failwithf "Limit value has type %A but expression data type is %A."
                          (value.GetType()) x.DataType
        let x =
            match minVal with
            | Some minVal -> 
                checkType minVal
                UExpr.maxElemwise (UExpr.scalar x.Dev minVal) x
            | None -> x
        let x =
            match maxVal with
            | Some maxVal -> 
                checkType maxVal
                UExpr.minElemwise (UExpr.scalar x.Dev maxVal) x
            | None -> x
        x

    /// Tensor of given shape filled with specified value.
    static member filled (dev: ITensorDevice) (shp: Shape) (value: obj) =
        let bcShp = shp |> List.map (fun _ -> Size.broadcastable)
        UExpr.scalar dev value |> UExpr.reshape bcShp |> UExpr.broadcast shp

    /// Zero tensor of given type and shape.
    static member zeros dataType dev (shp: Shape) =
        UExpr.filled dev shp (zeroOf dataType)

    /// zero tensor with same shape and type as given tensor
    static member zerosLike (expr: UExpr) = 
        UExpr.zeros expr.DataType expr.Dev expr.Shape

    /// Identity matrix of given type and size.
    static member identity dataType dev size =
        UExpr {Identity.Size=size; Type=TypeName.ofTypeInst dataType; Dev=dev}

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    static member invert (x: UExpr) =
        UExpr {Invert.X=x.BaseExprCh}

    /// Reverses the tensor in the specified dimension.
    static member reverseAxis axis (x: UExpr) =
        UExpr {ReverseAxis.Axis=axis; X=x.BaseExprCh} 

    /// Concatenates the sequence of tensors in the specified dimension.
    static member concat dim (es: UExpr seq) =
        // check that arguments are correctly sized
        let es = List.ofSeq es
        let shps = es |> List.map UExpr.shape
        match es with
        | [] -> failwithf "need at least one tensor to concatenate"
        | h :: ts ->
            if not (0 <= dim && dim < h.NDims) then
                failwithf "cannot concatenate over non-existant dimension %d given shapes %A" dim shps
            for t in ts do
                if t.TypeName <> h.TypeName then
                    failwithf "all arguments must have same type but got types %A" (es |> List.map (fun e -> e.DataType))
                if t.NDims <> h.NDims then
                    failwithf "all arguments must have same number of dimensions but shapes %A were specifed" shps                        
                for i, (sa, sb) in List.indexed (List.zip h.Shape t.Shape) do
                    if i <> dim && not (Size.equalIgnoringBc sa sb) then
                        failwithf "all arguments must have same shape expect in concatenation dimension %d but \
                                   shapes %A were specified" dim shps
                    
        // calculate shape of concatenation
        let totalElems = es |> Seq.sumBy (fun e -> e.Shape.[dim])
        let shp = es.Head.Shape |> Shape.set dim totalElems

        // build concatenation using iterative subtensor replacement
        let concatenated, _ =
            ((UExpr.zeros es.Head.DataType es.Head.Dev shp, Size.zero), es)
            ||> List.fold (fun (concatSoFar, pos) e ->
                let len = e.Shape.[dim]
                let slice: RangesSpec = 
                    List.replicate e.NDims RangeSpec.All
                    |> List.set dim (RangeSpec.SymStartSymEnd (Some pos, Some (pos + len - 1L)))
                UExpr.setSubtensor concatSoFar.[slice] e, pos + len)
        concatenated

    /// Extracts the diagonal along the given axes.
    static member diagAxis ax1 ax2 (x: UExpr) = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        UExpr {Diag.Axis1=ax1; Axis2=ax2; Diag.X=x.BaseExprCh} 
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    static member diag (x: UExpr) = 
        if x.NDims < 2 then 
            failwithf "Need at least a matrix to extract diagonal but got shape: %A" x.Shape
        x |> UExpr.diagAxis (x.NDims-2) (x.NDims-1)

    /// Creates a diagonal matrix by duplicating the given dimension.
    static member diagMatAxis ax1 ax2 (x: UExpr) = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        UExpr {DiagMat.Axis1=ax1; Axis2=ax2; X=x.BaseExprCh} 

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    static member diagMat (x: UExpr) =
        if x.NDims < 1 then 
            failwithf "Need at least a vector to build diagonal matrix but got shape: %A" x.Shape
        x |> UExpr.diagMatAxis (x.NDims-1) x.NDims

    /// summation over given dimension
    static member sumAxis (axis: int) (x: UExpr) = 
        UExpr {SumAxis.Axis=axis; X=x.BaseExprCh} 

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    static member sumKeepingAxis (axis: int) (x: UExpr) =
        x |> UExpr.sumAxis axis |> UExpr.insertBroadcastAxis axis

    /// summaiton of all elements
    static member sum (x: UExpr) = 
        x |> UExpr.flatten |> UExpr.sumAxis 0

    /// Computes the traces along the given axes.
    static member traceAxis (ax1: int) (ax2: int) (x: UExpr) =
        let tax = if ax1 < ax2 then ax1 else ax1 + 1
        x |> UExpr.diagAxis ax1 ax2 |> UExpr.sumAxis tax

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    static member trace (x: UExpr) =
        if x.NDims < 2 then
            failwithf "Need at least a matrix for trace but got shape: %A" x.Shape      
        x |> UExpr.traceAxis (x.NDims-2) (x.NDims-1) 
    
    /// product over given dimension
    static member productAxis (axis: int) (x: UExpr) = 
        UExpr {ProductAxis.Axis=axis; X=x.BaseExprCh} 

    /// product over given dimension, while keeping the axis with one (broadcastable) element
    static member productKeepingAxis (axis: int) (x: UExpr) =
        x |> UExpr.productAxis axis |> UExpr.insertBroadcastAxis axis

    /// product of all elements
    static member product (x: UExpr) = 
        x |> UExpr.flatten |> UExpr.productAxis 0

    /// Maximum over given dimension.
    static member maxAxis (axis: int) (x: UExpr) = 
        UExpr {MaxAxis.Axis=axis; X=x.BaseExprCh} 

    /// Maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member maxKeepingAxis (axis: int) (x: UExpr) =
        x |> UExpr.maxAxis axis |> UExpr.insertBroadcastAxis axis

    /// Maximum of all elements.
    static member max (x: UExpr) = 
        x |> UExpr.flatten |> UExpr.maxAxis 0

    /// Minimum over given dimension.
    static member minAxis (axis: int) (x: UExpr) = 
        UExpr {MinAxis.Axis=axis; X=x.BaseExprCh} 

    /// Minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member minKeepingAxis (axis: int) (x: UExpr) =
        x |> UExpr.minAxis axis |> UExpr.insertBroadcastAxis axis

    /// Minimum of all elements.
    static member min (x: UExpr) = 
        x |> UExpr.flatten |> UExpr.minAxis 0

    /// Index of maximum over given dimension.
    static member argMaxAxis (axis: int) (x: UExpr) = 
        UExpr {ArgMaxAxis.Axis=axis; X=x.BaseExprCh} 

    /// Index of maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMaxKeepingAxis (axis: int) (x: UExpr) =
        x |> UExpr.argMaxAxis axis |> UExpr.insertBroadcastAxis axis

    /// Index of minimum over given dimension.
    static member argMinAxis (axis: int) (x: UExpr) = 
        UExpr {MinAxis.Axis=axis; X=x.BaseExprCh} 

    /// Index of minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMinKeepingAxis (axis: int) (x: UExpr) =
        x |> UExpr.minAxis axis |> UExpr.insertBroadcastAxis axis

    /// mean over all elements
    static member mean (x: UExpr) = 
        let nElems = x.NElems |> UExpr.size x.Dev |> UExpr.convert x.DataType
        UExpr.sum x / nElems

    /// mean over given dimension
    static member meanAxis (axis: int) (x: UExpr) =
        let nElems = x.Shape.[axis] |> UExpr.size x.Dev |> UExpr.convert x.DataType
        UExpr.sumAxis axis x / nElems

    /// mean over given dimension, while keeping the axis with one (broadcastable) element
    static member meanKeepingAxis (axis: int) (x: UExpr) =
        x |> UExpr.meanAxis axis |> UExpr.insertBroadcastAxis axis

    /// Select elements according to the specified index tensors.
    static member gather (indices: UExpr option list) (x: UExpr) =
        let someIndices = indices |> List.choose id
        if List.isEmpty someIndices then
            failwith "Gather needs at least one specified index tensor."
        let bcSomeIndices = UExpr.broadcastToSameMany someIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs ->
                Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSomeIndices
        let bcIndices = bcIndices |> List.map (Option.map UExpr.baseExprCh)
        UExpr {Gather.Indices=bcIndices; X=x.BaseExprCh} 

    /// Disperses elements according to the specified index tensors.
    static member scatter (indices: UExpr option list) (trgtShp: Shape) (x: UExpr) =
        let indices = indices |> List.map (Option.map (UExpr.broadcastToShape x.Shape))
        let indices = indices |> List.map (Option.map UExpr.baseExprCh)
        UExpr {Scatter.Indices=indices; Shape=trgtShp; X=x.BaseExprCh} 

    /// Nullifies the Jacobian of its argument when calculating derivatives.
    static member assumeZeroDeriv (x: UExpr) =
        UExpr {AssumeZeroDeriv.X=x.BaseExprCh} 

    /// Assumes the specified Jacobian when calculating derivatives.
    static member assumeDeriv (deriv: UExpr) (x: UExpr) =
        UExpr {AssumeDeriv.Deriv=deriv.BaseExprCh; X=x.BaseExprCh} 

    /// Annotated expression (no influence on value).
    static member annotate label (x: UExpr) = 
        UExpr {Annotated.Label=label; X=x.BaseExprCh} 

    /// Print the result with the given label when evaluated.
    static member print (label: string) (x: UExpr) =
        UExpr {Print.Label=label; X=x.BaseExprCh} 

    /// Dumps the result into the given dataset in the active HDF5 dump file.
    static member dump (dataset: string) (x: UExpr) =
        UExpr {Dump.Dataset=dataset; X=x.BaseExprCh} 

    /// If the value contains NaNs or infinities, outputs their location and 
    /// stops the computation.
    static member checkFinite (label: string) (x: UExpr) =
        UExpr {CheckFinite.Label=label; X=x.BaseExprCh} 

    /// Tensor product.
    static member tensorProduct (x: UExpr) (y: UExpr) =
        UExpr {TensorProduct.X=x.BaseExprCh; Y=y.BaseExprCh} 

   /// Elementwise uses elements from `ifTrue` if `cond` is true for that element, otherwise elements from `ifFalse`.
    static member ifThenElse (cond: UExpr) (ifTrue: UExpr) (ifFalse: UExpr) =
        let shps = [cond.Shape; ifTrue.Shape; ifFalse.Shape]
        let pShps = Shape.padToSameMany shps
        let bcShps = Shape.broadcastToSameMany false pShps           
        match pShps, bcShps with
        | [condPShp; ifTruePShp; ifFalsePShp], [condBcShp; ifTrueBcShp; ifFalseBcShp] -> 
            let condBc = cond |> UExpr.reshape condPShp |> UExpr.broadcast condBcShp
            let ifTrueBc = ifTrue |> UExpr.reshape ifTruePShp |> UExpr.broadcast ifTrueBcShp
            let ifFalseBc = ifFalse |> UExpr.reshape ifFalsePShp |> UExpr.broadcast ifFalseBcShp
            UExpr {IfThenElse.Cond=condBc.BaseExprCh; IfTrue=ifTrueBc.BaseExprCh; IfFalse=ifFalseBc.BaseExprCh} 
        | _ -> failwith "impossible"

    /// Discards the results of all arguments.
    static member discard (xs: UExpr list) =
        let xs = xs |> List.map UExpr.baseExprCh
        UExpr {Discard.Xs=xs} 

    /// Build tensor from numeric ranges.
    static member internal buildTensor shape ranges (xs: UExpr list) =
        let xs = xs |> List.map UExpr.baseExprCh
        UExpr {BuildTensor.Shape=shape; Ranges=ranges; Xs=xs} 
    
    /// Calculates a tensor elementwise using the given element expression and result shape.
    static member elements shape elemExpr (xs: UExpr list) =
        let xs = xs |> List.map UExpr.baseExprCh
        UExpr {Elements.Shape=shape; ElemExpr=elemExpr; Xs=xs} 

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate interpolator (xs: UExpr list) =
        let xs = UExpr.broadcastToSameMany xs
        let xs = xs |> List.map UExpr.baseExprCh
        UExpr {Interpolate.Interpolator=interpolator; Xs=xs} 

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate1D interpolator (x: UExpr) =
        UExpr.interpolate interpolator [x]

    /// Element-wise two-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate2D interpolator (x: UExpr) (y: UExpr) =
        UExpr.interpolate interpolator [x; y]

    /// Element-wise three-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate3D interpolator (x: UExpr) (y: UExpr) (z: UExpr) =
        UExpr.interpolate interpolator [x; y; z]

    /// Substitutes the variables within the expression tree.
    static member substVars (env: Map<VarName, UExpr>) (expr: UExpr) =
        let env = env |> Map.map (fun _ sExpr -> sExpr.BaseExpr)
        expr.BaseExpr |> BaseExpr.substVars env |> UExpr

    /// Evaluates the expression into a numeric value using the specified evaluation envirnoment.
    static member evalWithEnv (evalEnv: EvalEnv) (expr: UExpr) : Tensor.ITensor =
        // Infer symbolic sizes from variable environment and substitute them into expression.
        let varValMap = VarValMap.make evalEnv.VarEnv expr.VarMap
        let symSizeEnv = varValMap |> VarValMap.inferSymSizes SymSizeEnv.empty
        let substExpr = expr |> UExpr.substSymSizes symSizeEnv

        // Evaluate.
        let chVals = BaseExprEval.eval evalEnv substExpr.BaseExpr
        chVals.[Ch.Default]        

    /// Evaluates the expression into a numeric value.
    static member eval (varEnv: VarEnv) (expr: UExpr) : Tensor.ITensor = 
        let evalEnv : EvalEnv = {VarEnv=varEnv; Tracer=NoTracer()}
        UExpr.evalWithEnv evalEnv expr


