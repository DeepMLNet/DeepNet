namespace SymTensor

open SymTensor.Ops
open DeepNet.Utils


module internal ExprHelpers =

    let (|SubtensorExpr|_|) (expr: BaseExpr) =
        match expr.Op with
        | :? Reshape as reshp ->
            let subtensorExpr = reshp.X
            match subtensorExpr.Op with
            | :? Subtensor as subtensor ->
                let trgtExpr = subtensor.X
                Some (subtensor.Range, subtensorExpr, trgtExpr)
            | _ -> None
        | _ -> None

open ExprHelpers


type Expr2 (op: IOp2) =    
    inherit BaseExpr(op)
    
    new (baseExpr: BaseExpr) =
        Expr2(baseExpr.Op)

    static member op (expr: Expr2) = expr.Op
    static member typeName (expr: Expr2) = expr.TypeName
    static member shape (expr: Expr2) = expr.Shape
    static member nDims (expr: Expr2) = expr.NDims
    static member nElems (expr: Expr2) = expr.NElems
    static member vars (expr: Expr2) = expr.Vars
    static member canEvalAllSymSizes (expr: Expr2) = expr.CanEvalAllSymSizes
    static member substSymSizes (env: SymSizeEnv) (expr: Expr2) : Expr2 =
        expr.SubstSymSizes env |> Expr2

    /// Checks that given axis is valid for specified expression
    static member internal checkAxis ax (expr: Expr2) =
        if not (0 <= ax && ax < expr.NDims) then
            failwithf "Specified axis %d is invalid for expression of shape %A." ax expr.Shape

    /// Reshapes the expression into the given shape.
    /// The element count must not change.
    static member reshape ss (expr: Expr2) =
        if ss = expr.Shape then expr 
        else Expr2 {Reshape.Shape=ss; X=expr}

    /// Broadcasts the expression into the given shape.
    static member broadcast ss (expr: Expr2) =
        if ss = expr.Shape then expr 
        else Expr2 {DoBroadcast.Shape=ss; X=expr}

    /// Inserts a broadcast axis at the given dimension.
    static member insertBroadcastAxis dim (expr: Expr2) =
        expr |> Expr2.reshape (expr.Shape |> ShapeSpec.insertBroadcastAxis dim)

    /// adds one broadcastable dimension to the left
    static member padLeft (a: Expr2) =
        a |> Expr2.reshape (ShapeSpec.padLeft a.Shape)

    /// adds one broadcastable dimension to the right
    static member padRight (a: Expr2) =
        a |> Expr2.reshape (ShapeSpec.padRight a.Shape)

    /// Reshapes the expression so that a single dimension remains.
    static member flatten (expr: Expr2) =
        expr |> Expr2.reshape (ShapeSpec.flatten expr.Shape)

    /// pads from the left and broadcasts the argument to the given shape if possible
    static member broadcastToShape shp (a: Expr2) =
        let psa = a.Shape |> ShapeSpec.padTo (ShapeSpec.nDim shp)
        let bsa = psa |> ShapeSpec.broadcastToShape shp
        a |> Expr2.reshape psa |> Expr2.broadcast bsa        

    /// pads and broadcasts all arguments to same shape if possible
    static member broadcastToSameMany (es: Expr2 list) =
        let ss = es |> List.map Expr2.shape
        let ps = ShapeSpec.padToSameMany ss
        let bs = ShapeSpec.broadcastToSameMany false ps
        List.zip3 es ps bs
        |> List.map (fun (e, p, b) -> e |> Expr2.reshape p |> Expr2.broadcast b)

    /// pads and broadcasts `a` and `b` to same shape if possible
    static member broadcastToSame (a: Expr2) (b: Expr2) =
        match Expr2.broadcastToSameMany [a; b] with
        | [bcA; bcB] -> bcA, bcB
        | _ -> failwith "impossible"

    /// scalar constant of given value
    static member scalar (f: obj) = 
        {Scalar.Value=Const.ofValue f} |> Expr2 

    /// scalar of given value converted to same type as given expression
    static member scalarOfSameType (expr: Expr2) f = 
        let v = System.Convert.ChangeType (box f, expr.TypeName.Type)
        Expr2.scalar v

    /// Scalar with value of given size and type int64.
    static member size (size: SizeSpec) = 
        {SizeValue.Value=size} |> Expr2

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    static member permuteAxes permutation (expr: Expr2) =
        Expr2 {PermuteAxes.Permutation=permutation; X=expr}

    /// Swaps two dimensions of a tensor.
    static member swapDim ax1 ax2 (expr: Expr2) = 
        expr |> Expr2.checkAxis ax1
        expr |> Expr2.checkAxis ax2
        if ax1 = ax2 then expr
        else
            let perm = 
                [0 .. expr.NDims - 1]
                |> List.map (function
                             | d when d=ax1 -> ax2
                             | d when d=ax2 -> ax1
                             | d -> d)
            expr |> Expr2.permuteAxes perm

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    static member transpose (expr: Expr2) =
        if expr.NDims < 2 then invalidArg "expr" "Need at least a matrix to transpose."
        expr |> Expr2.swapDim (expr.NDims - 2) (expr.NDims - 1)

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    member this.T = Expr2.transpose this

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
        let rec parseArgs (args: obj list) : RangesSpec =
            match args with
            // direct range specification
            | [:? RangesSpec as rngs] -> rngs

            // slices
            | (:? (SizeSpec option) as so)  :: (:? (SizeSpec option) as fo)    :: rest ->
                RangeSpec.SymStartSymEnd (so, fo) :: parseArgs rest
            | (:? (SizeSpec option) as so)  :: null                            :: rest ->
                RangeSpec.SymStartSymEnd (so, None) :: parseArgs rest
            | null                          :: (:? (SizeSpec option) as fo)    :: rest ->
                RangeSpec.SymStartSymEnd (None, fo) :: parseArgs rest
            | (:? (Expr2 option) as so)     :: (:? (PlusElems option) as fo)   :: rest ->
                if so.Value.TypeName <> TypeName.ofType<int64> then
                    failwith "Need expression of type int64 for range start."
                RangeSpec.DynStartSymSize (so.Value, fo.Value.Elems) :: parseArgs rest
            | null                           :: null                           :: rest ->
                RangeSpec.SymStartSymEnd (None, None) :: parseArgs rest

            // items
            | (:? SizeSpec as s)     :: rest -> RangeSpec.SymElem s :: parseArgs rest
            | (:? int64 as s)        :: rest when s = Tensor.TensorVal.NewAxis -> RangeSpec.NewAxis :: parseArgs rest
            | (:? int64 as s)        :: rest when s = Tensor.TensorVal.Fill ->    RangeSpec.AllFill :: parseArgs rest
            | (:? Expr2 as e)        :: rest -> if e.TypeName <> TypeName.ofType<int64> then
                                                    failwith "Need expression of type int64 for element index."               
                                                RangeSpec.DynElem e :: parseArgs rest
            | []                              -> []
            | _                               -> failwithf "Invalid item/slice specification: %A" allArgs

        /// converts a full range specification into a simple range specification
        let rec splitFRS (rngs: RangesSpec) (shps: ShapeSpec) (simpleRs: SimpleRangesSpec) (newShape: ShapeSpec) =
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
            | _ -> failwith "Item/slice processing error."

        // build full range specificaton
        let argList = allArgs |> Array.toList |> List.map intToSizeSpec

        let srs, reshp = 
            match argList with
            | [:? SimpleRangesSpec as srs] -> 
                // simplified range specification was specified, use directly
                srs, (Expr2 {Subtensor.Range=srs; X=this}).Shape
            | [:? RangesSpec as frs] ->
                // split into simplified range specification and reshape operation
                splitFRS frs this.Shape [] []
            | _ ->
                // parse, then split into simplified range specification and reshape operation
                splitFRS (argList |> parseArgs) this.Shape [] []

        // emit expression
        let sub = {Subtensor.Range=srs; X=this} |> Expr2
        let reshaped = {Reshape.Shape=reshp; X=sub} |> Expr2
        reshaped

    member this.Item 
        with get ([<System.ParamArray>] allArgs: obj []) = 
            this.GetSlice (allArgs)


    /// Expression a with the specified subtensor replaced with b.
    static member setSubtensor (trgt: Expr2) (src: Expr2) =
        match trgt with
        | SubtensorExpr (range, subtensorExpr, trgtExpr) ->
            let srcReshaped = Expr2 {Reshape.Shape=subtensorExpr.Shape; X=src}
            Expr2 {SetSubtensor.Range=range; X=trgtExpr; Y=srcReshaped}
        | _ ->
            invalidArg "trgt" "The first argument of setSubtensor must be an item or slice of an expression, i.e. a.[...]."                 

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    static member constructElementwise op (a: Expr2) (b: Expr2) =
        let psa, psb = ShapeSpec.padToSame a.Shape b.Shape
        let bsa, bsb = ShapeSpec.broadcastToSame false psa psb
        let ba = a |> Expr2.reshape psa |> Expr2.broadcast bsa
        let bb = b |> Expr2.reshape psb |> Expr2.broadcast bsb 
        let opInst: IOp2 = op ba bb
        Expr2 opInst

    // elementwise unary arithmetic
    static member (~+) (x: Expr2) = Expr2 {UnaryPlus.X=x}
    static member (~-) (x: Expr2) = Expr2 {Negate.X=x}
    static member Abs (x: Expr2) = Expr2 {Abs.X=x}
    static member SignT (x: Expr2) = Expr2 {SignT.X=x}
    static member Log (x: Expr2) = Expr2 {Log.X=x}
    static member Log10 (x: Expr2) = Expr2 {Log10.X=x}
    static member Exp (x: Expr2) = Expr2 {Exp.X=x}
    static member Sin (x: Expr2) = Expr2 {Sin.X=x}
    static member Cos (x: Expr2) = Expr2 {Cos.X=x}
    static member Tan (x: Expr2) = Expr2 {Tan.X=x}
    static member Asin (x: Expr2) = Expr2 {Asin.X=x}
    static member Acos (x: Expr2) = Expr2 {Acos.X=x}
    static member Atan (x: Expr2) = Expr2 {Atan.X=x}
    static member Sinh (x: Expr2) = Expr2 {Sinh.X=x}
    static member Cosh (x: Expr2) = Expr2 {Cosh.X=x}
    static member Tanh (x: Expr2) = Expr2 {Tanh.X=x}
    static member Sqrt (x: Expr2) = Expr2 {Sqrt.X=x}
    static member Ceiling (x: Expr2) = Expr2 {Ceiling.X=x}
    static member Floor (x: Expr2) = Expr2 {Floor.X=x}
    static member Round (x: Expr2) = Expr2 {Round.X=x}
    static member Truncate (x: Expr2) = Expr2 {Truncate.X=x}

    // element-wise unary logic
    static member (~~~~) (x: Expr2) = Expr2 {Not.X=x}

    // elementwise binary arithmetic
    static member (+) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Add.X=x; Y=y} :> IOp2) x y
    static member (-) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Subtract.X=x; Y=y} :> IOp2) x y
    static member (*) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Multiply.X=x; Y=y} :> IOp2) x y
    static member (/) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Divide.X=x; Y=y} :> IOp2) x y
    static member (%) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Modulo.X=x; Y=y} :> IOp2) x y
    static member Pow (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Pow.X=x; Y=y} :> IOp2) x y   
    static member ( *** ) (x: Expr2, y: Expr2) = x ** y

    // element-wise binary logic
    static member (&&&&) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {And.X=x; Y=y} :> IOp2) x y
    static member (||||) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Or.X=x; Y=y} :> IOp2) x y

    // element-wise binary comparison
    static member (====) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Equal.X=x; Y=y} :> IOp2) x y
    static member (<<<<) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Less.X=x; Y=y} :> IOp2) x y
    static member (<<==) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {LessOrEqual.X=x; Y=y} :> IOp2) x y
    static member (>>>>) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {Greater.X=x; Y=y} :> IOp2) x y
    static member (>>==) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {GreaterOrEqual.X=x; Y=y} :> IOp2) x y
    static member (<<>>) (x: Expr2, y: Expr2) = Expr2.constructElementwise (fun x y -> {NotEqual.X=x; Y=y} :> IOp2) x y

    // elementwise binary with basetype
    static member (+) (x: Expr2, y: System.IComparable) = x + (Expr2.scalar y)
    static member (-) (x: Expr2, y: System.IComparable) = x - (Expr2.scalar y)
    static member (*) (x: Expr2, y: System.IComparable) = x * (Expr2.scalar y)
    static member (/) (x: Expr2, y: System.IComparable) = x / (Expr2.scalar y)
    static member (%) (x: Expr2, y: System.IComparable) = x % (Expr2.scalar y)
    static member Pow (x: Expr2, y: System.IComparable) = x ** (Expr2.scalar y)
    static member ( *** ) (x: Expr2, y: System.IComparable) = x ** (Expr2.scalar y)   
    static member (====) (x: Expr2, y: System.IComparable) = x ==== (Expr2.scalar y)
    static member (<<<<) (x: Expr2, y: System.IComparable) = x <<<< (Expr2.scalar y)
    static member (<<==) (x: Expr2, y: System.IComparable) = x <<== (Expr2.scalar y)
    static member (>>>>) (x: Expr2, y: System.IComparable) = x >>>> (Expr2.scalar y)
    static member (>>==) (x: Expr2, y: System.IComparable) = x >>== (Expr2.scalar y)
    static member (<<>>) (x: Expr2, y: System.IComparable) = x <<>> (Expr2.scalar y)

    static member (+) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) + y
    static member (-) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) - y
    static member (*) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) * y
    static member (/) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) / y
    static member (%) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) % y
    static member Pow (x: System.IComparable, y: Expr2) = (Expr2.scalar x) ** y
    static member ( *** ) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) ** y
    static member (====) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) ==== y
    static member (<<<<) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) <<<< y
    static member (<<==) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) <<== y
    static member (>>>>) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) >>>> y
    static member (>>==) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) >>== y
    static member (<<>>) (x: System.IComparable, y: Expr2) = (Expr2.scalar x) <<>> y

    static member ( .* ) (x: Expr2, y: Expr2) = Expr2 {Dot.X=x; Y=y}

    /// Sign keeping type.
    static member signt (expr: Expr2) =
        Expr2.SignT expr

    /// Square root.
    static member sqrtt (expr: Expr2) =
        Expr2.Sqrt expr

    /// Tensor of given shape filled with specified value.
    static member filled (shp: ShapeSpec) value =
        let bcShp = shp |> List.map (fun _ -> SizeSpec.broadcastable)
        Expr2.scalar value |> Expr2.reshape bcShp |> Expr2.broadcast shp

    /// Zero tensor of given shape.
    [<RequiresExplicitTypeArguments>]
    static member zeros<'T> (shp: ShapeSpec) =
        Expr2.filled shp (conv<'T> 0)

    /// Zero tensor of given type and shape.
    static member zerosOfType typ shp =
        Expr2.filled shp (convTo typ 0)

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    static member invert (x: Expr2) =
        {Invert.X=x} |> Expr2

    /// Reverses the tensor in the specified dimension.
    static member reverseAxis axis (x: Expr2) =
        {ReverseAxis.Axis=axis; X=x} |> Expr2  

    /// Extracts the diagonal along the given axes.
    static member diagAxis ax1 ax2 (x: Expr2) = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        {Diag.Axis1=ax1; Axis2=ax2; Diag.X=x} |> Expr2
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    static member diag (x: Expr2) = 
        if x.NDims < 2 then 
            failwithf "Need at least a matrix to extract diagonal but got shape: %A" x.Shape
        x |> Expr2.diagAxis (x.NDims-2) (x.NDims-1)

    /// Creates a diagonal matrix by duplicating the given dimension.
    static member diagMatAxis ax1 ax2 (x: Expr2) = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        {DiagMat.Axis1=ax1; Axis2=ax2; X=x} |> Expr2

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    static member diagMat (x: Expr2) =
        if x.NDims < 1 then 
            failwithf "Need at least a vector to build diagonal matrix but got shape: %A" x.Shape
        x |> Expr2.diagMatAxis (x.NDims-1) x.NDims

    /// summation over given dimension
    static member sumAxis axis x = 
        {SumAxis.Axis=axis; X=x} |> Expr2

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    static member sumKeepingAxis axis x =
        x |> Expr2.sumAxis axis |> Expr2.insertBroadcastAxis axis

    /// summaiton of all elements
    static member sum x = 
        x |> Expr2.flatten |> Expr2.sumAxis 0

    /// Computes the traces along the given axes.
    static member traceAxis ax1 ax2 x =
        let tax = if ax1 < ax2 then ax1 else ax1 + 1
        x |> Expr2.diagAxis ax1 ax2 |> Expr2.sumAxis tax

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    static member trace (x: Expr2) =
        if x.NDims < 2 then
            failwithf "Need at least a matrix for trace but got shape: %A" x.Shape      
        x |> Expr2.traceAxis (x.NDims-2) (x.NDims-1) 
    
    /// product over given dimension
    static member productAxis axis x = 
        {ProductAxis.Axis=axis; X=x} |> Expr2

    /// product over given dimension, while keeping the axis with one (broadcastable) element
    static member productKeepingAxis axis x =
        x |> Expr2.productAxis axis |> Expr2.insertBroadcastAxis axis

    /// product of all elements
    static member product x = 
        x |> Expr2.flatten |> Expr2.productAxis 0

    /// Maximum over given dimension.
    static member maxAxis axis x = 
        {MaxAxis.Axis=axis; X=x} |> Expr2

    /// Maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member maxKeepingAxis axis x =
        x |> Expr2.maxAxis axis |> Expr2.insertBroadcastAxis axis

    /// Maximum of all elements.
    static member max x = 
        x |> Expr2.flatten |> Expr2.maxAxis 0

    /// Minimum over given dimension.
    static member minAxis axis x = 
        {MinAxis.Axis=axis; X=x} |> Expr2

    /// Minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member minKeepingAxis axis x =
        x |> Expr2.minAxis axis |> Expr2.insertBroadcastAxis axis

    /// Minimum of all elements.
    static member min x = 
        x |> Expr2.flatten |> Expr2.minAxis 0

    /// Index of maximum over given dimension.
    static member argMaxAxis axis x = 
        {ArgMaxAxis.Axis=axis; X=x} |> Expr2

    /// Index of maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMaxKeepingAxis axis x =
        x |> Expr2.argMaxAxis axis |> Expr2.insertBroadcastAxis axis

    /// Index of minimum over given dimension.
    static member argMinAxis axis x = 
        {MinAxis.Axis=axis; X=x} |> Expr2

    /// Index of minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMinKeepingAxis axis x =
        x |> Expr2.minAxis axis |> Expr2.insertBroadcastAxis axis

    /// Select elements according to the specified index tensors.
    static member gather (indices: Expr2 option list) (x: Expr2) =
        let someIndices = indices |> List.choose id
        if List.isEmpty someIndices then
            failwith "Gather needs at least one specified index tensor."
        let bcSomeIndices = Expr2.broadcastToSameMany someIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs ->
                Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSomeIndices
        let bcIndices = bcIndices |> List.map (Option.map (fun i -> i :> BaseExpr))
        {Gather.Indices=bcIndices; X=x} |> Expr2

    /// Disperses elements according to the specified index tensors.
    static member scatter (indices: Expr2 option list) (trgtShp: ShapeSpec) (x: Expr2) =
        let indices = indices |> List.map (Option.map (Expr2.broadcastToShape x.Shape))
        let indices = indices |> List.map (Option.map (fun i -> i :> BaseExpr))
        {Scatter.Indices=indices; Shape=trgtShp; X=x} |> Expr2

    /// Nullifies the Jacobian of its argument when calculating derivatives.
    static member assumeZeroDeriv x =
        {AssumeZeroDeriv.X=x} |> Expr2

    /// Assumes the specified Jacobian when calculating derivatives.
    static member assumeDeriv deriv x =
        {AssumeDeriv.Deriv=deriv; X=x} |> Expr2

    /// Annotated expression (no influence on value).
    static member annotate label x = 
        {Annotated.Label=label; X=x} |> Expr2

    /// Print the result with the given label when evaluated.
    static member print label x =
        {Print.Label=label; X=x} |> Expr2

    /// Dumps the result into the given dataset in the active HDF5 dump file.
    static member dump dataset x =
        {Dump.Dataset=dataset; X=x} |> Expr2

    /// If the value contains NaNs or infinities, outputs their location and 
    /// stops the computation.
    static member checkFinite label x =
        {CheckFinite.Label=label; X=x} |> Expr2

    /// Dot product.
    /// Behavior depends on the dimensionality of the arguments.
    /// Cases: 
    /// (1, 1) -> vector-vector dot product resulting in a scalar
    /// (2, 1) -> matrix-vector dot product resulting in a vector
    /// (2, 2) -> matrix-matrix dot product resulting in a matrix
    /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
    /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
    static member dot (a: Expr2) (b: Expr2) =
        let sa, sb = a.Shape, b.Shape
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
        | 1, 1 -> 
            // vector-vector dot product
            Expr2.sum (a * b)
        | 2, 1 -> 
            // matrix-vector dot product
            let bm = b |> Expr2.reshape (ShapeSpec.padRight sb)
            {Dot.X=a; Y=bm} |> Expr2 |> Expr2.reshape [sa.[0]]
        | 2, 2 -> 
            // matrix-matrix dot product
            {Dot.X=a; Y=b} |> Expr2
        | na, nb when na = nb -> 
            // batched matrix-matrix dot product
            let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa sb
            let ba = a |> Expr2.broadcast bsa
            let bb = b |> Expr2.broadcast bsb    
            {Dot.X=ba; Y=bb} |> Expr2
        | na, nb when na = nb + 1 ->
            // batched matrix-vector dot product
            let psb = ShapeSpec.padRight sb
            let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa psb
            let ba = a |> Expr2.broadcast bsa
            let bb = b |> Expr2.reshape psb |> Expr2.broadcast bsb    
            {Dot.X=ba; Y=bb} |> Expr2 |> Expr2.reshape bsa.[0 .. na-2]
        | _ -> failwithf "Cannot compute dot product between tensors of shapes %A and %A." sa sb  

    /// Tensor product.
    static member tensorProduct (x: Expr2) (y: Expr2) =
        {TensorProduct.X=x; Y=y} |> Expr2

   /// Elementwise uses elements from `ifTrue` if `cond` is true for that element, otherwise elements from `ifFalse`.
    static member ifThenElse (cond: Expr2) (ifTrue: Expr2) (ifFalse: Expr2) =
        let shps = [cond.Shape; ifTrue.Shape; ifFalse.Shape]
        let pShps = ShapeSpec.padToSameMany shps
        let bcShps = ShapeSpec.broadcastToSameMany false pShps           
        match pShps, bcShps with
        | [condPShp; ifTruePShp; ifFalsePShp], [condBcShp; ifTrueBcShp; ifFalseBcShp] -> 
            let condBc = cond |> Expr2.reshape condPShp |> Expr2.broadcast condBcShp
            let ifTrueBc = ifTrue |> Expr2.reshape ifTruePShp |> Expr2.broadcast ifTrueBcShp
            let ifFalseBc = ifFalse |> Expr2.reshape ifFalsePShp |> Expr2.broadcast ifFalseBcShp
            {IfThenElse.Cond=condBc; IfTrue=ifTrueBc; IfFalse=ifFalseBc} |> Expr2
        | _ -> failwith "impossible"

    /// Discards the results of all arguments.
    static member discard (xs: Expr2 list) =
        let xs = xs |> List.map (fun x -> x :> BaseExpr)
        {Discard.Xs=xs} |> Expr2

    /// Build tensor from numeric ranges.
    static member internal buildTensor shape ranges xs =
        {BuildTensor.Shape=shape; Ranges=ranges; Xs=xs} |> Expr2
    
    /// Calculates a tensor elementwise using the given element expression and result shape.
    static member elements shape elemExpr xs =
        {Elements.Shape=shape; ElemExpr=elemExpr; Xs=xs} |> Expr2

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate interpolator xs =
        let xs = Expr2.broadcastToSameMany xs
        let xs = xs |> List.map (fun x -> x :> BaseExpr)
        {Interpolate.Interpolator=interpolator; Xs=xs} |> Expr2

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate1D interpolator x =
        Expr2.interpolate interpolator [x]

    /// Element-wise two-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate2D interpolator x y =
        Expr2.interpolate interpolator [x; y]

    /// Element-wise three-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate3D interpolator x y z =
        Expr2.interpolate interpolator [x; y; z]

