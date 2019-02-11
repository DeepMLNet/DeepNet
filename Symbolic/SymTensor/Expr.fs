namespace SymTensor

open SymTensor.Ops
open DeepNet.Utils



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
        if ss = expr.Shape then expr else Expr2 (OpForwards.Reshape ss expr)

    /// Broadcasts the expression into the given shape.
    static member broadcast ss (expr: Expr2) =
        if ss = expr.Shape then expr else Expr2 (OpForwards.DoBroadcast ss expr)

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
        expr |> OpForwards.PermuteAxes permutation |> Expr2

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

    interface IDynElem

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
                srs, (Expr2 (OpForwards.Subtensor srs this)).Shape
            | [:? RangesSpec as frs] ->
                // split into simplified range specification and reshape operation
                splitFRS frs this.Shape [] []
            | _ ->
                // parse, then split into simplified range specification and reshape operation
                splitFRS (argList |> parseArgs) this.Shape [] []

        // emit expression
        let sub = OpForwards.Subtensor srs this |> Expr2
        let reshaped = OpForwards.Reshape reshp sub |> Expr2
        reshaped

    member this.Item 
        with get ([<System.ParamArray>] allArgs: obj []) = 
            this.GetSlice (allArgs)

    /// Expression a with the specified subtensor replaced with b.
    static member setSubtensor (trgt: Expr2) (src: Expr2) =
        match OpForwards.IsSubtensor trgt with
        | Some (range, subtensorExpr, trgtExpr) ->
            let srcReshaped = OpForwards.Reshape subtensorExpr.Shape src |> Expr2
            OpForwards.SetSubtensor range trgtExpr srcReshaped |> Expr2
        | None ->
            invalidArg "trgt" "The first argument of setSubtensor must be an item or slice of an expression, i.e. a.[...]."                 

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    static member constructElementwise op (a: Expr2) (b: Expr2) =
        let psa, psb = ShapeSpec.padToSame a.Shape b.Shape
        let bsa, bsb = ShapeSpec.broadcastToSame false psa psb
        let ba = a |> Expr2.reshape psa |> Expr2.broadcast bsa
        let bb = b |> Expr2.reshape psb |> Expr2.broadcast bsb    
        Expr2 (op ba bb)

    // elementwise unary arithmetic
    static member (~+) (x: Expr2) = Expr2 (OpForwards.UnaryPlus x)
    static member (~-) (x: Expr2) = Expr2 (OpForwards.Negate x)
    static member Abs (x: Expr2) = Expr2 (OpForwards.Abs x)
    static member SignT (x: Expr2) = Expr2 (OpForwards.SignT x)
    static member Log (x: Expr2) = Expr2 (OpForwards.Log x)
    static member Log10 (x: Expr2) = Expr2 (OpForwards.Log10 x)
    static member Exp (x: Expr2) = Expr2 (OpForwards.Exp x)
    static member Sin (x: Expr2) = Expr2 (OpForwards.Sin x)
    static member Cos (x: Expr2) = Expr2 (OpForwards.Cos x)
    static member Tan (x: Expr2) = Expr2 (OpForwards.Tan x)
    static member Asin (x: Expr2) = Expr2 (OpForwards.Asin x)
    static member Acos (x: Expr2) = Expr2 (OpForwards.Acos x)
    static member Atan (x: Expr2) = Expr2 (OpForwards.Atan x)
    static member Sinh (x: Expr2) = Expr2 (OpForwards.Sinh x)
    static member Cosh (x: Expr2) = Expr2 (OpForwards.Cosh x)
    static member Tanh (x: Expr2) = Expr2 (OpForwards.Tanh x)
    static member Sqrt (x: Expr2) = Expr2 (OpForwards.Sqrt x)
    static member Ceiling (x: Expr2) = Expr2 (OpForwards.Ceiling x)
    static member Floor (x: Expr2) = Expr2 (OpForwards.Floor x)
    static member Round (x: Expr2) = Expr2 (OpForwards.Round x)
    static member Truncate (x: Expr2) = Expr2 (OpForwards.Truncate x)

    // element-wise unary logic
    static member (~~~~) (x: Expr2) = Expr2 (OpForwards.Not x)

    // elementwise binary arithmetic
    static member (+) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Add x y
    static member (-) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Subtract x y
    static member (*) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Multiply x y
    static member (/) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Divide x y
    static member (%) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Modulo x y
    static member Pow (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Pow x y   
    static member ( *** ) (x: Expr2, y: Expr2) = x ** y

    // element-wise binary logic
    static member (&&&&) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.And x y
    static member (||||) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Or x y

    // element-wise binary comparison
    static member (====) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Equal x y
    static member (<<<<) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Less x y
    static member (<<==) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.LessOrEqual x y
    static member (>>>>) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.Greater x y
    static member (>>==) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.GreaterOrEqual x y
    static member (<<>>) (x: Expr2, y: Expr2) = Expr2.constructElementwise OpForwards.NotEqual x y

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

    static member ( .* ) (x: Expr2, y: Expr2) = OpForwards.Dot x y

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

