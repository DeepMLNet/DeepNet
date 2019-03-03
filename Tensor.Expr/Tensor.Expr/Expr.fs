namespace rec Tensor.Expr

open System

open DeepNet.Utils
open Tensor.Expr.Ops
open Tensor.Backend


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


type IExpr =
    inherit System.IEquatable<IExpr>
    inherit System.IComparable<IExpr>

    abstract BaseExpr: BaseExpr
    abstract BaseExprCh: BaseExprCh
    abstract Op: IOp
    abstract TypeName: TypeName
    abstract DataType: System.Type
    abstract Shape: ShapeSpec
    abstract NDims: int
    abstract NElems: SizeSpec
    abstract Dev: ITensorDevice
    abstract Args: Map<Ch, IExpr>
    abstract Vars: Set<BaseVar>
    abstract CanEvalAllSymSizes: bool

    abstract Pretty: string
    abstract ToString: maxLength:int -> string
    abstract ToString: unit -> string

    abstract Reshape: ShapeSpec -> IExpr
    abstract Broadcast: ShapeSpec -> IExpr



module IExpr =

    let ofBaseExpr (baseExpr: BaseExpr) : IExpr =
         if not (baseExpr.IsSingleChannel) then
            failwithf "IExpr is for single-channel expressions only, but got %A." baseExpr 
         let exprType = typedefof<Expr<_>>.MakeGenericType baseExpr.[Ch.Default].DataType
         Activator.CreateInstance(exprType, [|box baseExpr|]) :?> IExpr        

    let ofBaseExprCh (exprCh: BaseExprCh) : IExpr =
        let exprType = typedefof<Expr<_>>.MakeGenericType exprCh.DataType
        Activator.CreateInstance(exprType, [|box exprCh|]) :?> IExpr
 
    let ofOp (op: IOp) : IExpr =
        ofBaseExpr (BaseExpr.ofOp op)

    let ofBaseVar (var: BaseVar) : IExpr =
        ofOp {VarArg.Var=var}
        
    let baseExpr (expr: IExpr) = expr.BaseExpr
    let baseExprCh (expr: IExpr) = expr.BaseExprCh
    let op (expr: IExpr) = expr.Op
    let typeName (expr: IExpr) = expr.TypeName
    let dataType (expr: IExpr) = expr.DataType
    let shape (expr: IExpr) = expr.Shape
    let nDims (expr: IExpr) = expr.NDims
    let nElems (expr: IExpr) = expr.NElems
    let dev (expr: IExpr) = expr.Dev
    let args (expr: IExpr) = expr.Args
    let vars (expr: IExpr) = expr.Vars
    let canEvalAllSymSizes (expr: IExpr) = expr.CanEvalAllSymSizes

    let reshape (shape: ShapeSpec) (expr: IExpr) = expr.Reshape shape
    let broadcast (shape: ShapeSpec) (expr: IExpr) = expr.Broadcast shape


    /// pads and broadcasts all arguments to same shape if possible
    let broadcastToSameMany (es: IExpr list) =
        let ss = es |> List.map shape
        let ps = ShapeSpec.padToSameMany ss
        let bs = ShapeSpec.broadcastToSameMany false ps
        List.zip3 es ps bs
        |> List.map (fun (e, p, b) -> e |> reshape p |> broadcast b)      
      

/// An tensor-valued expression with a single output channel.
[<StructuredFormatDisplay("{Pretty}")>]
type Expr<'T> (baseExpr: BaseExpr) =    
    do 
        if not (baseExpr.IsSingleChannel) then
            failwithf "Expr<'T> is for single-channel expressions only, but got %A." baseExpr
        if baseExpr.[Ch.Default].DataType <> typeof<'T> then
            failwithf "Cannot use Expr<%A> for BaseExpr %A of data type %A." 
                      typeof<'T> baseExpr baseExpr.[Ch.Default].DataType
    
    /// Create expression from specified single-channel op.
    new (op: IOp) =
        Expr<'T> (BaseExpr.ofOp op)

    /// Create expression by accessing the specified channel of the BaseExpr.
    new (exprCh: BaseExprCh) =
        match exprCh with
        | BaseExprCh (Ch.Default, baseExpr) -> Expr baseExpr
        | BaseExprCh (Ch.Custom chName, baseExpr) ->
            Expr<'T> {Channel.X=baseExpr.[Ch.Custom chName]}

    /// Expression having the value of the specified variable.
    new (var: Var<'T>) =
        Expr<'T> {VarArg.Var=var.BaseVar}

    member this.BaseExpr = baseExpr
    static member baseExpr (expr: Expr<'T>) = expr.BaseExpr

    member this.BaseExprCh = baseExpr.[Ch.Default]
    static member baseExprCh (expr: Expr<'T>) = expr.BaseExprCh

    member this.Op = baseExpr.Op
    static member op (expr: Expr<'T>) = expr.Op

    member this.TypeName = baseExpr.[Ch.Default].TypeName
    static member typeName (expr: Expr<'T>) = expr.TypeName

    member this.DataType = baseExpr.[Ch.Default].DataType
    static member dataType (expr: Expr<'T>) = expr.DataType

    member this.Shape = baseExpr.[Ch.Default].Shape
    static member shape (expr: Expr<'T>) = expr.Shape

    member this.NDims = baseExpr.[Ch.Default].NDims
    static member nDims (expr: Expr<'T>) = expr.NDims

    member this.NElems = baseExpr.[Ch.Default].NElems
    static member nElems (expr: Expr<'T>) = expr.NElems

    member this.Dev = baseExpr.[Ch.Default].Dev
    static member dev (expr: Expr<'T>) = expr.Dev

    member this.Args = baseExpr.Args |> Map.map (fun _ arg -> IExpr.ofBaseExprCh arg)
    static member args (expr: Expr<'T>) = expr.Args

    //member this.Arg 
    //    with get (arg: Arg) : Expr<'A> = Expr<'A> baseExpr.Args.[arg]

    member this.Vars = baseExpr.Vars
    static member vars (expr: Expr<'T>) = expr.Vars

    member this.CanEvalAllSymSizes = baseExpr.CanEvalAllSymSizes
    static member canEvalAllSymSizes (expr: Expr<'T>) = expr.CanEvalAllSymSizes

    static member substSymSizes (env: SymSizeEnv) (expr: Expr<'T>) : Expr<'T> =
        expr.BaseExpr |> BaseExpr.substSymSizes env |> Expr

    interface System.IEquatable<Expr<'T>> with
        member this.Equals other = this.BaseExpr = other.BaseExpr

    override this.Equals other =
        match other with
        | :? Expr<'T> as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<Expr<'T>> with
        member this.CompareTo other = compare this.BaseExpr other.BaseExpr

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? Expr<'T> as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare Expr<%A> to type %A." this.DataType (other.GetType())

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
    static member internal checkAxis ax (expr: Expr<'T>) =
        if not (0 <= ax && ax < expr.NDims) then
            failwithf "Specified axis %d is invalid for expression of shape %A." ax expr.Shape

    /// Reshapes the expression into the given shape.
    /// The element count must not change.
    static member reshape ss (expr: Expr<'T>) : Expr<'T> =
        if ss = expr.Shape then expr 
        else Expr {Reshape.Shape=ss; X=expr.BaseExprCh}

    /// Broadcasts the expression into the given shape.
    static member broadcast ss (expr: Expr<'T>) : Expr<'T> =
        if ss = expr.Shape then expr 
        else Expr {DoBroadcast.Shape=ss; X=expr.BaseExprCh}

    /// Inserts a broadcast axis at the given dimension.
    static member insertBroadcastAxis dim (expr: Expr<'T>) : Expr<'T> =
        expr |> Expr.reshape (expr.Shape |> ShapeSpec.insertBroadcastAxis dim)

    /// adds one broadcastable dimension to the left
    static member padLeft (a: Expr<'T>) : Expr<'T> =
        a |> Expr.reshape (ShapeSpec.padLeft a.Shape)

    /// adds one broadcastable dimension to the right
    static member padRight (a: Expr<'T>) : Expr<'T> =
        a |> Expr.reshape (ShapeSpec.padRight a.Shape)

    /// Reshapes the expression so that a single dimension remains.
    static member flatten (expr: Expr<'T>) =
        expr |> Expr.reshape (ShapeSpec.flatten expr.Shape)

    /// pads from the left and broadcasts the argument to the given shape if possible
    static member broadcastToShape shp (a: Expr<'T>) =
        let psa = a.Shape |> ShapeSpec.padTo (ShapeSpec.nDim shp)
        let bsa = psa |> ShapeSpec.broadcastToShape shp
        a |> Expr.reshape psa |> Expr.broadcast bsa        

    /// pads and broadcasts all arguments to same shape if possible
    static member broadcastToSameMany (es: Expr<'T> list) =
        let ss = es |> List.map Expr.shape
        let ps = ShapeSpec.padToSameMany ss
        let bs = ShapeSpec.broadcastToSameMany false ps
        List.zip3 es ps bs
        |> List.map (fun (e, p, b) -> e |> Expr.reshape p |> Expr.broadcast b)

    /// pads and broadcasts `a` and `b` to same shape if possible
    static member broadcastToSame (a: Expr<'T>) (b: Expr<'T>) =
        match Expr.broadcastToSameMany [a; b] with
        | [bcA; bcB] -> bcA, bcB
        | _ -> failwith "impossible"

    /// enables broadcasting in the given dimension, it must be of size one
    static member enableBroadcast dim (a: Expr<'T>) = 
        a |> Expr.reshape (a.Shape |> ShapeSpec.enableBroadcast dim)

    /// disables broadcasting in the given dimension
    static member disableBroadcast dim (a: Expr<'T>) =
        a |> Expr.reshape (a.Shape |> ShapeSpec.disableBroadcast dim)
  
    /// scalar constant of given value
    static member scalar dev (value: 'T) : Expr<'T> = 
        Expr<'T> {Scalar.Value=Const value; Dev=dev} 

    ///// scalar of given value converted to same type as given expression
    //static member scalarLike (expr: Expr<'T>) value : Expr<'T> = 
    //    let v = System.Convert.ChangeType (box value, expr.DataType) :?> 'T
    //    Expr.scalar expr.Dev v

    /// Scalar with value of given size and type int64.
    static member size dev (size: SizeSpec) = 
        Expr<int64> {SizeValue.Value=size; Dev=dev} 

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    static member permuteAxes permutation (expr: Expr<'T>) =
        Expr<'T> {PermuteAxes.Permutation=permutation; X=expr.BaseExprCh}

    /// Swaps two dimensions of a tensor.
    static member swapDim ax1 ax2 (expr: Expr<'T>) = 
        expr |> Expr.checkAxis ax1
        expr |> Expr.checkAxis ax2
        if ax1 = ax2 then expr
        else
            let perm = 
                [0 .. expr.NDims - 1]
                |> List.map (function
                             | d when d=ax1 -> ax2
                             | d when d=ax2 -> ax1
                             | d -> d)
            expr |> Expr.permuteAxes perm

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    static member transpose (expr: Expr<'T>) =
        if expr.NDims < 2 then invalidArg "expr" "Need at least a matrix to transpose."
        expr |> Expr.swapDim (expr.NDims - 2) (expr.NDims - 1)

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    member this.T = Expr.transpose this

    // item / slicing
    member this.GetSlice ([<System.ParamArray>] allArgs: obj []) : Expr<'T> =

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
            | (:? (Expr<int64> option) as so)     :: (:? (PlusElems option) as fo)   :: rest ->
                RangeSpec.DynStartSymSize (so.Value.BaseExprCh, fo.Value.Elems) :: parseArgs rest
            | null                           :: null                           :: rest ->
                RangeSpec.SymStartSymEnd (None, None) :: parseArgs rest

            // items
            | (:? SizeSpec as s)     :: rest -> RangeSpec.SymElem s :: parseArgs rest
            | (:? int64 as s)        :: rest when s = Tensor.TensorVal.NewAxis -> RangeSpec.NewAxis :: parseArgs rest
            | (:? int64 as s)        :: rest when s = Tensor.TensorVal.Fill    -> RangeSpec.AllFill :: parseArgs rest
            | (:? Expr<int64> as e)  :: rest  -> RangeSpec.DynElem e.BaseExprCh :: parseArgs rest                                                             
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
                srs, (Expr {Subtensor.Range=srs; X=this.BaseExprCh}).Shape
            | [:? RangesSpec as frs] ->
                // split into simplified range specification and reshape operation
                splitFRS frs this.Shape [] []
            | _ ->
                // parse, then split into simplified range specification and reshape operation
                splitFRS (argList |> parseArgs) this.Shape [] []

        // emit expression
        let sub = {Subtensor.Range=srs; X=this.BaseExprCh} |> Expr
        let reshaped = {Reshape.Shape=reshp; X=sub.BaseExprCh} |> Expr
        reshaped

    member this.Item 
        with get ([<System.ParamArray>] allArgs: obj []) = 
            this.GetSlice (allArgs)

    /// Expression a with the specified subtensor replaced with b.
    static member setSubtensor (trgt: Expr<'T>) (src: Expr<'T>) =
        match trgt.BaseExpr with
        | ExprHelpers.SubtensorExpr (range, subtensorExpr, trgtExpr) ->
            let srcReshaped = Expr {Reshape.Shape=subtensorExpr.[Ch.Default].Shape; X=src.BaseExprCh}
            Expr<'T> {SetSubtensor.Range=range; X=trgtExpr; Y=srcReshaped.BaseExprCh}
        | _ ->
            invalidArg "trgt" "The first argument of setSubtensor must be an item or slice of an expression, i.e. a.[...]."                 

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    static member constructElementwise op (a: Expr<'I>) (b: Expr<'I>) =
        let psa, psb = ShapeSpec.padToSame a.Shape b.Shape
        let bsa, bsb = ShapeSpec.broadcastToSame false psa psb
        let ba = a |> Expr.reshape psa |> Expr.broadcast bsa
        let bb = b |> Expr.reshape psb |> Expr.broadcast bsb 
        let opInst: IOp = op ba bb
        Expr<'T> opInst

    // elementwise unary arithmetic
    static member (~+) (x: Expr<'T>) = Expr<'T> {UnaryPlus.X=x.BaseExprCh}
    static member (~-) (x: Expr<'T>) = Expr<'T> {Negate.X=x.BaseExprCh}
    static member Abs (x: Expr<'T>) = Expr<'T> {Abs.X=x.BaseExprCh}
    static member SignT (x: Expr<'T>) = Expr<'T> {SignT.X=x.BaseExprCh}
    static member Log (x: Expr<'T>) = Expr<'T> {Log.X=x.BaseExprCh}
    static member Log10 (x: Expr<'T>) = Expr<'T> {Log10.X=x.BaseExprCh}
    static member Exp (x: Expr<'T>) = Expr<'T> {Exp.X=x.BaseExprCh}
    static member Sin (x: Expr<'T>) = Expr<'T> {Sin.X=x.BaseExprCh}
    static member Cos (x: Expr<'T>) = Expr<'T> {Cos.X=x.BaseExprCh}
    static member Tan (x: Expr<'T>) = Expr<'T> {Tan.X=x.BaseExprCh}
    static member Asin (x: Expr<'T>) = Expr<'T> {Asin.X=x.BaseExprCh}
    static member Acos (x: Expr<'T>) = Expr<'T> {Acos.X=x.BaseExprCh}
    static member Atan (x: Expr<'T>) = Expr<'T> {Atan.X=x.BaseExprCh}
    static member Sinh (x: Expr<'T>) = Expr<'T> {Sinh.X=x.BaseExprCh}
    static member Cosh (x: Expr<'T>) = Expr<'T> {Cosh.X=x.BaseExprCh}
    static member Tanh (x: Expr<'T>) = Expr<'T> {Tanh.X=x.BaseExprCh}
    static member Sqrt (x: Expr<'T>) = Expr<'T> {Sqrt.X=x.BaseExprCh}
    static member Ceiling (x: Expr<'T>) = Expr<'T> {Ceiling.X=x.BaseExprCh}
    static member Floor (x: Expr<'T>) = Expr<'T> {Floor.X=x.BaseExprCh}
    static member Round (x: Expr<'T>) = Expr<'T> {Round.X=x.BaseExprCh}
    static member Truncate (x: Expr<'T>) = Expr<'T> {Truncate.X=x.BaseExprCh}

    // element-wise unary logic
    static member (~~~~) (x: Expr<bool>) = Expr<bool> {Not.X=x.BaseExprCh}

    // elementwise binary arithmetic
    static member (+) (x: Expr<'T>, y: Expr<'T>) = Expr<'T>.constructElementwise (fun x y -> {Add.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (-) (x: Expr<'T>, y: Expr<'T>) = Expr<'T>.constructElementwise (fun x y -> {Subtract.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (*) (x: Expr<'T>, y: Expr<'T>) = Expr<'T>.constructElementwise (fun x y -> {Multiply.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (/) (x: Expr<'T>, y: Expr<'T>) = Expr<'T>.constructElementwise (fun x y -> {Divide.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (%) (x: Expr<'T>, y: Expr<'T>) = Expr<'T>.constructElementwise (fun x y -> {Modulo.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member Pow (x: Expr<'T>, y: Expr<'T>) = Expr<'T>.constructElementwise (fun x y -> {Pow.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y   
    static member ( *** ) (x: Expr<'T>, y: Expr<'T>) = x ** y

    // element-wise binary logic
    static member (&&&&) (x: Expr<bool>, y: Expr<bool>) = Expr<bool>.constructElementwise (fun x y -> {And.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (||||) (x: Expr<bool>, y: Expr<bool>) = Expr<bool>.constructElementwise (fun x y -> {Or.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y

    // element-wise binary comparison
    static member (====) (x: Expr<'T>, y: Expr<'T>) = Expr<bool>.constructElementwise (fun x y -> {Equal.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (<<<<) (x: Expr<'T>, y: Expr<'T>) = Expr<bool>.constructElementwise (fun x y -> {Less.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (<<==) (x: Expr<'T>, y: Expr<'T>) = Expr<bool>.constructElementwise (fun x y -> {LessOrEqual.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (>>>>) (x: Expr<'T>, y: Expr<'T>) = Expr<bool>.constructElementwise (fun x y -> {Greater.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (>>==) (x: Expr<'T>, y: Expr<'T>) = Expr<bool>.constructElementwise (fun x y -> {GreaterOrEqual.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y
    static member (<<>>) (x: Expr<'T>, y: Expr<'T>) = Expr<bool>.constructElementwise (fun x y -> {NotEqual.X=x.BaseExprCh; Y=y.BaseExprCh} :> IOp) x y

    // element-wise binary logic with basetype
    static member (&&&&) (x: Expr<bool>, y: bool) = x &&&& (Expr.scalar x.Dev y)
    static member (||||) (x: Expr<bool>, y: bool) = x |||| (Expr.scalar x.Dev y)

    static member (&&&&) (x: bool, y: Expr<bool>) = (Expr.scalar y.Dev x) &&&& y
    static member (||||) (x: bool, y: Expr<bool>) = (Expr.scalar y.Dev x) |||| y

    // elementwise binary arithmetic with basetype
    static member (+) (x: Expr<'T>, y: 'T) = x + (Expr.scalar x.Dev y)
    static member (-) (x: Expr<'T>, y: 'T) = x - (Expr.scalar x.Dev y)
    static member (*) (x: Expr<'T>, y: 'T) = x * (Expr.scalar x.Dev y)
    static member (/) (x: Expr<'T>, y: 'T) = x / (Expr.scalar x.Dev y)
    static member (%) (x: Expr<'T>, y: 'T) = x % (Expr.scalar x.Dev y)
    static member Pow (x: Expr<'T>, y: 'T) = x ** (Expr.scalar x.Dev y)
    static member ( *** ) (x: Expr<'T>, y: 'T) = x ** (Expr.scalar x.Dev y)   

    static member (+) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) + y
    static member (-) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) - y
    static member (*) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) * y
    static member (/) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) / y
    static member (%) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) % y
    static member Pow (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) ** y
    static member ( *** ) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) ** y

    // element-wise binary comparison with basetype
    static member (====) (x: Expr<'T>, y: 'T) = x ==== (Expr.scalar x.Dev y)
    static member (<<<<) (x: Expr<'T>, y: 'T) = x <<<< (Expr.scalar x.Dev y)
    static member (<<==) (x: Expr<'T>, y: 'T) = x <<== (Expr.scalar x.Dev y)
    static member (>>>>) (x: Expr<'T>, y: 'T) = x >>>> (Expr.scalar x.Dev y)
    static member (>>==) (x: Expr<'T>, y: 'T) = x >>== (Expr.scalar x.Dev y)
    static member (<<>>) (x: Expr<'T>, y: 'T) = x <<>> (Expr.scalar x.Dev y)

    static member (====) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) ==== y
    static member (<<<<) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) <<<< y
    static member (<<==) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) <<== y
    static member (>>>>) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) >>>> y
    static member (>>==) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) >>== y
    static member (<<>>) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) <<>> y

    /// Dot product.
    static member ( .* ) (x: Expr<'T>, y: Expr<'T>) = Expr<'T> {Dot.X=x.BaseExprCh; Y=y.BaseExprCh}

    /// Sign keeping type.
    static member signt (expr: Expr<'T>) =
        Expr.SignT expr

    /// Square root.
    static member sqrtt (expr: Expr<'T>) =
        Expr.Sqrt expr

    /// Tensor of given shape filled with specified value.
    static member filled (dev: ITensorDevice) (shp: ShapeSpec) (value: 'T) =
        let bcShp = shp |> List.map (fun _ -> SizeSpec.broadcastable)
        Expr.scalar dev value |> Expr.reshape bcShp |> Expr.broadcast shp

    /// Zero tensor of given shape.
    static member zeros dev (shp: ShapeSpec) =
        Expr.filled dev shp (conv<'T> 0)

    ///// Zero tensor of given type and shape.
    //static member zerosOfType typ dev shp =
    //    Expr.filled dev shp (convTo typ 0)

    ///// zero tensor with same shape and type as given tensor
    //static member zerosLike (expr: Expr) = 
    //    Expr.zerosOfType expr.DataType expr.Dev expr.Shape

    ///// Identity matrix of given size and type.
    //static member identityOfType typ dev size =
    //    Expr {Identity.Size=size; Type=TypeName.ofTypeInst typ; Dev=dev}

    /// Identity matrix of given size.
    static member identity dev size = 
        Expr<'T> {Identity.Size=size; Type=TypeName.ofType<'T>; Dev=dev}

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    static member invert (x: Expr<'T>) =
        Expr<'T> {Invert.X=x.BaseExprCh}

    /// Reverses the tensor in the specified dimension.
    static member reverseAxis axis (x: Expr<'T>) =
        Expr<'T> {ReverseAxis.Axis=axis; X=x.BaseExprCh} 

    /// Concatenates the sequence of tensors in the specified dimension.
    static member concat dim (es: Expr<'T> seq) =
        // check that arguments are correctly sized
        let es = List.ofSeq es
        let shps = es |> List.map Expr.shape
        match es with
        | [] -> failwithf "need at least one tensor to concatenate"
        | h :: ts ->
            if not (0 <= dim && dim < h.NDims) then
                failwithf "cannot concatenate over non-existant dimension %d given shapes %A" dim shps
            for t in ts do
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
            ((Expr<'T>.zeros es.Head.Dev shp, SizeSpec.zero), es)
            ||> List.fold (fun (concatSoFar, pos) e ->
                let len = e.Shape.[dim]
                let slice: RangesSpec = 
                    List.replicate e.NDims RangeSpec.All
                    |> List.set dim (RangeSpec.SymStartSymEnd (Some pos, Some (pos + len - 1L)))
                Expr.setSubtensor concatSoFar.[slice] e, pos + len)
        concatenated

    /// Extracts the diagonal along the given axes.
    static member diagAxis ax1 ax2 (x: Expr<'T>) = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        Expr<'T> {Diag.Axis1=ax1; Axis2=ax2; Diag.X=x.BaseExprCh} 
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    static member diag (x: Expr<'T>) = 
        if x.NDims < 2 then 
            failwithf "Need at least a matrix to extract diagonal but got shape: %A" x.Shape
        x |> Expr.diagAxis (x.NDims-2) (x.NDims-1)

    /// Creates a diagonal matrix by duplicating the given dimension.
    static member diagMatAxis ax1 ax2 (x: Expr<'T>) = 
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        Expr<'T> {DiagMat.Axis1=ax1; Axis2=ax2; X=x.BaseExprCh} 

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    static member diagMat (x: Expr<'T>) =
        if x.NDims < 1 then 
            failwithf "Need at least a vector to build diagonal matrix but got shape: %A" x.Shape
        x |> Expr<'T>.diagMatAxis (x.NDims-1) x.NDims

    /// summation over given dimension
    static member sumAxis (axis: int) (x: Expr<'T>) = 
        Expr<'T> {SumAxis.Axis=axis; X=x.BaseExprCh} 

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    static member sumKeepingAxis (axis: int) (x: Expr<'T>) =
        x |> Expr.sumAxis axis |> Expr.insertBroadcastAxis axis

    /// summaiton of all elements
    static member sum (x: Expr<'T>) = 
        x |> Expr.flatten |> Expr.sumAxis 0

    /// Computes the traces along the given axes.
    static member traceAxis (ax1: int) (ax2: int) (x: Expr<'T>) =
        let tax = if ax1 < ax2 then ax1 else ax1 + 1
        x |> Expr.diagAxis ax1 ax2 |> Expr.sumAxis tax

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    static member trace (x: Expr<'T>) =
        if x.NDims < 2 then
            failwithf "Need at least a matrix for trace but got shape: %A" x.Shape      
        x |> Expr.traceAxis (x.NDims-2) (x.NDims-1) 
    
    /// product over given dimension
    static member productAxis (axis: int) (x: Expr<'T>) = 
        Expr<'T> {ProductAxis.Axis=axis; X=x.BaseExprCh} 

    /// product over given dimension, while keeping the axis with one (broadcastable) element
    static member productKeepingAxis (axis: int) (x: Expr<'T>) =
        x |> Expr.productAxis axis |> Expr.insertBroadcastAxis axis

    /// product of all elements
    static member product (x: Expr<'T>) = 
        x |> Expr.flatten |> Expr.productAxis 0

    /// Maximum over given dimension.
    static member maxAxis (axis: int) (x: Expr<'T>) = 
        Expr<'T> {MaxAxis.Axis=axis; X=x.BaseExprCh} 

    /// Maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member maxKeepingAxis (axis: int) (x: Expr<'T>) =
        x |> Expr.maxAxis axis |> Expr.insertBroadcastAxis axis

    /// Maximum of all elements.
    static member max (x: Expr<'T>) = 
        x |> Expr.flatten |> Expr.maxAxis 0

    /// Minimum over given dimension.
    static member minAxis (axis: int) (x: Expr<'T>) = 
        Expr<'T> {MinAxis.Axis=axis; X=x.BaseExprCh} 

    /// Minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member minKeepingAxis (axis: int) (x: Expr<'T>) =
        x |> Expr.minAxis axis |> Expr.insertBroadcastAxis axis

    /// Minimum of all elements.
    static member min (x: Expr<'T>) = 
        x |> Expr.flatten |> Expr.minAxis 0

    /// Index of maximum over given dimension.
    static member argMaxAxis (axis: int) (x: Expr<'T>) = 
        Expr<'T> {ArgMaxAxis.Axis=axis; X=x.BaseExprCh} 

    /// Index of maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMaxKeepingAxis (axis: int) (x: Expr<'T>) =
        x |> Expr.argMaxAxis axis |> Expr.insertBroadcastAxis axis

    /// Index of minimum over given dimension.
    static member argMinAxis (axis: int) (x: Expr<'T>) = 
        Expr<'T> {MinAxis.Axis=axis; X=x.BaseExprCh} 

    /// Index of minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMinKeepingAxis (axis: int) (x: Expr<'T>) =
        x |> Expr.minAxis axis |> Expr.insertBroadcastAxis axis

    /// Select elements according to the specified index tensors.
    static member gather (indices: Expr<int64> option list) (x: Expr<'T>) =
        let someIndices = indices |> List.choose id
        if List.isEmpty someIndices then
            failwith "Gather needs at least one specified index tensor."
        let bcSomeIndices = Expr.broadcastToSameMany someIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs ->
                Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSomeIndices
        let bcIndices = bcIndices |> List.map (Option.map Expr.baseExprCh)
        Expr<'T> {Gather.Indices=bcIndices; X=x.BaseExprCh} 

    /// Disperses elements according to the specified index tensors.
    static member scatter (indices: Expr<int64> option list) (trgtShp: ShapeSpec) (x: Expr<'T>) =
        let indices = indices |> List.map (Option.map (Expr.broadcastToShape x.Shape))
        let indices = indices |> List.map (Option.map Expr.baseExprCh)
        Expr<'T> {Scatter.Indices=indices; Shape=trgtShp; X=x.BaseExprCh} 

    /// Nullifies the Jacobian of its argument when calculating derivatives.
    static member assumeZeroDeriv (x: Expr<'T>) =
        Expr<'T> {AssumeZeroDeriv.X=x.BaseExprCh} 

    /// Assumes the specified Jacobian when calculating derivatives.
    static member assumeDeriv (deriv: Expr<'T>) (x: Expr<'T>) =
        Expr<'T> {AssumeDeriv.Deriv=deriv.BaseExprCh; X=x.BaseExprCh} 

    /// Annotated expression (no influence on value).
    static member annotate label (x: Expr<'T>) = 
        Expr<'T> {Annotated.Label=label; X=x.BaseExprCh} 

    /// Print the result with the given label when evaluated.
    static member print (label: string) (x: Expr<'T>) =
        Expr<'T> {Print.Label=label; X=x.BaseExprCh} 

    /// Dumps the result into the given dataset in the active HDF5 dump file.
    static member dump (dataset: string) (x: Expr<'T>) =
        Expr<'T> {Dump.Dataset=dataset; X=x.BaseExprCh} 

    /// If the value contains NaNs or infinities, outputs their location and 
    /// stops the computation.
    static member checkFinite (label: string) (x: Expr<'T>) =
        Expr<'T> {CheckFinite.Label=label; X=x.BaseExprCh} 

    /// Dot product.
    /// Behavior depends on the dimensionality of the arguments.
    /// Cases: 
    /// (1, 1) -> vector-vector dot product resulting in a scalar
    /// (2, 1) -> matrix-vector dot product resulting in a vector
    /// (2, 2) -> matrix-matrix dot product resulting in a matrix
    /// (n, n) with n>2 -> batched matrix-matrix dot product resulting in a matrix
    /// (n+1, n) with n>2 -> batched matrix-vector dot product resulting in a vector.
    static member dot (a: Expr<'T>) (b: Expr<'T>) =
        let sa, sb = a.Shape, b.Shape
        match ShapeSpec.nDim sa, ShapeSpec.nDim sb with
        | 1, 1 -> 
            // vector-vector dot product
            Expr.sum (a * b)
        | 2, 1 -> 
            // matrix-vector dot product
            let bm = b |> Expr.reshape (ShapeSpec.padRight sb)
            Expr<'T> {Dot.X=a.BaseExprCh; Y=bm.BaseExprCh} |> Expr.reshape [sa.[0]]
        | 2, 2 -> 
            // matrix-matrix dot product
            Expr<'T> {Dot.X=a.BaseExprCh; Y=b.BaseExprCh} 
        | na, nb when na = nb -> 
            // batched matrix-matrix dot product
            let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa sb
            let ba = a |> Expr.broadcast bsa
            let bb = b |> Expr.broadcast bsb    
            Expr<'T> {Dot.X=ba.BaseExprCh; Y=bb.BaseExprCh} 
        | na, nb when na = nb + 1 ->
            // batched matrix-vector dot product
            let psb = ShapeSpec.padRight sb
            let bsa, bsb = ShapeSpec.broadcastToSameInDims [0 .. na-3] false sa psb
            let ba = a |> Expr.broadcast bsa
            let bb = b |> Expr.reshape psb |> Expr.broadcast bsb    
            Expr<'T> {Dot.X=ba.BaseExprCh; Y=bb.BaseExprCh} |> Expr.reshape bsa.[0 .. na-2]
        | _ -> failwithf "Cannot compute dot product between tensors of shapes %A and %A." sa sb  

    /// Tensor product.
    static member tensorProduct (x: Expr<'T>) (y: Expr<'T>) =
        Expr<'T> {TensorProduct.X=x.BaseExprCh; Y=y.BaseExprCh} 

   /// Elementwise uses elements from `ifTrue` if `cond` is true for that element, otherwise elements from `ifFalse`.
    static member ifThenElse (cond: Expr<bool>) (ifTrue: Expr<'T>) (ifFalse: Expr<'T>) =
        let shps = [cond.Shape; ifTrue.Shape; ifFalse.Shape]
        let pShps = ShapeSpec.padToSameMany shps
        let bcShps = ShapeSpec.broadcastToSameMany false pShps           
        match pShps, bcShps with
        | [condPShp; ifTruePShp; ifFalsePShp], [condBcShp; ifTrueBcShp; ifFalseBcShp] -> 
            let condBc = cond |> Expr.reshape condPShp |> Expr.broadcast condBcShp
            let ifTrueBc = ifTrue |> Expr.reshape ifTruePShp |> Expr.broadcast ifTrueBcShp
            let ifFalseBc = ifFalse |> Expr.reshape ifFalsePShp |> Expr.broadcast ifFalseBcShp
            Expr<'T> {IfThenElse.Cond=condBc.BaseExprCh; IfTrue=ifTrueBc.BaseExprCh; IfFalse=ifFalseBc.BaseExprCh} 
        | _ -> failwith "impossible"

    ///// Discards the results of all arguments.
    //static member discard (xs: IExpr list) =
    //    let xs = xs |> List.map Expr.baseExprCh
    //    Expr<int32> {Discard.Xs=xs} 

    /// Build tensor from numeric ranges.
    static member internal buildTensor shape ranges (xs: Expr<'T> list) =
        let xs = xs |> List.map Expr.baseExprCh
        Expr<'T> {BuildTensor.Shape=shape; Ranges=ranges; Xs=xs} 
    
    /// Calculates a tensor elementwise using the given element expression and result shape.
    static member elements shape elemExpr (xs: Expr<'T> list) =
        let xs = xs |> List.map Expr.baseExprCh
        Expr<'T> {Elements.Shape=shape; ElemExpr=elemExpr; Xs=xs} 

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate interpolator (xs: Expr<'T> list) =
        let xs = Expr.broadcastToSameMany xs
        let xs = xs |> List.map Expr.baseExprCh
        Expr<'T> {Interpolate.Interpolator=interpolator; Xs=xs} 

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate1D interpolator (x: Expr<'T>) =
        Expr.interpolate interpolator [x]

    /// Element-wise two-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate2D interpolator (x: Expr<'T>) (y: Expr<'T>) =
        Expr.interpolate interpolator [x; y]

    /// Element-wise three-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate3D interpolator (x: Expr<'T>) (y: Expr<'T>) (z: Expr<'T>) =
        Expr.interpolate interpolator [x; y; z]

    /// Evaluates the expression into a numeric value.
    static member eval (varEnv: VarEnv) (expr: Expr<'T>) = 
        let evalEnv : EvalEnv = {VarEnv=varEnv}
        let chVals = BaseExprEval.eval evalEnv expr.BaseExpr
        chVals.[Ch.Default] :?> Tensor.Tensor<'T>




