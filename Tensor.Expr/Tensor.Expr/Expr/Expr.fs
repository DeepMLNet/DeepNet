namespace rec Tensor.Expr

open DeepNet.Utils
open Tensor.Expr.Ops
open Tensor.Backend



/// An tensor-valued expression with a single output channel.
[<StructuredFormatDisplay("{Pretty}")>]
type Expr<'T> (baseExpr: BaseExpr) =    

    do 
        if baseExpr.[Ch.Default].DataType <> typeof<'T> then
            failwithf "Cannot use Expr<%A> for expression %A of data type %A." 
                      typeof<'T> baseExpr baseExpr.[Ch.Default].DataType
    
    /// Create expression from specified single-channel op.
    new (op: IOp) =
        Expr<'T> (BaseExpr.ofOp op)

    /// Create expression by accessing the specified channel of the BaseExpr.
    new (exprCh: BaseExprCh) =
        match exprCh with
        | BaseExprCh (Ch.Default, baseExpr) -> Expr<'T> baseExpr
        | BaseExprCh (ch, baseExpr) ->
            Expr<'T> {Channel.X=baseExpr.[ch]}

    /// Create typed expression from specified untyped expression.
    new (expr: UExpr) =
        Expr<'T> expr.BaseExpr

    /// Expression having the value of the specified variable.
    new (var: Var<'T>) = 
        Expr<'T> (UExpr var.Untyped)

    /// Expression having the value of the specified tensor.
    /// A reference to that tensor is stored.
    new (tensor: Tensor.Tensor<'T>) =
        Expr<'T> {DataArg.Data=OrdRef (tensor :> Tensor.ITensor)}

    member this.Untyped = UExpr baseExpr
    static member untyped (expr: Expr<'T>) = expr.Untyped

    member this.BaseExpr = this.Untyped.BaseExpr
    static member baseExpr (expr: Expr<'T>) = expr.BaseExpr

    member this.BaseExprCh = this.Untyped.BaseExprCh
    static member baseExprCh (expr: Expr<'T>) = expr.BaseExprCh

    member this.Op = this.Untyped.Op
    static member op (expr: Expr<'T>) = expr.Op

    member this.TypeName = this.Untyped.TypeName
    static member typeName (expr: Expr<'T>) = expr.TypeName

    member this.DataType = this.Untyped.DataType
    static member dataType (expr: Expr<'T>) = expr.DataType

    member this.Shape = this.Untyped.Shape
    static member shape (expr: Expr<'T>) = expr.Shape

    member this.NDims = this.Untyped.NDims
    static member nDims (expr: Expr<'T>) = expr.NDims

    member this.NElems = this.Untyped.NElems
    static member nElems (expr: Expr<'T>) = expr.NElems

    member this.Dev = this.Untyped.Dev
    static member dev (expr: Expr<'T>) = expr.Dev

    member this.Args = this.Untyped.Args
    static member args (expr: Expr<'T>) = expr.Args

    /// Typed argument expression.
    member this.Arg (arg: Arg) : Expr<'A> = Expr<'A> baseExpr.Args.[arg]
    static member arg (arg: Arg) (expr: Expr<'T>) = expr.Arg arg

    member this.VarMap = baseExpr.VarMap
    static member varMap (expr: Expr<'T>) = expr.VarMap

    member this.Vars = this.Untyped.Vars
    static member vars (expr: Expr<'T>) = expr.Vars

    member this.CanEvalAllSymSizes = this.Untyped.CanEvalAllSymSizes
    static member canEvalAllSymSizes (expr: Expr<'T>) = expr.CanEvalAllSymSizes

    static member substSymSizes (env: SizeEnv) (expr: Expr<'T>) : Expr<'T> =
        UExpr.substSymSizes env expr.Untyped |> Expr<'T>

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
            | _ -> failwithf "Cannot compare Expr<%A> to type %A." typeof<'T> (other.GetType())

    override this.GetHashCode() = hash this.BaseExpr

    member this.ToString maxLength = this.Untyped.ToString maxLength
    override this.ToString () = this.Untyped.ToString ()
    member this.Pretty = this.Untyped.Pretty

    /// Reshapes the expression into the given shape.
    /// The element count must not change.
    static member reshape ss (expr: Expr<'T>) =
        UExpr.reshape ss expr.Untyped |> Expr<'T>

    /// Broadcasts the expression into the given shape.
    static member broadcast ss (expr: Expr<'T>) =
        UExpr.broadcast ss expr.Untyped |> Expr<'T>

    /// Inserts a broadcast axis at the given dimension.
    static member insertBroadcastAxis dim (expr: Expr<'T>) =
        UExpr.insertBroadcastAxis dim expr.Untyped |> Expr<'T>

    /// adds one broadcastable dimension to the left
    static member padLeft (a: Expr<'T>) =
        UExpr.padLeft a.Untyped |> Expr<'T>

    /// adds one broadcastable dimension to the right
    static member padRight (a: Expr<'T>) =
        UExpr.padRight a.Untyped |> Expr<'T>

    /// Reshapes the expression so that a single dimension remains.
    static member flatten (expr: Expr<'T>) =
        UExpr.flatten expr.Untyped |> Expr<'T>

    /// pads from the left and broadcasts the argument to the given shape if possible
    static member broadcastToShape shp (a: Expr<'T>) =
        UExpr.broadcastToShape shp a.Untyped |> Expr<'T>      

    /// pads and broadcasts all arguments to same shape if possible
    static member broadcastToSameMany (es: Expr<'T> list) =
        es
        |> List.map (fun expr -> expr.Untyped)
        |> UExpr.broadcastToSameMany
        |> List.map (fun expr -> expr |> Expr<'T>)

    /// pads and broadcasts `a` and `b` to same shape if possible
    static member broadcastToSame (a: Expr<'A>) (b: Expr<'B>) =
        let bcA, bcB = UExpr.broadcastToSame a.Untyped b.Untyped 
        bcA |> Expr<'A>, bcB |> Expr<'B>

    /// enables broadcasting in the given dimension, it must be of size one
    static member enableBroadcast dim (a: Expr<'T>) = 
        UExpr.enableBroadcast dim a.Untyped |> Expr<'T>

    /// disables broadcasting in the given dimension
    static member disableBroadcast dim (a: Expr<'T>) =
        UExpr.disableBroadcast dim a.Untyped |> Expr<'T>
  
    /// scalar constant of given value
    static member scalar dev (value: 'T) = 
        UExpr.scalar dev value |> Expr<'T>

    /// Scalar constant of same type and stored on same device as this expression.
    member this.Scalar (value: obj) =
        UExpr.scalar this.Dev (convTo this.DataType value) |> Expr<'T>

    /// Converts the expression data type from another type.
    static member convert (expr: Expr<'C>) =
        UExpr.convert typeof<'T> expr.Untyped |> Expr<'T>

    /// Transfers the data to the specified device.
    static member transfer (dev: ITensorDevice) (expr: Expr<'T>) =
        UExpr.transfer dev expr.Untyped |> Expr<'T>

    /// Scalar with value of given size and type int64.
    static member size dev (size: Size) = 
        UExpr.size dev size |> Expr<int64>

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    static member permuteAxes permutation (expr: Expr<'T>) =
        UExpr.permuteAxes permutation expr.Untyped |> Expr<'T>

    /// Swaps two dimensions of a tensor.
    static member swapDim ax1 ax2 (expr: Expr<'T>) = 
        UExpr.swapDim ax1 ax2 expr.Untyped |> Expr<'T>

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    static member transpose (expr: Expr<'T>) =
        UExpr.transpose expr.Untyped |> Expr<'T>

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    member this.T = this.Untyped.T |> Expr<'T>

    // item / slicing
    member this.GetSlice ([<System.ParamArray>] allArgs: obj []) : Expr<'T> =
        this.Untyped.GetSlice (allArgs) |> Expr<'T>

    member this.Item 
        with get ([<System.ParamArray>] allArgs: obj []) = 
            this.GetSlice (allArgs)

    /// Expression a with the specified subtensor replaced with b.
    static member setSubtensor (trgt: Expr<'T>) (src: Expr<'T>) =
        UExpr.setSubtensor trgt.Untyped src.Untyped |> Expr<'T>

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    static member constructElementwise op (a: Expr<'I>) (b: Expr<'I>) =
        UExpr.constructElementwise op a.Untyped b.Untyped |> Expr<'T>

    // elementwise unary arithmetic
    static member (~+) (x: Expr<'T>) = +(x.Untyped) |> Expr<'T>
    static member (~-) (x: Expr<'T>) = -(x.Untyped) |> Expr<'T>
    static member Abs (x: Expr<'T>) = abs (x.Untyped) |> Expr<'T>
    static member SignT (x: Expr<'T>) = UExpr.SignT (x.Untyped) |> Expr<'T>
    static member Log (x: Expr<'T>) = log (x.Untyped) |> Expr<'T>
    static member Log10 (x: Expr<'T>) = log10 (x.Untyped) |> Expr<'T>
    static member Exp (x: Expr<'T>) = exp (x.Untyped) |> Expr<'T>
    static member Sin (x: Expr<'T>) = sin (x.Untyped) |> Expr<'T>
    static member Cos (x: Expr<'T>) = cos (x.Untyped) |> Expr<'T>
    static member Tan (x: Expr<'T>) = tan (x.Untyped) |> Expr<'T>
    static member Asin (x: Expr<'T>) = asin (x.Untyped) |> Expr<'T>
    static member Acos (x: Expr<'T>) = acos (x.Untyped) |> Expr<'T>
    static member Atan (x: Expr<'T>) = atan (x.Untyped) |> Expr<'T>
    static member Sinh (x: Expr<'T>) = sinh (x.Untyped) |> Expr<'T>
    static member Cosh (x: Expr<'T>) = cosh (x.Untyped) |> Expr<'T>
    static member Tanh (x: Expr<'T>) = tanh (x.Untyped) |> Expr<'T>
    static member Sqrt (x: Expr<'T>) = sqrt (x.Untyped) |> Expr<'T>
    static member Ceiling (x: Expr<'T>) = ceil (x.Untyped) |> Expr<'T>
    static member Floor (x: Expr<'T>) = floor (x.Untyped) |> Expr<'T>
    static member Round (x: Expr<'T>) = round (x.Untyped) |> Expr<'T>
    static member Truncate (x: Expr<'T>) = truncate (x.Untyped) |> Expr<'T>

    // element-wise unary logic
    static member (~~~~) (x: Expr<bool>) = ~~~~(x.Untyped) |> Expr<bool>

    // elementwise binary arithmetic
    static member (+) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) + (y.Untyped) |> Expr<'T>
    static member (-) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) - (y.Untyped) |> Expr<'T>
    static member (*) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) * (y.Untyped) |> Expr<'T>
    static member (/) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) / (y.Untyped) |> Expr<'T>
    static member (%) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) % (y.Untyped) |> Expr<'T>
    static member Pow (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) ** (y.Untyped) |> Expr<'T>   
    static member ( *** ) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) *** (y.Untyped) |> Expr<'T>

    // element-wise binary logic
    static member (&&&&) (x: Expr<bool>, y: Expr<bool>) = (x.Untyped) &&&& (y.Untyped) |> Expr<bool>
    static member (||||) (x: Expr<bool>, y: Expr<bool>) = (x.Untyped) |||| (y.Untyped) |> Expr<bool>

    // element-wise binary comparison
    static member (====) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) ==== (y.Untyped) |> Expr<bool>
    static member (<<<<) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) <<<< (y.Untyped) |> Expr<bool>
    static member (<<==) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) <<== (y.Untyped) |> Expr<bool> 
    static member (>>>>) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) >>>> (y.Untyped) |> Expr<bool>
    static member (>>==) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) >>== (y.Untyped) |> Expr<bool>
    static member (<<>>) (x: Expr<'T>, y: Expr<'T>) = (x.Untyped) <<>> (y.Untyped) |> Expr<bool>

    // element-wise binary logic with basetype
    static member (&&&&) (x: Expr<bool>, y: bool) = (x.Untyped) &&&& (UExpr.scalar x.Dev y)
    static member (||||) (x: Expr<bool>, y: bool) = (x.Untyped) |||| (UExpr.scalar x.Dev y)

    static member (&&&&) (x: bool, y: Expr<bool>) = (UExpr.scalar y.Dev x) &&&& (y.Untyped)
    static member (||||) (x: bool, y: Expr<bool>) = (UExpr.scalar y.Dev x) |||| (y.Untyped)

    // elementwise binary arithmetic with basetype
    static member (+) (x: Expr<'T>, y: 'T) = (x.Untyped) + (UExpr.scalar x.Dev y) |> Expr<'T>   
    static member (-) (x: Expr<'T>, y: 'T) = (x.Untyped) - (UExpr.scalar x.Dev y) |> Expr<'T>   
    static member (*) (x: Expr<'T>, y: 'T) = (x.Untyped) * (UExpr.scalar x.Dev y) |> Expr<'T>   
    static member (/) (x: Expr<'T>, y: 'T) = (x.Untyped) / (UExpr.scalar x.Dev y) |> Expr<'T>   
    static member (%) (x: Expr<'T>, y: 'T) = (x.Untyped) % (UExpr.scalar x.Dev y) |> Expr<'T>   
    static member Pow (x: Expr<'T>, y: 'T) = (x.Untyped) ** (UExpr.scalar x.Dev y) |> Expr<'T>   
    static member ( *** ) (x: Expr<'T>, y: 'T) = (x.Untyped) *** (UExpr.scalar x.Dev y) |> Expr<'T>  

    static member (+) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) + (y.Untyped) |> Expr<'T>
    static member (-) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) - (y.Untyped) |> Expr<'T>
    static member (*) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) * (y.Untyped) |> Expr<'T>
    static member (/) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) / (y.Untyped) |> Expr<'T>
    static member (%) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) % (y.Untyped) |> Expr<'T>
    static member Pow (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) ** (y.Untyped) |> Expr<'T>
    static member ( *** ) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) *** (y.Untyped) |> Expr<'T>

    // element-wise binary comparison with basetype
    static member (====) (x: Expr<'T>, y: 'T) = (x.Untyped) ==== (UExpr.scalar x.Dev y) |> Expr<bool>
    static member (<<<<) (x: Expr<'T>, y: 'T) = (x.Untyped) <<<< (UExpr.scalar x.Dev y) |> Expr<bool>
    static member (<<==) (x: Expr<'T>, y: 'T) = (x.Untyped) <<== (UExpr.scalar x.Dev y) |> Expr<bool>
    static member (>>>>) (x: Expr<'T>, y: 'T) = (x.Untyped) >>>> (UExpr.scalar x.Dev y) |> Expr<bool>
    static member (>>==) (x: Expr<'T>, y: 'T) = (x.Untyped) >>== (UExpr.scalar x.Dev y) |> Expr<bool>
    static member (<<>>) (x: Expr<'T>, y: 'T) = (x.Untyped) <<>> (UExpr.scalar x.Dev y) |> Expr<bool>

    static member (====) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) ==== (y.Untyped) |> Expr<bool>
    static member (<<<<) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) <<<< (y.Untyped) |> Expr<bool>
    static member (<<==) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) <<== (y.Untyped) |> Expr<bool>
    static member (>>>>) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) >>>> (y.Untyped) |> Expr<bool>
    static member (>>==) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) >>== (y.Untyped) |> Expr<bool>
    static member (<<>>) (x: 'T, y: Expr<'T>) = (UExpr.scalar y.Dev x) <<>> (y.Untyped) |> Expr<bool>

    /// Dot product.
    static member ( .* ) (x: Expr<'T>, y: Expr<'T>) = 
        (x.Untyped) .* (y.Untyped) |> Expr<'T>

    /// Sign keeping type.
    static member signt (expr: Expr<'T>) =
        UExpr.signt expr.Untyped |> Expr<'T>

    /// Square root.
    static member sqrtt (expr: Expr<'T>) =
        UExpr.sqrtt expr.Untyped |> Expr<'T>

    /// Elementwise maximum.
    static member maxElemwise (x: Expr<'T>) (y: Expr<'T>) =
        UExpr.maxElemwise x.Untyped y.Untyped |> Expr<'T>

    /// Elementwise minimum.
    static member minElemwise (x: Expr<'T>) (y: Expr<'T>) =
        UExpr.minElemwise x.Untyped y.Untyped |> Expr<'T>

    /// Ensures that all elements are between minVal and maxVal.
    /// Values outside these limits are capped to the limits.
    static member limit (x: Expr<'T>, ?minVal: 'T, ?maxVal: 'T) =
        UExpr.limit (x.Untyped, ?minVal=Option.map box minVal, ?maxVal=Option.map box maxVal) |> Expr<'T> 

    /// Tensor of given shape filled with specified value.
    static member filled (dev: ITensorDevice) (shp: Shape) (value: 'T) =
        UExpr.filled dev shp (box value) |> Expr<'T>

    /// Zero tensor of given shape.
    static member zeros dev (shp: Shape) =
        UExpr.zeros typeof<'T> dev shp |> Expr<'T>

    /// Identity matrix of given size.
    static member identity dev size = 
        UExpr.identity typeof<'T> dev size |> Expr<'T>

    /// Vector counting from zero to given size minus one.
    static member counting dev size =
        UExpr {Counting.Size=size; Dev=dev} |> Expr<int64>

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    static member invert (x: Expr<'T>) =
        UExpr.invert x.Untyped |> Expr<'T>

    /// Reverses the tensor in the specified dimension.
    static member reverseAxis axis (x: Expr<'T>) =
        UExpr.reverseAxis axis x.Untyped |> Expr<'T>

    /// Concatenates the sequence of tensors in the specified dimension.
    static member concat dim (es: Expr<'T> seq) =
        let es = es |> Seq.map (fun expr -> expr.Untyped)
        UExpr.concat dim es |> Expr<'T>

    /// Extracts the diagonal along the given axes.
    static member diagAxis ax1 ax2 (x: Expr<'T>) = 
        UExpr.diagAxis ax1 ax2 x.Untyped |> Expr<'T>
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    static member diag (x: Expr<'T>) = 
        UExpr.diag x.Untyped |> Expr<'T>

    /// Creates a diagonal matrix by duplicating the given dimension.
    static member diagMatAxis ax1 ax2 (x: Expr<'T>) = 
        UExpr.diagMatAxis ax1 ax2 x.Untyped |> Expr<'T>

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    static member diagMat (x: Expr<'T>) =
        UExpr.diagMat x.Untyped |> Expr<'T>

    /// summation over given dimension
    static member sumAxis (axis: int) (x: Expr<'T>) = 
        UExpr.sumAxis axis x.Untyped |> Expr<'T>

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    static member sumKeepingAxis (axis: int) (x: Expr<'T>) =
        UExpr.sumKeepingAxis axis x.Untyped |> Expr<'T>

    /// summaiton of all elements
    static member sum (x: Expr<'T>) = 
        UExpr.sum x.Untyped |> Expr<'T>

    /// Computes the traces along the given axes.
    static member traceAxis (ax1: int) (ax2: int) (x: Expr<'T>) =
        UExpr.traceAxis ax1 ax2 x.Untyped |> Expr<'T>

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    static member trace (x: Expr<'T>) =
        UExpr.trace x.Untyped |> Expr<'T>
    
    /// product over given dimension
    static member productAxis (axis: int) (x: Expr<'T>) = 
        UExpr.productAxis axis x.Untyped |> Expr<'T>

    /// product over given dimension, while keeping the axis with one (broadcastable) element
    static member productKeepingAxis (axis: int) (x: Expr<'T>) =
        UExpr.productKeepingAxis axis x.Untyped |> Expr<'T>

    /// product of all elements
    static member product (x: Expr<'T>) = 
        UExpr.product x.Untyped |> Expr<'T>

    /// Maximum over given dimension.
    static member maxAxis (axis: int) (x: Expr<'T>) = 
        UExpr.maxAxis axis x.Untyped |> Expr<'T>

    /// Maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member maxKeepingAxis (axis: int) (x: Expr<'T>) =
        UExpr.maxKeepingAxis axis x.Untyped |> Expr<'T>

    /// Maximum of all elements.
    static member max (x: Expr<'T>) = 
        UExpr.max x.Untyped |> Expr<'T>

    /// Minimum over given dimension.
    static member minAxis (axis: int) (x: Expr<'T>) = 
        UExpr.minAxis axis x.Untyped |> Expr<'T>

    /// Minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member minKeepingAxis (axis: int) (x: Expr<'T>) =
        UExpr.minKeepingAxis axis x.Untyped |> Expr<'T>

    /// Minimum of all elements.
    static member min (x: Expr<'T>) = 
        UExpr.min x.Untyped |> Expr<'T>

    /// Index of maximum over given dimension.
    static member argMaxAxis (axis: int) (x: Expr<'T>) = 
        UExpr.argMaxAxis axis x.Untyped |> Expr<'T>

    /// Index of maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMaxKeepingAxis (axis: int) (x: Expr<'T>) =
        UExpr.argMaxKeepingAxis axis x.Untyped |> Expr<'T>

    /// Index of minimum over given dimension.
    static member argMinAxis (axis: int) (x: Expr<'T>) = 
        UExpr.argMinAxis axis x.Untyped |> Expr<'T>

    /// Index of minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMinKeepingAxis (axis: int) (x: Expr<'T>) =
        UExpr.argMinKeepingAxis axis x.Untyped |> Expr<'T>

    /// mean over all elements
    static member mean (x: Expr<'T>) = 
        UExpr.mean x.Untyped |> Expr<'T>

    /// mean over given dimension
    static member meanAxis (axis: int) (x: Expr<'T>) =
        UExpr.meanAxis axis x.Untyped |> Expr<'T>

    /// mean over given dimension, while keeping the axis with one (broadcastable) element
    static member meanKeepingAxis (axis: int) (x: Expr<'T>) =
        UExpr.meanKeepingAxis axis x.Untyped |> Expr<'T>

    /// Select elements according to the specified index tensors.
    static member gather (indices: Expr<int64> option list) (x: Expr<'T>) =
        let indices = indices |> List.map (Option.map (fun expr -> expr.Untyped))
        UExpr.gather indices x.Untyped |> Expr<'T>

    /// Disperses elements according to the specified index tensors.
    static member scatter (indices: Expr<int64> option list) (trgtShp: Shape) (x: Expr<'T>) =
        let indices = indices |> List.map (Option.map (fun expr -> expr.Untyped))
        UExpr.scatter indices trgtShp x.Untyped |> Expr<'T>

    /// Nullifies the Jacobian of its argument when calculating derivatives.
    static member assumeZeroDeriv (x: Expr<'T>) =
        UExpr.assumeZeroDeriv x.Untyped |> Expr<'T>

    /// Assumes the specified Jacobian when calculating derivatives.
    static member assumeDeriv (deriv: Expr<'T>) (x: Expr<'T>) =
        UExpr.assumeDeriv deriv.Untyped x.Untyped |> Expr<'T>

    /// Annotated expression (no influence on value).
    static member annotate label (x: Expr<'T>) = 
        UExpr.annotate label x.Untyped |> Expr<'T>

    /// Print the result with the given label when evaluated.
    static member print (label: string) (x: Expr<'T>) =
        UExpr.print label x.Untyped |> Expr<'T>

    /// Dumps the result into the given dataset in the active HDF5 dump file.
    static member dump (dataset: string) (x: Expr<'T>) =
        UExpr.dump dataset x.Untyped |> Expr<'T>

    /// If the value contains NaNs or infinities, outputs their location and 
    /// stops the computation.
    static member checkFinite (label: string) (x: Expr<'T>) =
        UExpr.checkFinite label x.Untyped |> Expr<'T>

    /// Tensor product.
    static member tensorProduct (x: Expr<'T>) (y: Expr<'T>) =
        UExpr.tensorProduct x.Untyped y.Untyped |> Expr<'T>

   /// Elementwise uses elements from `ifTrue` if `cond` is true for that element, otherwise elements from `ifFalse`.
    static member ifThenElse (cond: Expr<bool>) (ifTrue: Expr<'T>) (ifFalse: Expr<'T>) =
        UExpr.ifThenElse cond.Untyped ifTrue.Untyped ifFalse.Untyped |> Expr<'T>

    /// Build tensor from numeric ranges.
    static member internal buildTensor shape ranges (xs: Expr<'T> list) =
        let xs = xs |> List.map (fun expr -> expr.Untyped)
        UExpr.buildTensor shape ranges xs |> Expr<'T>
    
    /// Calculates a tensor elementwise using the given element expression and result shape.
    static member elements shape elemExpr (xs: Expr<'T> list) =
        let xs = xs |> List.map (fun expr -> expr.Untyped)
        UExpr.elements shape elemExpr xs |> Expr<'T>

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate interpolator (xs: Expr<'T> list) =
        let xs = xs |> List.map (fun expr -> expr.Untyped)
        UExpr.interpolate interpolator xs |> Expr<'T>

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate1D interpolator (x: Expr<'T>) =
        UExpr.interpolate1D interpolator x.Untyped |> Expr<'T>

    /// Element-wise two-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate2D interpolator (x: Expr<'T>) (y: Expr<'T>) =
        UExpr.interpolate2D interpolator x.Untyped y.Untyped |> Expr<'T>

    /// Element-wise three-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate3D interpolator (x: Expr<'T>) (y: Expr<'T>) (z: Expr<'T>) =
        UExpr.interpolate3D interpolator x.Untyped y.Untyped z.Untyped |> Expr<'T>

    /// Evaluates the expression into a numeric value.
    static member eval (varEnv: VarEnv) (expr: Expr<'T>) : Tensor.Tensor<'T> = 
        UExpr.eval varEnv expr.Untyped :?> Tensor.Tensor<'T>

    /// Substitutes the variables within the expression tree.
    static member substVars (env: Map<VarName, UExpr>) (expr: Expr<'T>) =
        let env = env |> Map.map (fun _ sExpr -> sExpr.BaseExpr)
        expr.BaseExpr |> BaseExpr.substVars env |> Expr<'T>







