namespace rec Tensor.Expr

open System

open DeepNet.Utils
open Tensor.Expr.Ops
open Tensor.Backend



/// An tensor-valued expression with a single output channel.
[<StructuredFormatDisplay("{Pretty}")>]
type Expr<'T> (baseExpr: BaseExpr) =    
    inherit Expr (baseExpr)

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
        | BaseExprCh (Ch.Custom chName, baseExpr) ->
            Expr<'T> {Channel.X=baseExpr.[Ch.Custom chName]}

    /// Create typed expression from specified untyped expression.
    new (expr: Expr) =
        Expr<'T> expr.BaseExpr

    /// Expression having the value of the specified variable.
    static member var (var: Var<'T>) = 
        Expr.baseVar var.BaseVar |> Expr<'T>

    /// Argument expression.
    member this.Arg (arg: Arg) : Expr<'A> = Expr<'A> baseExpr.Args.[arg]

    static member substSymSizes (env: SymSizeEnv) (expr: Expr<'T>) : Expr<'T> =
        expr.BaseExpr |> BaseExpr.substSymSizes env |> Expr<'T>

    interface System.IEquatable<Expr<'T>> with
        member this.Equals other = this.BaseExpr = other.BaseExpr

    interface System.IComparable<Expr<'T>> with
        member this.CompareTo other = compare this.BaseExpr other.BaseExpr

    /// Reshapes the expression into the given shape.
    /// The element count must not change.
    static member reshape ss (expr: Expr<'T>) =
        Expr.reshape ss expr |> Expr<'T>

    /// Broadcasts the expression into the given shape.
    static member broadcast ss (expr: Expr<'T>) =
        Expr.broadcast ss expr |> Expr<'T>

    /// Inserts a broadcast axis at the given dimension.
    static member insertBroadcastAxis dim (expr: Expr<'T>) =
        Expr.insertBroadcastAxis dim expr |> Expr<'T>

    /// adds one broadcastable dimension to the left
    static member padLeft (a: Expr<'T>) =
        Expr.padLeft a |> Expr<'T>

    /// adds one broadcastable dimension to the right
    static member padRight (a: Expr<'T>) =
        Expr.padRight a |> Expr<'T>

    /// Reshapes the expression so that a single dimension remains.
    static member flatten (expr: Expr<'T>) =
        Expr.flatten expr |> Expr<'T>

    /// pads from the left and broadcasts the argument to the given shape if possible
    static member broadcastToShape shp (a: Expr<'T>) =
        Expr.broadcastToShape shp a |> Expr<'T>      

    /// pads and broadcasts all arguments to same shape if possible
    static member broadcastToSameMany (es: Expr<'T> list) =
        es
        |> List.map (fun expr -> expr :> Expr)
        |> Expr.broadcastToSameMany
        |> List.map (fun expr -> expr |> Expr<'T>)

    /// pads and broadcasts `a` and `b` to same shape if possible
    static member broadcastToSame (a: Expr<'A>) (b: Expr<'B>) =
        let bcA, bcB = Expr.broadcastToSame a b 
        bcA |> Expr<'A>, bcB |> Expr<'B>

    /// enables broadcasting in the given dimension, it must be of size one
    static member enableBroadcast dim (a: Expr<'T>) = 
        Expr.enableBroadcast dim a |> Expr<'T>

    /// disables broadcasting in the given dimension
    static member disableBroadcast dim (a: Expr<'T>) =
        Expr.disableBroadcast dim a |> Expr<'T>
  
    /// scalar constant of given value
    static member scalar dev (value: 'T) = 
        Expr.scalar dev value |> Expr<'T>

    /// Scalar with value of given size and type int64.
    static member size dev (size: SizeSpec) = 
        Expr.size dev size |> Expr<int64>

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the *new* position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    static member permuteAxes permutation (expr: Expr<'T>) =
        Expr.permuteAxes permutation expr |> Expr<'T>

    /// Swaps two dimensions of a tensor.
    static member swapDim ax1 ax2 (expr: Expr<'T>) = 
        Expr.swapDim ax1 ax2 expr |> Expr<'T>

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    static member transpose (expr: Expr<'T>) =
        Expr.transpose expr |> Expr<'T>

    /// Transpose matrix.
    /// If the input has more than two dimensions, the last two axes are transposed.
    member this.T = Expr.transpose this

    // item / slicing
    member this.GetSlice ([<System.ParamArray>] allArgs: obj []) : Expr<'T> =
        base.GetSlice (allArgs) |> Expr<'T>

    member this.Item 
        with get ([<System.ParamArray>] allArgs: obj []) = 
            this.GetSlice (allArgs)

    /// Expression a with the specified subtensor replaced with b.
    static member setSubtensor (trgt: Expr<'T>) (src: Expr<'T>) =
        Expr.setSubtensor trgt src |> Expr<'T>

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    static member constructElementwise op (a: Expr<'I>) (b: Expr<'I>) =
        Expr.constructElementwise op a b |> Expr<'T>

    // elementwise unary arithmetic
    static member (~+) (x: Expr<'T>) = +(x :> Expr) |> Expr<'T>
    static member (~-) (x: Expr<'T>) = -(x :> Expr) |> Expr<'T>
    static member Abs (x: Expr<'T>) = abs (x :> Expr) |> Expr<'T>
    static member SignT (x: Expr<'T>) = Expr.SignT (x :> Expr) |> Expr<'T>
    static member Log (x: Expr<'T>) = log (x :> Expr) |> Expr<'T>
    static member Log10 (x: Expr<'T>) = log10 (x :> Expr) |> Expr<'T>
    static member Exp (x: Expr<'T>) = exp (x :> Expr) |> Expr<'T>
    static member Sin (x: Expr<'T>) = sin (x :> Expr) |> Expr<'T>
    static member Cos (x: Expr<'T>) = cos (x :> Expr) |> Expr<'T>
    static member Tan (x: Expr<'T>) = tan (x :> Expr) |> Expr<'T>
    static member Asin (x: Expr<'T>) = asin (x :> Expr) |> Expr<'T>
    static member Acos (x: Expr<'T>) = acos (x :> Expr) |> Expr<'T>
    static member Atan (x: Expr<'T>) = atan (x :> Expr) |> Expr<'T>
    static member Sinh (x: Expr<'T>) = sinh (x :> Expr) |> Expr<'T>
    static member Cosh (x: Expr<'T>) = cosh (x :> Expr) |> Expr<'T>
    static member Tanh (x: Expr<'T>) = tanh (x :> Expr) |> Expr<'T>
    static member Sqrt (x: Expr<'T>) = sqrt (x :> Expr) |> Expr<'T>
    static member Ceiling (x: Expr<'T>) = ceil (x :> Expr) |> Expr<'T>
    static member Floor (x: Expr<'T>) = floor (x :> Expr) |> Expr<'T>
    static member Round (x: Expr<'T>) = round (x :> Expr) |> Expr<'T>
    static member Truncate (x: Expr<'T>) = truncate (x :> Expr) |> Expr<'T>

    // element-wise unary logic
    static member (~~~~) (x: Expr<bool>) = ~~~~(x :> Expr) |> Expr<bool>

    // elementwise binary arithmetic
    static member (+) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) + (y :> Expr) |> Expr<'T>
    static member (-) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) - (y :> Expr) |> Expr<'T>
    static member (*) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) * (y :> Expr) |> Expr<'T>
    static member (/) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) / (y :> Expr) |> Expr<'T>
    static member (%) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) % (y :> Expr) |> Expr<'T>
    static member Pow (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) ** (y :> Expr) |> Expr<'T>   
    static member ( *** ) (x: Expr<'T>, y: Expr<'T>) = x ** y

    // element-wise binary logic
    static member (&&&&) (x: Expr<bool>, y: Expr<bool>) = (x :> Expr) &&&& (y :> Expr) |> Expr<bool>
    static member (||||) (x: Expr<bool>, y: Expr<bool>) = (x :> Expr) |||| (y :> Expr) |> Expr<bool>

    // element-wise binary comparison
    static member (====) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) ==== (y :> Expr) |> Expr<bool>
    static member (<<<<) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) <<<< (y :> Expr) |> Expr<bool>
    static member (<<==) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) <<== (y :> Expr) |> Expr<bool> 
    static member (>>>>) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) >>>> (y :> Expr) |> Expr<bool>
    static member (>>==) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) >>== (y :> Expr) |> Expr<bool>
    static member (<<>>) (x: Expr<'T>, y: Expr<'T>) = (x :> Expr) <<>> (y :> Expr) |> Expr<bool>

    // element-wise binary logic with basetype
    static member (&&&&) (x: Expr<bool>, y: bool) = (x :> Expr) &&&& y
    static member (||||) (x: Expr<bool>, y: bool) = (x :> Expr) |||| y

    static member (&&&&) (x: bool, y: Expr<bool>) = x &&&& (y :> Expr)
    static member (||||) (x: bool, y: Expr<bool>) = x |||| (y :> Expr)

    // elementwise binary arithmetic with basetype
    static member (+) (x: Expr<'T>, y: 'T) = (x :> Expr) + (Expr.scalar x.Dev y) |> Expr<'T>   
    static member (-) (x: Expr<'T>, y: 'T) = (x :> Expr) - (Expr.scalar x.Dev y) |> Expr<'T>   
    static member (*) (x: Expr<'T>, y: 'T) = (x :> Expr) * (Expr.scalar x.Dev y) |> Expr<'T>   
    static member (/) (x: Expr<'T>, y: 'T) = (x :> Expr) / (Expr.scalar x.Dev y) |> Expr<'T>   
    static member (%) (x: Expr<'T>, y: 'T) = (x :> Expr) % (Expr.scalar x.Dev y) |> Expr<'T>   
    static member Pow (x: Expr<'T>, y: 'T) = (x :> Expr) ** (Expr.scalar x.Dev y) |> Expr<'T>   
    static member ( *** ) (x: Expr<'T>, y: 'T) = (x :> Expr) ** (Expr.scalar x.Dev y) |> Expr<'T>  

    static member (+) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) + (y :> Expr) |> Expr<'T>
    static member (-) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) - (y :> Expr) |> Expr<'T>
    static member (*) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) * (y :> Expr) |> Expr<'T>
    static member (/) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) / (y :> Expr) |> Expr<'T>
    static member (%) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) % (y :> Expr) |> Expr<'T>
    static member Pow (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) ** (y :> Expr) |> Expr<'T>
    static member ( *** ) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) ** (y :> Expr) |> Expr<'T>

    // element-wise binary comparison with basetype
    static member (====) (x: Expr<'T>, y: 'T) = (x :> Expr) ==== (Expr.scalar x.Dev y) |> Expr<bool>
    static member (<<<<) (x: Expr<'T>, y: 'T) = (x :> Expr) <<<< (Expr.scalar x.Dev y) |> Expr<bool>
    static member (<<==) (x: Expr<'T>, y: 'T) = (x :> Expr) <<== (Expr.scalar x.Dev y) |> Expr<bool>
    static member (>>>>) (x: Expr<'T>, y: 'T) = (x :> Expr) >>>> (Expr.scalar x.Dev y) |> Expr<bool>
    static member (>>==) (x: Expr<'T>, y: 'T) = (x :> Expr) >>== (Expr.scalar x.Dev y) |> Expr<bool>
    static member (<<>>) (x: Expr<'T>, y: 'T) = (x :> Expr) <<>> (Expr.scalar x.Dev y) |> Expr<bool>

    static member (====) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) ==== (y :> Expr) |> Expr<bool>
    static member (<<<<) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) <<<< (y :> Expr) |> Expr<bool>
    static member (<<==) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) <<== (y :> Expr) |> Expr<bool>
    static member (>>>>) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) >>>> (y :> Expr) |> Expr<bool>
    static member (>>==) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) >>== (y :> Expr) |> Expr<bool>
    static member (<<>>) (x: 'T, y: Expr<'T>) = (Expr.scalar y.Dev x) <<>> (y :> Expr) |> Expr<bool>

    /// Dot product.
    static member ( .* ) (x: Expr<'T>, y: Expr<'T>) = 
        (x :> Expr) .* (y :> Expr) |> Expr<'T>

    /// Sign keeping type.
    static member signt (expr: Expr<'T>) =
        Expr.signt expr |> Expr<'T>

    /// Square root.
    static member sqrtt (expr: Expr<'T>) =
        Expr.sqrtt expr |> Expr<'T>

    /// Tensor of given shape filled with specified value.
    static member filled (dev: ITensorDevice) (shp: ShapeSpec) (value: 'T) =
        Expr.filled dev shp (box value) |> Expr<'T>

    /// Zero tensor of given shape.
    static member zeros dev (shp: ShapeSpec) =
        Expr.zeros typeof<'T> dev shp |> Expr<'T>

    /// Identity matrix of given size.
    static member identity dev size = 
        Expr.identity typeof<'T> dev size |> Expr<'T>

    /// Computes the inverse of a matrix.
    /// If the input has more than two dimensions, the inverses
    /// along the last two dimensions are returned.
    /// The inverse of a singular matrix is undefinied.
    /// No error is raised in that case.
    static member invert (x: Expr<'T>) =
        Expr.invert x |> Expr<'T>

    /// Reverses the tensor in the specified dimension.
    static member reverseAxis axis (x: Expr<'T>) =
        Expr.reverseAxis axis x |> Expr<'T>

    /// Concatenates the sequence of tensors in the specified dimension.
    static member concat dim (es: Expr<'T> seq) =
        let es = es |> Seq.map (fun expr -> expr :> Expr)
        Expr.concat dim es |> Expr<'T>

    /// Extracts the diagonal along the given axes.
    static member diagAxis ax1 ax2 (x: Expr<'T>) = 
        Expr.diagAxis ax1 ax2 x |> Expr<'T>
                             
    /// Extracts the diagonal of a matrix.
    /// If the expression has more than two dimensions, the diagonals
    /// are extracted along the last two dimensions.
    static member diag (x: Expr<'T>) = 
        Expr.diag x |> Expr<'T>

    /// Creates a diagonal matrix by duplicating the given dimension.
    static member diagMatAxis ax1 ax2 (x: Expr<'T>) = 
        Expr.diagMatAxis ax1 ax2 x |> Expr<'T>

    /// Creates a matrix with the given vector on its diagonal. 
    /// All other elements are zeros.
    /// If the input has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    static member diagMat (x: Expr<'T>) =
        Expr.diagMat x |> Expr<'T>

    /// summation over given dimension
    static member sumAxis (axis: int) (x: Expr<'T>) = 
        Expr.sumAxis axis x |> Expr<'T>

    /// summation over given dimension, while keeping the axis with one (broadcastable) element
    static member sumKeepingAxis (axis: int) (x: Expr<'T>) =
        Expr.sumKeepingAxis axis x |> Expr<'T>

    /// summaiton of all elements
    static member sum (x: Expr<'T>) = 
        Expr.sum x |> Expr<'T>

    /// Computes the traces along the given axes.
    static member traceAxis (ax1: int) (ax2: int) (x: Expr<'T>) =
        Expr.traceAxis ax1 ax2 x |> Expr<'T>

    /// Computes the trace of a matrix.
    /// If the input has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    static member trace (x: Expr<'T>) =
        Expr.trace x |> Expr<'T>
    
    /// product over given dimension
    static member productAxis (axis: int) (x: Expr<'T>) = 
        Expr.productAxis axis x |> Expr<'T>

    /// product over given dimension, while keeping the axis with one (broadcastable) element
    static member productKeepingAxis (axis: int) (x: Expr<'T>) =
        Expr.productKeepingAxis axis x |> Expr<'T>

    /// product of all elements
    static member product (x: Expr<'T>) = 
        Expr.product x |> Expr<'T>

    /// Maximum over given dimension.
    static member maxAxis (axis: int) (x: Expr<'T>) = 
        Expr.maxAxis axis x |> Expr<'T>

    /// Maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member maxKeepingAxis (axis: int) (x: Expr<'T>) =
        Expr.maxKeepingAxis axis x |> Expr<'T>

    /// Maximum of all elements.
    static member max (x: Expr<'T>) = 
        Expr.max x |> Expr<'T>

    /// Minimum over given dimension.
    static member minAxis (axis: int) (x: Expr<'T>) = 
        Expr.minAxis axis x |> Expr<'T>

    /// Minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member minKeepingAxis (axis: int) (x: Expr<'T>) =
        Expr.minKeepingAxis axis x |> Expr<'T>

    /// Minimum of all elements.
    static member min (x: Expr<'T>) = 
        Expr.min x |> Expr<'T>

    /// Index of maximum over given dimension.
    static member argMaxAxis (axis: int) (x: Expr<'T>) = 
        Expr.argMaxAxis axis x |> Expr<'T>

    /// Index of maximum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMaxKeepingAxis (axis: int) (x: Expr<'T>) =
        Expr.argMaxKeepingAxis axis x |> Expr<'T>

    /// Index of minimum over given dimension.
    static member argMinAxis (axis: int) (x: Expr<'T>) = 
        Expr.argMinAxis axis x |> Expr<'T>

    /// Index of minimum over given dimension, while keeping the axis with one (broadcastable) element.
    static member argMinKeepingAxis (axis: int) (x: Expr<'T>) =
        Expr.argMinKeepingAxis axis x |> Expr<'T>

    /// Select elements according to the specified index tensors.
    static member gather (indices: Expr<int64> option list) (x: Expr<'T>) =
        let indices = indices |> List.map (Option.map (fun expr -> expr :> Expr))
        Expr.gather indices x |> Expr<'T>

    /// Disperses elements according to the specified index tensors.
    static member scatter (indices: Expr<int64> option list) (trgtShp: ShapeSpec) (x: Expr<'T>) =
        let indices = indices |> List.map (Option.map (fun expr -> expr :> Expr))
        Expr.scatter indices trgtShp x |> Expr<'T>

    /// Nullifies the Jacobian of its argument when calculating derivatives.
    static member assumeZeroDeriv (x: Expr<'T>) =
        Expr.assumeZeroDeriv x |> Expr<'T>

    /// Assumes the specified Jacobian when calculating derivatives.
    static member assumeDeriv (deriv: Expr<'T>) (x: Expr<'T>) =
        Expr.assumeDeriv deriv x |> Expr<'T>

    /// Annotated expression (no influence on value).
    static member annotate label (x: Expr<'T>) = 
        Expr.annotate label x |> Expr<'T>

    /// Print the result with the given label when evaluated.
    static member print (label: string) (x: Expr<'T>) =
        Expr.print label x |> Expr<'T>

    /// Dumps the result into the given dataset in the active HDF5 dump file.
    static member dump (dataset: string) (x: Expr<'T>) =
        Expr.dump dataset x |> Expr<'T>

    /// If the value contains NaNs or infinities, outputs their location and 
    /// stops the computation.
    static member checkFinite (label: string) (x: Expr<'T>) =
        Expr.checkFinite label x |> Expr<'T>

    /// Tensor product.
    static member tensorProduct (x: Expr<'T>) (y: Expr<'T>) =
        Expr.tensorProduct x y |> Expr<'T>

   /// Elementwise uses elements from `ifTrue` if `cond` is true for that element, otherwise elements from `ifFalse`.
    static member ifThenElse (cond: Expr<bool>) (ifTrue: Expr<'T>) (ifFalse: Expr<'T>) =
        Expr.ifThenElse cond ifTrue ifFalse |> Expr<'T>

    /// Discards the results of all arguments.
    static member discard (xs: Expr<'T> list) =
        let xs = xs |> List.map (fun expr -> expr :> Expr)
        Expr.discard xs 

    /// Build tensor from numeric ranges.
    static member internal buildTensor shape ranges (xs: Expr<'T> list) =
        let xs = xs |> List.map (fun expr -> expr :> Expr)
        Expr.buildTensor shape ranges xs |> Expr<'T>
    
    /// Calculates a tensor elementwise using the given element expression and result shape.
    static member elements shape elemExpr (xs: Expr<'T> list) =
        let xs = xs |> List.map (fun expr -> expr :> Expr)
        Expr.elements shape elemExpr xs |> Expr<'T>

    /// Element-wise n-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate interpolator (xs: Expr<'T> list) =
        let xs = xs |> List.map (fun expr -> expr :> Expr)
        Expr.interpolate interpolator xs |> Expr<'T>

    /// Element-wise one-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate1D interpolator (x: Expr<'T>) =
        Expr.interpolate1D interpolator x |> Expr<'T>

    /// Element-wise two-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate2D interpolator (x: Expr<'T>) (y: Expr<'T>) =
        Expr.interpolate2D interpolator x y |> Expr<'T>

    /// Element-wise three-dimensional interpolation using the specified interpolator.
    /// The interpolator is created using the Interpolator.create function.
    static member interpolate3D interpolator (x: Expr<'T>) (y: Expr<'T>) (z: Expr<'T>) =
        Expr.interpolate3D interpolator x y z |> Expr<'T>

    /// Evaluates the expression into a numeric value.
    static member eval (varEnv: VarEnv) (expr: Expr<'T>) : Tensor.Tensor<'T> = 
        Expr.eval varEnv expr :?> Tensor.Tensor<'T>





