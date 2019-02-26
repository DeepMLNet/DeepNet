namespace rec Tensor.Expr.Ops

open System
open Tensor.Expr
open DeepNet.Utils
open Tensor
open Tensor.Backend


/// A channel name (one output of an expression/op with multiple outputs).
[<RequireQualifiedAccess>]
type Ch = 
    /// Only channel.
    | Default
    /// Custom channel name.
    | Custom of string


/// An argument name.
[<RequireQualifiedAccess>]
type Arg = 
    /// Only argument of unary op.
    | Only
    /// First argument of binary op.
    | X
    /// Second argument of binary op.
    | Y
    /// N-th argument of n-ary op.
    | N of int
    /// Custom argument.
    | Custom of string


/// Map containing argument expression by name.
//type ArgsMap = Map<Arg, BaseExprCh>

/// Map containing argument expression by name for multi-channel arguments.
//type MultiChannelArgsMap = Map<string, string * BaseMultiChannelExpr>


/// Information necessary to evaluate an expression.
/// Currently this just holds the variable values, but may contain further information in the future.
type EvalEnv = {
    /// Values of variables.
    VarEnv: VarEnv
    /// Device to store result on.
    Dev:    ITensorDevice
    /// Argument values.
    Args:   Map<Arg, ITensor>
}


/// start plus the specified number of (symbolic elements)
type internal PlusElems (elems: SizeSpec) =
    new (intElems: int64) = PlusElems (SizeSpec.fix intElems)
    member this.Elems = elems


/// A mathematical operation in an expression with multiple output values.
/// This models a mathematical function or operator that takes one or more tensors
/// and returns multiple tensors.
type IOp =
    inherit System.IComparable
      
    /// Should check if the types and shapes of the arguments are acceptable and,
    /// if not, raise an exception.
    abstract Check: unit -> unit
       
    /// Returns the arguments of this op.
    abstract Args: Map<Arg, BaseExprCh>

    /// Creates a new op with the arguments replaced by the specified arguments.
    abstract ReplaceArgs: Map<Arg, BaseExprCh> -> IOp

    /// Should return the expression with all symbolic sizes substituted using the specified
    /// substitution table.
    /// Return a *new* op with substitution applied. Do not apply the mapping in-place.
    abstract SubstSymSizes: env: SymSizeEnv -> IOp

    /// Should be true, if all symbolic sizes can be evaluated to numeric sizes.
    /// This is the case if the function ShapeSpec.canEval or SizeSpec.canEval respectively
    /// return true on all sizes used in this op.
    abstract CanEvalAllSymSizes: bool
     
    /// The output channels of this operation.
    abstract Channels: Set<Ch>

    /// Should return the types of the results.
    abstract TypeNames: Map<Ch, TypeName>

    /// Should return the shapes of the results.
    abstract Shapes: Map<Ch, ShapeSpec>      

    /// Should evaluate the numerical value of this op given the numerical values of its arguments.
    /// This evaluation should be done on the host using the simplest means possible and is used
    /// as a reference implementation for verifying the correctness of optimized (e.g. CUDA) 
    /// implementations. This method may be omitted when no verification will be done.
    abstract Eval: env:EvalEnv -> Map<Ch, Tensor.ITensor>


/// An op that contains variables.
type IVarContainingOp =
    /// Variables contained in that op.
    abstract Vars: Set<Var>


/// An op that has a custom print format.
type IOpFormat =
    /// Text to output for this op when expression is printed.
    abstract Text: string



/// Base for single-channel and multi-channel expressions.
/// BaseExpr is reference-unique, i.e. all expressions that are structurally equal 
/// are also reference equal.
/// No conatined variables must have the same name but different types or shapes.
type BaseExpr private (op: IOp) =   
    do op.Check()

    let _singleCh = op.Channels = Set [Ch.Default]
    let _hash = lazy (hash op)
    let _typeNames = lazy (op.TypeNames)
    let _shapes = lazy (op.Shapes)      

    let _varMap = 
        let add (m: Map<VarName, Var>) (var: Var) =
            match m |> Map.tryFind var.Name with
            | Some otherVar when otherVar = var -> m
            | Some otherVar ->
                failwithf "Expression contains inconsistent variable %A with specifications %A and %A."
                          var.Name var otherVar
            | None -> m |> Map.add var.Name var
        let merge (a: Map<VarName, Var>) b =
            (a, Map.toSeq b)||> Seq.fold (fun m (varName, var) -> add m var)

        let opVarSet =
            match op with
            | :? IVarContainingOp as op -> op.Vars
            | _ -> Set.empty
        let opVars = (Map.empty, Set.toSeq opVarSet) ||> Seq.fold add
        let argVars = 
            op.Args
            |> Map.toSeq
            |> Seq.map (fun (_, BaseExprCh (_, argExpr)) -> argExpr.VarMap)

        if Seq.isEmpty argVars then 
            opVars
        else
            Seq.append argVars [opVars] |> Seq.reduce merge

    let _canEvalAllSymSizes = lazy (
        let argsEvalable =
            op.Args
            |> Map.forall (fun _ (BaseExprCh (_, argExpr)) -> argExpr.CanEvalAllSymSizes)
        argsEvalable && op.CanEvalAllSymSizes)

    let checkSingleCh () =
        if not _singleCh then
            failwithf "This operation requires a single-channel op, but op %A has channels %A." 
                      op op.Channels

    /// Unique expression instance for each op.
    static let uniqueExprs = new ConcurrentWeakDict<IOp, BaseExpr> (BaseExpr.op, BaseExpr)

    /// Creates a base expression for the specified op.
    static member ofOp (op: IOp) = uniqueExprs.[op]

    /// Op of this expression.
    member this.Op = op
    /// Op of this expression.
    static member op (expr: BaseExpr) = expr.Op

    /// Arguments.
    member this.Args = op.Args
    /// Arguments.
    static member args (expr: BaseExpr) = expr.Args

    /// Type names of channels.
    member this.TypeNames = _typeNames.Force()
    /// Type names of channels.
    static member typeNames (expr: BaseExpr) = expr.TypeNames

    /// Data types of channels.
    member this.DataTypes = this.TypeNames |> Map.map (fun _ tn -> tn.Type)
    /// Data types of channels.
    static member dataTypes (expr: BaseExpr) = expr.DataTypes
    
    /// Shapes of channels.
    member this.Shapes = _shapes.Force()
    /// Shapes of channels.
    static member shapes (expr: BaseExpr) = expr.Shapes

    /// Number of dimensions of channels.
    member this.NDims = this.Shapes |> Map.map (fun _ s -> List.length s)
    /// Number of dimensions of channels.
    static member nDims (expr: BaseExpr) = expr.NDims

    /// Number of elements of channels.
    member this.NElems = this.Shapes |> Map.map (fun _ s -> List.fold (*) SizeSpec.one s)
    /// Number of elements of channels.
    static member nElems (expr: BaseExpr) = expr.NElems

    /// Channels.
    member this.Channels = op.Channels
    /// Channels.
    static member channels (expr: BaseExpr) = expr.Channels

    /// True if the op of this expression is an op with a single output channel.
    member this.IsSingleChannel = _singleCh
    /// True if the op of this expression is an op with a single output channel.
    static member isSingleChannel (expr: BaseExpr) = expr.IsSingleChannel
    
    /// Returns all variables contained in an op and its arguments as a map from variable name to variable.
    member private this.VarMap = _varMap

    /// Returns all variables contained in an op and its arguments.
    member this.Vars = this.VarMap |> Map.toSeq |> Seq.map snd |> Set.ofSeq
    /// Returns all variables contained in an op and its arguments.
    static member vars (expr: BaseExpr) = expr.Vars

    /// Returns true, if all symbolic sizes of the op and its arguments can be evaluated to numeric values.
    member this.CanEvalAllSymSizes = _canEvalAllSymSizes.Force()
    /// Returns true, if all symbolic sizes of the op and its arguments can be evaluated to numeric values.
    static member canEvalAllSymSizes (expr: BaseExpr) = expr.CanEvalAllSymSizes

    /// Substitutes the symbolic sizes within this expression and all its subexpressions.
    static member substSymSizes (env: SymSizeEnv) (expr: BaseExpr) =
        let op = expr.Op.SubstSymSizes env
        let subsArgs = 
            op.Args 
            |> Map.map (fun _ arg -> arg |> BaseExprCh.map (BaseExpr.substSymSizes env))
        op.ReplaceArgs subsArgs |> BaseExpr.ofOp

    /// Maps the arguments of this expression using the specified function.
    static member mapArgs (fn: BaseExprCh -> BaseExprCh) (expr: BaseExpr) =
        expr.Op.Args
        |> Map.map (fun _ arg -> fn arg)
        |> expr.Op.ReplaceArgs 
        |> BaseExpr.ofOp

    /// Access to specified channel of this expression.
    member this.Item
        with get (channel: Ch) = BaseExprCh.make channel this

    /// Access to the only channel of this expression.
    member this.OnlyCh =
        checkSingleCh()
        this.[Ch.Default]

    interface System.IEquatable<BaseExpr> with
        member this.Equals other = Object.ReferenceEquals (this, other)

    override this.Equals other =
        match other with
        | :? BaseExpr as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<BaseExpr> with
        member this.CompareTo other = compare this.Op other.Op

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? BaseExpr as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare BaseExpr to type %A." (other.GetType())

    override this.GetHashCode() = _hash.Force ()

    override this.Finalize () =
        uniqueExprs.Finalized this



/// A channel of a multi-channel expression.
type BaseExprCh = private {
    /// Channel
    _Channel: Ch
    /// Expression
    _Expr: BaseExpr 
} with

    interface IDynElem

    /// Create from specified channel and expression.
    static member make (channel: Ch) (expr: BaseExpr) : BaseExprCh = 
        if not (expr.Channels.Contains channel) then
            failwithf "Expression %A does not provide channel %A." expr channel
        {_Channel=channel; _Expr=expr}

    /// Channel.
    member this.Channel = this._Channel
    /// Channel.
    static member channel (bec: BaseExprCh) = bec.Channel

    /// Expression.
    member this.Expr = this._Expr
    /// Expression.
    static member expr (bec: BaseExprCh) = bec.Expr

    /// Type name.
    member this.TypeName = this.Expr.TypeNames.[this.Channel]
    /// Type name.
    static member typeName (this: BaseExprCh) = this.TypeName

    /// Data type.
    member this.DataType = this.Expr.DataTypes.[this.Channel]
    /// Data type.
    static member dataType (this: BaseExprCh) = this.DataType

    /// Shape.
    member this.Shape = this.Expr.Shapes.[this.Channel]
    /// Shape.
    static member shape (this: BaseExprCh) = this.Shape

    /// Number of dimensions.
    member this.NDims = this.Expr.NDims.[this.Channel]
    /// Number of dimensions.
    static member nDims (this: BaseExprCh) = this.NDims

    /// Number of elements.
    member this.NElems = this.Expr.NElems.[this.Channel]
    /// Number of elements.
    static member nElems (this: BaseExprCh) = this.NElems

    /// Apply mapping function to contained expression.
    static member map (fn: BaseExpr -> BaseExpr) (bec: BaseExprCh) =
        BaseExprCh.make bec.Channel (fn bec.Expr)
    

[<AutoOpen>]
module BaseExprChRecognizier =
    /// Decomposes a BaseExprCh into channel and expression.
    let (|BaseExprCh|) (arg: BaseExprCh) : Ch * BaseExpr =
        arg.Channel, arg.Expr

