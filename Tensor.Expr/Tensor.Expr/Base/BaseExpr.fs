namespace rec Tensor.Expr.Base

open System
open System.IO
open MBrace.FsPickler

open DeepNet.Utils
open Tensor.Expr
open Tensor
open Tensor.Backend



/// A channel name (one output of an expression/op with multiple outputs).
[<RequireQualifiedAccess; StructuredFormatDisplay("{Pretty}")>]
type Ch = 
    /// Only channel.
    | Default
    /// Custom channel name.
    | Custom of string
    /// N-th channel.
    | N of int

    override this.ToString() =
        match this with
        | Default -> "Default"
        | Custom name -> sprintf "Custom:%s" name
        | N idx -> sprintf "%d" idx     
        
    member this.Pretty = this.ToString()
        
    static member tryParse (str: string) =
        match str with
        | "Default" -> Some Default
        | String.Prefixed "Custom:" ch -> Some (Custom ch)
        | String.Int n -> Some (N n)
        | _ -> None


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


/// Custom data for a trace event.
[<RequireQualifiedAccess>]
type TraceCustomData =
    | Bool of bool
    | Int of int
    | Int64 of int64
    | Single of single
    | Double of double
    | String of string
    | Tensor of ITensor
    | DateTime of DateTime

    /// The contained value.
    member this.Value =
        match this with
        | Bool value -> box value
        | Int value -> box value
        | Int64 value -> box value
        | Single value -> box value
        | Double value -> box value
        | String value -> box value
        | Tensor value -> box value
        | DateTime value -> box value


/// A trace event.
[<RequireQualifiedAccess>]
type TraceEvent =
    /// Expression to associate with.
    | Expr of BaseExpr
    /// Parent expression to assoicate with.
    | ParentExpr of BaseExpr
    /// A log message.
    | Msg of string
    /// Evaluated expression value.
    | EvalValue of Map<Ch, ITensor>
    /// Evaluation start time.
    | EvalStart of DateTime
    /// Evaluation end time.
    | EvalEnd of DateTime
    /// Loop iteration.
    | LoopIter of int64
    /// Custom data.
    | Custom of key:string * data:TraceCustomData
    /// Event assoicated with an expression.
    | ForExpr of BaseExpr * TraceEvent
    /// Event from a sub tracer.
    | Subtrace of id:int * TraceEvent


/// Interface for tracing operations on expression trees.
type ITracer =
    /// Logs data associated with a base expression.
    abstract Log: TraceEvent -> unit
    /// Creates a sub-tracer.
    abstract GetSubTracer: unit -> ITracer
    
/// Tracer that performs no trace.
type NoTracer() =
    interface ITracer with
        member this.Log event = ()
        member this.GetSubTracer () = NoTracer() :> _


/// Information necessary to evaluate an expression.
/// Currently this just holds the variable values, but may contain further information in the future.
type EvalEnv = {
    /// Values of variables.
    VarEnv: VarEnv
    /// Tracer for logging of evaluation.
    Tracer: ITracer
}



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
    abstract SubstSymSizes: env: SizeEnv -> IOp

    /// Should be true, if all symbolic sizes can be evaluated to numeric sizes.
    /// This is the case if the function Shape.canEval or Size.canEval respectively
    /// return true on all sizes used in this op.
    abstract CanEvalAllSymSizes: bool
     
    /// The output channels of this operation.
    abstract Channels: Set<Ch>

    /// The data types of the channels.
    abstract TypeNames: Map<Ch, TypeName>

    /// The shapes of the channels.
    abstract Shapes: Map<Ch, Shape>      

    /// The storage devices of the channels.
    abstract Devs: Map<Ch, ITensorDevice>

    /// Should evaluate the numerical value of this op given the numerical values of its arguments.
    abstract Eval: env:EvalEnv -> argVals:Map<Arg, ITensor> -> Map<Ch, Tensor.ITensor>


/// Helper functions for ops.
module IOp =
    /// Checks two ops for equality.
    let inline equal (a: IOp) (b: IOp) =
        a.GetType() = b.GetType() && a = b

    /// Compares two ops.
    let inline compare (a: IOp) (b: IOp) =
        let aType, bType = a.GetType(), b.GetType()
        if aType = bType then
            Operators.compare a b
        else
            Operators.compare aType.FullName bType.FullName


/// An op that can have multiple output channels.
type IMultiChannelOp =
    interface end


/// An op that represents a variable.
type IVarOp =
    /// Variable represented by this op.
    abstract Var: Var


/// An operation that works element-wise on its operands.
type IElemwiseOp =
    interface end

    
/// An op that has a custom print format.
type IOpFormat =
    /// Text to output for this op when expression is printed.
    abstract Text: string



module private BaseExprTools =

    let collectMap (getOp: IOp -> 'V seq) (getExpr: BaseExpr -> Map<'K,'V>) (getKey: 'V -> 'K) (op: IOp) =
        let add (map: Map<'K, 'V>) (value: 'V) =
            match map |> Map.tryFind (getKey value) with
            | Some otherVar when otherVar = value -> map
            | Some otherVar ->
                failwithf "Expression is inconsistent in %A with specifications %A and %A."
                    (getKey value) value otherVar
            | None -> map |> Map.add (getKey value) value

        let merge (a: Map<'K, 'V>) b =
            (a, Map.toSeq b) ||> Seq.fold (fun map (key, value) -> add map value)

        let opMap = (Map.empty, getOp op) ||> Seq.fold add
        let argMaps = 
            op.Args
            |> Map.toSeq
            |> Seq.map (fun (_, BaseExprCh (_, argExpr)) -> getExpr argExpr)

        if Seq.isEmpty argMaps then 
            opMap
        else
            Seq.append argMaps [opMap] |> Seq.reduce merge



/// Base for single-channel and multi-channel expressions.
/// BaseExpr is reference-unique, i.e. all expressions that are structurally equal 
/// are also reference equal.
/// No conatined variables must have the same name but different types or shapes.
[<StructuredFormatDisplay("{Pretty}"); CustomPickler>]
type BaseExpr private (op: IOp) =   
    do op.Check()

    let _hash = lazy (hash op)
    let _typeNames = lazy (op.TypeNames)
    let _shapes = lazy (op.Shapes)    
    let _devs = lazy (op.Devs)

    let _singleCh = 
        match op with
        | :? IMultiChannelOp -> false
        | _ -> true
    do if _singleCh && not (op.Channels = Set [Ch.Default]) then
        failwith "A single-channel op must provide only Ch.Default."

    let _varMap = 
        let getOp (op: IOp) =
            match op with
            | :? IVarOp as op -> Seq.singleton op.Var
            | _ -> Seq.empty
        let getExpr (expr: BaseExpr) = expr.VarMap
        let getKey (var: Var) = var.Name
        BaseExprTools.collectMap getOp getExpr getKey op

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
    static let uniqueExprs = new ConcurrentWeakDict<IOp, BaseExpr> (BaseExpr.op, IOp.equal, BaseExpr)

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
    member this.NElems = this.Shapes |> Map.map (fun _ s -> List.fold (*) Size.one s)
    /// Number of elements of channels.
    static member nElems (expr: BaseExpr) = expr.NElems

    /// Storage devices of channels.
    member this.Devs = _devs.Force()
    /// Storage devices of channels.
    static member devs (expr: BaseExpr) = expr.Devs

    /// Channels.
    member this.Channels = op.Channels
    /// Channels.
    static member channels (expr: BaseExpr) = expr.Channels

    /// True if the op of this expression is an op with a single output channel.
    member this.IsSingleChannel = _singleCh
    /// True if the op of this expression is an op with a single output channel.
    static member isSingleChannel (expr: BaseExpr) = expr.IsSingleChannel
    
    /// Returns all variables contained in an op and its arguments as a map from variable name to variable.
    member this.VarMap = _varMap
    /// Returns all variables contained in an op and its arguments as a map from variable name to variable.
    static member varMap (expr: BaseExpr) = expr.VarMap

    /// Returns all variables contained in an op and its arguments.
    member this.Vars = this.VarMap |> Map.values
    /// Returns all variables contained in an op and its arguments.
    static member vars (expr: BaseExpr) = expr.Vars

    /// Returns true, if all symbolic sizes of the op and its arguments can be evaluated to numeric values.
    member this.CanEvalAllSymSizes = _canEvalAllSymSizes.Force()
    /// Returns true, if all symbolic sizes of the op and its arguments can be evaluated to numeric values.
    static member canEvalAllSymSizes (expr: BaseExpr) = expr.CanEvalAllSymSizes

    /// Maps the arguments of this expression using the specified function.
    static member mapArgs (fn: BaseExprCh -> BaseExprCh) (expr: BaseExpr) =
        expr.Op.Args
        |> Map.map (fun _ arg -> fn arg)
        |> expr.Op.ReplaceArgs 
        |> BaseExpr.ofOp

    /// Recursively applies a function to each op in the expression tree.
    /// Mapping is performed depth-first, i.e. first the arguments of an expression are mapped, 
    /// then the expression itself.
    /// The function is called only once per unique subexpression.
    static member mapOpRec (fn: IOp -> IOp) (expr: BaseExpr) =
        let replacement = Dictionary<BaseExpr, BaseExpr> ()
        let rec mapStep (subExpr: BaseExpr) =
            replacement.IGetOrAdd (subExpr, fun _ ->
                subExpr.Op.Args
                |> Map.map (fun _ arg -> arg |> BaseExprCh.map mapStep)
                |> subExpr.Op.ReplaceArgs 
                |> fn
                |> BaseExpr.ofOp)
        mapStep expr       

    /// Substitutes the symbolic sizes within the expression tree.
    static member substSymSizes (env: SizeEnv) (expr: BaseExpr) =
        expr |> BaseExpr.mapOpRec (fun op -> op.SubstSymSizes env)

    /// Substitutes the variables within the expression tree.
    static member substVars (env: Map<VarName, BaseExpr>) (expr: BaseExpr) = 
        expr |> BaseExpr.mapOpRec (fun op ->
            match op with
            | :? IVarOp as varOp -> 
                match env |> Map.tryFind varOp.Var.Name with
                | Some replExpr -> replExpr.Op 
                | None -> op
            | _ -> op)

    /// Deterministically assigns a numeric id to each expression within the expression tree.
    /// The order is based on argument names.
    static member enumerate (expr: BaseExpr) = 
        let ids = Dictionary<BaseExpr, int> ()
        let mutable nextId = 0
        let rec enumerate (expr: BaseExpr) =
            if not (ids.ContainsKey expr) then
                ids.[expr] <- nextId
                nextId <- nextId + 1
                let args = expr.Args |> Map.toList |> List.sortBy fst
                for (_arg, argExprCh) in args do
                    enumerate argExprCh.Expr
        enumerate expr
        ids |> Map.ofDictionary
 
    /// Access to specified channel of this expression.
    member this.Item
        with get (channel: Ch) = BaseExprCh.make channel this

    interface System.IEquatable<BaseExpr> with
        member this.Equals other = Object.ReferenceEquals (this, other)

    override this.Equals other =
        match other with
        | :? BaseExpr as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<BaseExpr> with
        member this.CompareTo other = IOp.compare this.Op other.Op

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? BaseExpr as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare BaseExpr to type %A." (other.GetType())

    override this.GetHashCode() = _hash.Force ()

    override this.Finalize () =
        uniqueExprs.Finalized this

    static member CreatePickler (resolver : IPicklerResolver) =
        let xp = resolver.Resolve<IOp> ()
        let writer (ws : WriteState) (baseExpr: BaseExpr) =
            xp.Write ws "op" baseExpr.Op
        let reader (rs : ReadState) =
            let op = xp.Read rs "op"
            BaseExpr.ofOp op
        Pickler.FromPrimitives(reader, writer)

    /// Converts expression to string with specified approximate maximum length.
    member this.ToString maxLength =     
        let opStr =
            match this.Op with
            | :? IOpFormat as opFormat -> opFormat.Text
            | _ -> this.Op.GetType().Name
        let args = this.Args
        let argList = args |> Map.keys |> List.ofSeq |> List.sortBy (sprintf "%A")
        String.limited maxLength [
            yield String.Formatter (fun _ -> opStr)
            if not argList.IsEmpty then
                yield String.Delim " ("
                for i, arg in List.indexed argList do
                    if i > 0 then
                        yield String.Delim ", "
                    yield String.Formatter (fun _ -> sprintf "%A=" arg)
                    yield String.Formatter args.[arg].ToString
                yield String.Delim ")"
        ]

    /// Converts expression to string with unlimited length.
    override this.ToString () = this.ToString System.Int32.MaxValue

    /// Pretty string.
    member this.Pretty = this.ToString 80
    
    /// Writes the expression to a TextWriter.
    static member dump (writer: TextWriter) (getId: BaseExpr -> int) (expr: BaseExpr) =
        fprintf writer "#%d := " (getId expr)
        fprintf writer "%s" (expr.Op.GetType().Name)
        
        match expr.Op with
        | :? IOpFormat as opFormat ->
            fprintf writer " {%A}" opFormat.Text
        | _ -> ()
        
        let args = expr.Args
        let argStr =
            args
            |> Map.keys
            |> List.ofSeq
            |> List.sortBy (fun arg -> arg.ToString())
            |> List.map (fun arg ->
                let (BaseExprCh (argCh, argExpr)) = args.[arg]
                sprintf "%A=#%d[%A]" arg (getId argExpr) argCh)
            |> String.concat ", "
        fprintf writer " (%s)\n" argStr      



/// A channel of a multi-channel expression.
[<StructuredFormatDisplay("{Pretty}")>]
type BaseExprCh = private {
    /// Expression
    _Expr: BaseExpr 
    /// Channel
    _Channel: Ch
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

    /// Storage device.
    member this.Dev = this.Expr.Devs.[this.Channel]
    /// Storage device.
    static member dev (this: BaseExprCh) = this.Dev

    /// Apply mapping function to contained expression.
    static member map (fn: BaseExpr -> BaseExpr) (bec: BaseExprCh) =
        BaseExprCh.make bec.Channel (fn bec.Expr)
    
    /// Converts expression to string with specified approximate maximum length.
    member this.ToString maxLength =     
        String.limited maxLength [
            yield String.Formatter this.Expr.ToString
            if this.Channel <> Ch.Default then
                yield String.Delim "["
                yield String.Formatter (fun _ -> sprintf "%A" this.Channel)
                yield String.Delim "]"
        ]

    /// Converts expression to string with unlimited length.
    override this.ToString () = this.ToString System.Int32.MaxValue

    /// Pretty string.
    member this.Pretty = this.ToString 80




[<AutoOpen>]
module BaseExprChRecognizier =
    /// Decomposes a BaseExprCh into channel and expression.
    let (|BaseExprCh|) (arg: BaseExprCh) : Ch * BaseExpr =
        arg.Channel, arg.Expr

