namespace rec SymTensor.Ops

open SymTensor
open DeepNet.Utils
open Tensor
open Tensor.Backend
open System

    
/// Map containing argument expression by name.
type ArgsMap = Map<string, BaseExpr>

/// Map containing argument expression by name for multi-channel arguments.
type MultiChannelArgsMap = Map<string, BaseMultiChannelExpr>


/// Information necessary to evaluate an expression.
/// Currently this just holds the variable values, but may contain further information in the future.
type EvalEnv = {
    /// Values of variables.
    VarEnv: VarEnv
    /// Device to store result on.
    Dev:    ITensorDevice
    /// Argument values.
    Args:   Map<string, ITensor>
    /// Multi-channel argument values.
    MultiChannelArgs:   Map<string, Map<string, ITensor>>
}


/// start plus the specified number of (symbolic elements)
type internal PlusElems (elems: SizeSpec) =
    new (intElems: int64) = PlusElems (SizeSpec.fix intElems)
    member this.Elems = elems


/// Base interface for a mathematical operation in an expression.
type IBaseOp =
    inherit System.IComparable
      
    /// Should check if the types and shapes of the arguments are acceptable and,
    /// if not, raise an exception.
    abstract Check: unit -> unit
       
    /// Returns the arguments of this op.
    abstract Args: ArgsMap

    /// Creates a new op with the arguments replaced by the specified arguments.
    abstract ReplaceArgs: ArgsMap -> IBaseOp

    /// Should return the expression with all symbolic sizes substituted using the specified
    /// substitution table.
    /// Return a *new* op with substitution applied. Do not apply the mapping in-place.
    abstract SubstSymSizes: env: SymSizeEnv -> IBaseOp

    /// Should be true, if all symbolic sizes can be evaluated to numeric sizes.
    /// This is the case if the function ShapeSpec.canEval or SizeSpec.canEval respectively
    /// return true on all sizes used in this op.
    abstract CanEvalAllSymSizes: bool


/// Operation that uses multi-channel expressions as its arguments.
type IMultiChannelArgsOp = 
    /// Returns the multi-channel arguments of this op.
    abstract MultiChannelArgs: MultiChannelArgsMap

    /// Replaces the multi-channel arguments of this op.
    abstract ReplaceMultiChannelArgs: MultiChannelArgsMap -> IMultiChannelArgsOp


/// A mathematical operation in an expression with a single output value.
/// This models a mathematical function or operator that takes one or more tensors
/// and returns one tensor.
type IOp =
    inherit IBaseOp
      
    /// Should return the type of the result.
    abstract TypeName: TypeName

    /// Should return the shape of the result.
    abstract Shape: ShapeSpec             

    /// Should evaluate the numerical value of this op given the numerical values of its arguments.
    /// This evaluation should be done on the host using the simplest means possible and is used
    /// as a reference implementation for verifying the correctness of optimized (e.g. CUDA) 
    /// implementations. This method may be omitted when no verification will be done.
    abstract Eval: env:EvalEnv -> Tensor.ITensor


/// A mathematical operation in an expression with multiple output values.
/// This models a mathematical function or operator that takes one or more tensors
/// and returns multiple tensors.
type IMultiChannelOp =
    inherit IBaseOp
      
    /// The output channels of this operation.
    abstract Channels: string List

    /// Should return the types of the results.
    abstract TypeNames: Map<string, TypeName>

    /// Should return the shapes of the results.
    abstract Shapes: Map<string, ShapeSpec>      

    /// Should evaluate the numerical value of this op given the numerical values of its arguments.
    /// This evaluation should be done on the host using the simplest means possible and is used
    /// as a reference implementation for verifying the correctness of optimized (e.g. CUDA) 
    /// implementations. This method may be omitted when no verification will be done.
    abstract Eval: env:EvalEnv -> Map<string, Tensor.ITensor>


/// An op that contains variables.
type IVarContainingOp =
    /// Variables contained in that op.
    abstract Vars: Set<Var>


/// Provides a derivative for an op.
type IDerivableOp =    
    /// Computes the derivative w.r.t. each argument given the derivative w.r.t. the op.
    ///
    /// `dOp` is the incoming derivative, i.e. the derivative with respect to this op.
    /// Assuming that N is the number of elements of the function the derivative is being taken and
    /// the output shape of this op is M1xM2x...xMD, the incoming derivative will be of shape
    /// NxM1xM2x...xMD.
    ///
    /// The outgoing derivatives should be of shape NxK1xK2x...xKD where K1xK2x...xKD is the
    /// shape of the respective argument.
    abstract Deriv: dOp:BaseExpr -> Map<string, BaseExpr>


/// Provides the derivative for a multi-channel op.
type IDerivableMultiChannelOp =
    /// Computes the derivative w.r.t. each argument given the derivative w.r.t. the op.
    /// The derivative is always an NxM matrix where N is the number of elements of the function
    /// the derivative of which is being taken and M is the number of elements of the argument
    /// w.r.t. which the derivative is being taken. 
    /// Thus, if dOp is an NxK matrix and an argument has M elements, the derivative matrix
    /// you return w.r.t. that argument must have NxM elements.
    abstract Deriv: dOp:Map<string, BaseExpr> -> Map<string, BaseExpr>


/// Declares that the type is extending an op by implementing additional interfaces.
/// The type's constructor is called with the op instance as argument.
[<AttributeUsage(AttributeTargets.Class)>]
type OpExtenderAttribute () =
    inherit System.Attribute()


/// A single-channel or multi-channel expression.
type IExpr = 
    inherit System.IComparable
    inherit System.IComparable<IExpr>
    inherit System.IEquatable<IExpr>


module internal ExprTools =
    /// Returns all variables contained in an op and its arguments.
    let extractVars (op: IBaseOp) =
        let opVars =
            match op with
            | :? IVarContainingOp as op -> op.Vars
            | _ -> Set.empty
        let argVars =
            op.Args |> Map.toSeq |> Seq.map (fun (_, arg) -> arg.Vars) |> Set.unionMany
        let mcArgVars =
            match op with
            | :? IMultiChannelArgsOp as op ->
                op.MultiChannelArgs |> Map.toSeq |> Seq.map (fun (_, mcArg) -> mcArg.Vars) |> Set.unionMany
            | _ -> Set.empty
        Set.unionMany [opVars; argVars; mcArgVars]

    /// Returns true, if all symbolic sizes of the op and its arguments can be evaluated to numeric values.
    let canEvalAllSymSizes (op: IBaseOp) =
        let argsEvalable = 
            op.Args |> Map.forall (fun _ arg -> arg.CanEvalAllSymSizes)
        let mcArgsEvalable = 
            match op with
            | :? IMultiChannelArgsOp as op ->
                op.MultiChannelArgs |> Map.forall (fun _ mcArg -> mcArg.CanEvalAllSymSizes)
            | _ -> true
        argsEvalable && mcArgsEvalable && op.CanEvalAllSymSizes

    /// Recursively substitues all symbolic sizes within the given op and its arguments.
    let substSymSizes (env: SymSizeEnv) (op: IBaseOp) =
        let op = op.SubstSymSizes env
        let subsArgs = op.Args |> Map.map (fun _ arg -> arg.SubstSymSizes env)
        let op = op.ReplaceArgs subsArgs
        let op =
            match op with
            | :? IMultiChannelArgsOp as op ->
                let subsArgs = 
                    op.MultiChannelArgs |> Map.map (fun _ mcArg -> mcArg.SubstSymSizes env)
                op.ReplaceMultiChannelArgs subsArgs :?> IBaseOp
            | _ -> op
        op


/// Base for single-channel expressions.
type BaseExpr (op: IOp) =
    do op.Check()
        
    let _vars = lazy (ExprTools.extractVars op)
    let _canEvalAllSymSizes = lazy (ExprTools.canEvalAllSymSizes op)

    interface IExpr
    interface IDynElem

    member this.Op = op
    member this.TypeName = op.TypeName   
    member this.DataType = this.TypeName.Type
    member this.Shape = op.Shape
    member this.NDims = List.length this.Shape
    member this.NElems = List.fold (*) SizeSpec.one this.Shape
    member this.Vars = _vars.Force()
    member this.CanEvalAllSymSizes = _canEvalAllSymSizes.Force()
    member this.SubstSymSizes (env: SymSizeEnv) =
        ExprTools.substSymSizes env op :?> IOp |> BaseExpr

    interface System.IEquatable<IExpr> with
        member this.Equals other = 
            match other with
            | :? BaseExpr as other -> this.Op.Equals other.Op
            | _ -> false

    override this.Equals other =
        match other with
        | :? IExpr as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<IExpr> with
        member this.CompareTo other =
            match other with
            | :? BaseExpr as other -> compare this.Op other.Op
            | :? BaseMultiChannelExpr -> 1
            | _ -> failwithf "Cannot compare BaseExpr to type %A." (other.GetType())

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? IExpr as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare BaseExpr to type %A." (other.GetType())

    override this.GetHashCode() =
        hash this.Op


/// Base for multi-channel expressions.
type BaseMultiChannelExpr (op: IMultiChannelOp) =   
    do op.Check()
        
    let _vars = lazy (ExprTools.extractVars op)
    let _canEvalAllSymSizes = lazy (ExprTools.canEvalAllSymSizes op)

    member this.Op = op
    member this.TypeNames = op.TypeNames
    member this.DataTypes = this.TypeNames |> Map.map (fun _ tn -> tn.Type)
    member this.Shapes = op.Shapes
    member this.NDims = this.Shapes |> Map.map (fun _ s -> List.length s)
    member this.NElems = this.Shapes |> Map.map (fun _ s -> List.fold (*) SizeSpec.one s)
    member this.Channels = op.Channels
    member this.Vars = _vars.Force()
    member this.CanEvalAllSymSizes = _canEvalAllSymSizes.Force()
    member this.SubstSymSizes (env: SymSizeEnv) =
        ExprTools.substSymSizes env this.Op :?> IMultiChannelOp |> BaseMultiChannelExpr

    interface IExpr

    interface System.IEquatable<IExpr> with
        member this.Equals other = 
            match other with
            | :? BaseMultiChannelExpr as other -> this.Op.Equals other.Op
            | _ -> false

    override this.Equals other =
        match other with
        | :? IExpr as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<IExpr> with
        member this.CompareTo other =
            match other with
            | :? BaseMultiChannelExpr as other -> compare this.Op other.Op
            | :? BaseExpr -> -1
            | _ -> failwithf "Cannot compate MultiChannelExpr to type %A." (other.GetType())

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? IExpr as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare MultiChannelExpr to type %A." (other.GetType())

    override this.GetHashCode() =
        hash this.Op

