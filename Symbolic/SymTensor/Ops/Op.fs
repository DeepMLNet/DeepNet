namespace rec SymTensor.Ops

open SymTensor
open DeepNet.Utils
open Tensor
open Tensor.Backend


///// A mathematical operation in an expression.
///// This models a mathematical function or operator that takes one or more tensors
///// and returns one tensor.
//type IOp =
//    inherit System.IComparable
      
//    /// Should return the type of the result, given the types of the arguments.
//    abstract TypeName: argTypes: TypeName list -> TypeName

//    /// Should return the shape of the result, given the shape of the arguments.
//    abstract Shape: argShapes: ShapeSpec list -> ShapeSpec      
        
//    /// Should check if the shapes of the arguments are acceptable and,
//    /// if not, raise an exception.
//    abstract CheckArgs: argShapes: ShapeSpec list -> unit      

//    /// Should return the op with all symbolic sizes substituted using the specified
//    /// substitution table.
//    /// Return a *new* op with substitution applied. Do not apply the mapping in-place.
//    abstract SubstSymSizes: symSizes: SymSizeEnv -> IOp

//    /// Should be true, if all symbolic sizes can be evaluated to numeric sizes.
//    /// This is the case if the function ShapeSpec.canEval or SizeSpec.canEval respectively
//    /// return true on all sizes used in this op.
//    abstract CanEvalAllSymSizes: bool

//    /// Should compute the derivative w.r.t. each argument given the derivative w.r.t. the op.
//    /// The derivative is always an NxM matrix where N is the number of elements of the function
//    /// the derivative of which is being taken and M is the number of elements of the argument
//    /// w.r.t. which the derivative is being taken. 
//    /// Thus, if dOp is an NxK matrix and an argument has M elements, the derivative matrix
//    /// you return w.r.t. that argument must have NxM elements.
//    abstract Deriv: dOp:Expr -> args:Expr list -> Expr list

//    /// Should evaluate the numerical value of this op given the numerical values of its arguments.
//    /// This evaluation should be done on the host using the simplest means possible and is used
//    /// as a reference implementation for verifying the correctness of optimized (e.g. CUDA) 
//    /// implementations. This method may be omitted when no verification will be done.
//    abstract EvalSimple: args:Tensor.Tensor<'T> list -> Tensor.Tensor<'T>

//    /// Should return the set of variables that this op instance depends on.
//    abstract ContainedVars: Set<Var>


    
type ArgsMap = Map<string, Expr2>

/// A mathematical operation in an expression.
/// This models a mathematical function or operator that takes one or more tensors
/// and returns one tensor.
type IOp2 =
    inherit System.IComparable
      
    /// Should check if the types and shapes of the arguments are acceptable and,
    /// if not, raise an exception.
    abstract Check: unit -> unit

    /// Should return the type of the result.
    abstract TypeName: TypeName

    /// Should return the shape of the result.
    abstract Shape: ShapeSpec      
        
    /// Returns the arguments of this op.
    abstract Args: ArgsMap

    /// Creates a new op with the arguments replaced by the specified arguments.
    abstract ReplaceArgs: ArgsMap -> IOp2

    /// Should return the expression with all symbolic sizes substituted using the specified
    /// substitution table.
    /// Return a *new* op with substitution applied. Do not apply the mapping in-place.
    abstract SubstSymSizes: env: SymSizeEnv -> IOp2

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
    abstract Deriv: dOp:Expr2 -> Map<string, Expr2>

    /// Should evaluate the numerical value of this op given the numerical values of its arguments.
    /// This evaluation should be done on the host using the simplest means possible and is used
    /// as a reference implementation for verifying the correctness of optimized (e.g. CUDA) 
    /// implementations. This method may be omitted when no verification will be done.
    abstract Eval: dev:ITensorDevice -> args:Map<string, Tensor.ITensor> -> Tensor.ITensor

    /// Should return the set of variables that this op instance depends on.
    //abstract ContainedVars: Set<Var>

//type Expr = Expr of IOp * (Expr list)

type Expr2 (op: IOp2) =
    
    do op.Check()
        
    member this.Op = op
    static member op (expr: Expr2) = expr.Op

    member this.TypeName = op.TypeName   
    static member typeName (expr: Expr2) = expr.TypeName

    member this.Shape = op.Shape
    static member shape (expr: Expr2) = expr.Shape

    interface System.IEquatable<Expr2> with
        member this.Equals other = 
            this.Op.Equals other.Op

    override this.Equals other =
        match other with
        | :? Expr2 as other -> (this :> System.IEquatable<_>).Equals other
        | _ -> false

    interface System.IComparable<Expr2> with
        member this.CompareTo other =
            compare this.Op other.Op

    interface System.IComparable with
        member this.CompareTo other =
            match other with
            | :? Expr2 as other -> (this :> System.IComparable<_>).CompareTo other
            | _ -> failwithf "Cannot compare Expr to type %A." (other.GetType())

    override this.GetHashCode() =
        hash this.Op

    /// Wraps the given op in a Reshape op if its shape does not match ss.
    static member reshapeIfNecessary ss (expr: Expr2) =
        if ss = expr.Shape then expr else Expr2(OpForwards.Reshape ss expr)

    /// Wraps the given op in a Broadcast op if its shape does not match ss.
    static member broadcastIfNecessary ss (expr: Expr2) =
        if ss = expr.Shape then expr else Expr2(OpForwards.DoBroadcast ss expr)

    /// emits an elementwise binary operation with broadcasting of the inputs if necessary
    static member constructElementwise op (a: Expr2) (b: Expr2) =
        let psa, psb = ShapeSpec.padToSame a.Shape b.Shape
        let bsa, bsb = ShapeSpec.broadcastToSame false psa psb
        let ba = a |> Expr2.reshapeIfNecessary psa |> Expr2.broadcastIfNecessary bsa
        let bb = b |> Expr2.reshapeIfNecessary psb |> Expr2.broadcastIfNecessary bsb    
        Expr2 (op ba bb)


    static member (~-) (x: Expr2) = Expr2 (OpForwards.Negate x)

    static member (+) (a: Expr2, b: Expr2) = Expr2.constructElementwise OpForwards.Add a b


[<AllowNullLiteral>]
type internal IOpForwards =   
    abstract Reshape: shp:ShapeSpec -> x:Expr2 -> IOp2
    abstract DoBroadcast: shp:ShapeSpec -> x:Expr2 -> IOp2
    abstract Negate: x:Expr2 -> IOp2
    abstract Add: x:Expr2 -> y:Expr2 -> IOp2

[<AutoOpen>]
module internal OpForwardTypes = 
    let OpForwards : IOpForwards = 
        let typ = System.Type.GetType("OpForwards")
        System.Activator.CreateInstance(typ) :?> IOpForwards



