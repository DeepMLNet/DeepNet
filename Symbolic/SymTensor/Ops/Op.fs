namespace rec SymTensor.Ops

open SymTensor
open DeepNet.Utils
open Tensor
open Tensor.Backend
open System.Drawing


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

/// Information necessary to evaluate an expression.
/// Currently this just holds the variable values, but may contain further information in the future.
type EvalEnv = {
    /// Values of variables.
    VarEnv: VarEnv
    /// Device to store result on.
    Dev:    ITensorDevice
    /// Argument values.
    Args:   Map<string, ITensor>
}

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
    abstract Eval: env:EvalEnv -> Tensor.ITensor

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

    /// scalar constant of given value
    static member scalar (f: obj) = 
        Expr2 (OpForwards.ScalarConst (Const.ofValue f)) 

    /// scalar of given value converted to same type as given expression
    static member scalarOfSameType (expr: Expr2) f = 
        let v = System.Convert.ChangeType (box f, expr.TypeName.Type)
        Expr2.scalar v

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


[<AllowNullLiteral>]
type internal IOpForwards =   

    abstract ScalarConst: value:Const -> IOp2
    abstract Reshape: shp:ShapeSpec -> x:Expr2 -> IOp2
    abstract DoBroadcast: shp:ShapeSpec -> x:Expr2 -> IOp2

    abstract UnaryPlus: x:Expr2 -> IOp2
    abstract Negate: x:Expr2 -> IOp2
    abstract Abs: x:Expr2 -> IOp2
    abstract SignT: x:Expr2 -> IOp2
    abstract Log: x:Expr2 -> IOp2
    abstract Log10: x:Expr2 -> IOp2
    abstract Exp: x:Expr2 -> IOp2
    abstract Sin: x:Expr2 -> IOp2
    abstract Cos: x:Expr2 -> IOp2
    abstract Tan: x:Expr2 -> IOp2
    abstract Asin: x:Expr2 -> IOp2
    abstract Acos: x:Expr2 -> IOp2
    abstract Atan: x:Expr2 -> IOp2
    abstract Sinh: x:Expr2 -> IOp2
    abstract Cosh: x:Expr2 -> IOp2
    abstract Tanh: x:Expr2 -> IOp2
    abstract Sqrt: x:Expr2 -> IOp2
    abstract Ceiling: x:Expr2 -> IOp2
    abstract Floor: x:Expr2 -> IOp2
    abstract Round: x:Expr2 -> IOp2
    abstract Truncate: x:Expr2 -> IOp2

    abstract Not: x:Expr2 -> IOp2

    abstract Add: x:Expr2 -> y:Expr2 -> IOp2
    abstract Subtract: x:Expr2 -> y:Expr2 -> IOp2
    abstract Multiply: x:Expr2 -> y:Expr2 -> IOp2
    abstract Divide: x:Expr2 -> y:Expr2 -> IOp2
    abstract Pow: x:Expr2 -> y:Expr2 -> IOp2
    abstract Modulo: x:Expr2 -> y:Expr2 -> IOp2

    abstract And: x:Expr2 -> y:Expr2 -> IOp2
    abstract Or: x:Expr2 -> y:Expr2 -> IOp2
    abstract Xor: x:Expr2 -> y:Expr2 -> IOp2

    abstract Equal: x:Expr2 -> y:Expr2 -> IOp2
    abstract NotEqual: x:Expr2 -> y:Expr2 -> IOp2
    abstract Less: x:Expr2 -> y:Expr2 -> IOp2
    abstract LessOrEqual: x:Expr2 -> y:Expr2 -> IOp2
    abstract Greater: x:Expr2 -> y:Expr2 -> IOp2
    abstract GreaterOrEqual: x:Expr2 -> y:Expr2 -> IOp2

    abstract Dot: x:Expr2 -> y:Expr2 -> Expr2

[<AutoOpen>]
module internal OpForwardTypes = 
    let OpForwards : IOpForwards = 
        let typ = System.Type.GetType("OpForwards")
        System.Activator.CreateInstance(typ) :?> IOpForwards



