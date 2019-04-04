namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Expr.Ops


/// Active recognizers for untyped single-channel expressions.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module UExpr =

    /// Matches an expression with an element-wise operation.
    let (|UnaryElemwise|_|) (expr: UExpr) =
        match expr.Op with
        | :? IElemwiseOp when Map.keys expr.Args = Set [Arg.Only] -> 
            Some (expr.Op, expr.Args.[Arg.Only])
        | _ -> None

    let (|BinaryElemwise|_|) (expr: UExpr) =
        match expr.Op with
        | :? IElemwiseOp when Map.keys expr.Args = Set [Arg.X; Arg.Y] ->
            Some (expr.Op, expr.Args.[Arg.X], expr.Args.[Arg.Y])
        | _ -> None

    let (|Scalar|_|) (expr: UExpr) =
        match expr.Op with
        | :? Scalar as this -> Some this.Value.Value
        | _ -> None

    let (|SizeValue|_|) (expr: UExpr) =
        match expr.Op with
        | :? SizeValue as this -> Some this.Value
        | _ -> None

    let (|Identity|_|) (expr: UExpr) =
        match expr.Op with
        | :? Identity as this -> Some this
        | _ -> None

    let (|Counting|_|) (expr: UExpr) =
        match expr.Op with
        | :? Counting as this -> Some this
        | _ -> None

    let (|VarArg|_|) (expr: UExpr) =
        match expr.Op with
        | :? VarArg as this -> Some this.Var
        | _ -> None

    let varArg expr = 
        match expr with
        | VarArg v -> v
        | _ -> failwithf "Not an expression consisting solely of a variable."

    let (|DataArg|_|) (expr: UExpr) =
        match expr.Op with
        | :? DataArg as this -> Some this.Data.Value
        | _ -> None

    let (|UnaryPlus|_|) (expr: UExpr) =
        match expr.Op with
        | :? UnaryPlus as this -> Some (UExpr this.X)
        | _ -> None

    let (|Negate|_|) (expr: UExpr) =
        match expr.Op with
        | :? Negate as this -> Some (UExpr this.X)
        | _ -> None

    let (|Abs|_|) (expr: UExpr) =
        match expr.Op with
        | :? Abs as this -> Some (UExpr this.X)
        | _ -> None

    let (|SignT|_|) (expr: UExpr) =
        match expr.Op with
        | :? SignT as this -> Some (UExpr this.X)
        | _ -> None

    let (|Log|_|) (expr: UExpr) =
        match expr.Op with
        | :? Log as this -> Some (UExpr this.X)
        | _ -> None

    let (|Log10|_|) (expr: UExpr) =
        match expr.Op with
        | :? Log10 as this -> Some (UExpr this.X)
        | _ -> None

    let (|Exp|_|) (expr: UExpr) =
        match expr.Op with
        | :? Exp as this -> Some (UExpr this.X)
        | _ -> None

    let (|Sin|_|) (expr: UExpr) =
        match expr.Op with
        | :? Sin as this -> Some (UExpr this.X)
        | _ -> None

    let (|Cos|_|) (expr: UExpr) =
        match expr.Op with
        | :? Cos as this -> Some (UExpr this.X)
        | _ -> None

    let (|Tan|_|) (expr: UExpr) =
        match expr.Op with
        | :? Tan as this -> Some (UExpr this.X)
        | _ -> None

    let (|Asin|_|) (expr: UExpr) =
        match expr.Op with
        | :? Asin as this -> Some (UExpr this.X)
        | _ -> None

    let (|Acos|_|) (expr: UExpr) =
        match expr.Op with
        | :? Acos as this -> Some (UExpr this.X)
        | _ -> None

    let (|Atan|_|) (expr: UExpr) =
        match expr.Op with
        | :? Atan as this -> Some (UExpr this.X)
        | _ -> None

    let (|Sinh|_|) (expr: UExpr) =
        match expr.Op with
        | :? Sinh as this -> Some (UExpr this.X)
        | _ -> None

    let (|Cosh|_|) (expr: UExpr) =
        match expr.Op with
        | :? Cosh as this -> Some (UExpr this.X)
        | _ -> None

    let (|Tanh|_|) (expr: UExpr) =
        match expr.Op with
        | :? Tanh as this -> Some (UExpr this.X)
        | _ -> None

    let (|Sqrt|_|) (expr: UExpr) =
        match expr.Op with
        | :? Sqrt as this -> Some (UExpr this.X)
        | _ -> None

    let (|Ceiling|_|) (expr: UExpr) =
        match expr.Op with
        | :? Ceiling as this -> Some (UExpr this.X)
        | _ -> None

    let (|Floor|_|) (expr: UExpr) =
        match expr.Op with
        | :? Floor as this -> Some (UExpr this.X)
        | _ -> None

    let (|Round|_|) (expr: UExpr) =
        match expr.Op with
        | :? Round as this -> Some (UExpr this.X)
        | _ -> None

    let (|Truncate|_|) (expr: UExpr) =
        match expr.Op with
        | :? Truncate as this -> Some (UExpr this.X)
        | _ -> None

    let (|Invert|_|) (expr: UExpr) =
        match expr.Op with
        | :? Invert as this -> Some (UExpr this.X)
        | _ -> None

    let (|Not|_|) (expr: UExpr) =
        match expr.Op with
        | :? Not as this -> Some (UExpr this.X)
        | _ -> None

    let (|Reshape|_|) (expr: UExpr) =
        match expr.Op with
        | :? Reshape as this -> Some (this.Shape, UExpr this.X)
        | _ -> None

    let (|DoBroadcast|_|) (expr: UExpr) =
        match expr.Op with
        | :? DoBroadcast as this -> Some (this.Shape, UExpr this.X)
        | _ -> None

    let (|PermuteAxes|_|) (expr: UExpr) =
        match expr.Op with
        | :? PermuteAxes as this -> Some (this.Permutation, UExpr this.X)
        | _ -> None

    let (|Subtensor|_|) (expr: UExpr) =
        match expr.Op with
        | :? Subtensor as this -> Some (this.Range, UExpr this.X)
        | _ -> None

    let (|SetSubtensor|_|) (expr: UExpr) =
        match expr.Op with
        | :? SetSubtensor as this -> Some (this.Range, UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|ReverseAxis|_|) (expr: UExpr) =
        match expr.Op with
        | :? ReverseAxis as this -> Some (this.Axis, UExpr this.X)
        | _ -> None   

    let (|Diag|_|) (expr: UExpr) =
        match expr.Op with
        | :? Diag as this -> Some (this.Axis1, this.Axis2, UExpr this.X)
        | _ -> None   

    let (|DiagMat|_|) (expr: UExpr) =
        match expr.Op with
        | :? DiagMat as this -> Some (this.Axis1, this.Axis2, UExpr this.X)
        | _ -> None   

    let (|SumAxis|_|) (expr: UExpr) =
        match expr.Op with
        | :? SumAxis as this -> Some (this.Axis, UExpr this.X)
        | _ -> None

    let (|ProductAxis|_|) (expr: UExpr) =
        match expr.Op with
        | :? ProductAxis as this -> Some (this.Axis, UExpr this.X)
        | _ -> None

    let (|MaxAxis|_|) (expr: UExpr) =
        match expr.Op with
        | :? MaxAxis as this -> Some (this.Axis, UExpr this.X)
        | _ -> None

    let (|MinAxis|_|) (expr: UExpr) =
        match expr.Op with
        | :? MinAxis as this -> Some (this.Axis, UExpr this.X)
        | _ -> None

    let (|ArgMaxAxis|_|) (expr: UExpr) =
        match expr.Op with
        | :? ArgMaxAxis as this -> Some (this.Axis, UExpr this.X)
        | _ -> None

    let (|ArgMinAxis|_|) (expr: UExpr) =
        match expr.Op with
        | :? ArgMinAxis as this -> Some (this.Axis, UExpr this.X)
        | _ -> None

    let (|Gather|_|) (expr: UExpr) =
        match expr.Op with
        | :? Gather as this -> 
            let indices = this.Indices |> List.map (Option.map UExpr)
            Some (indices, UExpr this.X)
        | _ -> None    

    let (|Scatter|_|) (expr: UExpr) =
        match expr.Op with
        | :? Scatter as this -> 
            let indices = this.Indices |> List.map (Option.map UExpr)
            Some (this.Shape, indices, UExpr this.X)
        | _ -> None   

    let (|Store|_|) (expr: UExpr) =
        match expr.Op with
        | :? Store as this -> Some (this.Var, UExpr this.X)
        | _ -> None

    let (|AssumeZeroDeriv|_|) (expr: UExpr) =
        match expr.Op with
        | :? AssumeZeroDeriv as this -> Some (UExpr this.X)
        | _ -> None

    let (|AssumeDeriv|_|) (expr: UExpr) =
        match expr.Op with
        | :? AssumeDeriv as this -> Some (UExpr this.Deriv, UExpr this.X)
        | _ -> None

    let (|Annotated|_|) (expr: UExpr) =
        match expr.Op with
        | :? Annotated as this -> Some (this.Label, UExpr this.X)
        | _ -> None

    let (|Print|_|) (expr: UExpr) =
        match expr.Op with
        | :? Print as this -> Some (this.Label, UExpr this.X)
        | _ -> None

    let (|Dump|_|) (expr: UExpr) =
        match expr.Op with
        | :? Dump as this -> Some (this.Dataset, UExpr this.X)
        | _ -> None

    let (|CheckFinite|_|) (expr: UExpr) =
        match expr.Op with
        | :? CheckFinite as this -> Some (this.Label, UExpr this.X)
        | _ -> None

    let (|Convert|_|) (expr: UExpr) =
        match expr.Op with
        | :? Convert as this -> Some (this.ToType, UExpr this.X)
        | _ -> None

    let (|Transfer|_|) (expr: UExpr) =
        match expr.Op with
        | :? Transfer as this -> Some (this.ToDev, UExpr this.X)
        | _ -> None

    let (|Channel|_|) (expr: UExpr) =
        match expr.Op with
        | :? Channel as this -> Some (this.X.Channel, MultiChannelExpr this.X.Expr)
        | _ -> None

    let (|Add|_|) (expr: UExpr) =
        match expr.Op with
        | :? Add as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Subtract|_|) (expr: UExpr) =
        match expr.Op with
        | :? Subtract as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Multiply|_|) (expr: UExpr) =
        match expr.Op with
        | :? Multiply as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Divide|_|) (expr: UExpr) =
        match expr.Op with
        | :? Divide as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Pow|_|) (expr: UExpr) =
        match expr.Op with
        | :? Pow as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Modulo|_|) (expr: UExpr) =
        match expr.Op with
        | :? Modulo as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|MaxElemwise|_|) (expr: UExpr) =
        match expr.Op with
        | :? MaxElemwise as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|MinElemwise|_|) (expr: UExpr) =
        match expr.Op with
        | :? MinElemwise as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|And|_|) (expr: UExpr) =
        match expr.Op with
        | :? And as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Or|_|) (expr: UExpr) =
        match expr.Op with
        | :? Or as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Xor|_|) (expr: UExpr) =
        match expr.Op with
        | :? Xor as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Equal|_|) (expr: UExpr) =
        match expr.Op with
        | :? Equal as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|NotEqual|_|) (expr: UExpr) =
        match expr.Op with
        | :? NotEqual as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Less|_|) (expr: UExpr) =
        match expr.Op with
        | :? Less as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|LessOrEqual|_|) (expr: UExpr) =
        match expr.Op with
        | :? LessOrEqual as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Greater|_|) (expr: UExpr) =
        match expr.Op with
        | :? Greater as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|GreaterOrEqual|_|) (expr: UExpr) =
        match expr.Op with
        | :? GreaterOrEqual as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Dot|_|) (expr: UExpr) =
        match expr.Op with
        | :? Dot as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|TensorProduct|_|) (expr: UExpr) =
        match expr.Op with
        | :? TensorProduct as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|IfThenElse|_|) (expr: UExpr) =
        match expr.Op with
        | :? IfThenElse as this -> Some (UExpr this.Cond, UExpr this.IfTrue, UExpr this.IfFalse)
        | _ -> None

    let (|Discard|_|) (expr: UExpr) =
        match expr.Op with
        | :? Discard as this -> Some (this.Xs |> List.map UExpr)
        | _ -> None

    let (|BuildTensor|_|) (expr: UExpr) =
        match expr.Op with
        | :? BuildTensor as this -> Some (this.Shape, this.Ranges, this.Xs |> List.map UExpr)
        | _ -> None

    let (|Elements|_|) (expr: UExpr) =
        match expr.Op with
        | :? Elements as this -> Some (this.Shape, this.ElemExpr, this.Xs |> List.map UExpr)
        | _ -> None

    let (|Interpolate|_|) (expr: UExpr) =
        match expr.Op with
        | :? Interpolate as this -> Some (this.Interpolator, this.Xs |> List.map UExpr)
        | _ -> None

