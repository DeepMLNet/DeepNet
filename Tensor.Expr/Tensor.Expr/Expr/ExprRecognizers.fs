namespace Tensor.Expr

open Tensor.Expr.Ops


/// Active recognizers for untyped single-channel expressions.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Expr =

    let (|Scalar|_|) (expr: Expr) =
        match expr.Op with
        | :? Scalar as this -> Some this.Value.Value
        | _ -> None

    let (|SizeValue|_|) (expr: Expr) =
        match expr.Op with
        | :? SizeValue as this -> Some this.Value
        | _ -> None

    let (|Identity|_|) (expr: Expr) =
        match expr.Op with
        | :? Identity as this -> Some this
        | _ -> None

    let (|Arange|_|) (expr: Expr) =
        match expr.Op with
        | :? Arange as this -> Some this
        | _ -> None

    let (|VarArg|_|) (expr: Expr) =
        match expr.Op with
        | :? VarArg as this -> Some this.Var
        | _ -> None

    let varArg expr = 
        match expr with
        | VarArg v -> v
        | _ -> failwithf "Not an expression consisting solely of a variable."

    let (|UnaryPlus|_|) (expr: Expr) =
        match expr.Op with
        | :? UnaryPlus as this -> Some (Expr this.X)
        | _ -> None

    let (|Negate|_|) (expr: Expr) =
        match expr.Op with
        | :? Negate as this -> Some (Expr this.X)
        | _ -> None

    let (|Abs|_|) (expr: Expr) =
        match expr.Op with
        | :? Abs as this -> Some (Expr this.X)
        | _ -> None

    let (|SignT|_|) (expr: Expr) =
        match expr.Op with
        | :? SignT as this -> Some (Expr this.X)
        | _ -> None

    let (|Log|_|) (expr: Expr) =
        match expr.Op with
        | :? Log as this -> Some (Expr this.X)
        | _ -> None

    let (|Log10|_|) (expr: Expr) =
        match expr.Op with
        | :? Log10 as this -> Some (Expr this.X)
        | _ -> None

    let (|Exp|_|) (expr: Expr) =
        match expr.Op with
        | :? Exp as this -> Some (Expr this.X)
        | _ -> None

    let (|Sin|_|) (expr: Expr) =
        match expr.Op with
        | :? Sin as this -> Some (Expr this.X)
        | _ -> None

    let (|Cos|_|) (expr: Expr) =
        match expr.Op with
        | :? Cos as this -> Some (Expr this.X)
        | _ -> None

    let (|Tan|_|) (expr: Expr) =
        match expr.Op with
        | :? Tan as this -> Some (Expr this.X)
        | _ -> None

    let (|Asin|_|) (expr: Expr) =
        match expr.Op with
        | :? Asin as this -> Some (Expr this.X)
        | _ -> None

    let (|Acos|_|) (expr: Expr) =
        match expr.Op with
        | :? Acos as this -> Some (Expr this.X)
        | _ -> None

    let (|Atan|_|) (expr: Expr) =
        match expr.Op with
        | :? Atan as this -> Some (Expr this.X)
        | _ -> None

    let (|Sinh|_|) (expr: Expr) =
        match expr.Op with
        | :? Sinh as this -> Some (Expr this.X)
        | _ -> None

    let (|Cosh|_|) (expr: Expr) =
        match expr.Op with
        | :? Cosh as this -> Some (Expr this.X)
        | _ -> None

    let (|Tanh|_|) (expr: Expr) =
        match expr.Op with
        | :? Tanh as this -> Some (Expr this.X)
        | _ -> None

    let (|Sqrt|_|) (expr: Expr) =
        match expr.Op with
        | :? Sqrt as this -> Some (Expr this.X)
        | _ -> None

    let (|Ceiling|_|) (expr: Expr) =
        match expr.Op with
        | :? Ceiling as this -> Some (Expr this.X)
        | _ -> None

    let (|Floor|_|) (expr: Expr) =
        match expr.Op with
        | :? Floor as this -> Some (Expr this.X)
        | _ -> None

    let (|Round|_|) (expr: Expr) =
        match expr.Op with
        | :? Round as this -> Some (Expr this.X)
        | _ -> None

    let (|Truncate|_|) (expr: Expr) =
        match expr.Op with
        | :? Truncate as this -> Some (Expr this.X)
        | _ -> None

    let (|Invert|_|) (expr: Expr) =
        match expr.Op with
        | :? Invert as this -> Some (Expr this.X)
        | _ -> None

    let (|Not|_|) (expr: Expr) =
        match expr.Op with
        | :? Not as this -> Some (Expr this.X)
        | _ -> None

    let (|Reshape|_|) (expr: Expr) =
        match expr.Op with
        | :? Reshape as this -> Some (this.Shape, Expr this.X)
        | _ -> None

    let (|DoBroadcast|_|) (expr: Expr) =
        match expr.Op with
        | :? DoBroadcast as this -> Some (this.Shape, Expr this.X)
        | _ -> None

    let (|PermuteAxes|_|) (expr: Expr) =
        match expr.Op with
        | :? PermuteAxes as this -> Some (this.Permutation, Expr this.X)
        | _ -> None

    let (|Subtensor|_|) (expr: Expr) =
        match expr.Op with
        | :? Subtensor as this -> Some (this.Range, Expr this.X)
        | _ -> None

    let (|SetSubtensor|_|) (expr: Expr) =
        match expr.Op with
        | :? SetSubtensor as this -> Some (this.Range, Expr this.X, Expr this.Y)
        | _ -> None

    let (|ReverseAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? ReverseAxis as this -> Some (this.Axis, Expr this.X)
        | _ -> None   

    let (|Diag|_|) (expr: Expr) =
        match expr.Op with
        | :? Diag as this -> Some (this.Axis1, this.Axis2, Expr this.X)
        | _ -> None   

    let (|DiagMat|_|) (expr: Expr) =
        match expr.Op with
        | :? DiagMat as this -> Some (this.Axis1, this.Axis2, Expr this.X)
        | _ -> None   

    let (|SumAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? SumAxis as this -> Some (this.Axis, Expr this.X)
        | _ -> None

    let (|ProductAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? ProductAxis as this -> Some (this.Axis, Expr this.X)
        | _ -> None

    let (|MaxAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? MaxAxis as this -> Some (this.Axis, Expr this.X)
        | _ -> None

    let (|MinAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? MinAxis as this -> Some (this.Axis, Expr this.X)
        | _ -> None

    let (|ArgMaxAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? ArgMaxAxis as this -> Some (this.Axis, Expr this.X)
        | _ -> None

    let (|ArgMinAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? ArgMinAxis as this -> Some (this.Axis, Expr this.X)
        | _ -> None

    let (|Gather|_|) (expr: Expr) =
        match expr.Op with
        | :? Gather as this -> 
            let indices = this.Indices |> List.map (Option.map Expr)
            Some (indices, Expr this.X)
        | _ -> None    

    let (|Scatter|_|) (expr: Expr) =
        match expr.Op with
        | :? Scatter as this -> 
            let indices = this.Indices |> List.map (Option.map Expr)
            Some (this.Shape, indices, Expr this.X)
        | _ -> None   

    let (|Store|_|) (expr: Expr) =
        match expr.Op with
        | :? Store as this -> Some (this.Var, Expr this.X)
        | _ -> None

    let (|AssumeZeroDeriv|_|) (expr: Expr) =
        match expr.Op with
        | :? AssumeZeroDeriv as this -> Some (Expr this.X)
        | _ -> None

    let (|AssumeDeriv|_|) (expr: Expr) =
        match expr.Op with
        | :? AssumeDeriv as this -> Some (Expr this.Deriv, Expr this.X)
        | _ -> None

    let (|Annotated|_|) (expr: Expr) =
        match expr.Op with
        | :? Annotated as this -> Some (this.Label, Expr this.X)
        | _ -> None

    let (|Print|_|) (expr: Expr) =
        match expr.Op with
        | :? Print as this -> Some (this.Label, Expr this.X)
        | _ -> None

    let (|Dump|_|) (expr: Expr) =
        match expr.Op with
        | :? Dump as this -> Some (this.Dataset, Expr this.X)
        | _ -> None

    let (|CheckFinite|_|) (expr: Expr) =
        match expr.Op with
        | :? CheckFinite as this -> Some (this.Label, Expr this.X)
        | _ -> None

    let (|Channel|_|) (expr: Expr) =
        match expr.Op with
        | :? Channel as this -> Some (this.X.Channel, MultiChannelExpr this.X.Expr)
        | _ -> None

    let (|Add|_|) (expr: Expr) =
        match expr.Op with
        | :? Add as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Subtract|_|) (expr: Expr) =
        match expr.Op with
        | :? Subtract as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Multiply|_|) (expr: Expr) =
        match expr.Op with
        | :? Multiply as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Divide|_|) (expr: Expr) =
        match expr.Op with
        | :? Divide as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Pow|_|) (expr: Expr) =
        match expr.Op with
        | :? Pow as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Modulo|_|) (expr: Expr) =
        match expr.Op with
        | :? Modulo as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|MaxElemwise|_|) (expr: Expr) =
        match expr.Op with
        | :? MaxElemwise as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|MinElemwise|_|) (expr: Expr) =
        match expr.Op with
        | :? MinElemwise as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|And|_|) (expr: Expr) =
        match expr.Op with
        | :? And as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Or|_|) (expr: Expr) =
        match expr.Op with
        | :? Or as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Xor|_|) (expr: Expr) =
        match expr.Op with
        | :? Xor as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Equal|_|) (expr: Expr) =
        match expr.Op with
        | :? Equal as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|NotEqual|_|) (expr: Expr) =
        match expr.Op with
        | :? NotEqual as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Less|_|) (expr: Expr) =
        match expr.Op with
        | :? Less as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|LessOrEqual|_|) (expr: Expr) =
        match expr.Op with
        | :? LessOrEqual as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Greater|_|) (expr: Expr) =
        match expr.Op with
        | :? Greater as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|GreaterOrEqual|_|) (expr: Expr) =
        match expr.Op with
        | :? GreaterOrEqual as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|Dot|_|) (expr: Expr) =
        match expr.Op with
        | :? Dot as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|TensorProduct|_|) (expr: Expr) =
        match expr.Op with
        | :? TensorProduct as this -> Some (Expr this.X, Expr this.Y)
        | _ -> None

    let (|IfThenElse|_|) (expr: Expr) =
        match expr.Op with
        | :? IfThenElse as this -> Some (Expr this.Cond, Expr this.IfTrue, Expr this.IfFalse)
        | _ -> None

    let (|Discard|_|) (expr: Expr) =
        match expr.Op with
        | :? Discard as this -> Some (this.Xs |> List.map Expr)
        | _ -> None

    let (|BuildTensor|_|) (expr: Expr) =
        match expr.Op with
        | :? BuildTensor as this -> Some (this.Shape, this.Ranges, this.Xs |> List.map Expr)
        | _ -> None

    let (|Elements|_|) (expr: Expr) =
        match expr.Op with
        | :? Elements as this -> Some (this.Shape, this.ElemExpr, this.Xs |> List.map Expr)
        | _ -> None

    let (|Interpolate|_|) (expr: Expr) =
        match expr.Op with
        | :? Interpolate as this -> Some (this.Interpolator, this.Xs |> List.map Expr)
        | _ -> None

