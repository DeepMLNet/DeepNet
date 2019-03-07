namespace Tensor.Expr

open Tensor.Expr.Ops


/// Active recognizers for typed single-channel expressions.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Expr =

    let (|Scalar|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Scalar as this -> Some (this.Value.Value :?> 'T)
        | _ -> None

    let (|SizeValue|_|) (expr: Expr<int64>) =
        match expr.Op with
        | :? SizeValue as this -> Some this.Value
        | _ -> None

    let (|Identity|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Identity as this -> Some this
        | _ -> None

    let (|Arange|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Arange as this -> Some this
        | _ -> None

    let (|VarArg|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? VarArg as this -> Some (Var<'T> this.Var)
        | _ -> None

    let varArg (expr: Expr<'T>) = 
        match expr with
        | VarArg v -> v
        | _ -> failwithf "Not an expression consisting solely of a variable."

    let (|DataArg|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? DataArg as this -> Some (Data<'T> this.Data)
        | _ -> None

    let (|UnaryPlus|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? UnaryPlus as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Negate|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Negate as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Abs|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Abs as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|SignT|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? SignT as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Log|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Log as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Log10|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Log10 as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Exp|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Exp as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Sin|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Sin as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Cos|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Cos as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Tan|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Tan as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Asin|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Asin as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Acos|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Acos as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Atan|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Atan as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Sinh|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Sinh as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Cosh|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Cosh as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Tanh|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Tanh as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Sqrt|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Sqrt as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Ceiling|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Ceiling as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Floor|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Floor as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Round|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Round as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Truncate|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Truncate as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Invert|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Invert as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|Not|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? Not as this -> Some (Expr<bool> this.X)
        | _ -> None

    let (|Reshape|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Reshape as this -> Some (this.Shape, Expr<'T> this.X)
        | _ -> None

    let (|DoBroadcast|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? DoBroadcast as this -> Some (this.Shape, Expr<'T> this.X)
        | _ -> None

    let (|PermuteAxes|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? PermuteAxes as this -> Some (this.Permutation, Expr<'T> this.X)
        | _ -> None

    let (|Subtensor|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Subtensor as this -> Some (this.Range, Expr<'T> this.X)
        | _ -> None

    let (|SetSubtensor|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? SetSubtensor as this -> Some (this.Range, Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|ReverseAxis|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? ReverseAxis as this -> Some (this.Axis, Expr<'T> this.X)
        | _ -> None   

    let (|Diag|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Diag as this -> Some (this.Axis1, this.Axis2, Expr<'T> this.X)
        | _ -> None   

    let (|DiagMat|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? DiagMat as this -> Some (this.Axis1, this.Axis2, Expr<'T> this.X)
        | _ -> None   

    let (|SumAxis|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? SumAxis as this -> Some (this.Axis, Expr<'T> this.X)
        | _ -> None

    let (|ProductAxis|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? ProductAxis as this -> Some (this.Axis, Expr<'T> this.X)
        | _ -> None

    let (|MaxAxis|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? MaxAxis as this -> Some (this.Axis, Expr<'T> this.X)
        | _ -> None

    let (|MinAxis|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? MinAxis as this -> Some (this.Axis, Expr<'T> this.X)
        | _ -> None

    let (|ArgMaxAxis|_|) (expr: Expr<int64>) =
        match expr.Op with
        | :? ArgMaxAxis as this -> Some (this.Axis, Expr<'T> this.X)
        | _ -> None

    let (|ArgMinAxis|_|) (expr: Expr<int64>) =
        match expr.Op with
        | :? ArgMinAxis as this -> Some (this.Axis, Expr<'T> this.X)
        | _ -> None

    let (|Gather|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Gather as this -> 
            let indices = this.Indices |> List.map (Option.map Expr<int64>)
            Some (indices, Expr<'T> this.X)
        | _ -> None    

    let (|Scatter|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Scatter as this -> 
            let indices = this.Indices |> List.map (Option.map Expr<int64>)
            Some (this.Shape, indices, Expr<'T> this.X)
        | _ -> None   

    let (|Store|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Store as this -> Some (Var<'T> this.Var, Expr<'T> this.X)
        | _ -> None

    let (|AssumeZeroDeriv|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? AssumeZeroDeriv as this -> Some (Expr<'T> this.X)
        | _ -> None

    let (|AssumeDeriv|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? AssumeDeriv as this -> Some (Expr<'T> this.Deriv, Expr<'T> this.X)
        | _ -> None

    let (|Annotated|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Annotated as this -> Some (this.Label, Expr<'T> this.X)
        | _ -> None

    let (|Print|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Print as this -> Some (this.Label, Expr<'T> this.X)
        | _ -> None

    let (|Dump|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Dump as this -> Some (this.Dataset, Expr<'T> this.X)
        | _ -> None

    let (|CheckFinite|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? CheckFinite as this -> Some (this.Label, Expr<'T> this.X)
        | _ -> None

    let (|Channel|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Channel as this -> Some (this.X.Channel, MultiChannelExpr this.X.Expr)
        | _ -> None

    let (|Add|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Add as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|Subtract|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Subtract as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|Multiply|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Multiply as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|Divide|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Divide as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|Pow|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Pow as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|Modulo|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Modulo as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|MaxElemwise|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? MaxElemwise as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|MinElemwise|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? MinElemwise as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|And|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? And as this -> Some (Expr<bool> this.X, Expr<bool> this.Y)
        | _ -> None

    let (|Or|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? Or as this -> Some (Expr<bool> this.X, Expr<bool> this.Y)
        | _ -> None

    let (|Xor|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? Xor as this -> Some (Expr<bool> this.X, Expr<bool> this.Y)
        | _ -> None

    let (|Equal|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? Equal as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|NotEqual|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? NotEqual as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|Less|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? Less as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|LessOrEqual|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? LessOrEqual as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|Greater|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? Greater as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|GreaterOrEqual|_|) (expr: Expr<bool>) =
        match expr.Op with
        | :? GreaterOrEqual as this -> Some (UExpr this.X, UExpr this.Y)
        | _ -> None

    let (|Dot|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Dot as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|TensorProduct|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? TensorProduct as this -> Some (Expr<'T> this.X, Expr<'T> this.Y)
        | _ -> None

    let (|IfThenElse|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? IfThenElse as this -> Some (Expr<bool> this.Cond, Expr<'T> this.IfTrue, Expr<'T> this.IfFalse)
        | _ -> None

    let (|BuildTensor|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? BuildTensor as this -> Some (this.Shape, this.Ranges, this.Xs |> List.map Expr<'T>)
        | _ -> None

    let (|Elements|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Elements as this -> Some (this.Shape, this.ElemExpr, this.Xs |> List.map Expr<'T>)
        | _ -> None

    let (|Interpolate|_|) (expr: Expr<'T>) =
        match expr.Op with
        | :? Interpolate as this -> Some (this.Interpolator, this.Xs |> List.map Expr<'T>)
        | _ -> None

