namespace SymTensor

open SymTensor.Ops


/// Active recognizers for single-channel expressions.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Expr2 =

    let (|Scalar|_|) (expr: Expr) =
        match expr.Op with
        | :? Scalar as this -> Some this.Value
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

    let (|UnaryPlus|_|) (expr: Expr) =
        match expr.Op with
        | :? UnaryPlus as this -> Some this.X
        | _ -> None

    let (|Negate|_|) (expr: Expr) =
        match expr.Op with
        | :? Negate as this -> Some this.X
        | _ -> None

    let (|Abs|_|) (expr: Expr) =
        match expr.Op with
        | :? Abs as this -> Some this.X
        | _ -> None

    let (|SignT|_|) (expr: Expr) =
        match expr.Op with
        | :? SignT as this -> Some this.X
        | _ -> None

    let (|Log|_|) (expr: Expr) =
        match expr.Op with
        | :? Log as this -> Some this.X
        | _ -> None

    let (|Log10|_|) (expr: Expr) =
        match expr.Op with
        | :? Log10 as this -> Some this.X
        | _ -> None

    let (|Exp|_|) (expr: Expr) =
        match expr.Op with
        | :? Exp as this -> Some this.X
        | _ -> None

    let (|Sin|_|) (expr: Expr) =
        match expr.Op with
        | :? Sin as this -> Some this.X
        | _ -> None

    let (|Cos|_|) (expr: Expr) =
        match expr.Op with
        | :? Cos as this -> Some this.X
        | _ -> None

    let (|Tan|_|) (expr: Expr) =
        match expr.Op with
        | :? Tan as this -> Some this.X
        | _ -> None

    let (|Asin|_|) (expr: Expr) =
        match expr.Op with
        | :? Asin as this -> Some this.X
        | _ -> None

    let (|Acos|_|) (expr: Expr) =
        match expr.Op with
        | :? Acos as this -> Some this.X
        | _ -> None

    let (|Atan|_|) (expr: Expr) =
        match expr.Op with
        | :? Atan as this -> Some this.X
        | _ -> None

    let (|Sinh|_|) (expr: Expr) =
        match expr.Op with
        | :? Sinh as this -> Some this.X
        | _ -> None

    let (|Cosh|_|) (expr: Expr) =
        match expr.Op with
        | :? Cosh as this -> Some this.X
        | _ -> None

    let (|Tanh|_|) (expr: Expr) =
        match expr.Op with
        | :? Tanh as this -> Some this.X
        | _ -> None

    let (|Sqrt|_|) (expr: Expr) =
        match expr.Op with
        | :? Sqrt as this -> Some this.X
        | _ -> None

    let (|Ceiling|_|) (expr: Expr) =
        match expr.Op with
        | :? Ceiling as this -> Some this.X
        | _ -> None

    let (|Floor|_|) (expr: Expr) =
        match expr.Op with
        | :? Floor as this -> Some this.X
        | _ -> None

    let (|Round|_|) (expr: Expr) =
        match expr.Op with
        | :? Round as this -> Some this.X
        | _ -> None

    let (|Truncate|_|) (expr: Expr) =
        match expr.Op with
        | :? Truncate as this -> Some this.X
        | _ -> None

    let (|Invert|_|) (expr: Expr) =
        match expr.Op with
        | :? Invert as this -> Some this.X
        | _ -> None

    let (|Not|_|) (expr: Expr) =
        match expr.Op with
        | :? Not as this -> Some this.X
        | _ -> None

    let (|Reshape|_|) (expr: Expr) =
        match expr.Op with
        | :? Reshape as this -> Some this
        | _ -> None

    let (|DoBroadcast|_|) (expr: Expr) =
        match expr.Op with
        | :? DoBroadcast as this -> Some this
        | _ -> None

    let (|PermuteAxes|_|) (expr: Expr) =
        match expr.Op with
        | :? PermuteAxes as this -> Some this
        | _ -> None

    let (|Subtensor|_|) (expr: Expr) =
        match expr.Op with
        | :? Subtensor as this -> Some this
        | _ -> None

    let (|SetSubtensor|_|) (expr: Expr) =
        match expr.Op with
        | :? SetSubtensor as this -> Some this
        | _ -> None

    let (|ReverseAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? ReverseAxis as this -> Some this
        | _ -> None   

    let (|Diag|_|) (expr: Expr) =
        match expr.Op with
        | :? Diag as this -> Some this
        | _ -> None   

    let (|DiagMat|_|) (expr: Expr) =
        match expr.Op with
        | :? DiagMat as this -> Some this
        | _ -> None   

    let (|SumAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? SumAxis as this -> Some this
        | _ -> None

    let (|ProductAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? ProductAxis as this -> Some this
        | _ -> None

    let (|MaxAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? MaxAxis as this -> Some this
        | _ -> None

    let (|MinAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? MinAxis as this -> Some this
        | _ -> None

    let (|ArgMaxAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? ArgMaxAxis as this -> Some this
        | _ -> None

    let (|ArgMinAxis|_|) (expr: Expr) =
        match expr.Op with
        | :? ArgMinAxis as this -> Some this
        | _ -> None

    let (|Gather|_|) (expr: Expr) =
        match expr.Op with
        | :? Gather as this -> Some this
        | _ -> None    

    let (|Scatter|_|) (expr: Expr) =
        match expr.Op with
        | :? Scatter as this -> Some this
        | _ -> None   

    let (|Store|_|) (expr: Expr) =
        match expr.Op with
        | :? Store as this -> Some this
        | _ -> None

    let (|AssumeZeroDeriv|_|) (expr: Expr) =
        match expr.Op with
        | :? AssumeZeroDeriv as this -> Some this.X
        | _ -> None

    let (|AssumeDeriv|_|) (expr: Expr) =
        match expr.Op with
        | :? AssumeDeriv as this -> Some this
        | _ -> None

    let (|Annotated|_|) (expr: Expr) =
        match expr.Op with
        | :? Annotated as this -> Some this
        | _ -> None

    let (|Print|_|) (expr: Expr) =
        match expr.Op with
        | :? Print as this -> Some this
        | _ -> None

    let (|Dump|_|) (expr: Expr) =
        match expr.Op with
        | :? Dump as this -> Some this
        | _ -> None

    let (|CheckFinite|_|) (expr: Expr) =
        match expr.Op with
        | :? CheckFinite as this -> Some this
        | _ -> None

    let (|Channel|_|) (expr: Expr) =
        match expr.Op with
        | :? Channel as this -> Some this
        | _ -> None

    let (|Add|_|) (expr: Expr) =
        match expr.Op with
        | :? Add as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Subtract|_|) (expr: Expr) =
        match expr.Op with
        | :? Subtract as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Multiply|_|) (expr: Expr) =
        match expr.Op with
        | :? Multiply as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Divide|_|) (expr: Expr) =
        match expr.Op with
        | :? Divide as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Pow|_|) (expr: Expr) =
        match expr.Op with
        | :? Pow as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Modulo|_|) (expr: Expr) =
        match expr.Op with
        | :? Modulo as this -> Some (this.X, this.Y)
        | _ -> None

    let (|MaxElemwise|_|) (expr: Expr) =
        match expr.Op with
        | :? MaxElemwise as this -> Some (this.X, this.Y)
        | _ -> None

    let (|MinElemwise|_|) (expr: Expr) =
        match expr.Op with
        | :? MinElemwise as this -> Some (this.X, this.Y)
        | _ -> None

    let (|And|_|) (expr: Expr) =
        match expr.Op with
        | :? And as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Or|_|) (expr: Expr) =
        match expr.Op with
        | :? Or as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Xor|_|) (expr: Expr) =
        match expr.Op with
        | :? Xor as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Equal|_|) (expr: Expr) =
        match expr.Op with
        | :? Equal as this -> Some (this.X, this.Y)
        | _ -> None

    let (|NotEqual|_|) (expr: Expr) =
        match expr.Op with
        | :? NotEqual as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Less|_|) (expr: Expr) =
        match expr.Op with
        | :? Less as this -> Some (this.X, this.Y)
        | _ -> None

    let (|LessOrEqual|_|) (expr: Expr) =
        match expr.Op with
        | :? LessOrEqual as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Greater|_|) (expr: Expr) =
        match expr.Op with
        | :? Greater as this -> Some (this.X, this.Y)
        | _ -> None

    let (|GreaterOrEqual|_|) (expr: Expr) =
        match expr.Op with
        | :? GreaterOrEqual as this -> Some (this.X, this.Y)
        | _ -> None

    let (|Dot|_|) (expr: Expr) =
        match expr.Op with
        | :? Dot as this -> Some (this.X, this.Y)
        | _ -> None

    let (|TensorProduct|_|) (expr: Expr) =
        match expr.Op with
        | :? TensorProduct as this -> Some (this.X, this.Y)
        | _ -> None

    let (|IfThenElse|_|) (expr: Expr) =
        match expr.Op with
        | :? IfThenElse as this -> Some this
        | _ -> None

    let (|Discard|_|) (expr: Expr) =
        match expr.Op with
        | :? Discard as this -> Some (this.Xs)
        | _ -> None

    let (|BuildTensor|_|) (expr: Expr) =
        match expr.Op with
        | :? BuildTensor as this -> Some this
        | _ -> None

    let (|Elements|_|) (expr: Expr) =
        match expr.Op with
        | :? Elements as this -> Some this
        | _ -> None

    let (|Interpolate|_|) (expr: Expr) =
        match expr.Op with
        | :? Interpolate as this -> Some this
        | _ -> None


/// Active recognizers for multi-channel expressions.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module MultiChannelExpr =

    let (|Loop|_|) (expr: MultiChannelExpr) =
        match expr.Op with
        | :? Loop as this -> Some this
        | _ -> None
