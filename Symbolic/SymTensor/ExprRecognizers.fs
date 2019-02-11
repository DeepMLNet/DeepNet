namespace SymTensor

open SymTensor.Ops


/// Active recognizers for single-channel expressions.
module Expr =

    let (|Scalar|_|) (expr: Expr2) =
        match expr.Op with
        | :? Scalar as this -> Some this.Value
        | _ -> None

    let (|SizeValue|_|) (expr: Expr2) =
        match expr.Op with
        | :? SizeValue as this -> Some this.Value
        | _ -> None

    let (|Identity|_|) (expr: Expr2) =
        match expr.Op with
        | :? Identity as this -> Some this
        | _ -> None

    let (|Arange|_|) (expr: Expr2) =
        match expr.Op with
        | :? Arange as this -> Some this
        | _ -> None

    let (|VarArg|_|) (expr: Expr2) =
        match expr.Op with
        | :? VarArg as this -> Some this.Var
        | _ -> None

    let (|UnaryPlus|_|) (expr: Expr2) =
        match expr.Op with
        | :? UnaryPlus as this -> Some this.X
        | _ -> None

    let (|Negate|_|) (expr: Expr2) =
        match expr.Op with
        | :? Negate as this -> Some this.X
        | _ -> None

    let (|Abs|_|) (expr: Expr2) =
        match expr.Op with
        | :? Abs as this -> Some this.X
        | _ -> None

    let (|SignT|_|) (expr: Expr2) =
        match expr.Op with
        | :? SignT as this -> Some this.X
        | _ -> None



