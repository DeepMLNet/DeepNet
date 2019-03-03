namespace Tensor.Expr

open Tensor.Expr.Ops



/// Active recognizers for multi-channel expressions.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module MultiChannelExpr =

    let (|Loop|_|) (expr: MultiChannelExpr) =
        match expr.Op with
        | :? Loop as this -> Some (this.Length, this.Vars, this.Channels, this.Xs |> List.map Expr)
        | _ -> None

