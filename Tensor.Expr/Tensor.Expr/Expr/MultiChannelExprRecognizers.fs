namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Expr.Ops



/// Active recognizers for multi-channel expressions.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module MultiChannelExpr =

    let (|Bundle|_|) (expr: MultiChannelExpr) =
        match expr.Op with
        | :? Bundle as this ->
            Some (this.ChExprs |> Map.map (fun _ e -> Expr e))
        | _ -> None

    let (|Loop|_|) (expr: MultiChannelExpr) =
        match expr.Op with
        | :? Loop as this -> Some (this.Length, this.Vars, this.Channels, this.Xs |> List.map Expr)
        | _ -> None

