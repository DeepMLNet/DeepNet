namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Expr.Base
open Tensor.Expr.Ops



/// Active recognizers for multi-channel expressions.
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module MultiChannelExpr =

    let (|Bundle|_|) (expr: MultiChannelExpr) =
        match expr.Op with
        | :? Bundle as this ->
            Some (this.ChExprs |> Map.map (fun _ e -> UExpr e))
        | _ -> None

    let (|Loop|_|) (expr: MultiChannelExpr) =
        match expr.Op with
        | :? Loop as this -> Some this
        | _ -> None


/// Active recognizers for single- and multi-channel expressions.
module ExprChs = 

    /// Discriminates between single- and multi-channel expressions.
    let (|Single|Multi|) (baseExpr: BaseExpr) =
        if baseExpr.IsSingleChannel then
            Single (UExpr baseExpr)
        else
            Multi (MultiChannelExpr baseExpr)

