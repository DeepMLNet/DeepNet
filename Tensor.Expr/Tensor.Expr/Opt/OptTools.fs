namespace Tensor.Expr.Opt

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops



/// Common utility functions for optimizers.
module Tools =

    /// Apply single-channel optimization function.
    let apply (optData: OptimizerData) (baseExpr: BaseExpr) (opt: OptimizerData -> UExpr -> UExpr) =
        match baseExpr with
        | ExprChs.Single uExpr -> opt optData uExpr |> UExpr.baseExpr       
        | _ -> baseExpr

    /// Apply multi-channel optimization function.
    let applyMultiChannel (optData: OptimizerData) (baseExpr: BaseExpr) (opt: OptimizerData -> MultiChannelExpr -> MultiChannelExpr) =
        match baseExpr with
        | ExprChs.Multi uExpr -> opt optData uExpr |> MultiChannelExpr.baseExpr       
        | _ -> baseExpr

    /// Apply the optimizers with the current settings to the expression tree.
    let subOpt (data: OptimizerData) (expr: UExpr) =
        data.SubOptimize expr.BaseExpr |> UExpr    
        
    /// Apply the optimizers with the current settings to the expression tree.
    let subOptMultiChannel (data: OptimizerData) (expr: MultiChannelExpr) =
        data.SubOptimize expr.BaseExpr |> MultiChannelExpr  

    /// Replace the argument of the unary operation.
    let replaceUnaryArg (op: IOp) (newArg: UExpr) =
        op.ReplaceArgs (Map [Arg.Only, newArg.BaseExprCh]) |> UExpr

    /// Replace the argument of the binary operation.
    let replaceBinaryArgs (op: IOp) (newX: UExpr) (newY: UExpr) =
        op.ReplaceArgs (Map [Arg.X, newX.BaseExprCh; Arg.Y, newY.BaseExprCh]) |> UExpr

    /// Broadcast information
    type BcInfo =
        /// axis is broadcasted to specific size
        | Bc of Size
        /// axis is not broadcasted and has specific size
        | NotBc of Size

        /// true if axis is broadcasted
        member this.IsBC =
            match this with
            | Bc _ -> true
            | NotBc _ -> false

        /// final size
        member this.Size = 
            match this with
            | Bc s 
            | NotBc s -> s

    /// Returns a list containing one element each axis of the expression.
    /// The element is true if the axis is broadcasted.
    let axesBroadcasted expr =
        match expr with
        | UExpr.DoBroadcast (bc, a) ->
            List.zip bc a.Shape
            |> List.map (fun (bcDim, aDim) -> 
                if aDim = Size.broadcastable && bcDim <> aDim then Bc bcDim
                else NotBc aDim)
        | _ ->
            expr.Shape
            |> List.map (fun aDim -> NotBc aDim)
