namespace Tensor.Expr.Opt

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Base
open Tensor.Expr.Ops
open Tensor.Expr.Opt.Tools



module BuildTensorRecognizer =
    let (|BuildTensor|_|) (expr: UExpr) =
        match expr.Op with
        | :? BuildTensor as this -> Some (this.Shape, this.Ranges, this.Xs)
        | _ -> None

open BuildTensorRecognizer


/// Combines multiple SetSubtensor invocations into one SetSubtensor operation.
[<Optimizer>]
type BuildTensorOptimizer() =
    interface IOptimizer with
        member __.Order = 20
        member __.Optimize opt expr = apply opt expr __.Optimize

    member __.Optimize opt expr =
        match expr with

        // tranform SetSubtensor(Zero, X) into BuildTensor(X)
        | UExpr.SetSubtensor (SimpleRanges.Static as rngs, UExpr.ZeroValued, part) ->
            UExpr {
                BuildTensor.Shape = expr.Shape 
                BuildTensor.Ranges = [SimpleRanges.toBaseRanges expr.Shape rngs]
                BuildTensor.Xs = [part.BaseExprCh]
            } 

        // combine Add(BuildTensor, BuildTensor) into BuildTensor if ranges are not overlapping
        | UExpr.Add (BuildTensor (aShp, aRngs, aParts), BuildTensor (bShp, bRngs, bParts)) when 
                aShp=bShp && not (BaseRanges.areOverlapping (aRngs @ bRngs)) ->
            UExpr {
                BuildTensor.Shape = aShp 
                BuildTensor.Ranges = aRngs @ bRngs
                BuildTensor.Xs = aParts @ bParts
            } 
          
        | _ -> expr

