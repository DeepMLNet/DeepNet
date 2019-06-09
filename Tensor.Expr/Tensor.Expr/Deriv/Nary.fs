namespace Tensor.Expr.DerivOps

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Base
open Tensor.Expr.Ops



[<OpExtender>]
type BundleDeriv(op: Bundle) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            dOp |> Map.mapKeyValue (fun ch dCh -> Bundle.chToArg ch, dCh)



[<OpExtender>]
type ElementsDeriv(op: Elements) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dXsElemExprs = 
                Elem.Deriv.buildDerivElemExpr op.ElemExpr op.Shape op.Xs.Length
            let dXs =
                List.zip env.Xs dXsElemExprs
                |> List.map (fun (x, dXElemExpr) -> 
                    let dXShp = env.FunElems :: x.Shape
                    let dXArgs = env.Xs @ [env.DOp]
                    UExpr.elements dXShp dXElemExpr dXArgs)
            DerivTools.nary dXs



[<OpExtender>]
type InterpolateDeriv(op: Interpolate) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dXs =
                match op.Interpolator.Mode with
                | InterpolationMode.Linear ->
                    List.indexed env.Xs
                    |> List.map (fun (d, x) ->
                        let ipd = op.Interpolator |> Interpolator.getDeriv d 
                        env.DOp * UExpr.padLeft (UExpr.interpolate ipd env.Xs))
                | InterpolationMode.ToLeft -> 
                    env.Xs |> List.map (fun x -> env.Zeros x)
            DerivTools.nary dXs



