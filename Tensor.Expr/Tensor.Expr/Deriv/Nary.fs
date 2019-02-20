namespace Tensor.Expr.Deriv

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops


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
                    Expr.elements dXShp dXElemExpr dXArgs)
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
                        let ipd = op.Interpolator |> Interpolator.getDerivative d 
                        env.DOp * Expr.padLeft (Expr.interpolate ipd env.Xs))
                | InterpolationMode.ToLeft -> 
                    env.Xs |> List.map (fun x -> env.Zeros x)
            DerivTools.nary dXs



