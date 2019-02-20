namespace Tensor.Expr.DerivOps

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops


[<OpExtender>]
type ScalarDeriv(op: Scalar) =
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty


[<OpExtender>]
type SizeValueDeriv(op: SizeValue) =
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty


[<OpExtender>]
type IdentityDeriv(op: Identity) =
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty


[<OpExtender>]
type ArangeDeriv(op: Arange) =
    interface IDerivableOp with
        member this.Deriv dOp = Map.empty

