namespace SymTensor.Deriv

open DeepNet.Utils
open SymTensor
open SymTensor.Ops


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

