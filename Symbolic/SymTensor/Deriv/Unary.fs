namespace SymTensor.Deriv

open SymTensor
open SymTensor.Ops


[<OpExtender>]
type UnaryPlusDeriv(op: UnaryPlus) =
    interface IDerivableOp with
        member this.Deriv dOp = dOp |> Args.unary


[<OpExtender>]
type NegateDeriv(op: Negate) =
    interface IDerivableOp with
        member this.Deriv dOp = -dOp |> Args.unary


[<OpExtender>]
type AbsDeriv(op: Abs) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            dOp * Expr2.padLeft (Expr2.signt op.X) |> Args.unary


[<OpExtender>]
type SignTDeriv(op: Abs) =
    interface IDerivableOp with
        member this.Deriv dOp = 
            Deriv.zeros dOp this.X |> Args.unary



