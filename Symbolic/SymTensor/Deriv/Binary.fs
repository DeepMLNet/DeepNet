namespace SymTensor.Deriv

open DeepNet.Utils
open SymTensor
open SymTensor.Ops


[<OpExtender>]
type AddDeriv(op: Add) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let dX = env.DOp 
            let dY = env.DOp 
            Deriv.binary dX dY


[<OpExtender>]
type SubtractDeriv(op: Subtract) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let dX = env.DOp 
            let dY = -env.DOp 
            Deriv.binary dX dY


[<OpExtender>]
type MultiplyDeriv(op: Multiply) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let dX = env.DOp * (Expr2.padLeft env.Y)
            let dY = env.DOp * (Expr2.padLeft env.X)
            Deriv.binary dX dY


[<OpExtender>]
type DivideDeriv(op: Divide) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let dX = env.DOp * (Expr2.padLeft env.Y)
            let dY = env.DOp * (Expr2.padLeft env.X)
            Deriv.binary dX dY


[<OpExtender>]
type ModuloDeriv(op: Modulo) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            failwith "TODO: FIX"
            let env = Deriv.Env.make op dOp 
            let dX = env.DOp 
            let dY = env.DOp * Expr2.padLeft (-truncate (env.X / env.Y))
            Deriv.binary dX dY


[<OpExtender>]
type PowerDeriv(op: Pow) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let dX = env.DOp * Expr2.padLeft (env.Y * env.X**(env.Y - env.One))
            let dY = env.DOp * Expr2.padLeft (env.X**env.Y * log env.X)
            Deriv.binary dX dY


[<OpExtender>]
type SetSubtensorDeriv(op: SetSubtensor) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let dYExp = env.DOp.[SimpleRangeSpec.All :: op.Range]
            let zeros = Expr2.zerosOfType dYExp.DataType dYExp.Shape
            let dXExp = Expr2.setSubtensor env.DOp.[SimpleRangeSpec.All :: op.Range] zeros
            Deriv.binary dXExp dYExp


