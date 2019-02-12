namespace SymTensor.Deriv

open DeepNet.Utils
open SymTensor
open SymTensor.Ops


[<OpExtender>]
type SetSubtensorDeriv(op: SetSubtensor) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = Deriv.Env.make op dOp 
            let dYExp = env.DOp.[SimpleRangeSpec.All :: op.Range]
            let zeros = Expr2.zerosOfType dYExp.DataType dYExp.Shape
            let dXExp = Expr2.setSubtensor env.DOp.[SimpleRangeSpec.All :: op.Range] zeros
            Deriv.binary dXExp dYExp


