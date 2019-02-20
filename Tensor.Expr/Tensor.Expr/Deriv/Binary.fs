namespace Tensor.Expr.DerivOps

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops


module private IfThenElseDeriv =
    let ifThenElseDeriv (dOp: Expr) (cond: Expr) =
        let dOpZeros = Expr.zerosLike dOp
        let dX = Expr.ifThenElse (Expr.padLeft cond) dOp dOpZeros 
        let dY = Expr.ifThenElse (Expr.padLeft cond) dOpZeros dOp 
        DerivTools.binary dX dY

open IfThenElseDeriv


[<OpExtender>]
type AddDeriv(op: Add) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dX = env.DOp 
            let dY = env.DOp 
            DerivTools.binary dX dY


[<OpExtender>]
type SubtractDeriv(op: Subtract) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dX = env.DOp 
            let dY = -env.DOp 
            DerivTools.binary dX dY


[<OpExtender>]
type MultiplyDeriv(op: Multiply) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dX = env.DOp * (Expr.padLeft env.Y)
            let dY = env.DOp * (Expr.padLeft env.X)
            DerivTools.binary dX dY


[<OpExtender>]
type DivideDeriv(op: Divide) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dX = env.DOp * (Expr.padLeft env.Y)
            let dY = env.DOp * (Expr.padLeft env.X)
            DerivTools.binary dX dY


[<OpExtender>]
type PowDeriv(op: Pow) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dX = env.DOp * Expr.padLeft (env.Y * env.X**(env.Y - env.One))
            let dY = env.DOp * Expr.padLeft (env.X**env.Y * log env.X)
            DerivTools.binary dX dY


[<OpExtender>]
type ModuloDeriv(op: Modulo) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            failwith "TODO: FIX"
            let env = DerivTools.Env.make op dOp 
            let dX = env.DOp 
            let dY = env.DOp * Expr.padLeft (-truncate (env.X / env.Y))
            DerivTools.binary dX dY


[<OpExtender>]
type MaxElemwiseDeriv(op: MaxElemwise) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            ifThenElseDeriv env.DOp (env.X >>>> env.Y) 
            

[<OpExtender>]
type MinElemwiseDeriv(op: MinElemwise) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            ifThenElseDeriv env.DOp (env.X <<<< env.Y) 
        

[<OpExtender>]
type IfThenElseDeriv(op: IfThenElse) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            ifThenElseDeriv env.DOp (Expr op.Cond)
        

[<OpExtender>]
type AndDeriv(op: And) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type OrDeriv(op: Or) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type XorDeriv(op: Xor) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type EqualDeriv(op: Equal) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type NotEqualDeriv(op: NotEqual) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type LessDeriv(op: Less) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type LessOrEqualDeriv(op: LessOrEqual) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type GreaterDeriv(op: Greater) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type GreaterOrEqualDeriv(op: GreaterOrEqual) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            DerivTools.binary (env.Zeros env.X) (env.Zeros env.Y)


[<OpExtender>]
type DotDeriv(op: Dot) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 

            /// Helper function that computes derivative of y = m .* x wrt x.
            let mxWrtX (m: Expr) x y dy =
                let xShp, yShp, dyShp = Expr.shape x, Expr.shape y, Expr.shape dy
                let nd = ShapeSpec.nDim xShp
                let batchShp = xShp.[0..nd-3]
                let batchElems = ShapeSpec.nElem batchShp
                let xSmplShp, ySmplShp = xShp.[nd-2..], yShp.[nd-2..]
                let funElems = dyShp.[0]
                let dyMat = 
                    dy 
                    |> Expr.swapDim 0 1 
                    |> Expr.reshape (batchShp @ [ySmplShp.[0]; ySmplShp.[1] * funElems])
                let dxMat = m.T .* dyMat
                let dx = 
                    dxMat 
                    |> Expr.reshape [batchElems * xSmplShp.[0] * xSmplShp.[1]; funElems] 
                    |> Expr.swapDim 1 0
                dx

            // Jacobian wrt Y.
            let dYJac = mxWrtX env.X env.Y env.Expr env.DOpJac

            // Calculate Jacobian wrt X by transposing expression and resulting Jacobian.
            let xShp = Expr.shape env.X
            let nd = ShapeSpec.nDim xShp
            let batchShp = xShp.[0..nd-3]
            let egT = env.DOp.T |> DerivTools.collapse
            let dXT = mxWrtX (env.Y.T) (env.X.T) (env.Expr.T) egT
            let dXJac = 
                dXT 
                |> Expr.reshape ([env.FunElems] @ batchShp @ [xShp.[nd-1]; xShp.[nd-2]]) 
                |> Expr.transpose 
                |> DerivTools.collapse

            // Expand jacobians into derivative shape.
            DerivTools.binary (DerivTools.expand dXJac env.X) (DerivTools.expand dYJac env.Y)


[<OpExtender>]
type TensorProductDeriv(op: TensorProduct) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            //let env = DerivTools.Env.make op dOp 
            //let dX = 
            //let dY = 
            //DerivTools.binary dX dY
            failwith "TODO: not implemented"


[<OpExtender>]
type SetSubtensorDeriv(op: SetSubtensor) =
    interface IDerivableOp with      
        member this.Deriv dOp =
            let env = DerivTools.Env.make op dOp 
            let dYExp = env.DOp.[SimpleRangeSpec.All :: op.Range]
            let zeros = Expr.zerosOfType dYExp.DataType dYExp.Shape
            let dXExp = Expr.setSubtensor env.DOp.[SimpleRangeSpec.All :: op.Range] zeros
            DerivTools.binary dXExp dYExp


