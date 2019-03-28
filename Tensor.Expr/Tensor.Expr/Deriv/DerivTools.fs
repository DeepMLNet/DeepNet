namespace Tensor.Expr.DerivOps

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops


/// Internal derivative utils.
module internal DerivTools =

    /// Expands the second dimension of the Jacobian into the shape of this expression.
    let inline expand (dOpJac: UExpr) (expr: UExpr) = 
        let funElems = dOpJac.Shape.[0]
        dOpJac |> UExpr.reshape (funElems :: expr.Shape)

    /// Flattens all but the first dimension of the Jacobian into one dimension.
    let inline collapse (dOp: UExpr) =
        let funElems = dOp.Shape.[0]
        let wrtElems = dOp.Shape.[1..] |> Shape.nElem
        dOp |> UExpr.reshape [funElems; wrtElems]

    /// Returns a zero derivative for the specified argument.
    let inline zeros (dOp: UExpr) (arg: UExpr) =
        let shape = dOp.Shape.[0] :: arg.Shape
        UExpr.zeros arg.DataType arg.Dev shape

    /// Zero of same type as arg.
    let inline zero (arg: UExpr) =
        UExpr.scalar arg.Dev (convTo arg.DataType 0)

    /// One of same type as arg.
    let inline one (arg: UExpr) =
        UExpr.scalar arg.Dev (convTo arg.DataType 1) 

    /// Two of same type as arg.
    let inline two (arg: UExpr) =
        UExpr.scalar arg.Dev (convTo arg.DataType 2)

    /// Ten of same type as arg.
    let inline ten (arg: UExpr) =
        UExpr.scalar arg.Dev (convTo arg.DataType 10)

    /// Unary derivative result.
    let inline unary x = ArgValue.unary x
    
    /// Binary derivative result.
    let inline binary x y = ArgValue.binary x y

    /// N-ary derivative result.
    let inline nary xs = ArgValue.nary xs

    /// N-ary derivative with option values result.
    let inline naryOpt xs = ArgValue.naryOpt xs

    /// Environment with derivative helper values.
    type Env = {
        Op: IOp
        DOp: UExpr
    } with
        member inline this.Expr = UExpr this.Op
        member inline this.FunElems = this.DOp.Shape.[0]
        member inline this.DOpJac =
            let wrtElems = this.DOp.Shape.[1..] |> Shape.nElem
            this.DOp |> UExpr.reshape [this.FunElems; wrtElems]
        member inline this.Only = this.Op.Args |> Args.unaryX |> UExpr
        member inline this.X = this.Op.Args |> Args.binaryX |> UExpr
        member inline this.Y = this.Op.Args |> Args.binaryY |> UExpr
        member inline this.Xs = this.Op.Args |> Args.naryXs |> List.map UExpr
        member inline this.Zeros arg = zeros this.DOp arg 
        member inline this.Zero = zero this.DOp
        member inline this.One = one this.DOp
        member inline this.Two = two this.DOp
        member inline this.Ten = ten this.DOp
        
        static member inline make op (dOp: Map<Ch, UExpr>) = {
            Op = op
            DOp = dOp.[Ch.Default]
        }




/// Internal derivative utils for multi-channel ops.
module internal MultiChannelDerivTools =

    /// Environment with derivative helper values.
    type Env = {
        Op: IOp
        DOp: Map<Ch, UExpr>
    } with
        member private this.DOpCh0 =
            this.DOp |> Map.toSeq |> Seq.head |> snd
        member inline this.Expr = MultiChannelExpr this.Op
        member inline this.FunElems = this.DOpCh0.Shape.[0]
        member inline this.Only = this.Op.Args |> Args.unaryX |> UExpr
        member inline this.X = this.Op.Args |> Args.binaryX |> UExpr
        member inline this.Y = this.Op.Args |> Args.binaryY |> UExpr
        member inline this.Xs = this.Op.Args |> Args.naryXs |> List.map UExpr
        member inline this.Zeros arg = DerivTools.zeros this.DOpCh0 arg 
        
        static member inline make op (dOp: Map<Ch, UExpr>) = {
            Op = op
            DOp = dOp 
        }

