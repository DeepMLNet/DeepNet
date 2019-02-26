namespace Tensor.Expr.DerivOps

open DeepNet.Utils
open Tensor.Expr
open Tensor.Expr.Ops


/// Internal derivative utils.
module internal DerivTools =

    /// Expands the second dimension of the Jacobian into the shape of this expression.
    let inline expand (dOpJac: Expr) (expr: Expr) = 
        let funElems = dOpJac.Shape.[0]
        dOpJac |> Expr.reshape (funElems :: expr.Shape)

    /// Flattens all but the first dimension of the Jacobian into one dimension.
    let inline collapse (dOp: Expr) =
        let funElems = dOp.Shape.[0]
        let wrtElems = dOp.Shape.[1..] |> ShapeSpec.nElem
        dOp |> Expr.reshape [funElems; wrtElems]

    /// Returns a zero derivative for the specified argument.
    let inline zeros (dOp: Expr) (arg: Expr) =
        let shape = dOp.Shape.[0] :: arg.Shape
        Expr.zerosOfType arg.DataType shape

    /// Zero of same type as arg.
    let inline zero (arg: Expr) =
        (convTo arg.DataType 0) |> Expr.scalar

    /// One of same type as arg.
    let inline one (arg: Expr) =
        (convTo arg.DataType 1) |> Expr.scalar

    /// Two of same type as arg.
    let inline two (arg: Expr) =
        (convTo arg.DataType 2) |> Expr.scalar

    /// Ten of same type as arg.
    let inline ten (arg: Expr) =
        (convTo arg.DataType 10) |> Expr.scalar

    /// Unary derivative result.
    let inline unary x = ArgValue.unary x
    
    /// Binary derivative result.
    let inline binary x y = ArgValue.binary x y

    /// N-ary derivative result.
    let inline nary xs = ArgValue.nary xs

    /// Environment with derivative helper values.
    type Env = {
        Op: IOp
        DOp: Expr
    } with
        member inline this.Expr = Expr this.Op
        member inline this.FunElems = this.DOp.Shape.[0]
        member inline this.DOpJac =
            let wrtElems = this.DOp.Shape.[1..] |> ShapeSpec.nElem
            this.DOp |> Expr.reshape [this.FunElems; wrtElems]
        member inline this.Only = this.Op.Args |> Args.unaryX |> Expr
        member inline this.X = this.Op.Args |> Args.binaryX |> Expr
        member inline this.Y = this.Op.Args |> Args.binaryY |> Expr
        member inline this.Xs = this.Op.Args |> Args.naryXs |> List.map Expr
        member inline this.Zeros arg = zeros this.DOp arg 
        member inline this.Zero = zero this.DOp
        member inline this.One = one this.DOp
        member inline this.Two = two this.DOp
        member inline this.Ten = ten this.DOp
        
        static member inline make op (dOp: Map<Ch, Expr>) = {
            Op = op
            DOp = dOp.[Ch.Default]
        }




/// Internal derivative utils for multi-channel ops.
module internal MultiChannelDerivTools =

    /// Environment with derivative helper values.
    type Env = {
        Op: IOp
        DOp: Map<Ch, Expr>
    } with
        member private this.DOpCh0 =
            this.DOp |> Map.toSeq |> Seq.head |> snd
        member inline this.Expr = MultiChannelExpr this.Op
        member inline this.FunElems = this.DOpCh0.Shape.[0]
        member inline this.Only = this.Op.Args |> Args.unaryX |> Expr
        member inline this.X = this.Op.Args |> Args.binaryX |> Expr
        member inline this.Y = this.Op.Args |> Args.binaryY |> Expr
        member inline this.Xs = this.Op.Args |> Args.naryXs |> List.map Expr
        member inline this.Zeros arg = DerivTools.zeros this.DOpCh0 arg 
        
        static member inline make op (dOp: Map<Ch, Expr>) = {
            Op = op
            DOp = dOp 
        }

