namespace SymTensor.Deriv 

open DeepNet.Utils
open SymTensor
open SymTensor.Ops


/// Internal derivative utils.
module internal Deriv =

    /// Expands the second dimension of the Jacobian into the shape of this expression.
    let expand (dOpJac: Expr) (expr: Expr) = 
        let funElems = dOpJac.Shape.[0]
        dOpJac |> Expr.reshape (funElems :: expr.Shape)

    /// Flattens all but the first dimension of the Jacobian into one dimension.
    let collapse (dOp: Expr) =
        let funElems = dOp.Shape.[0]
        let wrtElems = dOp.Shape.[1..] |> ShapeSpec.nElem
        dOp |> Expr.reshape [funElems; wrtElems]

    /// Returns a zero derivative for the specified argument.
    let zeros (dOp: BaseExpr) (arg: BaseExpr) =
        let shape = dOp.Shape.[0] :: arg.Shape
        Expr.zerosOfType arg.DataType shape

    /// Zero of same type as arg.
    let zero (arg: BaseExpr) =
        (convTo arg.DataType 0) |> Expr.scalar

    /// One of same type as arg.
    let one (arg: BaseExpr) =
        (convTo arg.DataType 1) |> Expr.scalar

    /// Two of same type as arg.
    let two (arg: BaseExpr) =
        (convTo arg.DataType 2) |> Expr.scalar

    /// Ten of same type as arg.
    let ten (arg: BaseExpr) =
        (convTo arg.DataType 10) |> Expr.scalar

    let expr (dOp: BaseExpr) =
        dOp :?> Expr

    /// Unary derivative result.
    let unary x =
        Args.unary (x :> BaseExpr)

    let unaryExpr (dOp: BaseExpr) (f: Expr -> Expr) =
        dOp |> expr |> f |> unary

    /// Binary derivative result.
    let binary x y =
        Args.binary (x :> BaseExpr) (y: BaseExpr)

    /// N-ary derivative result.
    let nary xs = 
        xs 
        |> List.map (fun x -> x :> BaseExpr)
        |> Args.nary 

    /// Environment with derivative helper values.
    type Env = {
        Op: IOp
        DOp: Expr
    } with
        member this.Expr = Expr this.Op
        member this.FunElems =
            this.DOp.Shape.[0]
        member this.DOpJac =
            let wrtElems = this.DOp.Shape.[1..] |> ShapeSpec.nElem
            this.DOp |> Expr.reshape [this.FunElems; wrtElems]
        member this.X = this.Op.Args |> Args.unaryX :?> Expr
        member this.Y = this.Op.Args |> Args.binaryY :?> Expr
        member this.Xs =
            this.Op.Args
            |> Args.naryXs
            |> List.map (fun x -> x :?> Expr)
        member this.Zeros arg = zeros this.DOp arg 
        member this.Zero = zero this.DOp
        member this.One = one this.DOp
        member this.Two = two this.DOp
        member this.Ten = ten this.DOp
        
        static member make op (dOp: BaseExpr) = {
            Op = op
            DOp = dOp :?> Expr
        }




/// Internal derivative utils for multi-channel ops.
module internal MultiChannelDeriv =

    /// Environment with derivative helper values.
    type Env = {
        Op: IMultiChannelOp
        DOp: Map<string, Expr>
    } with
        member private this.DOpCh0 =
            this.DOp |> Map.toSeq |> Seq.head |> snd
        member this.Expr = MultiChannelExpr this.Op
        member this.FunElems = this.DOpCh0.Shape.[0]
        member this.X = this.Op.Args |> Args.unaryX :?> Expr
        member this.Y = this.Op.Args |> Args.binaryY :?> Expr
        member this.Xs =
            this.Op.Args
            |> Args.naryXs
            |> List.map (fun x -> x :?> Expr)
        member this.Zeros arg = Deriv.zeros this.DOpCh0 arg 
        
        static member make op (dOp: Map<string, BaseExpr>) = {
            Op = op
            DOp = dOp |> Map.map (fun _ be -> be :?> Expr)
        }

