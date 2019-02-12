namespace SymTensor.Deriv 

open DeepNet.Utils
open SymTensor
open SymTensor.Ops


module internal Deriv =
    ///// Expands the second dimension of the Jacobian into the shape of this expression.
    //let expand (dOp: Expr2) (expr: IOp2) = 
    //    let funElems = dOp.Shape.[0]
    //    dOp |> Expr2.reshape (funElems :: expr.Shape)

    ///// Flattens all but the first dimension of the Jacobian into one dimension.
    //let collapse (g: Expr2) =
    //    let funElems = g.Shape.[0]
    //    let wrtElems = g.Shape.[1..] |> ShapeSpec.nElem
    //    g |> Expr2.reshape [funElems; wrtElems]

    /// Returns a zero derivative for the specified argument.
    let zeros (dOp: BaseExpr) (arg: BaseExpr) =
        let shape = dOp.Shape.[0] :: arg.Shape
        Expr2.zerosOfType arg.DataType shape

    let zero (arg: BaseExpr) =
        (convTo arg.DataType 0) |> Expr2.scalar

    let one (arg: BaseExpr) =
        (convTo arg.DataType 1) |> Expr2.scalar

    let two (arg: BaseExpr) =
        (convTo arg.DataType 2) |> Expr2.scalar

    let ten (arg: BaseExpr) =
        (convTo arg.DataType 10) |> Expr2.scalar

    type Env = {
        Op: IOp2
        DOp: Expr2
    } with
        member this.Expr = Expr2 this.Op
        member this.FunElems =
            this.DOp.Shape.[0]
        member this.DOpJac =
            let wrtElems = this.DOp.Shape.[1..] |> ShapeSpec.nElem
            this.DOp |> Expr2.reshape [this.FunElems; wrtElems]
        member this.X = this.Op.Args |> Args.unaryX :?> Expr2
        member this.Y = this.Op.Args |> Args.binaryY :?> Expr2
        member this.Xs =
            this.Op.Args
            |> Args.naryXs
            |> List.map (fun x -> x :?> Expr2)
        member this.Zeros arg = zeros this.DOp arg 
        member this.Zero = zero this.DOp
        member this.One = one this.DOp
        member this.Two = two this.DOp
        member this.Ten = ten this.DOp
        
        static member make op (dOp: BaseExpr) = {
            Op = op
            DOp = dOp :?> Expr2
        }
    

    let expr (dOp: BaseExpr) =
        dOp :?> Expr2

    let unary x =
        Args.unary (x :> BaseExpr)

    let unaryExpr (dOp: BaseExpr) (f: Expr2 -> Expr2) =
        dOp |> expr |> f |> unary

    let binary x y =
        Args.binary (x :> BaseExpr) (y: BaseExpr)

    let nary xs = 
        xs 
        |> List.map (fun x -> x :> BaseExpr)
        |> Args.nary 



