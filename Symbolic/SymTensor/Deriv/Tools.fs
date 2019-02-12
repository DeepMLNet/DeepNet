namespace SymTensor.Deriv 

open DeepNet.Utils
open SymTensor
open SymTensor.Ops


module internal Deriv =
    ///// Expands the second dimension of the Jacobian into the shape of this expression.
    //let expand (dOp: Expr2) (expr: IOp) = 
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
        Expr.zerosOfType arg.DataType shape

    let zero (arg: BaseExpr) =
        (convTo arg.DataType 0) |> Expr.scalar

    let one (arg: BaseExpr) =
        (convTo arg.DataType 1) |> Expr.scalar

    let two (arg: BaseExpr) =
        (convTo arg.DataType 2) |> Expr.scalar

    let ten (arg: BaseExpr) =
        (convTo arg.DataType 10) |> Expr.scalar

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
    

    let expr (dOp: BaseExpr) =
        dOp :?> Expr

    let unary x =
        Args.unary (x :> BaseExpr)

    let unaryExpr (dOp: BaseExpr) (f: Expr -> Expr) =
        dOp |> expr |> f |> unary

    let binary x y =
        Args.binary (x :> BaseExpr) (y: BaseExpr)

    let nary xs = 
        xs 
        |> List.map (fun x -> x :> BaseExpr)
        |> Args.nary 



