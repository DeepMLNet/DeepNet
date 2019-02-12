namespace SymTensor.Deriv 

open DeepNet.Utils
open SymTensor


module internal Consts =
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
    let zeros (dOp: Expr2) (arg: Expr2) =
        let shape = dOp.Shape.[0] :: arg.Shape
        Expr2.zerosOfType arg.DataType shape

    let zero (arg: Expr2) =
        (convTo arg.DataType 0) |> Expr2.scalar

    let one (arg: Expr2) =
        (convTo arg.DataType 1) |> Expr2.scalar

    let two (arg: Expr2) =
        (convTo arg.DataType 2) |> Expr2.scalar

    let ten (arg: Expr2) =
        (convTo arg.DataType 10) |> Expr2.scalar





