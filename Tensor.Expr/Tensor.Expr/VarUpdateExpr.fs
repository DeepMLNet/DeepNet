namespace Tensor.Expr

open DeepNet.Utils
open Tensor.Expr.Ops


type VarUpdateExpr = {
    Values:      Map<Ch, Expr>
    Updates:     Map<BaseVar, Expr>
}



