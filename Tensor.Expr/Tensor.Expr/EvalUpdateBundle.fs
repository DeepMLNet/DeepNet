namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Expr.Ops


/// Evaluates a set of expressions and updates variables.
type EvalUpdateBundle = {
    /// Expressions to evaluate.
    Exprs:       Map<Ch, Expr>
    /// New values for variables.
    Updates:     Map<Var, Expr>
} 


/// Evaluate a set of expressions and update variables.
module EvalUpdateBundle =

    let internal varToCh (var: Var) =
        let (VarName name) = var.Name
        let chName = sprintf "VarUpdate:%s" name
        Ch.Custom chName

    /// Executes an EvalUpdateBundle. 
    /// The variables are updated in-place the VarEnv,
    let exec (varEnv: VarEnv) (bundle: EvalUpdateBundle) : Map<Ch, ITensor> =
        // create channels for variable updates
        let updateChs = 
            bundle.Updates 
            |> Map.mapKeyValue (fun var expr -> varToCh var, expr)

        // bundle all expressions
        let allChs = Map.join bundle.Exprs updateChs
        let evalBundle = MultiChannelExpr.bundle allChs

        // evaluate bundle
        let vals = MultiChannelExpr.eval varEnv evalBundle
        
        // perform variable updates
        for var in Map.keys bundle.Updates do
            varEnv.[var.Name].CopyFrom vals.[varToCh var]

        // extract expression values
        bundle.Exprs |> Map.map (fun ch _ -> vals.[ch])







