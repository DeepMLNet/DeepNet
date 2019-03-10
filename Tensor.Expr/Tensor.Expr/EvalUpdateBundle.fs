namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Expr.Ops


/// Evaluates a set of expressions and updates variables.
type EvalUpdateBundle = private {
    /// Expressions to evaluate.
    Exprs:       Map<Ch, UExpr>
    /// New values for variables.
    VarUpdates:  Map<VarName, UExpr>
    /// New values for data.
    DataUpdates: Map<OrdRef<ITensor>, UExpr>
} with

    static member internal varToCh (var: VarName) = 
        Ch.Custom (sprintf "VarUpdate:%s" var.Str)

    static member internal dataToCh (dataIdx: int) =
        Ch.Custom (sprintf "DataUpdate:%d" dataIdx)        

    static member empty = {
        Exprs = Map.empty
        VarUpdates = Map.empty
        DataUpdates = Map.empty
    }

    static member addExpr (ch: Ch) (expr: UExpr) (bndl: EvalUpdateBundle) = 
        { bndl with
            Exprs = bndl.Exprs |> Map.add ch expr
        }

    static member addVar (var: Var) (expr: UExpr) (bndl: EvalUpdateBundle) = 
        if var.DataType <> expr.DataType then
            failwithf "Variable %A does not match expression data type %A."
                var expr.DataType
        if var.Dev <> expr.Dev then
            failwithf "Variable %A does not match expression device %A."
                var expr.Dev
        if var.Shape <> expr.Shape then
            failwithf "Variable %A does not match expression shape %A."
                var expr.Shape
        { bndl with
            VarUpdates = bndl.VarUpdates |> Map.add var.Name expr
        }        

    static member addData (data: ITensor) (expr: UExpr) (bndl: EvalUpdateBundle) = 
        if data.DataType <> expr.DataType then
            failwithf "Tensor data type %A does not match expression data type %A."
                data.DataType expr.DataType
        if data.Dev <> expr.Dev then
            failwithf "Tensor device %A does not match expression device %A."
                data.Dev expr.Dev
        match ShapeSpec.tryEval expr.Shape with
        | Some exprShp when exprShp <> data.Shape ->
            failwithf "Tensor shape %A does not match expression shape %A."
                data.Shape exprShp
        | Some exprShp -> ()
        | None ->
            failwithf "Cannot evaluate expression shape %A." expr.Shape
        { bndl with
            DataUpdates = bndl.DataUpdates |> Map.add (OrdRef data) expr
        }       

    static member merge (a: EvalUpdateBundle) (b: EvalUpdateBundle) =
        {
            Exprs = Map.join a.Exprs b.Exprs
            VarUpdates = Map.join a.VarUpdates b.VarUpdates
            DataUpdates = Map.join a.DataUpdates b.DataUpdates
        }

    static member mergeMany (bndls: EvalUpdateBundle seq) =
        bndls |> Seq.fold EvalUpdateBundle.merge EvalUpdateBundle.empty

    /// Executes an EvalUpdateBundle. 
    /// The variables are updated in-place the VarEnv,
    static member exec (varEnv: VarEnv) (bundle: EvalUpdateBundle) : Map<Ch, ITensor> =
        // create channels for variable updates
        let varUpdateChs = 
            bundle.VarUpdates 
            |> Map.mapKeyValue (fun var expr -> EvalUpdateBundle.varToCh var, expr)

        // create channel for data updates
        let dataUpdateList = bundle.DataUpdates |> Map.toList
        let dataUpdateChs =
            dataUpdateList
            |> Seq.indexed
            |> Seq.map (fun (idx, (_, expr)) -> EvalUpdateBundle.dataToCh idx, expr)
            |> Map.ofSeq

        // bundle all expressions
        let allChs = Map.joinMany [bundle.Exprs; varUpdateChs; dataUpdateChs]
        let evalBundle = MultiChannelExpr.bundle allChs

        // evaluate bundle
        let vals = MultiChannelExpr.eval varEnv evalBundle
        
        // perform variable updates
        for var in Map.keys bundle.VarUpdates do
            varEnv.[var].CopyFrom vals.[EvalUpdateBundle.varToCh var]

        // perform data updates
        for dataIdx, (data, _) in Seq.indexed dataUpdateList do
            data.Value.CopyFrom vals.[EvalUpdateBundle.dataToCh dataIdx]

        // extract expression values
        bundle.Exprs |> Map.map (fun ch _ -> vals.[ch])







