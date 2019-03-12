namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Expr.Ops


/// Contains evaluated values for expressions.
type ExprVals = ExprVals of Map<UExpr, ITensor> with

    /// All contained values.
    member this.Values =
        let (ExprVals values) = this
        values

    /// Get value of untyped expression.
    member this.Get (expr: UExpr) = 
        match this.Values |> Map.tryFind expr with
        | Some value -> value
        | None ->
            failwithf "The expression was not evaluated: %A" expr

    /// Get value of typed expression.
    member this.Get (expr: Expr<'T>) =
        this.Get expr.Untyped :?> Tensor<'T>



/// Evaluates a set of expressions and updates variables.
type EvalUpdateBundle = private {
    /// Expressions to evaluate.
    Exprs:       Set<UExpr>
    /// New values for variables.
    VarUpdates:  Map<VarName, UExpr>
    /// New values for data.
    DataUpdates: Map<OrdRef<ITensor>, UExpr>
} with

    static member internal exprToCh (exprIdx: int) =
        Ch.Custom (sprintf "Expr:%d" exprIdx)   

    static member internal varToCh (var: VarName) = 
        Ch.Custom (sprintf "VarUpdate:%s" var.Str)

    static member internal dataToCh (dataIdx: int) =
        Ch.Custom (sprintf "DataUpdate:%d" dataIdx)        

    static member empty = {
        Exprs = Set.empty
        VarUpdates = Map.empty
        DataUpdates = Map.empty
    }

    static member addUExpr (expr: UExpr) (bndl: EvalUpdateBundle) = 
        { bndl with
            Exprs = bndl.Exprs |> Set.add expr
        }

    static member addExpr (expr: Expr<'T>) (bndl: EvalUpdateBundle) = 
        EvalUpdateBundle.addUExpr expr.Untyped bndl

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
            Exprs = Set.union a.Exprs b.Exprs
            VarUpdates = Map.join a.VarUpdates b.VarUpdates
            DataUpdates = Map.join a.DataUpdates b.DataUpdates
        }

    static member mergeMany (bndls: EvalUpdateBundle seq) =
        bndls |> Seq.fold EvalUpdateBundle.merge EvalUpdateBundle.empty

    /// Executes an EvalUpdateBundle. 
    /// The variables are updated in-place the VarEnv,
    static member exec (varEnv: VarEnv) (bundle: EvalUpdateBundle) : ExprVals =
        // create channels for expression evaluation
        let exprList = bundle.Exprs |> Set.toList
        let exprChs =
            exprList
            |> Seq.indexed
            |> Seq.map (fun (idx, expr) -> EvalUpdateBundle.exprToCh idx, expr)
            |> Map.ofSeq

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
        let allChs = Map.joinMany [exprChs; varUpdateChs; dataUpdateChs]
        let evalBundle = MultiChannelExpr.bundle allChs

        // evaluate bundle
        let vals = MultiChannelExpr.eval varEnv evalBundle
        
        // extract expression values
        let exprVals =
            exprList
            |> Seq.indexed
            |> Seq.map (fun (idx, expr) -> expr, vals.[EvalUpdateBundle.exprToCh idx])
            |> Map.ofSeq

        // perform variable updates
        for var in Map.keys bundle.VarUpdates do
            varEnv.[var].CopyFrom vals.[EvalUpdateBundle.varToCh var]

        // perform data updates
        for dataIdx, (data, _) in Seq.indexed dataUpdateList do
            data.Value.CopyFrom vals.[EvalUpdateBundle.dataToCh dataIdx]

        ExprVals exprVals








