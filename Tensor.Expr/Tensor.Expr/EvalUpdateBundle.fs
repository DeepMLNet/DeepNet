namespace Tensor.Expr

open DeepNet.Utils
open Tensor
open Tensor.Expr.Base
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
    _Exprs:       Set<UExpr>
    /// New values for variables.
    _VarUpdates:  Map<VarName, UExpr>
    /// New values for data.
    _DataUpdates: Map<OrdRef<ITensor>, UExpr>
} with

    member this.Exprs = this._Exprs
    member this.VarUpdates = this._VarUpdates
    member this.DataUpdates = this._DataUpdates
    
    static member internal exprToCh (exprIdx: int) =
        Ch.Custom (sprintf "Expr:%d" exprIdx)   

    static member internal varToCh (var: VarName) = 
        Ch.Custom (sprintf "VarUpdate:%s" var.Str)

    static member internal dataToCh (dataIdx: int) =
        Ch.Custom (sprintf "DataUpdate:%d" dataIdx)        

    static member empty = {
        _Exprs = Set.empty
        _VarUpdates = Map.empty
        _DataUpdates = Map.empty
    }

    static member addUExpr (expr: UExpr) (bndl: EvalUpdateBundle) = 
        { bndl with
            _Exprs = bndl._Exprs |> Set.add expr
        }

    static member addExpr (expr: Expr<'T>) (bndl: EvalUpdateBundle) = 
        EvalUpdateBundle.addUExpr expr.Untyped bndl

    static member addVarName (varName: VarName) (expr: UExpr) (bndl: EvalUpdateBundle) = 
        { bndl with
            _VarUpdates = bndl._VarUpdates |> Map.add varName expr
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
        EvalUpdateBundle.addVarName var.Name expr bndl

    static member addDataRef (dataRef: OrdRef<ITensor>) (expr: UExpr) (bndl: EvalUpdateBundle) = 
        let data = dataRef.Value
        if data.DataType <> expr.DataType then
            failwithf "Tensor data type %A does not match expression data type %A."
                data.DataType expr.DataType
        if data.Dev <> expr.Dev then
            failwithf "Tensor device %A does not match expression device %A."
                data.Dev expr.Dev
        match Shape.tryEval expr.Shape with
        | Some exprShp when exprShp <> data.Shape ->
            failwithf "Tensor shape %A does not match expression shape %A."
                data.Shape exprShp
        | Some exprShp -> ()
        | None ->
            failwithf "Cannot evaluate expression shape %A." expr.Shape
        { bndl with
            _DataUpdates = bndl._DataUpdates |> Map.add dataRef expr
        }       

    static member addData (data: ITensor) (expr: UExpr) (bndl: EvalUpdateBundle) = 
        EvalUpdateBundle.addDataRef (OrdRef data) expr bndl

    static member make (exprs: UExpr seq) (varUpdates: Map<VarName, UExpr>) (dataUpdates: Map<OrdRef<ITensor>, UExpr>) =
        let bndl = EvalUpdateBundle.empty
        let bndl = (bndl, exprs) ||> Seq.fold (fun bndl expr -> bndl |> EvalUpdateBundle.addUExpr expr)
        let bndl = (bndl, varUpdates) ||> Map.fold (fun bndl varName expr -> bndl |> EvalUpdateBundle.addVarName varName expr)
        let bndl = (bndl, dataUpdates) ||> Map.fold (fun bndl data expr -> bndl |> EvalUpdateBundle.addDataRef data expr)
        bndl

    static member merge (a: EvalUpdateBundle) (b: EvalUpdateBundle) =
        {
            _Exprs = Set.union a._Exprs b._Exprs
            _VarUpdates = Map.join a._VarUpdates b._VarUpdates
            _DataUpdates = Map.join a._DataUpdates b._DataUpdates
        }

    static member mergeMany (bndls: EvalUpdateBundle seq) =
        bndls |> Seq.fold EvalUpdateBundle.merge EvalUpdateBundle.empty

    /// Executes an EvalUpdateBundle. 
    /// The variables are updated in-place the VarEnv,
    static member exec (varEnv: VarEnv) (bundle: EvalUpdateBundle) : ExprVals =
        // create channels for expression evaluation
        let exprList = bundle._Exprs |> Set.toList
        let exprChs =
            exprList
            |> Seq.indexed
            |> Seq.map (fun (idx, expr) -> EvalUpdateBundle.exprToCh idx, expr)
            |> Map.ofSeq

        // create channels for variable updates
        let varUpdateChs = 
            bundle._VarUpdates 
            |> Map.mapKeyValue (fun var expr -> EvalUpdateBundle.varToCh var, expr)

        // create channel for data updates
        let dataUpdateList = bundle._DataUpdates |> Map.toList
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
        for var in Map.keys bundle._VarUpdates do
            varEnv.[var].CopyFrom vals.[EvalUpdateBundle.varToCh var]

        // perform data updates
        for dataIdx, (data, _) in Seq.indexed dataUpdateList do
            data.Value.CopyFrom vals.[EvalUpdateBundle.dataToCh dataIdx]

        ExprVals exprVals








