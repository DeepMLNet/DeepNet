namespace SymTensor

open Basics
open ArrayNDNS


[<AutoOpen>]
module VarEnvTypes = 
    /// variable environment
    type VarEnvT = Map<IVarSpec, IArrayNDT>

module VarEnv = 
    /// add variable value to environment
    let add (var: Expr.ExprT<'T>) (value: ArrayNDT<'T>) (varEnv: VarEnvT) =
        let vs = Expr.extractVar var
        Map.add (vs :> IVarSpec) (value :> IArrayNDT) varEnv

    /// get variable value from environment
    let getVarSpecT (vs: VarSpecT<'T>) (varEnv: VarEnvT) : ArrayNDT<'T> =
        varEnv.[vs] :?> ArrayNDT<'T>

    /// get variable value from environment
    let get (var: Expr.ExprT<'T>) (varEnv: VarEnvT) : ArrayNDT<'T> =
        getVarSpecT (Expr.extractVar var) varEnv

    /// empty variable environment
    let (empty: VarEnvT) =
        Map.empty

    /// joins two variable environments
    let join (a: VarEnvT) (b: VarEnvT) =
        Map.join a b

    /// infers symbol sizes from the variable environment
    let inferSymSizes (varEnv: VarEnvT) =
        varEnv |> Map.fold 
            (fun env vSym vVal ->
                let symShape = VarSpec.shape vSym
                let valShape = ArrayND.shape vVal |> List.map (Fixed >> Base)
                SymSizeEnv.needEqualShape symShape valShape env)
            SymSizeEnv.empty 

    /// substitues the given symbol sizes into the variable environment
    let substSymSizes symSizes (varEnv: VarEnvT) : VarEnvT =
        varEnv |> Map.fold 
            (fun env vSym vVal ->
                Map.add (vSym.SubstSymSizes symSizes) vVal env)
            Map.empty

    /// gets the type names of the variable value arrays
    let valueTypeNames (varEnv: VarEnvT) =
        varEnv |> Map.map (fun _ vVal -> TypeName.ofObject vVal)


[<AutoOpen>]
module EvalEnvTypes =
    /// Information neccessary to evaluate an expression.
    type EvalEnvT = {
        VarEnv:             VarEnvT; 
    }


module EvalEnv = 
    open VarEnv

    /// empty EvalEnvT
    let empty = 
        {VarEnv = Map.empty;}

    /// Create an EvalEnvT using the specified VarEnvT and infers the size symbol values
    /// from the given expressions.
    let create varEnv = 
        {VarEnv = varEnv;}

    /// Enhances an existing EvalEnvT using the specified VarEnvT and infers the size symbol values
    /// from the given expressions.
    let enhance varEnv evalEnv =
        let joinedVarEnv = VarEnv.join evalEnv.VarEnv varEnv
        {VarEnv = joinedVarEnv;}

    /// get variable value from environment
    let getVarSpecT vs (evalEnv: EvalEnvT)  =
        VarEnv.getVarSpecT vs evalEnv.VarEnv

    /// get variable value from environment
    let get var (evalEnv: EvalEnvT) =
        VarEnv.get var evalEnv.VarEnv




module Func =

    /// a function that evaluates unified expressions in an EvalEnvT
    type UExprEvaluatorT = UExprT list -> EvalEnvT -> IArrayNDT list

    type private UExprGenT = SymSizeEnvT -> (UExprT * Set<IVarSpec>)

    let private uExprGen baseExpr symSizes =
        let expr = baseExpr |> Expr.substSymSizes symSizes 
        if not (Expr.canEvalAllSymSizes expr) then
            failwith "cannot evaluate expression because not all symbolic sizes could be resolved"
            
        let vars = Expr.extractVars expr |> Set.map (fun vs -> vs :> IVarSpec)
        UExpr.toUExpr expr, vars

    let private evalWrapper (evaluator: UExprEvaluatorT) (baseExprGens: UExprGenT list) (varEnv: VarEnvT) : IArrayNDT list =
        // infer size symbols from variables and substitute into expression and variables
        let symSizes = VarEnv.inferSymSizes varEnv
        let uexprs, vars = List.map (fun gen -> gen symSizes) baseExprGens |> List.unzip
        let varEnv = VarEnv.substSymSizes symSizes varEnv

        printfn "specified vars:%A" varEnv
        printfn "exprs to eval:%A" uexprs

        // check if evaluation is possible
        let vars = Set.unionMany vars
        let missingVars = Set.filter (fun v -> not (Map.containsKey v varEnv)) vars
        if not (Set.isEmpty missingVars) then
            failwithf "cannot evaluate expression because values for the variable(s) %A have not been specified"
                (missingVars |> Set.toList)

        // evaluate
        let evalEnv = EvalEnv.create varEnv
        evaluator uexprs evalEnv
                    

    /// makes a function that evaluates the given expression 
    let make factory (expr0: ExprT<'T0>)  =   
        let evalAll = evalWrapper factory [uExprGen expr0]        
        fun (varEnv: VarEnvT) ->
            let res = evalAll varEnv
            res.[0] :?> ArrayNDT<'T0>

    let make2 factory (expr0: ExprT<'T0>) (expr1: ExprT<'T1>) =    
        let evalAll = evalWrapper factory [uExprGen expr0; uExprGen expr1]        
        fun (varEnv: VarEnvT) ->
            let res = evalAll varEnv
            res.[0] :?> ArrayNDT<'T0>, res.[1] :?> ArrayNDT<'T1>

    let make3 factory (expr0: ExprT<'T0>) (expr1: ExprT<'T1>) (expr2: ExprT<'T2>) =    
        let evalAll = evalWrapper factory [uExprGen expr0; uExprGen expr1; uExprGen expr2]        
        fun (varEnv: VarEnvT) ->
            let res = evalAll varEnv
            res.[0] :?> ArrayNDT<'T0>, res.[1] :?> ArrayNDT<'T1>, res.[2] :?> ArrayNDT<'T2>

[<AutoOpen>]
module FuncTypes = 

    let arg (vs0: ExprT<'T0>) f =
        fun (val0: ArrayNDT<'T0>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> f

    let arg2 (vs0: ExprT<'T0>) (vs1: ExprT<'T1>) f =
        fun (val0: ArrayNDT<'T0>) (val1: ArrayNDT<'T1>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> f

    let arg3 (vs0: ExprT<'T0>) (vs1: ExprT<'T1>) (vs2: ExprT<'T2>) f =
        fun (val0: ArrayNDT<'T0>) (val1: ArrayNDT<'T1>) (val2: ArrayNDT<'T2>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> f

//    let inline (.|.) (varEnv: VarEnvT) (var: ExprT<'T>) =
//        fun (varValue: ArrayNDT<'T>) ->
//            varEnv |> VarEnv.add var varValue


//    let inline (.^.) func (var: ExprT<'T>) =
//        fun (varEnv: VarEnvT) (varValue: ArrayNDT<'T>) ->
//            func (varEnv |> VarEnv.add var varValue)
////
////    let inline (.||.) func (var: ExprT<'T>) =
////        fun varEnvBuildFunc (varValue: ArrayNDT<'T>) ->
////            func (varEnv |> VarEnv.add var varValue)
//
////    let inline (.^) func (varEnv: VarEnvT) =
////        func varEnv
////        fun (varEnv: VarEnvT) (firstVarValue: ArrayNDT<'T>) ->
////            func (varEnv |> VarEnv.add var firstVarValue)
//
//    let END : VarEnvT = 
//        VarEnv.empty
//
//    let addArg (var: ExprT<'T>) =
//        fun (varEnv: VarEnvT) ->
//            fun (varValue: ArrayNDT<'T>) -> varEnv |> VarEnv.add var varValue

//    let tst a b =
//        a .|. b


