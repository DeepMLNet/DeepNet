namespace SymTensor

open Basics
open ArrayNDNS


[<AutoOpen>]
module VarEnvTypes = 
    /// variable environment
    type VarEnvT = Map<IVarSpec, IArrayNDT>

    /// specification of variable storage locations
    type VarLocsT = Map<IVarSpec, ArrayLocT>


module VarEnv = 
    /// add variable value to environment
    let add (var: Expr.ExprT<'T>) (value: ArrayNDT<'T>) (varEnv: VarEnvT) : VarEnvT =
        let vs = Expr.extractVar var
        Map.add (vs :> IVarSpec) (value :> IArrayNDT) varEnv

    let addIVarSpec (vs: IVarSpec) (value: IArrayNDT) (varEnv: VarEnvT) : VarEnvT =
        Map.add vs value varEnv

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

    /// gets the locations of the variable value arrays
    let valueLocations (varEnv: VarEnvT) =
        varEnv |> Map.map (fun _ vVal -> ArrayND.location vVal)


[<AutoOpen>]
module EnvTypes =

    /// Information neccessary to evaluate an expression.
    /// Currently this just holds the variable values, but may contain further information in the future.
    type EvalEnvT = {
        VarEnv:             VarEnvT; 
    }

    /// Information necessary to compile an expression.
    /// Currently this contains the variable locations.
    type CompileEnvT = {
        SymSizes:           SymSizeEnvT;
        VarLocs:            VarLocsT;
        ResultLoc:          ArrayLocT;
        CanDelay:           bool;
    }

    /// a function that evaluates into a numeric value given variable values
    type CompiledUExprT = EvalEnvT -> IArrayNDT list

    /// a function that compiles a unified expression into a function
    type UExprCompilerT = CompileEnvT -> UExprT list -> CompiledUExprT

    type CompileSpecT = UExprCompilerT * CompileEnvT


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


module CompileEnv =

    /// empty compile environment
    let empty =
        {VarLocs=Map.empty; ResultLoc=LocHost; SymSizes=SymSizeEnv.empty; CanDelay=true;}


     

module Func =

    type private UExprGenT = SymSizeEnvT -> (UExprT * Set<IVarSpec> * bool) 

    let private uExprGen baseExpr symSizes =
        let expr = baseExpr |> Expr.substSymSizes symSizes 
        let vars = Expr.extractVars expr |> Set.map (fun vs -> vs :> IVarSpec)
        UExpr.toUExpr expr, vars, Expr.canEvalAllSymSizes expr

    let private evalWrapper (compileSpec: CompileSpecT) (baseExprGens: UExprGenT list) : (VarEnvT -> IArrayNDT list) =      
        let compiler, baseCompileEnv = compileSpec

        /// Tries to compile the expression using the given CompileEnv.
        let tryCompile compileEnv failIfImpossible = 
            let failIfImpossible = failIfImpossible || not compileEnv.CanDelay

            // substitute symbol sizes into expressions and convert to unified expressons
            let uexprs, vars, sizeAvail = 
                baseExprGens 
                |> List.map (fun gen -> gen compileEnv.SymSizes) 
                |> List.unzip3
            let neededVars = Set.unionMany vars            

            // check that all necessary symbol sizes are avilable
            let allSizesAvail = sizeAvail |> List.forall id
            if failIfImpossible && not allSizesAvail then
                failwith "cannot compile expression because not all symbolic sizes could be resolved"

            // substitute symbol sizes into variable locations
            let varLocs =
                compileEnv.VarLocs
                |> Map.toSeq
                |> Seq.map (fun (vs, loc) -> (vs |> VarSpec.substSymSizes compileEnv.SymSizes, loc))
                |> Map.ofSeq
            let compileEnv = {compileEnv with VarLocs=varLocs}

            // check that all neccessary variable locations are avilable
            let allKnownLocs = 
                varLocs
                |> Map.toSeq
                |> Seq.map (fun (vs, _) -> vs)
                |> Set.ofSeq
            let allLocsAvail = Set.isEmpty (neededVars - allKnownLocs)
            if failIfImpossible && not allLocsAvail then
                failwithf "cannot compile expression because location of variable(s) %A is missing"                
                    (neededVars - allKnownLocs |> Set.toList)

            if allSizesAvail && allLocsAvail then Some (compiler compileEnv uexprs, neededVars)
            else None

        /// Performs evaluation of a compiled function.
        let performEval compileEnv evaluator neededVars varEnv = 
            // substitute symbol sizes
            let varEnv = varEnv |> VarEnv.substSymSizes compileEnv.SymSizes

            // check if evaluation is possible
            let missingVars = Set.filter (fun v -> not (Map.containsKey v varEnv)) neededVars
            if not (Set.isEmpty missingVars) then
                failwithf "cannot evaluate expression because values for variable(s) %A is missing"
                    (missingVars |> Set.toList)

            // evaluate using compiled function
            let evalEnv = EvalEnv.create varEnv 
            evaluator evalEnv

        // If all size symbols and variable storage locations are known, then we can immedietly compile
        // the expression. Otherwise we have to wait for a VarEnv to infer the missing sizes and locations.
        match tryCompile baseCompileEnv false with
        | Some (evaluator, neededVars) -> performEval baseCompileEnv evaluator neededVars
        | None ->
            let mutable evaluators = Map.empty
            fun varEnv ->
                // infer size symbols from variables and substitute into expression and variables
                let symSizes = VarEnv.inferSymSizes varEnv
                let varLocs = VarEnv.valueLocations varEnv
                let compileEnv = {baseCompileEnv with SymSizes = SymSizeEnv.merge baseCompileEnv.SymSizes symSizes
                                                      VarLocs  = Map.join baseCompileEnv.VarLocs varLocs}

                // compile and cache compiled function if necessary
                if not (Map.containsKey compileEnv evaluators) then
                    evaluators <- evaluators |> Map.add compileEnv (tryCompile compileEnv true).Value

                // evaluate
                let evaluator, neededVars = evaluators.[compileEnv]
                performEval compileEnv evaluator neededVars varEnv


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

    //let makeMany factory ()

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


