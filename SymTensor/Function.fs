namespace SymTensor

open Basics
open ArrayNDNS


[<AutoOpen>]
module VarEnvTypes = 
    /// variable environment
    type VarEnvT = Map<UVarSpecT, IArrayNDT>

    /// specification of variable storage locations
    type VarLocsT = Map<UVarSpecT, ArrayLocT>


module VarEnv = 

    /// add variable value to environment
    let addUVarSpec (vs: UVarSpecT) (value: IArrayNDT) (varEnv: VarEnvT) : VarEnvT =
        Map.add vs value varEnv

    /// add variable value to environment
    let addVarSpecT (vs: VarSpecT<'T>) (value: ArrayNDT<'T>) (varEnv: VarEnvT) : VarEnvT =
        addUVarSpec (UVarSpec.ofVarSpec vs) (value :> IArrayNDT) varEnv        

    /// add variable value to environment
    let add (var: Expr.ExprT<'T>) (value: ArrayNDT<'T>) (varEnv: VarEnvT) : VarEnvT =
        addVarSpecT (Expr.extractVar var) value varEnv

    /// get variable value from environment
    let getUnified (vs: UVarSpecT) (varEnv: VarEnvT) : IArrayNDT =
        varEnv.[vs]

    /// get variable value from environment
    let getVarSpecT (vs: VarSpecT<'T>) (varEnv: VarEnvT) : ArrayNDT<'T> =
        getUnified (UVarSpec.ofVarSpec vs) varEnv :?> ArrayNDT<'T>

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
    let inferSymSizes (varEnv: VarEnvT) : SymSizeEnvT =
        varEnv |> Map.fold 
            (fun env vSym vVal ->                
                (UVarSpec.shape vSym, ArrayND.shape vVal)
                ||> List.zip
                |> List.fold (fun env (svSym, svVal) ->
                    match SizeSpec.simplify svSym with
                    | Base (Sym sym) -> env |> SymSizeEnv.add sym (Base (Fixed svVal))
                    | Base (Fixed f) -> 
                        if f = svVal then env
                        else failwithf "%A <> %d" svSym svVal
                    | Broadcast ->
                        if 1 = svVal then env
                        else failwithf "1 <> %d" svVal
                    | Multinom m -> failwithf "%A <> %d" m svVal
                ) env)
            SymSizeEnv.empty 

    /// substitues the given symbol sizes into the variable environment
    let checkAndSubstSymSizes symSizes (varEnv: VarEnvT) : VarEnvT =
        varEnv 
        |> Map.toSeq
        |> Seq.map (fun (vs, value) -> 
            let ss = UVarSpec.shape vs
            let ns = ss |> SymSizeEnv.substShape symSizes |> ShapeSpec.eval
            if ArrayND.shape value <> ns then
                failwithf "variable %A was expected to be of shape %A (%A) but a \
                           value with shape %A was provided" vs ns ss (ArrayND.shape value)
            UVarSpec.substSymSizes symSizes vs, value)
        |> Map.ofSeq
        
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
    type IUExprCompiler = 
        abstract Name:     string
        abstract Compile:  CompileEnvT -> UExprT list -> CompiledUExprT

    /// compile specification, consisting of a compiler and a compile environment
    type CompileSpecT = IUExprCompiler * CompileEnvT


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

    type private UExprGenT = SymSizeEnvT -> (UExprT * Set<UVarSpecT> * bool) 

    let private uExprGen baseExpr symSizes =
        let expr = baseExpr |> Expr.substSymSizes symSizes 
        let vars = Expr.extractVars expr |> Set.map UVarSpec.ofVarSpec
        UExpr.toUExpr expr, vars, Expr.canEvalAllSymSizes expr

    type private CompileResultT = {
        Exprs:      UExprT list;
        Eval:       CompiledUExprT;
        NeededVars: Set<UVarSpecT>;
        CompileEnv: CompileEnvT;
    }

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
                |> Seq.map (fun (vs, loc) -> (vs |> UVarSpec.substSymSizes compileEnv.SymSizes, loc))
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

            if allSizesAvail && allLocsAvail then 
                Some {
                    Exprs=uexprs; 
                    CompileEnv=compileEnv;
                    Eval=compiler.Compile compileEnv uexprs; 
                    NeededVars=neededVars
                }
            else None

        /// Performs evaluation of a compiled function.
        let performEval compileRes varEnv = 
            // substitute and check symbol sizes
            let varEnv = varEnv |> VarEnv.checkAndSubstSymSizes compileRes.CompileEnv.SymSizes

            // check that variable locations match with compile environment
            let varLocs = VarEnv.valueLocations varEnv
            for vs in compileRes.NeededVars do
                match varLocs |> Map.tryFind vs with
                | Some loc when loc <> UVarSpec.findByName vs compileRes.CompileEnv.VarLocs ->
                    failwithf "variable %A was expected to be in location %A but a value in \
                               location %A was specified" vs compileRes.CompileEnv.VarLocs.[vs] loc
                | Some _ -> ()
                | None -> 
                    failwithf "cannot evaluate expression because value for variable %A is missing" vs

            // start tracing
            Trace.startExprEval compileRes.Exprs compiler.Name

            // evaluate using compiled function
            let evalEnv = EvalEnv.create varEnv 
            let res = compileRes.Eval evalEnv

            // stop tracing
            Trace.endExprEval ()

            res

        // If all size symbols and variable storage locations are known, then we can immedietly compile
        // the expression. Otherwise we have to wait for a VarEnv to infer the missing sizes and locations.
        match tryCompile baseCompileEnv false with
        | Some compileRes -> performEval compileRes
        | None ->
            let mutable variants = Map.empty
            fun varEnv ->
                // infer size symbols from variables and substitute into expression and variables
                let symSizes = VarEnv.inferSymSizes varEnv
                let varLocs = VarEnv.valueLocations varEnv
                let compileEnv = {baseCompileEnv with SymSizes = SymSizeEnv.merge baseCompileEnv.SymSizes symSizes
                                                      VarLocs  = Map.join baseCompileEnv.VarLocs varLocs}

                // compile and cache compiled function if necessary
                if not (Map.containsKey compileEnv variants) then
                    variants <- variants |> Map.add compileEnv (tryCompile compileEnv true).Value

                // evaluate
                performEval variants.[compileEnv] varEnv


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


