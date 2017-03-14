namespace SymTensor

open System.Diagnostics

open Basics
open ArrayNDNS
open UExprTypes



[<AutoOpen>]
module VarEnvTypes = 
    /// variable value collection
    type VarEnvT = Map<VarSpecT, IArrayNDT>

    /// specification of variable storage locations
    type VarLocsT = Map<VarSpecT, ArrayLocT>

    /// specification of variable strides
    type VarStridesT = Map<VarSpecT, int64 list>

    /// specification of channel strides
    type ChannelStridesT = Map<ChannelT, int64 list>


/// Variable value collection.
module VarEnv = 

    /// add variable value to environment
    let addVarSpec (vs: VarSpecT) (value: #IArrayNDT) (varEnv: VarEnvT) : VarEnvT =
        Map.add vs (value :> IArrayNDT) varEnv

    /// remove variable value from environment
    let removeVarSpec (vs: VarSpecT) (varEnv: VarEnvT) : VarEnvT =
        Map.remove vs varEnv

    /// get variable value from environment
    let getVarSpec (vs: VarSpecT) (varEnv: VarEnvT) : #IArrayNDT =
        match varEnv |> Map.tryFind vs with
        | Some v -> v |> box |> unbox
        | None -> failwithf "variable %A is not present in the specified VarEnv" vs

    /// add variable value to environment
    let add (var: Expr.ExprT) (value: #IArrayNDT) (varEnv: VarEnvT) : VarEnvT =
        addVarSpec (Expr.extractVar var) value varEnv

    /// remove variable value from environment
    let remove (var: Expr.ExprT) (varEnv: VarEnvT) : VarEnvT =
        removeVarSpec (Expr.extractVar var) varEnv

    /// get variable value from environment
    let get (var: Expr.ExprT) (varEnv: VarEnvT) : #IArrayNDT =
        getVarSpec (Expr.extractVar var) varEnv

    /// empty variable environment
    let (empty: VarEnvT) =
        Map.empty

    /// joins two variable environments
    let join (a: VarEnvT) (b: VarEnvT) =
        Map.join a b

    /// infers symbol sizes from the variable environment
    let inferSymSizes (symSizeEnv: SymSizeEnvT) (varEnv: VarEnvT) : SymSizeEnvT =
        (symSizeEnv, varEnv) ||> Map.fold 
            (fun env vSym vVal ->   
                if VarSpec.nDims vSym <> ArrayND.nDims vVal then
                    failwithf "dimensionality mismatch: a value of shape %A was provided for variable %A"
                        (ArrayND.shape vVal) vSym

                (VarSpec.shape vSym, ArrayND.shape vVal)
                ||> List.zip
                |> List.fold (fun env (svSym, svVal) ->
                    let failShape () =
                        let vSymShp = vSym.Shape |> ShapeSpec.substSymbols env 
                        failwithf "expected variable %A with (inferred) shape %A but got value with shape %A"
                            vSym vSymShp vVal.Shape
                    match svSym |> SizeSpec.substSymbols env |> SizeSpec.simplify  with
                    | Base (Sym sym) -> env |> SymSizeEnv.add sym (SizeSpec.fix svVal)
                    | Base (Fixed f) -> 
                        if f .= svVal then env
                        else failShape ()
                    | Broadcast ->
                        if 1L = svVal then env
                        else failShape ()
                    | Multinom m -> failShape ()
                ) env)


    /// substitues the given symbol sizes into the variable environment
    let substSymSizes symSizes (varEnv: VarEnvT) : VarEnvT =
        varEnv 
        |> Map.toSeq
        |> Seq.map (fun (vs, value) -> VarSpec.substSymSizes symSizes vs, value)
        |> Map.ofSeq

    /// checks that the values are valid in type and shape for the variables
    let check (varEnv: VarEnvT) =
        varEnv |> Map.iter (fun vs value ->
            if TypeName.ofTypeInst value.DataType <> vs.TypeName then
                failwithf "variable %A was expected to be of type %A but a \
                           value with type %A was provided" vs.Name vs.TypeName.Type value.DataType

            let ss = VarSpec.shape vs
            match ShapeSpec.tryEval ss with
            | Some ns when ArrayND.shape value <> ns ->
                failwithf "variable %A was expected to be of shape %A (%A) but a \
                           value with shape %A was provided" vs.Name ns ss (ArrayND.shape value)
            | None -> failwithf "variable %A contains size symbols that cannot be evaluated" vs
            | _ -> ()
        )
        
    /// gets the type names of the variable value arrays
    let valueTypeNames (varEnv: VarEnvT) =
        varEnv |> Map.map (fun _ vVal -> TypeName.ofObject vVal)

    /// gets the locations of the variable value arrays
    let valueLocations (varEnv: VarEnvT) : VarLocsT =
        varEnv |> Map.map (fun _ vVal -> ArrayND.location vVal)

    /// gets the strides of the variable value arrays
    let valueStrides (varEnv: VarEnvT) : VarStridesT =
        varEnv |> Map.map (fun _ vVal -> ArrayND.stride vVal)

    /// Constructs a VarEnvT from a sequence of variable, value tuples.
    let ofSeq (entries: (ExprT * #IArrayNDT) seq) =
        (empty, entries)
        ||> Seq.fold (fun ve (var, value) -> ve |> add var value)


[<AutoOpen>]
module EnvTypes =

    open SymTensor.Compiler

    /// Information necessary to evaluate an expression.
    /// Currently this just holds the variable values, but may contain further information in the future.
    type EvalEnvT = {
        VarEnv:             VarEnvT
    }

    /// Information necessary to compile an expression.
    type CompileEnvT = {
        SymSizes:           SymSizeEnvT
        VarLocs:            VarLocsT
        VarStrides:         VarStridesT
        ChannelStrides:     ChannelStridesT
        ResultLoc:          ArrayLocT
        CanDelay:           bool
    }

    /// an evaluation function
    type EvalFn = EvalEnvT -> IArrayNDT list

    /// The result of the compilation of an expression.
    type CompiledUExprsT = {
        /// the expressions involved in this compilation
        Exprs:              UExprT list
        /// the CompileEnvT used for this compilation
        CompileEnv:         CompileEnvT
        /// the variables necessary to evaluate the expressions
        NeededVars:         Set<VarSpecT>
        /// the evaluation function
        Eval:               EvalFn
        /// diagnostic information
        Diagnostics:        CompileDiagnosticsT option
    }

    /// a function that compiles a unified expression into a function
    type IUExprCompiler = 
        abstract Name:     string
        abstract Compile:  CompileEnvT -> UExprT list -> EvalFn * (CompileDiagnosticsT option)

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
        VarEnv.getVarSpec vs evalEnv.VarEnv

    /// get variable value from environment
    let get var (evalEnv: EvalEnvT) =
        VarEnv.get var evalEnv.VarEnv


/// Function compilation parameters
module CompileEnv =

    /// empty compile environment
    let empty = {
        VarLocs         = Map.empty 
        VarStrides      = Map.empty
        ChannelStrides  = Map.empty
        ResultLoc       = LocHost
        SymSizes        = SymSizeEnv.empty
        CanDelay        = true
    }

 
/// Generates F# function from expressions.
module Func =

    type private UExprGenT = {
        Generate:               SymSizeEnvT -> ExprT
        UVarSpecsAndEvalable:   bool -> SymSizeEnvT -> Set<VarSpecT> * bool       
    }

    let private exprGenerate baseExpr symSizes =
        Expr.checkExpr baseExpr

        let sw = Stopwatch.StartNew ()
        if Debug.TraceCompile then printfn "Substituting symbolic sizes..."
        let substExpr = baseExpr |> Expr.substSymSizes symSizes 
        if Debug.Timing then printfn "Substituting symbolic sizes took %A" sw.Elapsed
        Expr.checkExpr substExpr

        if Debug.TraceCompile then printfn "Releasing held expressions..."
        let releasedExpr = Hold.tryRelease substExpr
        if Debug.Timing then printfn "Releasing held expressions took %A" sw.Elapsed        
        Expr.checkExpr releasedExpr

        releasedExpr

    let private uExprVarSpecsAndEvalable baseExpr failIfNotEvalable symSizes =
        let expr = baseExpr |> Expr.substSymSizes symSizes |> Hold.tryRelease
        let vars = Expr.extractVars expr 
        if failIfNotEvalable then Expr.failOnNotEvalableSymSize expr
        vars, Expr.canEvalAllSymSizes expr

    let private evalWrapper (compileSpec: CompileSpecT) (baseExprGens: UExprGenT list) 
            : (VarEnvT -> IArrayNDT list) =     
             
        let compiler, baseCompileEnv = compileSpec

        /// Tries to compile the expression using the given CompileEnv.
        let tryCompile compileEnv failIfImpossible = 
            let failIfImpossible = failIfImpossible || not compileEnv.CanDelay

            // substitute symbol sizes into expressions and convert to unified expressions
            let vars, sizeAvail = 
                baseExprGens 
                |> List.map (fun gen -> gen.UVarSpecsAndEvalable failIfImpossible compileEnv.SymSizes) 
                |> List.unzip
            let neededVars = Set.unionMany vars   
            if Debug.PrintInstantiations then
                printfn "Vars needed for instantiation:\n%A" (neededVars |> Set.toList)

            // check that all necessary symbol sizes are available
            let allSizesAvail = sizeAvail |> List.forall id
            if failIfImpossible && not allSizesAvail then
                failwith "cannot compile expression because not all symbolic sizes could be resolved"

            // substitute symbol sizes into variable locations and strides
            let substVarSizes varMap =
                varMap
                |> Map.toSeq
                |> Seq.map (fun (vs, value) -> (vs |> VarSpec.substSymSizes compileEnv.SymSizes, value))
                |> Map.ofSeq
            let compileEnv = {compileEnv with VarLocs=substVarSizes compileEnv.VarLocs
                                              VarStrides=substVarSizes compileEnv.VarStrides}

            // check that all necessary variable locations and strides are available
            let checkMissing varMap what =
                let allKnown = 
                    varMap |> Map.toSeq |> Seq.map (fun (vs, _) -> vs) |> Set.ofSeq
                let allAvail = Set.isEmpty (neededVars - allKnown)
                if failIfImpossible && not allAvail then
                    failwithf "cannot compile expression because %s of variable(s) %A is missing"                
                        what (neededVars - allKnown |> Set.toList)
                allAvail
            let allLocsAvail = checkMissing compileEnv.VarLocs "location"
            let allStridesAvail = checkMissing compileEnv.VarStrides "strides"

            // if everything is available, then compile
            if allSizesAvail && allLocsAvail && allStridesAvail then 
                // build exprs with substituted sizes
                let exprs = 
                    baseExprGens 
                    |> List.map (fun gen -> gen.Generate compileEnv.SymSizes) 

                // optimize expression group
                let sw = Stopwatch.StartNew ()
                if Debug.TraceCompile then printfn "Optimizing expression group..." 
                let optExprs = 
                    if Debug.DisableOptimizer then exprs
                    else Optimizer.optimize exprs
                if Debug.Timing then printfn "Optimizing expression group took %A" sw.Elapsed
                if Debug.PrintOptimizerStatistics then
                    printfn "Optimization:    ops: %6d => %6d    unique ops: %6d => %6d" 
                        (List.sumBy Expr.countOps exprs) (List.sumBy Expr.countOps optExprs)
                        (List.sumBy Expr.countUniqueOps exprs) (List.sumBy Expr.countUniqueOps optExprs)

                // convert to UExprs
                let sw = Stopwatch.StartNew ()
                if Debug.TraceCompile then printfn "Converting to UExprs..."
                let uexprs = UExpr.toUExprs optExprs
                if Debug.Timing then printfn "Converting to UExprs took %A" sw.Elapsed

                // compile
                let evalFn, diagnostics = compiler.Compile compileEnv uexprs
                let compileRes = {
                    Exprs       = uexprs
                    CompileEnv  = compileEnv
                    NeededVars  = neededVars
                    Eval        = evalFn
                    Diagnostics = diagnostics
                }

                // show diagnostics visualization, if requested
                if Debug.VisualizeUExpr then 
                    match compileRes.Diagnostics with
                    | Some diag ->
                        printfn "Visualizing UExpr in new window..."
                        DiagnosticsVisualizer.visualize diag
                    | None -> 
                        printfn "No compilation diagnostics available."

                // terminate program, if requested for debugging
                if Debug.TerminateAfterCompilation then exit 0

                Some compileRes
            else None

        /// Performs evaluation of a compiled function.
        let performEval compileRes varEnv = 
            // substitute and check symbol sizes
            let varEnv = varEnv |> VarEnv.substSymSizes compileRes.CompileEnv.SymSizes 
            VarEnv.check varEnv
            let varLocs = varEnv |> VarEnv.valueLocations 
            let varStrides = varEnv |> VarEnv.valueStrides 

            // check that variable locations and strides match with compile environment
            for vs in compileRes.NeededVars do
                if not (varLocs.ContainsKey vs) then
                    failwithf "cannot evaluate expression because value for variable %A is missing" vs
                
                let cmplLoc = VarSpec.findByName vs compileRes.CompileEnv.VarLocs
                if varLocs.[vs] <> cmplLoc then
                    failwithf "variable %A was expected to be in location %A but a value in \
                               location %A was specified" vs cmplLoc varLocs.[vs]

                let cmplStrides = VarSpec.findByName vs compileRes.CompileEnv.VarStrides
                if varStrides.[vs] <> cmplStrides then
                    failwithf "variable %A was expected to have strides %A but a value with \
                               strides %A was specified" vs cmplStrides varStrides.[vs]

            // start tracing
            Trace.startExprEval compileRes.Exprs compiler.Name

            try
                // evaluate using compiled function
                let evalEnv = EvalEnv.create varEnv 
                compileRes.Eval evalEnv
            finally
                // stop tracing
                Trace.endExprEval ()

        // If all size symbols and variable storage locations are known, then we can immediately compile
        // the expression. Otherwise we have to wait for a VarEnv to infer the missing sizes and locations.
        match tryCompile baseCompileEnv false with
        | Some compileRes -> performEval compileRes
        | None ->
            let mutable variants = Map.empty
            fun varEnv ->
                // infer information from VarEnv
                let compileEnv = 
                    {baseCompileEnv with SymSizes   = varEnv |> VarEnv.inferSymSizes baseCompileEnv.SymSizes 
                                         VarLocs    = varEnv |> VarEnv.valueLocations
                                         VarStrides = varEnv |> VarEnv.valueStrides}

                // compile and cache compiled function if necessary
                if not (Map.containsKey compileEnv variants) then 
                    if Debug.PrintInstantiations then 
                        printfn "Instantiating new function variant for %A" compileEnv
                    variants <- variants |> Map.add compileEnv (tryCompile compileEnv true).Value
                else
                    if Debug.PrintInstantiations then 
                        printfn "Using cached function variant for %A" compileEnv

                // evaluate
                performEval variants.[compileEnv] varEnv

    [<RequiresExplicitTypeArguments>]
    let private checkType<'T> pos (expr: ExprT) =
        if typeof<'T> = typeof<obj> then
            failwith "The result type of the function was not inferred (is ArrayNDT<obj>) and \
                      therefore must be specified manually."
        if typeof<'T> <> expr.Type then
            failwithf "Result of %s expression is of type %A and cannot be stored in array of type %A."
                pos expr.Type typeof<'T>

    /// makes a function that evaluates the given expression 
    let make<'T0> factory (expr0: ExprT)  =
        checkType<'T0> "" expr0
        let expr0gen = {Generate=exprGenerate expr0; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr0}   
        let evalAll = evalWrapper factory [expr0gen]        
        fun (varEnv: VarEnvT) ->
            let res = evalAll varEnv
            res.[0] :?> ArrayNDT<'T0>

    let make2<'T0, 'T1> factory (expr0: ExprT) (expr1: ExprT) =    
        checkType<'T0> "first" expr0
        checkType<'T1> "second" expr1
        let expr0gen = {Generate=exprGenerate expr0; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr0}   
        let expr1gen = {Generate=exprGenerate expr1; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr1}   
        let evalAll = evalWrapper factory [expr0gen; expr1gen]        
        fun (varEnv: VarEnvT) ->
            let res = evalAll varEnv
            res.[0] :?> ArrayNDT<'T0>, res.[1] :?> ArrayNDT<'T1>

    let make3<'T0, 'T1, 'T2> factory (expr0: ExprT) (expr1: ExprT) (expr2: ExprT) =    
        checkType<'T0> "first" expr0
        checkType<'T1> "second" expr1
        checkType<'T2> "third" expr2
        let expr0gen = {Generate=exprGenerate expr0; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr0}   
        let expr1gen = {Generate=exprGenerate expr1; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr1}   
        let expr2gen = {Generate=exprGenerate expr2; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr2}   
        let evalAll = evalWrapper factory [expr0gen; expr1gen; expr2gen]        
        fun (varEnv: VarEnvT) ->
            let res = evalAll varEnv
            res.[0] :?> ArrayNDT<'T0>, res.[1] :?> ArrayNDT<'T1>, res.[2] :?> ArrayNDT<'T2>

    let make4<'T0, 'T1, 'T2, 'T3> factory (expr0: ExprT) (expr1: ExprT) (expr2: ExprT) (expr3: ExprT) =    
        checkType<'T0> "first" expr0
        checkType<'T1> "second" expr1
        checkType<'T2> "third" expr2
        checkType<'T3> "fourth" expr3
        let expr0gen = {Generate=exprGenerate expr0; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr0}   
        let expr1gen = {Generate=exprGenerate expr1; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr1}   
        let expr2gen = {Generate=exprGenerate expr2; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr2}   
        let expr3gen = {Generate=exprGenerate expr3; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr3}   
        let evalAll = evalWrapper factory [expr0gen; expr1gen; expr2gen; expr3gen]        
        fun (varEnv: VarEnvT) ->
            let res = evalAll varEnv
            res.[0] :?> ArrayNDT<'T0>, res.[1] :?> ArrayNDT<'T1>, res.[2] :?> ArrayNDT<'T2>, res.[3] :?> ArrayNDT<'T3>

    let make5<'T0, 'T1, 'T2, 'T3, 'T4> factory (expr0: ExprT) (expr1: ExprT) (expr2: ExprT) (expr3: ExprT) (expr4: ExprT) =    
        checkType<'T0> "first" expr0
        checkType<'T1> "second" expr1
        checkType<'T2> "third" expr2
        checkType<'T3> "fourth" expr3
        checkType<'T4> "fifth" expr4
        let expr0gen = {Generate=exprGenerate expr0; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr0}   
        let expr1gen = {Generate=exprGenerate expr1; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr1}   
        let expr2gen = {Generate=exprGenerate expr2; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr2}   
        let expr3gen = {Generate=exprGenerate expr3; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr3}   
        let expr4gen = {Generate=exprGenerate expr4; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr4}   
        let evalAll = evalWrapper factory [expr0gen; expr1gen; expr2gen; expr3gen; expr4gen]        
        fun (varEnv: VarEnvT) ->
            let res = evalAll varEnv
            res.[0] :?> ArrayNDT<'T0>, res.[1] :?> ArrayNDT<'T1>, res.[2] :?> ArrayNDT<'T2>, res.[3] :?> ArrayNDT<'T3>, res.[4] :?> ArrayNDT<'T4>

    let makeMany<'T> factory (exprs: ExprT list) =
        exprs |> List.iter (checkType<'T> "all")
        let exprsGen =
            exprs
            |> List.map (fun expr -> 
                {Generate=exprGenerate expr; UVarSpecsAndEvalable=uExprVarSpecsAndEvalable expr})
        let evalAll = evalWrapper factory exprsGen
        fun (varEnv: VarEnvT) ->
            let reses = evalAll varEnv
            reses |> List.map (fun res -> res :?> ArrayNDT<'T>)


[<AutoOpen>]
module FuncTypes = 

    type Arg0Func<'TR> = unit -> 'TR
    let arg0<'TR> (f: VarEnvT -> 'TR) : Arg0Func<'TR> =
        fun () -> 
            VarEnv.empty |> f

    type Arg1Func<'T0, 'TR> = ArrayNDT<'T0> -> 'TR
    let arg1<'T0, 'TR> (vs0: ExprT) (f: VarEnvT -> 'TR) : Arg1Func<'T0, 'TR> =
        fun (val0: ArrayNDT<'T0>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> f

    type Arg2Func<'T0, 'T1, 'TR> = ArrayNDT<'T0> -> ArrayNDT<'T1> -> 'TR
    let arg2<'T0, 'T1, 'TR> (vs0: ExprT) (vs1: ExprT) (f: VarEnvT -> 'TR) : Arg2Func<'T0, 'T1, 'TR> =
        fun (val0: ArrayNDT<'T0>) (val1: ArrayNDT<'T1>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> f

    type Arg3Func<'T0, 'T1, 'T2, 'TR> = ArrayNDT<'T0> -> ArrayNDT<'T1> -> ArrayNDT<'T2> -> 'TR
    let arg3<'T0, 'T1, 'T2, 'TR> (vs0: ExprT) (vs1: ExprT) (vs2: ExprT) f : Arg3Func<'T0, 'T1, 'T2, 'TR> =
        fun (val0: ArrayNDT<'T0>) (val1: ArrayNDT<'T1>) (val2: ArrayNDT<'T2>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> f           

    type Arg4Func<'T0, 'T1, 'T2, 'T3, 'TR> = ArrayNDT<'T0> -> ArrayNDT<'T1> -> ArrayNDT<'T2> -> ArrayNDT<'T3> -> 'TR
    let arg4<'T0, 'T1, 'T2, 'T3, 'TR> (vs0: ExprT) (vs1: ExprT) (vs2: ExprT) (vs3: ExprT) f : Arg4Func<'T0, 'T1, 'T2, 'T3, 'TR> =
        fun (val0: ArrayNDT<'T0>) (val1: ArrayNDT<'T1>) (val2: ArrayNDT<'T2>) (val3: ArrayNDT<'T3>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> VarEnv.add vs3 val3 |> f           
   
    type Arg5Func<'T0, 'T1, 'T2, 'T3, 'T4, 'TR> = ArrayNDT<'T0> -> ArrayNDT<'T1> -> ArrayNDT<'T2> -> ArrayNDT<'T3> -> ArrayNDT<'T4> -> 'TR
    let arg5<'T0, 'T1, 'T2, 'T3, 'T4, 'TR> (vs0: ExprT) (vs1: ExprT) (vs2: ExprT) (vs3: ExprT) (vs4: ExprT)f : Arg5Func<'T0, 'T1, 'T2, 'T3, 'T4, 'TR> =
        fun (val0: ArrayNDT<'T0>) (val1: ArrayNDT<'T1>) (val2: ArrayNDT<'T2>) (val3: ArrayNDT<'T3>) (val4: ArrayNDT<'T4>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> VarEnv.add vs3 val3 |> VarEnv.add vs4 val4 |> f  
    
    type Arg6Func<'T0, 'T1, 'T2, 'T3, 'T4, 'T5, 'TR> = ArrayNDT<'T0> -> ArrayNDT<'T1> -> ArrayNDT<'T2> -> ArrayNDT<'T3> -> ArrayNDT<'T4> -> ArrayNDT<'T5> -> 'TR
    let arg6<'T0, 'T1, 'T2, 'T3, 'T4, 'T5, 'TR> (vs0: ExprT) (vs1: ExprT) (vs2: ExprT) (vs3: ExprT) (vs4: ExprT) (vs5: ExprT) f : Arg6Func<'T0, 'T1, 'T2, 'T3, 'T4, 'T5, 'TR> =
        fun (val0: ArrayNDT<'T0>) (val1: ArrayNDT<'T1>) (val2: ArrayNDT<'T2>) (val3: ArrayNDT<'T3>) (val4: ArrayNDT<'T4>) (val5: ArrayNDT<'T5>)  -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> VarEnv.add vs3 val3 |> VarEnv.add vs4 val4 |> VarEnv.add vs5 val5 |> f  

    type Arg7Func<'T0, 'T1, 'T2, 'T3, 'T4, 'T5, 'T6, 'TR> = ArrayNDT<'T0> -> ArrayNDT<'T1> -> ArrayNDT<'T2> -> ArrayNDT<'T3> -> ArrayNDT<'T4> -> ArrayNDT<'T5> -> ArrayNDT<'T6> -> 'TR
    let arg7<'T0, 'T1, 'T2, 'T3, 'T4, 'T5, 'T6, 'TR> (vs0: ExprT) (vs1: ExprT) (vs2: ExprT) (vs3: ExprT) (vs4: ExprT) (vs5: ExprT) (vs6: ExprT) f : Arg7Func<'T0, 'T1, 'T2, 'T3, 'T4, 'T5, 'T6, 'TR> =
        fun (val0: ArrayNDT<'T0>) (val1: ArrayNDT<'T1>) (val2: ArrayNDT<'T2>) (val3: ArrayNDT<'T3>) (val4: ArrayNDT<'T4>) (val5: ArrayNDT<'T5>) (val6: ArrayNDT<'T6>) -> 
            VarEnv.empty |> VarEnv.add vs0 val0 |> VarEnv.add vs1 val1 |> VarEnv.add vs2 val2 |> VarEnv.add vs3 val3 |> VarEnv.add vs4 val4 |> VarEnv.add vs5 val5 |> VarEnv.add vs6 val6 |> f  

    let addArg<'T, 'TR> (vs: ExprT) (f: VarEnvT -> 'TR) =
        fun (ve: VarEnvT) (value: ArrayNDT<'T>) ->
            f (ve |> VarEnv.add vs value)

    let addVarEnv (varEnv: VarEnvT) f =
        fun (ve: VarEnvT) ->
            f (VarEnv.join ve varEnv)

//    let (^@^) a b =
//        (addArg a) b
//            
//    let (^<|) a b =
//        a b
