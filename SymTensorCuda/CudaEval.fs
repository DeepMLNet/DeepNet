namespace SymTensor.Compiler.Cuda

open ArrayNDNS

open SymTensor


module CudaEval =

    /// CUDA GPU expression evaluator
    let cudaEvaluator resultLoc (uexprs: UExprT list)  =
          
        /// unified expression
        let transferUExprs, resVars, resArrays = 
            uexprs
            |> List.mapi (fun i (UExpr (_, tn, shp, _) as uexpr) ->
                let nshp = ShapeSpec.eval shp

                // create arrays for results
                let layout = ArrayNDLayout.newContiguous nshp
                let resArray = 
                    match resultLoc with
                    | LocHost -> ArrayNDHost.newOfType (TypeName.getType tn) layout   
                    | LocDev -> ArrayNDCuda.newOfType (TypeName.getType tn) layout   

                if List.fold (*) 1 nshp > 0 then
                    // expression has data that needs to be stored       
                    // create variable that will be inserted into expression
                    let resVarName = sprintf "__RESULT%d__" i
                    let resVar = VarSpec.ofNameShapeAndTypeName resVarName shp tn

                    // insert StoreToVar op in expression
                    UExpr (UUnaryOp (StoreToVar resVar), tn, shp, [uexpr]), Some resVar, resArray
                else
                    // no data needs to be transferred back
                    uexpr, None, resArray)
            |> List.unzip3

        /// unified expression containing all expressions to evaluate
        let mergedUexpr =
            match uexprs with
            | [] -> UExpr (UNaryOp Discard, TypeName.ofType<int>, ShapeSpec.emptyVector, [])
            | [uexpr] -> uexpr
            | UExpr (_, tn, _, _) :: _ ->
                UExpr (UNaryOp Discard, tn, ShapeSpec.emptyVector, uexprs)                       

        /// active workspaces
        let mutable workspaces = Map.empty

        fun (evalEnv: EvalEnvT) ->

            /// variable locations
            let vsLoc =
                evalEnv.VarEnv 
                |> Map.map (fun vs ary ->
                    match ary with
                    | :? IArrayNDCudaT -> LocDev
                    | :? IArrayNDHostT -> LocHost
                    | _ -> failwithf "unknown variable value type")

            // partition variables depending on location
            let devVars, hostVars =
                evalEnv.VarEnv
                |> Map.partition (fun vs _ -> vsLoc.[vs] = LocDev)

            /// the CUDA workspace
            let workspace =
                // obtain (cached) workspace 
                let compileEnv = {VarStorLoc = vsLoc;}
                match Map.tryFind compileEnv workspaces with
                | Some ws -> ws
                | None ->
                    let rcpt = CudaRecipe.build compileEnv mergedUexpr
                    let ws = new CudaExprWorkspace (rcpt)
                    workspaces <- workspaces |> Map.add compileEnv ws
                    ws

            // evaluate
            workspace.Eval (devVars, hostVars)
            resArrays

           


[<AutoOpen>]
module CudaEvalTypes =

    /// evaluates expression on CUDA GPU using compiler and returns results as ArrayNDHostTs
    let onCudaWithHostResults expr = 
        CudaEval.cudaEvaluator LocHost

    /// evaluates expression on CUDA GPU using compiler and returns results as ArrayNDDevTs
    let onCudaWithDevResults expr = 
        CudaEval.cudaEvaluator LocDev




