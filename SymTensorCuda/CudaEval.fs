﻿namespace SymTensor.Compiler.Cuda

open System
open System.Reflection

open Basics
open ArrayNDNS

open SymTensor


module CudaEval =

    /// CUDA GPU expression evaluator
    let cudaEvaluator (compileEnv: CompileEnvT) (uexprs: UExprT list)  =
          
        // add storage op for results
        let transferUExprs, resVars, resAllocators = 
            uexprs
            |> List.mapi (fun i (UExpr (_, tn, shp, _) as uexpr) ->
                let nshp = ShapeSpec.eval shp
                let layout = ArrayNDLayout.newContiguous nshp

                // create result storage allocator
                let resAllocator = fun () ->
                    match compileEnv.ResultLoc with
                    | LocHost -> ArrayNDHost.newOfType (TypeName.getType tn) layout   
                    | LocDev  -> ArrayNDCuda.newOfType (TypeName.getType tn) layout   
                    | l -> failwithf "CUDA cannot work with result location %A" l      

                if List.fold (*) 1 nshp > 0 then
                    // expression has data that needs to be stored       
                    // create variable that will be inserted into expression
                    let resVarName = sprintf "__RESULT%d__" i
                    let resVar = VarSpec.ofNameShapeAndTypeName resVarName shp tn                     

                    // insert StoreToVar op in expression
                    UExpr (UUnaryOp (StoreToVar resVar), tn, shp, [uexpr]), Some resVar, resAllocator
                else
                    // no data needs to be transferred back
                    uexpr, None, resAllocator)
            |> List.unzip3

        /// result variable locations
        let resVarLocs = 
            (Map.empty, resVars)
            ||> Seq.fold (fun locs resVar ->
                match resVar with
                | Some vs -> locs |> Map.add vs compileEnv.ResultLoc
                | None -> locs)

        /// unified expression containing all expressions to evaluate
        let mergedUexpr =
            match transferUExprs with
            | [] -> UExpr (UNaryOp Discard, TypeName.ofType<int>, ShapeSpec.emptyVector, [])
            | [uexpr] -> uexpr
            | UExpr (_, tn, _, _) :: _ ->
                UExpr (UNaryOp Discard, tn, ShapeSpec.emptyVector, transferUExprs)                       

        // build variable locations
        let varLocs =
            compileEnv.VarLocs
            |> Map.toSeq
            |> Seq.map (fun (vs, loc) -> UVarSpec.ofVarSpec vs, loc)
            |> Map.ofSeq
            |> Map.join resVarLocs

        // compile expression and create workspace
        let cudaCompileEnv = {VarStorLoc = varLocs}
        let rcpt = CudaRecipe.build cudaCompileEnv mergedUexpr
        let workspace = new CudaExprWorkspace (rcpt)


        fun (evalEnv: EvalEnvT) ->           
            // create arrays for results and add them to VarEnv
            let resArrays = resAllocators |> List.map (fun a -> a())
            let varEnv =
                (resVars, resArrays)
                ||> List.zip
                |> List.fold (fun varEnv (var, value) -> 
                        match var with
                        | Some vs -> varEnv |> VarEnv.addIVarSpec vs value
                        | None -> varEnv)
                    evalEnv.VarEnv               

            // partition variables depending on location
            let vsLoc = VarEnv.valueLocations varEnv
            let devVars, hostVars =
                varEnv
                |> Map.partition (fun vs _ -> vsLoc.[vs] = LocDev)

            // evaluate
            workspace.Eval (devVars, hostVars)
            resArrays

       


[<AutoOpen>]
module CudaEvalTypes =

    type private AllocatorT =
        static member Allocator<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> 
                (shp: NShapeSpecT) : ArrayNDT<'T> =
            let ary : ArrayNDCudaT<'T> = ArrayNDCuda.newContiguous shp 
            ary :> ArrayNDT<'T>

    /// Evaluates the model on a CUDA GPU.
    let DevCuda = { 
        new IDevice with
            member this.Allocator shp : ArrayNDT<'T> = 
                #if !CUDA_DUMMY
                let gm = typeof<AllocatorT>.GetMethod ("Allocator", 
                                                       BindingFlags.NonPublic ||| 
                                                       BindingFlags.Public ||| 
                                                       BindingFlags.Static)
                let m = gm.MakeGenericMethod ([|typeof<'T>|])
                m.Invoke(null, [|shp|]) :?> ArrayNDT<'T>
                #else
                ArrayNDHost.newContiguous shp
                #endif
                
            member this.Compiler = CudaEval.cudaEvaluator
            member this.DefaultLoc = LocDev
    }


