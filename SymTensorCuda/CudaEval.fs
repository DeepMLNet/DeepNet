namespace SymTensor.Compiler.Cuda

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
                let layout = ArrayNDLayout.newC nshp

                // create result storage allocator
                let resAllocator = fun () ->
                    match compileEnv.ResultLoc with
                    | LocHost -> ArrayNDHost.newOfType (TypeName.getType tn) layout :> IArrayNDT
                    | LocDev  -> ArrayNDCuda.newOfType (TypeName.getType tn) layout :> IArrayNDT  
                    | l -> failwithf "CUDA cannot work with result location %A" l      

                if List.fold (*) 1 nshp > 0 then
                    // expression has data that needs to be stored       
                    // create variable that will be inserted into expression
                    let resVarName = sprintf "__RESULT%d__" i
                    let resVar = UVarSpec.ofNameShapeAndTypeName resVarName shp tn                     

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
        let varLocs = Map.join compileEnv.VarLocs resVarLocs

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
                        | Some vs -> varEnv |> VarEnv.addUVarSpec vs value
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

    type private HelperT =
        static member Allocator<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> 
                (shp: NShapeSpecT) : ArrayNDT<'T> =
            let ary : ArrayNDCudaT<'T> = ArrayNDCuda.newC shp 
            ary :> ArrayNDT<'T>

        static member ToDev<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> 
                (ary: ArrayNDHostT<'T>) : ArrayNDT<'T> =
            ArrayNDCuda.toDev ary :> ArrayNDT<'T>

        static member ToHost<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> 
                (ary: ArrayNDT<'T>) : ArrayNDHostT<'T> =
            ArrayNDCuda.toHost (ary :?> ArrayNDCudaT<'T>)

    let private invokeHelperMethod<'T> name args = 
        let gm = typeof<HelperT>.GetMethod (name, 
                                            BindingFlags.NonPublic ||| 
                                            BindingFlags.Public ||| 
                                            BindingFlags.Static)
        let m = gm.MakeGenericMethod ([|typeof<'T>|])
        m.Invoke(null, args)  


    /// Evaluates the model on a CUDA GPU.
    let DevCuda = { 
        new IDevice with
            
            #if !CUDA_DUMMY
            member this.Allocator shp : ArrayNDT<'T>    = invokeHelperMethod<'T> "Allocator" [|shp|] |> unbox            
            member this.ToDev ary : ArrayNDT<'T1>       = invokeHelperMethod<'T1> "ToDev" [|ary|] |> unbox 
            member this.ToHost ary : ArrayNDHostT<'T2>  = invokeHelperMethod<'T2> "ToHost" [|ary|] |> unbox 

            #else

            member this.Allocator shp : ArrayNDT<'T> = ArrayNDHost.newContiguous shp
            member this.ToDev ary : ArrayNDT<'T1> = box |> unbox 
            member this.ToHost ary : ArrayNDHostT<'T2>  = box |> unbox 
            #endif
            
            member this.Compiler = { new IUExprCompiler with 
                                        member this.Name = "Cuda"
                                        member this.Compile env exprs = CudaEval.cudaEvaluator env exprs }
            member this.DefaultLoc = LocDev
    }


