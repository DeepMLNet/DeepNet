namespace Tensor.Expr.Compiler.Cuda

open System
open System.Reflection

open Tensor
open Tensor.Cuda
open Tensor.Utils
open DeepNet.Utils

open Tensor.Expr
open UExprTypes
open Tensor.Expr.Compiler


module CudaEval =

    /// CUDA GPU expression evaluator
    let cudaEvaluator (compileEnv: CompileEnvT) (uexprs: UExprT list)  =
        let channels = uexprs |> List.mapi (fun i _ -> sprintf "EXPR%d" i)
        let channelExprs = List.zip channels uexprs |> Map.ofList

        // build recipe and create workspace
        let rcpt, diagnostics = 
            {CompileEnv=compileEnv; UExprs=channelExprs; OwnerUExpr=None}
            |> CudaRecipe.buildFromDesc 
        let workspace = new CudaExprWorkspace (rcpt)

        // evaluator
        let evalFn = fun (evalEnv: EvalEnv) ->           
            // create arrays for results and add them to VarEnv
            let resArrays = rcpt.ChannelAllocators |> Map.map (fun _ alloc -> alloc ())
            let varEnv =
                (evalEnv.VarEnv, channels)
                ||> List.fold (fun varEnv ch -> 
                                    match rcpt.ChannelVars.[ch] with
                                    | Some vs -> varEnv |> VarEnv.addVarSpec vs resArrays.[ch]
                                    | None -> varEnv)

            // evaluate
            workspace.Eval varEnv
            channels |> List.map (fun ch -> resArrays.[ch])

        evalFn, Some diagnostics


[<AutoOpen>]
module CudaEvalTypes =

    type private HelperT =
        static member Allocator<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                (shp: NShape) : Tensor<'T> =
            Tensor<'T> (shp, CudaTensor.Dev)

        static member ToDev<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                (ary: Tensor<'T>) : Tensor<'T> =
            CudaTensor.transfer ary

        static member ToHost<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                (ary: Tensor<'T>) : Tensor<'T> =
            HostTensor.transfer ary


    /// Evaluates the model on a CUDA GPU.
    let DevCuda = { 
        new IDevice with
            
            #if !CUDA_DUMMY
            member this.Allocator shp : Tensor<'T>    = Generic.callGeneric<HelperT, Tensor<'T>> "Allocator" [typeof<'T>] (shp)
            member this.ToDev ary : Tensor<'T1>       = Generic.callGeneric<HelperT, Tensor<'T1>> "ToDev" [typeof<'T1>] (ary)
            member this.ToHost ary : Tensor<'T2>      = Generic.callGeneric<HelperT, Tensor<'T2>> "ToHost" [typeof<'T2>] (ary)

            #else

            member this.Allocator shp : ArrayNDT<'T> = ArrayNDHost.newContiguous shp
            member this.ToDev ary : ArrayNDT<'T1> = box |> unbox 
            member this.ToHost ary : ArrayNDHostT<'T2>  = box |> unbox 
            #endif
            
            member this.Compiler = { new IUExprCompiler with 
                                        member this.Name = "Cuda"
                                        member this.Compile env exprs = CudaEval.cudaEvaluator env exprs }
            member this.DefaultLoc = CudaTensor.Dev
            member this.DefaultFactory = this.Compiler, {CompileEnv.empty with ResultLoc=this.DefaultLoc}

    }


