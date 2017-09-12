namespace SymTensor.Compiler.Cuda


open System
open Tensor
open SymTensor


/// Locks and registers all variables in a VarEnv for use with CUDA.
type VarEnvReg (varEnv: VarEnvT) =   
    let varLocks =
        varEnv
        |> Map.toList
        |> List.map (fun (name, ary) -> CudaRegMem.register (ary.Storage :?> ITensorHostStorage))

    interface IDisposable with
        member this.Dispose () =
            for lck in varLocks do
                (lck :> IDisposable).Dispose()
