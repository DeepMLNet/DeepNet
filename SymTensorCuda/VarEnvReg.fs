namespace SymTensor.Compiler.Cuda


open System
open ArrayNDNS
open SymTensor


/// Locks and registers all variables in a VarEnv for use with CUDA.
type VarEnvReg (varEnv: VarEnvT) =   
    let varLocks =
        varEnv
        |> Map.toList
        |> List.map (fun (name, ary) -> ArrayNDHostReg.lock (ary :?> IArrayNDHostT))

    interface IDisposable with
        member this.Dispose () =
            for lck in varLocks do
                (lck :> IDisposable).Dispose()
