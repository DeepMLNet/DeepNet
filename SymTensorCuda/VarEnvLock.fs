namespace SymTensor.Compiler.Cuda


open ArrayNDNS
open SymTensor



/// Locks all variables in a VarEnv.
type VarEnvLock (varEnv: VarEnvT) =   
    let varLocks =
        varEnv
        |> Map.toList
        |> List.map (fun (name, ary) -> NDArrayLock.lock ary)

    interface IDisposable with
        member this.Dispose () =
            for lck in varLocks do
                (lck :> IDisposable).Dispose()
