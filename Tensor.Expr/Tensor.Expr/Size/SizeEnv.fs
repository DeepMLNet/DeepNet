namespace Tensor.Expr

open DeepNet.Utils


/// Environment for resolving symbolic sizes.
type SizeEnv = Map<SizeSym, Size>


/// Environment for resolving symbolic sizes.
module SizeEnv =

    /// empty size symbol environment    
    let empty = Map.empty

    ///// prints the size symbol environment
    //let dump (env: SizeEnv) =
    //    for KeyValue(sym, value) in env do
    //        printfn "%-30s = %A" (SizeSym.name sym) value

    /// Add symbol value.
    let add sym value (env: SizeEnv) =
        env |> Map.add sym value 

    /// Remove symbol value.
    let remove sym (env: SizeEnv) =
        env |> Map.remove sym 

    /// merges two environments
    let merge (aEnv: SizeEnv) (bEnv: SizeEnv) =
        Map.join aEnv bEnv


