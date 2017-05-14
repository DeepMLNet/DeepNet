namespace SymTensor

open Tensor.Utils

[<AutoOpen>]
module SymSizeEnvTypes =
    type SymSizeEnvT = Map<SizeSymbolT, SizeSpecT>


module SymSizeEnv =

    /// empty size symbol environment    
    let empty = Map.empty

    /// prints the size symbol environment
    let dump env =
        for KeyValue(sym, value) in env do
            printfn "%-30s = %A" (SizeSymbol.name sym) value

    /// substitutes all symbols into the size and simplifies it
    let subst env size =
        SizeSpec.substSymbols env size

    /// substitutes all symbols into the shape and simplifies it
    let substShape env (shape: ShapeSpecT) : ShapeSpecT =
        List.map (subst env) shape

    /// substitutes all symbols into the simplified range specification
    let substRange env (srs: SimpleRangesSpecT<_>) = 
        srs
        |> List.map (function
                     | SRSSymStartSymEnd (s, fo) -> 
                         SRSSymStartSymEnd (subst env s, Option.map (subst env) fo)
                     | SRSDynStartSymSize (s, elems) ->
                         SRSDynStartSymSize (s, subst env elems))

    /// adds inferred symbol value
    let add sym value env =
        if subst env value = Base (Sym sym) then
            failwithf "inferrering %A = %A would introduce a loop" sym value

        match Map.tryFind sym env with
        | Some other ->
            if other = value then env
            else failwithf "%A must be %A, but was inferred to be %A previously" sym value other
        | None -> 
            env |> Map.add sym value 
            
    let tryGetInferred sym env = 
        Map.tryFind sym env

    /// merges two environments
    let merge aEnv bEnv =
        Seq.fold (fun mEnv (a, b) -> add a b mEnv) aEnv (bEnv |> Map.toSeq)


    

