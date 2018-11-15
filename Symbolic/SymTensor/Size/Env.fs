namespace SymTensor

open DeepNet.Utils


/// Environment for resolving symbolic sizes.
type SymSizeEnv = Map<SizeSymbol, SizeSpec>

/// Functions for working with SymSizeEnv.
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
    let substShape env (shape: ShapeSpec) : ShapeSpec =
        List.map (subst env) shape

    /// substitutes all symbols into the simplified range specification
    let substRange env (srs: SimpleRangesSpec<_>) = 
        srs
        |> List.map (function
                     | SimpleRangeSpec.SymStartSymEnd (s, fo) -> 
                         SimpleRangeSpec.SymStartSymEnd (subst env s, Option.map (subst env) fo)
                     | SimpleRangeSpec.DynStartSymSize (s, elems) ->
                         SimpleRangeSpec.DynStartSymSize (s, subst env elems))

    /// adds inferred symbol value
    let add sym value env =
        if subst env value = SizeSpec.Base (BaseSize.Sym sym) then
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

