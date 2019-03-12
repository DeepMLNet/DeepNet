namespace Tensor.Expr

open DeepNet.Utils


/// Environment for resolving symbolic sizes.
type SymSizeEnv = Map<SizeSym, Size>

/// Functions for working with SymSizeEnv.
module SymSizeEnv =

    /// empty size symbol environment    
    let empty = Map.empty

    /// prints the size symbol environment
    let dump env =
        for KeyValue(sym, value) in env do
            printfn "%-30s = %A" (SizeSym.name sym) value

    /// substitutes all symbols into the size and simplifies it
    let subst env size =
        Size.substSyms env size

    /// substitutes all symbols into the shape and simplifies it
    let substShape env (shape: ShapeSpec) : ShapeSpec =
        List.map (subst env) shape

    /// substitutes all symbols into the simplified range specification
    let substRange env (srs: SimpleRangesSpec) = 
        srs
        |> List.map (function
                     | SimpleRangeSpec.SymStartSymEnd (s, fo) -> 
                         SimpleRangeSpec.SymStartSymEnd (subst env s, Option.map (subst env) fo)
                     | SimpleRangeSpec.DynStartSymSize (s, elems) ->
                         SimpleRangeSpec.DynStartSymSize (s, subst env elems))

    /// Add symbol value.
    let add sym value env =
        env |> Map.add sym value 

    /// Remove symbol value.
    let remove sym env =
        env |> Map.remove sym

    /// merges two environments
    let merge aEnv bEnv =
        Seq.fold (fun mEnv (a, b) -> add a b mEnv) aEnv (bEnv |> Map.toSeq)

