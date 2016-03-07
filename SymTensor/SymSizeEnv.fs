namespace SymTensor

open Basics


[<AutoOpen>]
module SymSizeEnvTypes =

    [<StructuredFormatDisplay("{Pretty}")>]
    type SymSizeEnvT = 
        {Inferred:       Map<SizeSymbolT, SizeSpecT>;
         Equalities:     (SizeSpecT * SizeSpecT) list;}

        member this.Pretty =
            sprintf "%A" (this.Inferred |> Map.toList)



module SymSizeEnv =
    
    let contradiction fmt = 
        Printf.ksprintf (fun msg -> failwithf "size inference contradiction: %s" msg) fmt

    let empty =
        {Inferred = Map.empty; Equalities = [];}

    let dump env =
        for KeyValue(sym, value) in env.Inferred do
            printfn "%-30s = %A" (SizeSymbol.name sym) value

    /// substitutes all inferred symbols into size and simplifies it
    let private substInf inferred size = 
        SizeSpec.substSymbols inferred size

    /// substitutes all symbols into the size and simplifies it
    let subst env size =
        substInf env.Inferred size

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
    let addInferred sym value inferred =
        if substInf inferred value = Base (Sym sym) then
            failwithf "inferrering %A = %A would introduce a loop" sym value

        match Map.tryFind sym inferred with
        | Some other ->
            if other = value then inferred
            else contradiction "%A must be %A, but was inferred to be %A previously" sym value other
        | None -> 
            inferred |> Map.add sym value 
            

    let tryGetInferred sym env = 
        Map.tryFind sym env.Inferred 


    let rec infer env =      
        let rec inferOne inferred (a, b) =
            let a = SizeSpec.substSymbols inferred a
            let b = SizeSpec.substSymbols inferred b
            let flexible = SizeSpec.isFlexible a || SizeSpec.isFlexible b

            match a, b with
            | Base (Fixed av), Base (Fixed bv) ->
                if av = bv then inferred
                else contradiction "fixed %d <> fixed %d" av bv
            | Base (Fixed av), Base (Sym bsym) ->
                inferred |> addInferred bsym a
            | Base (Fixed av), Broadcast ->
                if av = 1 then inferred
                else contradiction "fixed %d <> broadcast 1" av
            | Base (Fixed av), Multinom bm ->
                if flexible then inferred
                else contradiction "fixed %d <> multinom %A" av bm
            | _, Base (Fixed _) ->
                inferOne inferred (b, a)

            | Base (Sym asym), Base (Sym bsym) ->
                match SizeSymbol.isFlexible asym, SizeSymbol.isFlexible bsym with
                | _ when asym = bsym -> inferred
                | true, true -> inferred |> addInferred asym b
                | true, false -> inferred |> addInferred asym b 
                | false, true -> inferred |> addInferred bsym a
                | _ -> contradiction "symbol %A <> symbol %A" asym bsym
            | Base (Sym asym), Broadcast ->
                inferred |> addInferred asym b
            | Base (Sym asym), Multinom bm ->
                if SizeSymbol.isFlexible asym then inferred |> addInferred asym b
                else contradiction "symbol %A <> multinom %A" a bm
            | _, Base (Sym _) ->
                inferOne inferred (b, a)

            | Broadcast, Broadcast -> inferred
            | Broadcast, Multinom bm ->
                if flexible then inferred
                else contradiction "broadcast 1 <> multinom %A" bm
            | _, Broadcast ->
                inferOne inferred (b, a)

            | Multinom am, Multinom bm ->
                if am = bm || flexible then inferred                                
                else contradiction "multinom %A <> multinom %A" am bm

        let newEnv = {env with Inferred = List.fold inferOne env.Inferred env.Equalities}
        if newEnv <> env then infer newEnv
        else env


    /// requires size a to be equal to size b
    let needEqual a b env =
        {env with Equalities = (a, b) :: env.Equalities} |> infer

    /// requires shape a to be equal to shape b
    let needEqualShape (a: ShapeSpecT) (b: ShapeSpecT) env =
        if List.length a <> List.length b then
            contradiction "shape %A must be of same dimensionality as shape %A" a b
        List.fold2 (fun env sa sb -> needEqual sa sb env) env a b 

    /// merges two environments
    let merge aEnv bEnv =
        List.fold (fun mEnv (a, b) -> needEqual a b mEnv) aEnv bEnv.Equalities


    

