namespace SymTensor

open Basics


[<AutoOpen>]
module SymSizeEnvTypes =

    type SymSizeEnvT = {
        Inferred:       Map<SizeSymbolT, SizeSpecT>;
        Equalities:     (SizeSpecT * SizeSpecT) list;
    }



module SymSizeEnv =
    
    let contradictionFail msg =
        failwithf "size inference contradiction: %s" msg

    let contradiction fmt = 
        Printf.ksprintf contradictionFail fmt

    let empty =
        {Inferred = Map.empty; Equalities = [];}

    let dump env =
        for KeyValue(sym, value) in env.Inferred do
            printfn "%15s = %A" (SizeSymbol.name sym) value

    /// substitutes all inferred symbols into size and simplifies it
    let rec subst inferred size = 
        match size with
        | Base (Sym sym) ->
            match Map.tryFind sym inferred with
            | Some s -> subst inferred s
            | None -> size
        | Base (Fixed _) -> size
        | Broadcast -> size
        | Product p ->
            let substProduct p sBase sPower = 
                match subst inferred (Base (Sym sBase)) with
                | Base b -> SizeProduct.mult b sPower p
                | Broadcast -> p
                | Product op ->
                    // product substitution is currently unsupported
                    printfn "Warning: product into product substituion is unsupported"
                    SizeProduct.mult (Sym sBase) sPower p

            p.Symbols
                |> Map.fold substProduct (SizeProductT p.Factor) 
                |> Product
        |> SizeSpec.simplify
//
//    let rec simplifyInferred inferred =
//        let subst =
//            inferred
//            |> Map.map (fun sym value ->
//                match value with
//                | Base (Sym otherSym) ->
//                    match Map.tryFind otherSym inferred with
//                    | Some otherVal -> otherVal
//                    | None -> value
//                | _ -> value)
//        if subst <> inferred then simplifyInferred subst
//        else inferred

    let addInferred sym value inferred =
        match Map.tryFind sym inferred with
        | Some other ->
            if other = value then inferred
            else contradiction "%A must be %A, but was inferred to be %A previously" sym value other
        | None -> 
            inferred |> Map.add sym value 
            //inferred |> Map.add sym value |> simplifyInferred
            

    let tryGetInferred sym env = 
        Map.tryFind sym env.Inferred 


    let rec infer env =      
        let rec inferOne inferred (a, b) =
            match subst inferred a, subst inferred b with
            | Base (Fixed av), Base (Fixed bv) ->
                if av = bv then inferred
                else contradiction "fixed %d <> fixed %d" av bv
            | Base (Fixed av), Base (Sym bsym) ->
                inferred |> addInferred bsym a
            | Base (Fixed av), Broadcast ->
                if av = 1 then inferred
                else contradiction "fixed %d <> broadcast 1" av
            | Base (Fixed av), Product bp ->
                contradiction "fixed %d <> product %A" av bp
            | _, Base (Fixed _) ->
                inferOne inferred (b, a)

            | Base (Sym asym), Base (Sym bsym) ->
                match SizeSymbol.flexible asym, SizeSymbol.flexible bsym with
                | true, true -> inferred |> addInferred asym b
                | true, false -> inferred |> addInferred asym b 
                | false, true -> inferred |> addInferred bsym a
                | false, false when asym = bsym -> inferred
                | _ -> contradiction "symbol %A <> symbol %A" asym bsym
            | Base (Sym asym), Broadcast ->
                inferred |> addInferred asym b
            | Base (Sym asym), Product bp ->
                if SizeSymbol.flexible asym then inferred |> addInferred asym b
                else contradiction "symbol %A <> product %A" a b 
            | _, Base (Sym _) ->
                inferOne inferred (b, a)

            | Broadcast, Broadcast -> inferred
            | Broadcast, Product bp ->
                contradiction "broadcast 1 <> product %A" b
            | _, Broadcast ->
                inferOne inferred (b, a)

            | Product ap, Product bp ->
                if ap = bp then inferred                                
                else contradiction "product %A <> product %A" a b

        let newEnv = {env with Inferred = List.fold inferOne env.Inferred env.Equalities}
        if newEnv <> env then infer newEnv
        else env


    /// requires size a to be equal to size b
    let needEqual a b env =
        {env with Equalities = (a, b) :: env.Equalities} |> infer

    /// requires shape a to be equal to shape b
    let needEqualShape (a: ShapeSpecT) (b: ShapeSpecT) env =
        List.fold2 (fun env sa sb -> needEqual sa sb env) env a b 

    /// merges two environments
    let merge aEnv bEnv =
        List.fold (fun mEnv (a, b) -> needEqual a b mEnv) aEnv bEnv.Equalities


    

