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

    let rec simplifyInferred inferred =
        let subst =
            inferred
            |> Map.map (fun sym value ->
                match value with
                | Base (Sym otherSym) ->
                    match Map.tryFind otherSym inferred with
                    | Some otherVal -> otherVal
                    | None -> value
                | _ -> value)
        if subst <> inferred then simplifyInferred subst
        else inferred

    let addInferred sym value inferred =
        match Map.tryFind sym inferred with
        | Some other ->
            if other = value then inferred
            else contradiction "%A must be %A, but was inferred to be %A previously" sym value other
        | None -> 
            inferred |> Map.add sym value |> simplifyInferred
            

    let tryGetInferred sym env = 
        Map.tryFind sym env.Inferred 


    let infer env =
       
        let rec inferOne inferred (a, b) =
            match a, b with
            | Base (Fixed av), Base (Fixed bv) ->
                if av = bv then inferred
                else contradiction "fixed size %d <> %d" av bv
            | Base (Fixed av), Base (Sym bsym) ->
                match Map.tryFind bsym inferred with
                | Some bval -> inferOne inferred (a, bval)
                | None -> inferred |> addInferred bsym a
            | Base (Fixed av), Broadcast ->
                if av = 1 then inferred
                else contradiction "fixed size %d <> 1" av
            | Base (Fixed av), Product bp ->
                // what is todo?
                //   - fill in all inferred sizes into product?
                inferred // TODO
            | _, Base (Fixed _) ->
                inferOne inferred (b, a)

            | Base (Sym asym), _ ->
                match Map.tryFind asym inferred with
                | Some aval -> inferOne inferred (aval, b)                 
                | None ->
                    match b with
                    | Base (Fixed _) -> failwith "already handled above"
                    | Base (Sym bsym) ->
                        match Map.tryFind bsym inferred with
                        | Some bval -> inferOne inferred (a, bval)
                        | None ->
                            match SizeSymbol.flexible asym, SizeSymbol.flexible bsym with
                            | true, true -> inferred |> addInferred asym b
                            | true, false -> inferred |> addInferred asym b 
                            | false, true -> inferred |> addInferred bsym a
                            | false, false -> 
                                if asym = bsym then inferred
                                else contradiction "size symbol %A <> %A" asym bsym
                    | Broadcast ->
                        inferred |> addInferred asym b
                    | Product bp ->
                        // TODO
                        if SizeSymbol.flexible asym then inferred |> addInferred asym b
                        else contradiction "size %A <> %A" a b 
            | _, Base (Sym _) ->
                inferOne inferred (b, a)

            | Broadcast, Broadcast -> inferred
            | Broadcast, Product bp ->
                // TODO
                inferred
            | _, Broadcast ->
                inferOne inferred (b, a)

            | Product ap, Product bp ->
                // TODO
                inferred                                

        List.fold inferOne env.Inferred env.Equalities


    let needEqual a b env =
        {env with Equalities = (a, b) :: env.Equalities}

    

