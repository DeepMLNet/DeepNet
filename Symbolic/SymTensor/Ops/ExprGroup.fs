namespace SymTensor.Ops

open DeepNet.Utils


/// Provides information about a set of expressions (dependencies, channel usage).
type BaseExprInfoT (exprs: BaseExpr list) =
    
    /// expression cache
    let knownExprs = Dictionary<BaseExpr, BaseExpr> () 

    // rebuilt expression so that equal subtrees point to the same object instance
    let exprs =
        let rec doUnify expr =
            match knownExprs.TryFind expr with
            | Some knownExpr -> knownExpr
            | None ->
                let unifiedExpr = expr.MapArgs doUnify
                knownExprs.[expr] <- unifiedExpr
                unifiedExpr
        exprs |> List.map doUnify 
    
    // build sets of dependants for each subexpression
    let dependants = 
        let processed = HashSet<BaseExpr> (HashIdentity.Reference)
        let dependants = Dictionary<BaseExpr, HashSet<BaseExpr>> (HashIdentity.Reference)              
        let addDependant node dependant =
            if not (dependants.ContainsKey node) then
                dependants.[node] <- HashSet<BaseExpr> (HashIdentity.Reference)
            dependants.[node].Add dependant |> ignore
        let rec doBuild expr =
            if not (processed.Contains expr) then
                // update dependants recursively
                for KeyValue(_, argExpr) in expr.Op.Args do
                    addDependant argExpr expr
                for KeyValue(_, argExpr) in expr.Op.Args do
                    doBuild argExpr
                processed.Add expr |> ignore

        for expr in exprs do
            doBuild expr
        dependants

    // build sets of used channels
    let usedChannels = lazy (
        let processed = HashSet<BaseExpr> (HashIdentity.Reference)
        let usedChannels = Dictionary<BaseMultiChannelExpr, HashSet<string>> (HashIdentity.Structural)      
        let addUsedChannel key channel =
            if not (usedChannels.ContainsKey key) then
                usedChannels.[key] <- HashSet<string> (HashIdentity.Structural)
            usedChannels.[key].Add channel |> ignore
        let rec doBuild (expr: BaseExpr) =
            if not (processed.Contains expr) then
                // update used channel info
                match expr.Op with
                | :? IMultiChannelArgsOp as mcArgsOp ->
                    for KeyValue(_, (argCh, argMCExpr)) in mcArgsOp.MultiChannelArgs do
                        addUsedChannel argMCExpr argCh
                | _ -> ()

                for KeyValue(_, argExpr) in expr.Op.Args do
                    doBuild argExpr
                processed.Add expr |> ignore

        for expr in exprs do
            doBuild expr
        usedChannels      
    )

    /// Contained expressions.
    /// It is ensured that equal sub-expression are the same object instance.
    member this.Exprs = exprs

    /// Returns all expressions that depend on expr.
    /// Comparison is done based on reference equality.
    member this.Dependants expr =
        match dependants.TryFind expr with
        | Some deps -> deps :> IReadOnlyCollection<_>
        | None -> HashSet<_> () :> IReadOnlyCollection<_>

    /// Returns all expressions that depend on expr.
    /// Comparison is done based on structural equality.
    member this.DependantsStructural expr =
        match knownExprs.TryFind expr with
        | Some expr -> this.Dependants expr
        | None -> HashSet<_> () :> IReadOnlyCollection<_>

    /// Returns the list of used channles for the multi-channel op
    /// with the specified arguments.
    member this.UsedChannels (mcExpr: BaseMultiChannelExpr) =
        match usedChannels.Force().TryFind mcExpr with
        | Some chnls -> chnls |> Set.ofSeq
        | None -> Set.empty

