namespace SymTensor.Ops

open DeepNet.Utils


/// Provides information about a set of expressions (dependencies, channel usage).
type BaseExprGroup (exprs: BaseExpr list) =
       
    // build sets of dependants for each subexpression
    let dependants = 
        let processed = HashSet<BaseExpr> ()
        let dependants = Dictionary<BaseExprCh, HashSet<BaseExpr>> ()              
        let addDependant node dependant =
            if not (dependants.ContainsKey node) then
                dependants.[node] <- HashSet<_> ()
            dependants.[node].Add dependant |> ignore 
        let rec doBuild expr =
            if not (processed.Contains expr) then
                // update dependants recursively
                for KeyValue(_, arg) in expr.Args do
                    addDependant arg expr
                for KeyValue(_, arg) in expr.Args do
                    doBuild arg.Expr
                processed.Add expr |> ignore

        for expr in exprs do
            doBuild expr
        dependants

    // build sets of used channels
    let usedChannels = lazy (
        let processed = HashSet<BaseExpr> ()
        let usedChannels = Dictionary<BaseExpr, HashSet<Ch>> ()      
        let addUsedChannel key channel =
            if not (usedChannels.ContainsKey key) then
                usedChannels.[key] <- HashSet<Ch> ()
            usedChannels.[key].Add channel |> ignore
        let rec doBuild (expr: BaseExpr) =
            if not (processed.Contains expr) then
                // update used channel info
                for KeyValue(_, BaseExprCh(argCh, argExpr)) in expr.Args do
                    addUsedChannel argExpr argCh

                for KeyValue(_, argExpr) in expr.Args do
                    doBuild argExpr.Expr
                processed.Add expr |> ignore

        for expr in exprs do
            doBuild expr
        usedChannels      
    )

    /// Contained expressions.
    member this.Exprs = exprs

    /// Returns all expressions that depend on expr.
    /// Comparison is done based on reference equality.
    member this.Dependants expr =
        match dependants.TryFind expr with
        | Some deps -> deps :> IReadOnlyCollection<_>
        | None -> HashSet<_> () :> IReadOnlyCollection<_>

    /// Returns the list of used channles for the multi-channel op
    /// with the specified arguments.
    member this.UsedChannels (mcExpr: BaseExpr) =
        match usedChannels.Force().TryFind mcExpr with
        | Some chnls -> chnls |> Set.ofSeq
        | None -> Set.empty

