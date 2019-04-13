namespace Tensor.Expr.Ops

open DeepNet.Utils


/// Provides information about a set of expressions (dependencies, channel usage).
type BaseExprGroup (exprs: BaseExpr list) =
       
    // build sets of dependants for each subexpression
    let dependants = lazy (
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
    )

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

    // build set of all subexpressions
    let allExprs = lazy (
        let processed = HashSet<BaseExpr> ()
        let allExprs = HashSet<BaseExpr> ()

        let rec build expr = 
            if not (processed.Contains expr) then
                allExprs.Add expr |> ignore
                for KeyValue(_, arg) in expr.Args do
                    allExprs.Add arg.Expr |> ignore
                    build arg.Expr
                processed.Add expr |> ignore

        for expr in exprs do
            build expr
        allExprs
    )

    /// Contained top-level expressions.
    member this.Exprs = exprs

    /// All (sub-)expressions contained within the expression.
    member this.AllExprs = 
        allExprs.Force() :> IReadOnlyCollection<_>

    /// Returns all expressions that depend directly on the specified expression channel.
    member this.Dependants (exprCh: BaseExprCh) : seq<BaseExpr> =
        match dependants.Force().TryFind exprCh with
        | Some deps -> deps :> _
        | None -> HashSet<_> () :> _

    /// Returns all expressions that depend directly on the specified expression.
    member this.Dependants (expr: BaseExpr) : seq<BaseExpr> =
        expr.Channels
        |> Set.toSeq
        |> Seq.collect (fun ch -> this.Dependants expr.[ch])

    /// Returns all expressions that expr directly depends on.
    member this.Depending (expr: BaseExpr) = seq {
        for KeyValue(_, arg) in expr.Args do
            yield arg.Expr
    }        

    /// Returns the list of used channles for the multi-channel op
    /// with the specified arguments.
    member this.UsedChannels (mcExpr: BaseExpr) =
        match usedChannels.Force().TryFind mcExpr with
        | Some chnls -> chnls |> Set.ofSeq
        | None -> Set.empty

