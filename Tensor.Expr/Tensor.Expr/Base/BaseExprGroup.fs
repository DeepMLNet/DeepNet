namespace Tensor.Expr.Base

open System.IO
open DeepNet.Utils


/// Provides information about a set of expressions (dependencies, channel usage) and
/// functions for efficient iteration over the expression tree.
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


    /// Walks over an expression tree in such a way that all arguments of an
    /// expression are processed first before that expression itself is evaluated.
    /// I.e. the expression tree is processed from leafs to roots.
    member group.Iter (fn: BaseExpr -> unit, ?allDepsOfExprChEvaled: BaseExprCh -> unit, ?allDepsOfExprEvaled: BaseExpr -> unit)  =

        let allDepsOfExprChEvaled = defaultArg allDepsOfExprChEvaled (fun _ -> ())
        let allDepsOfExprEvaled = defaultArg allDepsOfExprEvaled (fun _ -> ())

        /// BaseExprs that have all their arguments evaluated and thus can be evaluated themselves.
        let evalQueue = Queue<BaseExpr> ()
        /// Values that are still missing but needed for evaluation of a BaseExpr.
        let missingForExpr = Dictionary<BaseExpr, HashSet<BaseExprCh>> ()
        /// Dependants of an expression channel that are not yet evaluated.
        let unevaledDeps = Dictionary<BaseExprCh, HashSet<BaseExpr>> ()
        let unevaledDepExprs = Dictionary<BaseExpr, HashSet<Ch>> ()
                  
        // Build missing values for each expression and enqueue leafs.
        for expr in group.AllExprs do
            let dependingOn = HashSet<_> ()
            for KeyValue(_, arg) in expr.Args do
                dependingOn.Add arg |> ignore
            missingForExpr.[expr] <- dependingOn

            if dependingOn.Count = 0 then
                evalQueue.Enqueue expr

            for ch in expr.Channels do
                unevaledDeps.[expr.[ch]] <- HashSet (group.Dependants expr.[ch])
            unevaledDepExprs.[expr] <- HashSet expr.Channels

        // Loop until all expressions are evaluated.
        while evalQueue.Count > 0 do
            // Get expression to evaluate.
            let expr = evalQueue.Dequeue ()

            // Evaluate.
            fn expr 

            // Update the dependants of the evaluated expression.
            for ch in expr.Channels do
                // Update dependants.
                for dep in group.Dependants expr.[ch] do
                    let mv = missingForExpr.[dep]
                    mv.Remove expr.[ch] |> ignore

                    // Enqueue, if all arguments have evaluated values.
                    if mv.Count = 0 then
                        evalQueue.Enqueue dep

                // Update unevaled deps.
                for KeyValue(_, arg) in expr.Args do    
                    let ueDeps = unevaledDeps.[arg]
                    ueDeps.Remove expr |> ignore

                    // Remove value, if all dependants are evaluated.
                    if ueDeps.Count = 0 then
                        allDepsOfExprChEvaled arg

                        let ueDepExprs = unevaledDepExprs.[arg.Expr]
                        ueDepExprs.Remove arg.Channel |> ignore
                        if ueDepExprs.Count = 0 then
                            allDepsOfExprEvaled arg.Expr


    /// Evaluates an expression tree using the specified function for evaluation
    /// of each expression given its arguments.
    member group.Eval (fn: BaseExpr -> Map<Arg, 'D> -> Map<Ch, 'D>, ?clearInternalValues: bool) =
        let clearInternalValues = defaultArg clearInternalValues true

        /// Evaluated values of each expression channel.
        let exprChValues = Dictionary<BaseExprCh, 'D> ()

        /// Evaluates one expression.
        let evalFn (expr: BaseExpr) =
            let argVals = expr.Args |> Map.map (fun _ argExpr -> exprChValues.[argExpr])     
            let chVals = fn expr argVals            
            for KeyValue(ch, value) in chVals do
                exprChValues.[expr.[ch]] <- value

        /// Clears evaluated values that are not used anymore.
        let clearChValues (exprCh: BaseExprCh) =
            if clearInternalValues then 
                exprChValues.Remove exprCh |> ignore

        group.Iter (evalFn, allDepsOfExprChEvaled=clearChValues) 
        exprChValues |> Map.ofDictionary


    /// Walks over an expression tree in such a way that all dependants of an
    /// expression are processed first before that expression itself is evaluated.
    /// I.e. the expression tree is processed from roots towards leafs.
    member group.RevIter (fn: BaseExpr -> unit) =
        /// BaseExprs that have all their dependants evaluated and thus can be evaluated themselves.
        let evalQueue = Queue<BaseExpr> ()
        /// Values that are still missing but needed for evaluation of a BaseExpr.
        let missingForExpr = Dictionary<BaseExpr, HashSet<BaseExpr>> ()

        // Build missing values for each expression in the tree and enqueue roots.
        for expr in group.AllExprs do
            let dependingOn = HashSet<_> (group.Dependants expr)
            missingForExpr.[expr] <- dependingOn

            // enqueue roots
            if dependingOn.Count = 0 then
                evalQueue.Enqueue expr

        // Loop until all expressions are evaluated.
        while evalQueue.Count > 0 do
            // Get expression to evaluate.
            let expr = evalQueue.Dequeue ()

            // Evaluate.
            fn expr

            // Update arguments of evaluated expression.
            for KeyValue(_arg, argExprCh) in expr.Args do
                // Remove expression from the missing values set the argument expression.
                let argExpr = argExprCh.Expr
                let mv = missingForExpr.[argExpr]
                if mv.Remove expr && mv.Count = 0 then
                    // Enqueue, if all dependants have evaluated values.
                    evalQueue.Enqueue argExpr
                
    
    /// Writes the expression forest to a TextWriter and returns the ids assigned to each expression.
    static member dump (writer: TextWriter) (group: BaseExprGroup) =
        let mutable nextId = 0
        let exprIds = Dictionary<BaseExpr, int> ()
        let getId expr = exprIds.IGetOrAdd (expr, fun _ -> Util.returnAndIncr &nextId) 
        group.Iter (BaseExpr.dump writer getId)
        Map.ofDictionary exprIds



/// Incrementally builds transitive expression dependencies when iterating
/// over an expression tree.
type BaseExprTransitiveDependencies () =
    let depending = Dictionary<BaseExpr, Set<BaseExpr>> ()

    /// Call for each expression when iterating over an expression tree.
    member this.Process (expr: BaseExpr) =
        depending.[expr] <-
            if Map.isEmpty expr.Args then 
                Set.empty
            else
                expr.Args
                |> Map.toSeq
                |> Seq.map (fun (_arg, argExpr) -> Set.add argExpr.Expr depending.[argExpr.Expr])
                |> Set.unionMany

    /// Call once all the dependencies of an expression have been processed
    /// and its dependency information are no longer needed.
    member this.Remove (expr: BaseExpr) =
        depending.Remove expr |> ignore

    /// Gets the transitive dependencies for the specified expression.
    member this.Item
        with get (expr: BaseExpr) =
            depending.[expr]

