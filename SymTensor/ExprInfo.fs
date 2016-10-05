namespace SymTensor

open Basics
open Expr



type MultiChannelOpUsageT = MultiChannelOpT * List<ExprT>

type ExprInfoT (expr: ExprT) =
    
    // rebuilt expression so that equal subtrees point to the same object instance
    let expr =
        let knownExprs = Dictionary<ExprT, ExprT> (HashIdentity.Structural)   
        let rec doUnify expr =
            match knownExprs.TryFind expr with
            | Some knownExpr -> knownExpr
            | None ->
                let unifiedExpr =
                    match expr with
                    | Leaf _ -> expr
                    | Unary (op, a) -> Unary (op, doUnify a)
                    | Binary (op, a, b) -> Binary (op, doUnify a, doUnify b)
                    | Nary (op, es) -> Nary (op, es |> List.map doUnify)
                knownExprs.[expr] <- unifiedExpr
                unifiedExpr
        doUnify expr
    
    // build sets of dependants for each subexpression
    let dependants, usedChannels = 
        let processed = HashSet<ExprT> (HashIdentity.Reference)
        let dependants = Dictionary<ExprT, HashSet<ExprT>> (HashIdentity.Reference)       
        let usedChannels = Dictionary<MultiChannelOpUsageT, HashSet<ChannelT>> (HashIdentity.Structural)
        
        let addDependant node dependant =
            if not (dependants.ContainsKey node) then
                dependants.[node] <- HashSet<ExprT> (HashIdentity.Reference)
            dependants.[node].Add dependant |> ignore
        let addUsedChannel key channel =
            if not (usedChannels.ContainsKey key) then
                usedChannels.[key] <- HashSet<ChannelT> (HashIdentity.Structural)
            usedChannels.[key].Add channel |> ignore

        let rec doBuild expr =
            if not (processed.Contains expr) then
                // update used channel info
                match expr with
                | Nary (Channel (op, channel), es) -> addUsedChannel (op, es) channel
                | _ -> ()

                // update dependants recursively
                match expr with
                | Leaf _ -> ()
                | Unary (op, a) ->
                    addDependant a expr
                    doBuild a
                | Binary (op, a, b) ->
                    addDependant a expr
                    addDependant b expr
                    doBuild a
                    doBuild b
                | Nary (op, es) ->
                    for e in es do
                        addDependant e expr
                    for e in es do
                        doBuild e
                processed.Add expr |> ignore

        doBuild expr
        dependants, usedChannels      

    /// Contained expression.
    /// It is ensured that equal sub-expression are the same object instance.
    member this.Expr = expr 

    /// Returns all expressions that depend on expr.
    /// Comparison is done based on reference equality.
    member this.Dependants expr =
        match dependants.TryFind expr with
        | Some deps -> deps |> Set.ofSeq
        | None -> Set.empty

    /// Returns the list of used channles for the multi-channel op
    /// with the specified arguments.
    member this.UsedChannels ((op, args): MultiChannelOpUsageT) =
        match usedChannels.TryFind (op, args) with
        | Some chnls -> chnls |> Set.ofSeq
        | None -> Set.empty




