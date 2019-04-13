namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor
open Tensor.Expr



type CompileEnv = {
    VarOffsetStrides: Map<VarName, int64 * int64 list> 
}

type CompileData = {
    Alloc:              AllocReq -> AllocStub
    Env:                CompileEnv
    ArgStubs:           Map<Arg, TensorStub> 
    OverwritableArgs:   Set<Arg>
    Expr:               BaseExpr
}


type ICompilableOp =
    /// Should compute the output stubs given the input stubs.
    abstract ChStubs: CompileData -> Map<Ch, TensorStub>


module BaseExprCompiler =

    /// Evaluates an expression tree using the specified function for evaluation
    /// of each expression given its arguments.
    let eval (fn: BaseExpr -> Map<Arg, 'D> -> Map<Ch, 'D>) (group: BaseExprGroup) =
        /// Evaluated values for each BaseExpr channel.
        let exprChValues = Dictionary<BaseExprCh, 'D> ()
        /// BaseExprs that have all their arguments evaluated and thus can be evaluated themselves.
        let evalQueue = Queue<BaseExpr> ()
        /// Values that are still missing but needed for evaluation of a BaseExpr.
        let missingValuesForExpr = Dictionary<BaseExpr, HashSet<BaseExprCh>> ()
        /// Dependants of an expression channel that are not yet evaluated.
        let unevaledDeps = Dictionary<BaseExprCh, HashSet<BaseExpr>> ()
                  
        // Build missing values for each expression and enqueue leafs.
        for expr in group.AllExprs do
            let dependingOn = HashSet<_> ()
            for KeyValue(_, arg) in expr.Args do
                dependingOn.Add arg |> ignore
            missingValuesForExpr.[expr] <- dependingOn

            if dependingOn.Count = 0 then
                evalQueue.Enqueue expr

            for ch in expr.Channels do
                unevaledDeps.[expr.[ch]] <- HashSet (group.Dependants expr.[ch])

        // Loop until all expressions are evaluated.
        while evalQueue.Count > 0 do
            // Get expression to evaluate and its argument values.
            let expr = evalQueue.Dequeue ()
            let argVals = expr.Args |> Map.map (fun _ argExpr -> exprChValues.[argExpr])

            // Evaluate.
            let chVals = fn expr argVals

            // Store the result value of the expression and update its dependants.
            for KeyValue(ch, value) in chVals do
                // Store channel value.
                exprChValues.[expr.[ch]] <- value

                // Update dependants.
                for dep in group.Dependants expr.[ch] do
                    let mv = missingValuesForExpr.[dep]
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
                        exprChValues.Remove arg |> ignore

        exprChValues |> Map.ofDictionary


    /// Walks over an expression tree in such a way that all arguments of an
    /// expression are evaluated first before that expression itself is evaluated.
    let iter (fn: BaseExpr -> unit) (group: BaseExprGroup) =
        let evalFn (expr: BaseExpr) (_arg: Map<Arg, unit>) =
            fn expr
            expr.Channels |> Set.toSeq |> Seq.map (fun ch -> ch, ()) |> Map.ofSeq
        eval evalFn group |> ignore


    // step 1: perform allocations
    let performLayouting (env: CompileEnv) (rootExpr: BaseExpr) =
        let group = BaseExprGroup [rootExpr]

        /// All allocation stubs.
        let allocStubs = ResizeArray<AllocStub> ()
        let allocUsers = Dictionary<AllocStub, Set<BaseExpr>> ()
        let allocUserChs = Dictionary<AllocStub, Set<BaseExprCh>> ()
        let alloc expr req =
            let alloc = {Req = req}
            allocUsers.Add (alloc, Set<BaseExpr> [expr])
            allocUserChs.Add (alloc, Set.empty)
            allocStubs.Add alloc
            alloc

        /// All tensor stubs.
        let tensorStubs = Dictionary<BaseExprCh, TensorStub> ()

        /// All expressions that an expression depends on transitively.
        // TODO: might need to clear values when they are not used anymore to save memory.
        let depending = Dictionary<BaseExpr, Set<BaseExpr>> ()

        // Walk over expression tree.
        group |> iter (fun expr ->    
            let argStubs = expr.Args |> Map.map (fun _ argExpr -> tensorStubs.[argExpr])

            /// All expression channels this expression transitively depends on.
            let exprDepending =
                if Map.isEmpty expr.Args then 
                    Set.empty
                else
                    expr.Args
                    |> Map.toSeq
                    |> Seq.map (fun (_arg, argExpr) -> Set.add argExpr.Expr depending.[argExpr.Expr])
                    |> Set.unionMany
            depending.[expr] <- exprDepending

            // Add expression to users of its argument stubs.
            for KeyValue(_, argStub) in argStubs do
                match argStub.Storage with
                | StorageStub.Allocated allocStub -> 
                    allocUsers.[allocStub] <- allocUsers.[allocStub] |> Set.add expr 
                | _ -> ()

            // Compute set of overwritable arguments for in-place operations.
            // An argument value can be overwritten with the result of this expression
            // if no other expression uses this argument value after this expression.
            let overwritableArgs =
                expr.Args
                |> Map.toSeq
                |> Seq.filter (fun (arg, argExprCh) ->
                    // Arguments that share the same storage cannot be overwritten.
                    expr.Args |> Map.toSeq |> Seq.exists (fun (otherArg, otherArgExprCh) ->
                        arg <> otherArg && tensorStubs.[otherArgExprCh] = tensorStubs.[argExprCh])
                    |> not)
                |> Seq.filter (fun (_arg, argExprCh) ->
                    match tensorStubs.[argExprCh].Storage with
                    | StorageStub.Allocated argAlloc ->
                        // Arguments uses an allocated storage: overwriting it might be possible.
                        // Get all expression channels that use the allocated storage of that argument (so far).
                        let exprChsUsingArgStorage = allocUserChs.[argAlloc] 
                        // Get their (complete) dependants.
                        let deps =
                            exprChsUsingArgStorage
                            |> Set.toSeq
                            |> Seq.collect group.Dependants
                            |> Seq.filter ((<>) expr)
                        // Check if we depend on all their dependants.
                        // If this is the case, we can overwrite that argument because all other users
                        // have already been evaluated before we can be evaluated.
                        deps |> Seq.forall exprDepending.Contains
                    | _ -> 
                        // Argument uses other storage. Do not overwrite it.
                        false)
                |> Seq.map fst
                |> Set.ofSeq

            // TODO:
            // - verify that this works and write down proof
                
            // Compute channel stubs given argument stubs.
            let chStubs =
                match expr.Op with
                | :? ICompilableOp as cop -> 
                    // Let op compute its output stubs given argument stubs.
                    let data = {
                        Alloc = alloc expr
                        Env = env
                        ArgStubs = argStubs
                        OverwritableArgs = overwritableArgs
                        Expr = expr
                    }
                    cop.ChStubs data
                | _ ->
                    // Op cannot compute stubs, generate dynamic stubs for all
                    // its output channels. 
                    expr.Channels
                    |> Set.toSeq
                    |> Seq.map (fun ch -> 
                        ch, {
                            Shape = expr.Shapes.[ch] |> Shape.eval
                            TypeName = expr.TypeNames.[ch]
                            Dev = expr.Devs.[ch]
                            OffsetStride = None
                            Storage = StorageStub.Dynamic
                        })
                    |> Map.ofSeq

            // Add expression to users of its channel stubs.
            for KeyValue(ch, chStub) in chStubs do
                match chStub.Storage with
                | StorageStub.Allocated allocStub -> 
                    allocUsers.[allocStub] <- allocUsers.[allocStub] |> Set.add expr 
                    allocUserChs.[allocStub] <- allocUserChs.[allocStub] |> Set.add expr.[ch]
                | _ -> ()

            // Store channel stubs.
            for KeyValue(ch, chStub) in chStubs do
                tensorStubs.[expr.[ch]] <- chStub
            )


        // next step? 
        // - Try to write the allocation stub function for some ops.
        // - Before that: switch to op extenders here?

        ()


type CompileTools () =

    /// Allocates tensor stubs for all output channels of the op.
    /// Argument stubs will be reused if tryInplace is true.
    static member chStubs (data: CompileData, ?tryInplace: bool) =
        let tryInplace = defaultArg tryInplace false

        let op = data.Expr.Op
        let mutable availArgs = data.OverwritableArgs
        
        /// Returns an overwritable argument TensorStub that matches the
        /// given specifications, if available.
        let tryUseArg typeName dev shape =
            availArgs
            |> Set.toSeq
            |> Seq.filter (fun arg ->
                // only reuse stubs with allocated storage
                match data.ArgStubs.[arg].Storage with
                | StorageStub.Allocated _ -> true
                | _ -> false)
            |> Seq.filter (fun arg ->
                // only reuse stubs with non-aliased elements 
                TensorStub.isNotAliased data.ArgStubs.[arg])
            |> Seq.tryFind (fun arg ->
                let argExpr = op.Args.[arg]
                argExpr.TypeName = typeName &&
                argExpr.Dev = dev &&
                Shape.eval argExpr.Shape = shape)
            |> Option.map (fun arg -> 
                availArgs <- availArgs |> Set.remove arg
                data.ArgStubs.[arg])

        // Other problem:
        // - it could happen that two arguments use the same storage and thus get overwritten twice.
        // - also 

        op.Channels
        |> Set.toSeq
        |> Seq.map (fun ch ->
            let typeName = op.TypeNames.[ch]
            let shape = Shape.eval op.Shapes.[ch]
            let dev = op.Devs.[ch]
            let stub = 
                if tryInplace then
                    match tryUseArg typeName dev shape with
                    | Some inplaceStub -> inplaceStub
                    | None -> TensorStub.alloc (data.Alloc, typeName, shape, dev)
                else
                    TensorStub.alloc (data.Alloc, typeName, shape, dev)
            ch, stub)
        |> Map.ofSeq




