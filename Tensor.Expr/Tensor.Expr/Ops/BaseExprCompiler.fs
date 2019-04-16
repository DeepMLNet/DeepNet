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
    ChStubWishes:       Map<Ch, TensorStub>
    Expr:               BaseExpr
}


type ICompilableOp =
    /// Should compute the output stubs given the input stubs.
    abstract ChStubs: CompileData -> Map<Ch, TensorStub>

type ITensorStubWishPropagatingOp =
    abstract PropagateWishes: Map<Ch, TensorStub> -> Map<Arg, TensorStub>


module BaseExprCompiler =

    /// Propagates tensor stub wishes from the roots of the expression tree towards its nodes.
    let propagateStubWishes (env: CompileEnv) (stubWish: Map<BaseExprCh, TensorStub>) (group: BaseExprGroup) =

        /// Tensor stub wishes for an expression channel.
        let stubWish = Dictionary<BaseExprCh, TensorStub> stubWish

        /// Propagates wishes from an expression to its arguments.
        let propagate (expr: BaseExpr) =
            match expr.Op with
            | :? ITensorStubWishPropagatingOp as op ->
                // Op supports wish propagation.

                // Assemble all channel wishes of an expression.
                let chWishes =
                    expr.Channels
                    |> Seq.choose (fun ch ->
                        stubWish.TryFind expr.[ch]
                        |> Option.map (fun wish -> ch, wish))
                    |> Map.ofSeq

                // Let op calculate wishes for its arguments.
                let argWishes = op.PropagateWishes chWishes

                // Store argument wishes.
                for KeyValue(arg, argWish) in argWishes do
                    let argExprCh = expr.Args.[arg]
                    // The first wish for an expression channel wins and subsequent 
                    // wishes are ignored.
                    if not (stubWish.ContainsKey argExprCh) then
                        stubWish.[argExprCh] <- argWish

            | _ -> 
                // Op does not support wish propagation. Do nothing.
                ()

        // Walk over expression tree from roots to leafs and propagate wishes.
        group |> BaseExprGroup.revIter propagate

        stubWish |> Map.ofDictionary


    /// Assigns a tensor stub to each expression channel in the expression tree.
    let assignStubs (env: CompileEnv) (stubWish: Map<BaseExprCh, TensorStub>) (group: BaseExprGroup) =

        /// All allocation stubs.
        let allocStubs = ResizeArray<AllocStub> ()
        /// Users of an allocation, either as argument, internally or as channel.
        let allocUsers = Dictionary<AllocStub, Set<BaseExpr>> ()
        /// Expression channels that use an allocation.
        let allocUserChs = Dictionary<AllocStub, Set<BaseExprCh>> ()

        /// Allocate memory for the specified expression.
        let alloc expr req =
            let alloc = {Req = req}
            allocUsers.Add (alloc, Set<BaseExpr> [expr])
            allocUserChs.Add (alloc, Set.empty)
            allocStubs.Add alloc
            alloc

        /// Assigned tensor stub for each expression channel.
        let tensorStubs = Dictionary<BaseExprCh, TensorStub> ()

        /// All expressions that an expression depends on transitively.
        let depending = Dictionary<BaseExpr, Set<BaseExpr>> ()

        /// Processes an expression.
        let processExpr (expr: BaseExpr) =    
            /// Stubs for the arguments of the expression.
            let argStubs = expr.Args |> Map.map (fun _ argExpr -> tensorStubs.[argExpr])

            /// All expression channels this expression transitively depends on.
            /// Due to the walk order over the expression tree all these expressions
            /// have already been processed.
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
            // An argument value can be overwritten with the result of this expression,
            // if it is guaranteed that all other expressions using the argument value are
            // finished evaluating before the evaluation of this expression starts.
            // Additionally, the argument must not share the same storage with another
            // argument.
            let overwritableArgs =
                expr.Args
                |> Map.toSeq
                |> Seq.filter (fun (arg, argExprCh) ->
                    // Arguments that share the same storage must not be overwritten.
                    expr.Args |> Map.toSeq |> Seq.exists (fun (otherArg, otherArgExprCh) ->
                        otherArg <> arg && tensorStubs.[otherArgExprCh] = tensorStubs.[argExprCh])
                    |> not)
                |> Seq.filter (fun (_arg, argExprCh) ->
                    // Only overwrite arguments that do not have aliased elements,
                    // for example due to broadcasting.
                    TensorStub.isNotAliased tensorStubs.[argExprCh])
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
                        // Check if we depend on all their dependants *and* that they have been processed,
                        // i.e. if they exposed the allocation via an output channel it would already have been
                        // added to allocUserChs.
                        deps |> Seq.forall exprDepending.Contains
                    | _ -> 
                        // Argument uses other storage. Do not overwrite it.
                        false)
                |> Seq.map fst
                |> Set.ofSeq
                
            // Assemble all channel wishes for the expression.
            let chWishes =
                expr.Channels
                |> Seq.choose (fun ch ->
                    stubWish.TryFind expr.[ch]
                    |> Option.map (fun wish -> ch, wish))
                |> Map.ofSeq

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
                        ChStubWishes = chWishes
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

        /// Called after all dependencies of an expression have been processed.
        let allDepsOfExprEvaled (expr: BaseExpr) =
            // Clear dependency set of expression to save memory.
            depending.Remove expr |> ignore

        // Process expression tree.
        group |> BaseExprGroup.iter processExpr (fun _ -> ()) allDepsOfExprEvaled 

        // Return assigned tensor stub for each expression channel.
        Map.ofDictionary tensorStubs 


    /// Generates action items for each expression in the expression tree.
    let generateActions (env: CompileEnv) (stubs: Map<BaseExprCh, TensorStub>) (group: BaseExprGroup) =
        ()

        // Shall we build an execution plan or what would be next step?
        // One possible execution would be passing and storing of lambda functions,
        // but how would this be compatible with later code generation?
        // Also we need to make sure that it will work both for CUDA and host execution
        // and code generation...
    
        // We could return a list of IAction items that are then enqueued properly.
        // These IAction items could also have a method for code generation for
        // the appropriate language.
        // Synchronization would have to be handled by the overall code generator
        // for a specific language and technology.

        // What about the up-propagation of storage wishes?
        // This could be important to avoid copies for loop expressions and parameter 
        // updates of optimizers.

        // Where would wishes come from?
        // - basically passed in externally since StoreToVar is gone
        // - i.e. usually a bundle will be used for evaluation and it could
        //   pass in the desired layout for the variables
        // - so we could accept a map of desired TensorStubs for expressions
        //   and propagate them upwards.
        // - No allocations should be possible.
        // - So up-propagation would only go through to the first allocation.
        // - memory saving is probably no issue there, so the first op
        //   that does actual work can stop the propagation
        // - so, the up-propagation interface should be optional
        // - however, the wishes should be presented to every op




type CompileTools () =

    /// Allocates tensor stubs for all output channels of the op.
    /// Argument stubs will be reused if tryInplace is true.
    static member chStubs (data: CompileData, ?tryInplace: bool) =
        let tryInplace = defaultArg tryInplace false

        let op = data.Expr.Op
        let mutable overwritableArgs = data.OverwritableArgs
        
        /// Returns an overwritable argument TensorStub that matches the
        /// given specifications, if available.
        let tryUseArg typeName dev shape =
            overwritableArgs
            |> Set.toSeq
            |> Seq.tryFind (fun arg ->
                // match type, device and shape
                let argExpr = op.Args.[arg]
                argExpr.TypeName = typeName &&
                argExpr.Dev = dev &&
                Shape.eval argExpr.Shape = shape)
            |> Option.map (fun arg -> 
                // remove from set of overwritable arguments
                overwritableArgs <- overwritableArgs |> Set.remove arg
                data.ArgStubs.[arg])

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




