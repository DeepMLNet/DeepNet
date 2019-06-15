namespace Tensor.Expr.Compiler

open System.IO
open DeepNet.Utils
open Tensor.Backend
open Tensor.Expr
open Tensor.Expr.Base
open Tensor.Expr.Compiler



/// Propagates tensor stub wishes from the roots of the expression tree towards its nodes.
module StubWishing =

    /// Arguments for stub wish propagation.
    type Args = {
        /// Compile environment.
        Env:                CompileEnv
        /// Wishes for channel stubs.
        ChStubWishes:       Map<BaseExprCh, TensorStub>
        /// Expression forest to operate on.
        Group:              BaseExprGroup
        /// Allocation function.
        AllocFn:            AllocReq -> AllocStub        
    }
    
    type WishNode = {
        /// Corresponding expression
        Expr:                    BaseExpr
        /// Wishes made for tensor stubs of arguments.
        ArgStubWishes:           Map<Arg, TensorStub>
        /// Wishes made for tensors stubs of channels by dependants.
        ChStubWishes:            Map<Ch, TensorStub>
        /// Channel stubs commited to during wish propagation.
        ChStubs:                 Map<Ch, TensorStub>
        /// Performed allocations during wish propagation.
        Allocs:                  AllocStub list
    } with
        static member dump (writer: TextWriter) (wn: WishNode) =
            fprintfn writer "  ArgStubWishes:"
            for KeyValue(arg, stub) in wn.ArgStubWishes do
                fprintfn writer "    %A: %A" arg stub            
            fprintfn writer "  ChStubWishes:"
            for KeyValue(ch, stub) in wn.ChStubWishes do
                fprintfn writer "    [%A]: %A" ch stub
            fprintfn writer "  ChStubs:"
            for KeyValue(ch, stub) in wn.ChStubs do
                fprintfn writer "    [%A]: %A" ch stub
            let allocStr =
                wn.Allocs |> Seq.map (sprintf "%A") |> String.concat ","
            fprintfn writer "  Allocs: %s" allocStr
            
        static member dumpMap (writer: TextWriter) (exprIds: Map<BaseExpr, int>) (wns: Map<BaseExpr, WishNode>) =
            let ids =
                Map.keys wns
                |> Seq.map (fun expr -> exprIds.[expr])
                |> List.ofSeq
                |> List.sort
            let exprById =
                exprIds
                |> Map.toSeq
                |> Seq.map (fun (expr, id) -> id, expr)
                |> Map.ofSeq
            for id in ids do
                let expr = exprById.[id]
                fprintfn writer "#%d:" id
                WishNode.dump writer wns.[expr]
                fprintfn writer ""
            

    /// Perform propagation of tensor stubs from channels to arguments.
    /// Only wishes for non-aliased tensor stubs are honored.
    let perform (args: Args) : Map<BaseExpr, WishNode> =

        let filterWishes stubWishes =
            stubWishes
            |> Map.filter (fun _ ts -> TensorStub.isNotAliased ts)            

        /// Tensor stub wishes for an expression channel.
        let chStubWishes = 
            args.ChStubWishes 
            |> filterWishes 
            |> Dictionary<BaseExprCh, TensorStub> 
        
        // Wishes for each expression.
        let wishNodes = Dictionary<BaseExpr, WishNode> ()

        /// Propagates wishes from an expression to its arguments.
        let propagate (expr: BaseExpr) =
            // Assemble all channel wishes of an expression.
            let chWishes =
                expr.Channels
                |> Seq.choose (fun ch ->
                    chStubWishes.TryFind expr.[ch]
                    |> Option.map (fun wish -> ch, wish))
                |> Map.ofSeq

            // Allocate memory for the specified expression.
            let opAllocs = ResizeArray<AllocStub> ()
            let allocForOp req =
                let alloc = args.AllocFn req
                opAllocs.Add alloc
                alloc
                    
            let comp = 
                match expr.Op with
                | :? IStubWishingOp as op ->                    
                    // Op supports wish propagation.
                    // Let it calculate wishes for its arguments.
                    op.WishStubs {
                        Alloc = allocForOp 
                        Env = args.Env
                        Expr = expr
                        ChStubWishes = chWishes
                    }
                | _ ->
                    // Op does not support wish propagation. Do nothing.
                    {
                        ChStubs = Map.empty
                        ArgStubWishes = Map.empty
                    }

            // Check commited channel stubs for plausibility.
            for KeyValue(ch, chStub) in comp.ChStubs do
                let exprCh = expr.[ch]
                if not (chStub.Dev = exprCh.Dev && 
                        chStub.TypeName = exprCh.TypeName &&
                        chStub.Shape = Shape.eval exprCh.Shape) then
                    failwithf "Tensor stub %A for channel %A is not compatiable with expression %A."
                        chStub ch exprCh    

            // Propagate argument wishes to channels of argument expressions.
            for KeyValue(arg, argWish) in filterWishes comp.ArgStubWishes do
                let argExprCh = expr.Args.[arg]

                // Check wish for plausibility.
                if not (argWish.Dev = argExprCh.Dev && 
                        argWish.TypeName = argExprCh.TypeName &&
                        argWish.Shape = Shape.eval argExprCh.Shape) then
                    failwithf "Tensor stub wish %A for argument %A is not compatiable with %A."
                        argWish arg argExprCh

                // Check that, if a argument wish is made, the storage is either allocated
                // or comes from a channel with a commited stub.
                let isAlloced =
                    opAllocs |> Seq.exists (fun alloc -> argWish.Storage = StorageStub.Allocated alloc)
                let isFromCommitedCh =
                    comp.ChStubs |> Map.exists (fun ch chStub -> argWish.Storage = chStub.Storage)
                if not (isAlloced || isFromCommitedCh) then 
                    failwithf "Tensor stub wish %A for argument %A uses storage neither from a commited channel
                               nor self-allocated storage." argWish arg 
            
                // The first wish for an expression channel wins and subsequent 
                // wishes are ignored.
                if not (chStubWishes.ContainsKey argExprCh) then
                    chStubWishes.[argExprCh] <- argWish
                    
            // Assemble.
            wishNodes.[expr] <- {
                Expr = expr
                ArgStubWishes = comp.ArgStubWishes
                ChStubWishes = chWishes
                ChStubs = comp.ChStubs
                Allocs = List.ofSeq opAllocs
            }                        

        // Walk over expression tree from roots to leafs and propagate wishes.
        args.Group.RevIter propagate
        Map.ofDictionary wishNodes



/// Assignment of tensor stubs and action generation.
module StubAndActionAssignment =
  
    /// Arguments for tensor stub and action assignment.  
    type Args = {
        /// Compile environment.
        Env:                CompileEnv
        /// Results for tensor stub wishing.
        WishNodes:          Map<BaseExpr, StubWishing.WishNode>
        /// Expression forest to operate on.
        Group:              BaseExprGroup
        /// Allocation function.
        AllocFn:            AllocReq -> AllocStub             
    }
    
    type Result = {
        /// Stubs for all expression channels in the expression tree.
        ChStubs:            Map<BaseExprCh, TensorStub>
        /// All action nodes.
        ActionNodes:        ActionNode list
        /// Stubs for all expression channels of root expressions.
        ResultStubs:        Map<BaseExprCh, TensorStub>        
    }
    
    /// Perform tensor stub and action assignment for every expression in the forect.
    let perform (args: Args) : Result =

        /// Users of an allocation, either as argument, internally or as channel.
        let allocUsers = Dictionary<AllocStub, Set<BaseExpr>> ()
        /// Expression channels that use an allocation.
        let allocUserChs = Dictionary<AllocStub, Set<BaseExprCh>> ()
        /// Add expression channel as user of an allocation.
        let addAllocUserCh allocStub exprCh =
            let users = allocUserChs.GetOrDefault allocStub Set.empty
            allocUserChs.[allocStub] <- users |> Set.add exprCh 

        /// Assigned tensor stub for each expression channel.
        let tensorStubs = Dictionary<BaseExprCh, TensorStub> ()

        /// Action group for each expression.
        let actionGroupForExpr = Dictionary<BaseExpr, ActionNode> ()

        /// All expressions that an expression depends on transitively.
        let depending = BaseExprTransitiveDependencies ()

        /// Action groups that have no dependants.
        let leafActGrps = HashSet<ActionNode> ()

        // Store assigned channel stubs from wish propagation. 
        for KeyValue(expr, wishNode) in args.WishNodes do
            for KeyValue(ch, stub) in wishNode.ChStubs do
                tensorStubs.[expr.[ch]] <- stub
                match stub.Storage with
                | StorageStub.Allocated allocStub -> addAllocUserCh allocStub expr.[ch]
                | _ -> ()

        /// Processes an expression.
        let processExpr (expr: BaseExpr) =    
            /// All expression channels this expression transitively depends on.
            /// Due to the walk order over the expression tree all these expressions
            /// have already been processed.
            depending.Process expr
            
            /// Wishing node for this expression.
            let wishNode = args.WishNodes.[expr]

            /// Allocations for op.
            let opAllocs = ResizeArray<AllocStub> ()
            let allocForOp req =
                let alloc = args.AllocFn req
                opAllocs.Add alloc
                allocUsers.Add (alloc, Set<BaseExpr> [expr])
                allocUserChs.Add (alloc, Set.empty)            
                alloc                 
            
            /// Stubs for the arguments of the expression.
            let argStubs = expr.Args |> Map.map (fun _ argExprCh -> tensorStubs.[argExprCh])

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
                |> Seq.filter (fun (_arg, argExprCh) ->
                    // Only overwrite arguments with non-runtime tensor stubs to
                    // prevent the channels of the op from becoming runtime stubs.
                    not (TensorStub.isRuntime tensorStubs.[argExprCh]))
                |> Seq.filter (fun (arg, argExprCh) ->
                    // Arguments that share the same storage must not be overwritten.
                    expr.Args |> Map.toSeq |> Seq.exists (fun (otherArg, otherArgExprCh) ->
                        otherArg <> arg && tensorStubs.[otherArgExprCh].Storage = tensorStubs.[argExprCh].Storage)
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
                            |> Seq.collect args.Group.Dependants
                            |> Seq.filter ((<>) expr)
                        // Check if we depend on all their dependants *and* that they have been processed,
                        // i.e. if they exposed the allocation via an output channel it would already have been
                        // added to allocUserChs.
                        deps |> Seq.forall depending.[expr].Contains
                    | _ -> 
                        // Argument uses other storage. Do not overwrite it.
                        false)
                |> Seq.map fst
                |> Set.ofSeq                      
            
            // Compute channel stubs given argument stubs.
            let comp =
                match expr.Op with
                | :? ICompilableOp as cop -> 
                    // Let op compute its output stubs given argument stubs.
                    cop.Compile {
                        Alloc = allocForOp
                        Env = args.Env
                        ArgStubs = argStubs
                        OverwritableArgs = overwritableArgs
                        ChStubs = wishNode.ChStubs
                        ChStubWishes = wishNode.ChStubWishes
                        ArgStubWishes = wishNode.ArgStubWishes
                        Expr = expr
                    } 
                | _ ->
                    // Op cannot compute stubs, generate dynamic stubs for all
                    // its output channels. 
                    let chStubs =
                        expr.Channels
                        |> Set.toSeq
                        |> Seq.map (fun ch -> 
                            ch, {
                                Shape = expr.Shapes.[ch] |> Shape.eval
                                TypeName = expr.TypeNames.[ch]
                                Dev = expr.Devs.[ch]
                                OffsetStride = OffsetStride.Runtime (RuntimeStub ())
                                Storage = StorageStub.Temporary (RuntimeStub ())
                            })
                        |> Map.ofSeq
                    // TODO: ensure that channels do not use any argument storage
                    //       if they do, copy them.
                    let action = NonCompilableOpAction (expr, argStubs) :> IAction
                    {ChStubs=chStubs; Actions=action}

            // Check that pre-assigned channel stubs were accepted.
            for KeyValue(ch, preassignedChStub) in wishNode.ChStubs do
                match comp.ChStubs |> Map.tryFind ch with
                | Some stub when stub = preassignedChStub -> ()
                | _ ->
                    failwithf "Expression %A did not accept pre-assigned channel stub %A." 
                        expr preassignedChStub

            // Check channel stubs for plausibility and initialize dynamic stub users.
            for KeyValue(ch, chStub) in comp.ChStubs do
                let exprCh = expr.[ch]
                if not (chStub.Dev = exprCh.Dev && 
                        chStub.TypeName = exprCh.TypeName &&
                        chStub.Shape = Shape.eval exprCh.Shape) then
                    failwithf "Tensor stub %A for channel %A is not compatiable with expression %A."
                        chStub ch exprCh    

            // Compute dependencies.
            let dependsOn =
                args.Group.Depending expr
                |> Seq.map (fun dep -> actionGroupForExpr.[dep])  
                |> HashSet     
                
            // Store action group.
            let actGrp = {
                Expr = Some expr
                Action = comp.Actions
                DependsOn = dependsOn
                Dependants = HashSet<_> ()
                ChStubs = comp.ChStubs
                DevData = None
                Allocs = List.ofSeq opAllocs
            }
            actionGroupForExpr.[expr] <- actGrp

            // Add this action group to dependants of the action group it depends on.
            for dep in dependsOn do
                dep.Dependants.Add actGrp |> ignore

            // Add expression to users of its channel stubs.
            for KeyValue(ch, chStub) in comp.ChStubs do
                match chStub.Storage with
                | StorageStub.Allocated allocStub -> 
                    allocUsers.[allocStub] <- allocUsers.[allocStub] |> Set.add expr 
                    allocUserChs.[allocStub] <- allocUserChs.[allocStub] |> Set.add expr.[ch]
                | _ -> ()

            // Store channel stubs.
            for KeyValue(ch, chStub) in comp.ChStubs do
                tensorStubs.[expr.[ch]] <- chStub

        /// Called after all dependencies of an expression have been processed.
        let allDepsProcessed (expr: BaseExpr) =
            // Check if generated action group has no dependants.
            let actGrp = actionGroupForExpr.[expr]
            if actGrp.Dependants.Count = 0 then
                leafActGrps.Add actGrp |> ignore

            // Clear dependency set of expression to save memory.
            depending.Remove expr 

        // Process expression tree.
        args.Group.Iter (processExpr, allDepsOfExprEvaled=allDepsProcessed)

        // Create action group that depends on all action groups that have not dependants.
        // It signals that all action groups have been evaluated.
        let finishActGrp = {
            Expr = None
            Action = FinishAction ()
            DependsOn = leafActGrps
            Dependants = HashSet<_> ()
            ChStubs = Map.empty
            DevData = None
            Allocs = []
        }
        // Add finish action group as dependant.
        for leafActGrp in leafActGrps do
            leafActGrp.Dependants.Add finishActGrp |> ignore

        // Compose compilation result.
        let chStubs = Map.ofDictionary tensorStubs
        let resultStubs =
            args.Group.Exprs
            |> Seq.collect (fun resExpr -> 
                resExpr.Channels
                |> Seq.map (fun ch -> 
                    let resExprCh = resExpr.[ch]
                    resExprCh, chStubs.[resExprCh]))
            |> Map.ofSeq
        {
            ChStubs = Map.ofDictionary tensorStubs
            ActionNodes = actionGroupForExpr.Values |> Seq.toList
            ResultStubs = resultStubs 
        }



[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module ExecutionRecipe = 

    let make (env: CompileEnv) (group: BaseExprGroup) : ExecutionRecipe =
        
        let mutable nextAllocId = 0
        let allocFn (req: AllocReq) : AllocStub =
            {
                Id = Util.returnAndIncr &nextAllocId
                Req = req
            }
        
        // Dump expression.
        let sep = "\n\n" + String.replicate 80 "=" + "\n"
        let dump = 
            match env.DumpPath with
            | Some dumpPath ->
                let dumpDir = Path.GetDirectoryName dumpPath
                Directory.CreateDirectory dumpDir |> ignore
                let writer = new StreamWriter (dumpPath)
                fprintfn writer "Exprs:\n" 
                let exprIds = group |> BaseExprGroup.dump writer
                Some (writer, exprIds)
            | None -> None
                        
        // Create tensor stubs for external target values.
        let targetStubs =
            env.ExternalTargets
            |> Seq.map (fun exprCh ->
                let inGroup = group.Exprs |> List.exists (fun e -> e.Channels |> Set.contains exprCh.Channel)
                if not inGroup then
                    failwithf "External target %A is not part of expression group." exprCh
                exprCh, {
                    Shape = Shape.eval exprCh.Shape
                    TypeName = exprCh.TypeName
                    Dev = exprCh.Dev
                    OffsetStride = OffsetStride.Runtime (RuntimeStub())
                    Storage = StorageStub.External (RuntimeStub())
                })
            |> Map.ofSeq
            
        // Dump target stubs.
        match dump with
        | Some (writer, exprIds) ->
            fprintf writer "%s" sep
            fprintfn writer "TargetStubs:\n"
            for KeyValue(BaseExprCh(ch, expr), stub) in targetStubs do
                fprintfn writer "%d[%s]: %A" exprIds.[expr] (ch.ToString()) stub         
        | None -> ()
            
        // Propagate stub wishes.
        let wishNodes = StubWishing.perform {
            Env = env
            ChStubWishes = targetStubs
            Group = group
            AllocFn = allocFn
        }
        
        // Dump propagated stub wishes.
        match dump with
        | Some (writer, exprIds) ->
            fprintf writer "%s" sep            
            fprintfn writer "WishNodes:\n"
            StubWishing.WishNode.dumpMap writer exprIds wishNodes
        | None -> ()
        
        // Perform stub and action assignment.
        let stubAndActionAssignment = 
            StubAndActionAssignment.perform {
                Env = env
                WishNodes = wishNodes
                Group = group
                AllocFn = allocFn                
            }

        // Perform allocations.
        let allocs =
            env.AllocationRealizer env stubAndActionAssignment.ActionNodes
        
        // Create execute recipe.
        let recipe = {
            ChStubs = stubAndActionAssignment.ChStubs
            ActionNodes = stubAndActionAssignment.ActionNodes
            ResultStubs = stubAndActionAssignment.ResultStubs
            Allocs = allocs
        }
        
        // Dump execution recipe.
        match dump with
        | Some (writer, exprIds) ->
            fprintf writer "%s" sep            
            fprintfn writer "ExecutionRecipe:\n"
            ExecutionRecipe.dump writer exprIds recipe
            writer.Dispose()
        | None -> ()        

        recipe
    