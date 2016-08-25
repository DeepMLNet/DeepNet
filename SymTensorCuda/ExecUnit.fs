namespace SymTensor.Compiler

open System
open System.Diagnostics
open System.Collections.Generic

open Basics
open ArrayNDNS
open SymTensor

[<AutoOpen>]
module ExecUnitsTypes = 

    /// Id of an ExecUnitT
    type ExecUnitIdT = int

    /// a group of commands that must be executed sequentially
    type ExecUnitT<'e> = {
        Id:           ExecUnitIdT; 
        DependsOn:    ExecUnitIdT list; 
        Items:        'e list; 
        Expr:         UExprT;
        Manikins:     ArrayNDManikinT list;
        RerunAfter:   ExecUnitIdT list;
    }

    /// result of an evaluation request
    type EvalResultT = {
        ExecUnitId:     ExecUnitIdT; 
        Manikin:        ArrayNDManikinT; 
        Shared:         bool;
    }

    /// an evaluation request
    type EvalReqT = {
        Id:             int; 
        Expr:           UExprT; 
        Multiplicity:   int; 
        ReqManikin:     ArrayNDManikinT option; 
        OnCompletion:   EvalResultT -> unit
    }

    type MemAllocatorT = TypeNameT -> int -> MemAllocKindT -> MemManikinT

    type ExecItemsForOpArgs<'e> = {
        MemAllocator:       MemAllocatorT
        Target:             ArrayNDManikinT
        Op:                 UOpT
        Srcs:               (ArrayNDManikinT * bool) list
        SubmitInitItems:    'e list -> unit
    }

    type TraceItemsForExprArgs = {
        MemAllocator:       MemAllocatorT
        Target:             ArrayNDManikinT
        Expr:               UExprT
    }

    type TrgtGivenSrcsArgs = {
        MemAllocator:       MemAllocatorT
        TargetType:         TypeNameT
        TargetShape:        NShapeSpecT
        TargetRequest:      ArrayNDManikinT option
        Op:                 UOpT
        Srcs:               (ArrayNDManikinT * bool) list    
    }

    type SrcReqsArgs = {
        TargetShape:        NShapeSpecT
        TargetRequest:      ArrayNDManikinT option
        Op:                 UOpT
        SrcShapes:          NShapeSpecT list 
    }

    /// record containing functions called by the ExecUnitT generator
    type ExecUnitsGeneratorT<'e> = {
        ExecItemsForOp:         ExecItemsForOpArgs<'e> -> 'e list
        TraceItemsForExpr:      TraceItemsForExprArgs -> 'e list
        TrgtGivenSrcs:          TrgtGivenSrcsArgs -> ArrayNDManikinT * bool
        SrcReqs:                SrcReqsArgs -> ArrayNDManikinT option list
    }

    /// generated ExecUnits for an expression
    type ExecUnitsForExprT<'e> = {
        Expr:           UExprT
        ExecUnits:      ExecUnitT<'e> list
        Result:         EvalResultT
        MemAllocs:      MemAllocManikinT list
        InitItems:      'e list
    }


module ExecUnit =

    /// a collection of ExecUnits
    type Collection<'e> (eus: ExecUnitT<'e> seq) =
        
        let eus = eus |> List.ofSeq

        let byIdMap =
            eus
            |> Seq.map (fun eu -> eu.Id, eu)
            |> Map.ofSeq

        let dependants =
            let build =
                seq { for eu in eus -> eu.Id, List<ExecUnitT<_>>()}
                |> Map.ofSeq
            for eu in eus do
                for d in eu.DependsOn do
                    build.[d].Add eu
            build

        let sortedByDep =
            let unsorted = LinkedList eus
            let sorted = List<ExecUnitT<'e>> (Seq.length eus)
        
            while unsorted.Count > 0 do           
                let toAdd =
                    unsorted
                    |> Seq.filter (fun eu -> 
                        eu.DependsOn |> List.forall (fun depId ->
                            sorted |> Seq.exists (fun seu -> seu.Id = depId)))
            
                for eu in toAdd |> Seq.toList do
                    sorted.Add eu |> ignore
                    unsorted.Remove eu |> ignore

            sorted |> Seq.toList

        let storesByVar =
            let build = Dictionary<UVarSpecT, List<ExecUnitT<_>>> ()
            for eu in eus do
                match eu.Expr with
                | UExpr(UUnaryOp (StoreToVar vs), _, _, _) ->
                    if not (build.ContainsKey vs) then build.[vs] <- List ()
                    build.[vs].Add eu
                | _ -> ()
            build         
            

        /// list of all execution unit contained in this collection
        member this.ExecUnits = eus

        /// execution unit by id
        member this.ById id = byIdMap.[id]

        /// all ExecUnits that depend directly on eu
        member this.DependantsOf (eu: ExecUnitT<'e>) = dependants.[eu.Id].AsReadOnly () :> seq<_>

        /// all ExecUnits that eu directly depends on
        member this.DependsOn (eu: ExecUnitT<'e>) = eu.DependsOn |> Seq.map this.ById

        /// a list of ExecUnits in this collection so that an ExecUnit comes after all ExecUnits it depends on
        member this.SortedByDep = sortedByDep
       
        /// returns all successors of eu (execution units that depend (indirectly) on eu)
        member this.AllSuccessorsOf (eu: ExecUnitT<'e>) = seq {
            for deu in dependants.[eu.Id] do
                yield deu
                yield! this.AllSuccessorsOf deu
        }

        /// returns all predecessors of eu (execution units on which eu depends (indirectly))
        member this.AllPredecessorsOf (eu: ExecUnitT<'e>) = seq {
            for peuId in eu.DependsOn do
                let peu = this.ById peuId
                yield peu
                yield! this.AllPredecessorsOf peu
        }

        /// true if a is a successor of b. (a depends (indirectly) on b)
        member this.IsSuccessorOf (a: ExecUnitT<'e>) (b: ExecUnitT<'e>) =
            this.AllSuccessorsOf b |> Seq.exists (fun eu -> eu.Id = a.Id)

        /// Return all EUs above or equal to "eu" that access "storage" for the last time.
        member this.LastStorageAccess storage (eu: ExecUnitT<_>) = seq {
            let isAccessing =
                eu.Manikins
                |> List.exists (fun m -> m.Storage = storage)

            if isAccessing then yield eu
            else
                for deuId in eu.DependsOn do
                    yield! this.LastStorageAccess storage (this.ById deuId)
        }

        /// all StoreToVar ExecUnits that store into the given variable
        member this.StoresToVar vs =
            if storesByVar.ContainsKey vs then storesByVar.[vs].AsReadOnly () :> seq<_>
            else Seq.empty


        /// Walks all ExecUnits contained in this collection calling processFn for each.
        /// The order is so that each execution unit is visited after all the nodes it 
        /// depends on have been visited.
        member this.WalkByDeps (processFn: (ExecUnitT<'e> -> HashSet<ExecUnitIdT> -> unit))  =          
         
            // create list of all ExecUnits that have no predecessors
            let nodesWithoutPredecessors = 
                this.ExecUnits
                |> List.filter (fun eu -> this.DependsOn eu |> Seq.isEmpty)
                |> List.map (fun eu -> eu.Id)

            // initialize queues of ExecUnits to process with list of nodes that have no predecessors
            let toProcess = Queue nodesWithoutPredecessors

            // set of nodes that were already processed
            let processed = HashSet<ExecUnitIdT> ()

            while toProcess.Count > 0 do
                let eu = toProcess.Dequeue () |> this.ById

                // To process an ExecUnit we must already have processed all ExecUnits it depends on.
                // If this is not the case, it can be dequeued safely, because it will be added at a later 
                // time by one of the ExecUnits it depends on and that is yet to be processed.
                let canProcess =
                    this.DependsOn eu
                    |> Seq.forall (fun dep -> processed.Contains dep.Id)

                if canProcess && not (processed.Contains eu.Id) then
                    // process ExecUnit
                    processFn eu processed

                    // mark as processed
                    processed.Add eu.Id |> ignore

                    // add all our direct dependants to the to-process queue
                    for child in this.DependantsOf eu do
                        toProcess.Enqueue child.Id

            assert (processed.Count = this.ExecUnits.Length)


    /// Builds a map that for every storage contains a set of the ids of the ExecUnits
    /// that will access it last during execution.
    let private buildLastStorageAccess (coll: Collection<'e>) : Map<MemManikinT, Set<ExecUnitIdT>> =
                
        // map of last storage access taking into account the key ExecUnit and its predecessors
        let lastStorageAccessInOrAbove = 
            Dictionary<ExecUnitIdT, Dictionary<MemManikinT, HashSet<ExecUnitIdT>>> ()

        // Visit each ExecUnit eu so that, all ExecUnits that eu depends on are visited before
        // eu is visited.
        coll.WalkByDeps (fun eu processed -> 

            // all storages that eu is accessing directly
            let euStorages = 
                eu.Manikins
                |> List.map (fun m -> m.Storage)
                |> Set.ofList

            // build last storage access taking into account eu and its predecessors
            let lsa = Dictionary<MemManikinT, HashSet<ExecUnitIdT>> ()

            // for all storages that eu is accessing, it is the sole last storage accessor
            for storage in euStorages do
                lsa.[storage] <- HashSet<_> (Seq.singleton eu.Id)

            // the last accessors of the *other* storages are the joint last storage accessors of
            // eu's parents
            for parent in coll.DependsOn eu do
                for KeyValue (storage, lastAccessors) in lastStorageAccessInOrAbove.[parent.Id] do
                    if not (euStorages.Contains storage) then
                        if not (lsa.ContainsKey storage) then
                            lsa.[storage] <- HashSet<ExecUnitIdT> ()
                        lsa.[storage].UnionWith lastAccessors
                        
            // store 
            lastStorageAccessInOrAbove.[eu.Id] <- lsa

            // For all of the nodes we depend on, check if all nodes that depend on them have
            // been processed. If yes, then remove information about the parents because it
            // is no longer needed.
            for parent in coll.DependsOn eu do
                let allChildrenProcessed = 
                    coll.DependantsOf parent 
                    |> Seq.forall (fun child -> processed.Contains child.Id || eu.Id = child.Id)
                if allChildrenProcessed then 
                    lastStorageAccessInOrAbove.Remove parent.Id |> ignore
        )

        // In the end, lastStorageAccessInOrAbove must contain one element and this
        // element must be the (bottom-most) ExecUnit without any dependants.
        let lastEu = lastStorageAccessInOrAbove.Keys |> Seq.exactlyOne |> coll.ById
        assert (coll.DependantsOf lastEu |> Seq.isEmpty)
        let lastStorageAccess = lastStorageAccessInOrAbove.[lastEu.Id]

        // convert result into F# immutable Map and List
        seq { 
            for KeyValue (storage, lastAccessors) in lastStorageAccess do
                yield storage, Set.ofSeq lastAccessors            
        } |> Map.ofSeq


    /// Builds the rerun-after dependencies for the ExecUnits.
    let private buildRerunAfter execUnits = 
        // build ExecUnit collection
        let coll = Collection execUnits
            
        // For every storage lastStorageAccess contains a set of the ids of the ExecUnits
        // that will access it last during execution.
        let lastStorageAccess = buildLastStorageAccess coll
       
        // For each ExecUnit this contains a set of units we have to wait upon before 
        // being allowed to rerun.
        let rerunAfter = Dictionary<ExecUnitIdT, HashSet<ExecUnitIdT>> ()

        // For each ExecUnit this contains a set of units we are running after.
        let combinedRerunningAfter = Dictionary<ExecUnitIdT, HashSet<ExecUnitIdT>> ()

        // Visit each ExecUnit eu so that, all ExecUnits that eu depends on are visited before
        // eu is visited.
        coll.WalkByDeps (fun eu processed -> 
            
            // In the current situation, eu reruns after the ExecUnits it depends on
            // and all ExecUnits its parents are rerunning after.
            let euCombinedRerunningAfter = HashSet<ExecUnitIdT> ()
            for parent in coll.DependsOn eu do
                euCombinedRerunningAfter.Add parent.Id |> ignore
                euCombinedRerunningAfter.UnionWith combinedRerunningAfter.[parent.Id] |> ignore

            // An ExecUnit may not be rerun while the storage it uses is still in use by the previous invocation.
            let euMustRerunAfter = seq {
                // For each of our storages:
                for m in eu.Manikins do
                    // We may only run again after each last node in every tree branch that 
                    // accesses the storage has been executed.
                    yield! lastStorageAccess.[m.Storage]

                // If this execution unit is a variable read:
                match eu.Expr with
                | UExpr(ULeafOp (Var readVs), _, _, _) ->
                    // Find all StoreToVars to the same variable operations.
                    // We may only run again, after the previous variable write has been completed.
                    let stvs = coll.StoresToVar readVs                               
                    yield! stvs |> Seq.map (fun stv -> stv.Id)
                | _ -> ()
            }

            // Find all rerun-after ExecUnits that are missing so far.
            let euRerunAfter = HashSet<ExecUnitIdT> ()
            for mraEuId in euMustRerunAfter do
                if not (euCombinedRerunningAfter.Contains mraEuId) then
                    // mraEu is missing, we need to add it to eu's rerun-after list.
                    euRerunAfter.Add mraEuId |> ignore

                    // By rerunning after mraEu, we are also rerunning after all successors of it.
                    // Thus we add them to euCombinedRerunningAfter.
                    let rec addExecUnitAndSuccessors (eu: ExecUnitT<_>) =
                        euCombinedRerunningAfter.Add eu.Id |> ignore

                        for dep in coll.DependantsOf eu do
                            // Recurse over dependants, but skip those that are in 
                            // euCombinedRerunningAfter, because then their successors must be 
                            // present already in euCombinedRerunningAfter.
                            if not (euCombinedRerunningAfter.Contains dep.Id) then
                                addExecUnitAndSuccessors dep

                    addExecUnitAndSuccessors (coll.ById mraEuId)

            // store
            rerunAfter.[eu.Id] <- euRerunAfter
            combinedRerunningAfter.[eu.Id] <- euCombinedRerunningAfter

            // For all of the nodes we depend on, check if all nodes that depend on them have
            // been processed. If yes, then remove information about the parents because it
            // is no longer needed.
            for parent in coll.DependsOn eu do
                let allChildrenProcessed = 
                    coll.DependantsOf parent 
                    |> Seq.forall (fun child -> processed.Contains child.Id || eu.Id = child.Id)
                if allChildrenProcessed then 
                    combinedRerunningAfter.Remove parent.Id |> ignore
        )

        // update ExecUnits
        coll.SortedByDep
        |> List.map (fun eu -> {eu with RerunAfter=rerunAfter.[eu.Id] |> Seq.toList})
             

    /// generates execution units that will evaluate the given unified expression
    let exprToExecUnits (gen: ExecUnitsGeneratorT<'e>) (expr: UExprT) : ExecUnitsForExprT<'e> =

        // number of occurrences of subexpressions
        let sw = Stopwatch.StartNew()
        let exprOccurrences = UExpr.subExprOccurrences expr
        if SymTensor.Compiler.Cuda.Debug.Timing then
            printfn "Building SubExprOccurrences took %A" sw.Elapsed

        // calculates the numeric shape
        let numShapeOf expr = UExpr.shapeOf expr |> ShapeSpec.eval 

        // execution units
        let mutable execUnits = []
        let mutable execUnitIdCnt = 0
        let newExecUnitId () =
            execUnitIdCnt <- execUnitIdCnt + 1
            execUnitIdCnt
        let submitExecUnit eu =
            execUnits <- eu :: execUnits

        let mutable initItems = []
        let submitInitItems ii =
            initItems <- ii @ initItems

        // storage space
        let mutable memAllocIdCnt = 0
        let mutable memAllocs = []
        let newMemory typ elements kind = 
            let mem = {Id=List.length memAllocs; TypeName=typ; Elements=elements; Kind=kind}
            memAllocs <- mem :: memAllocs
            MemAlloc mem

        // evaluation request
        let mutable evalRequests : EvalReqT list = []
        let mutable evalRequestIdCnt = 0
        let submitEvalRequest expr multiplicity storage onCompletion =
            evalRequestIdCnt <- evalRequestIdCnt + 1
            evalRequests <- {Id=evalRequestIdCnt; Expr=expr; Multiplicity=multiplicity; 
                             ReqManikin=storage; OnCompletion=onCompletion} :: evalRequests

        // evaluated requests
        let mutable evaluatedExprs : Map<UExprT, EvalResultT> = Map.empty

        /// takes an evaluation request from the evaluation request queue and processes it
        let processEvalRequest () =   
            // find a request to process and target storage
            let erqToProcess, erqTarget, erqResult, erqMultiplicity, erqRequestors =
                // First, look if there are any expressions which are already computed.
                match evalRequests |> List.tryFind (fun erq -> evaluatedExprs |> Map.containsKey erq.Expr) with
                | Some computedErq -> computedErq, 
                                      Some evaluatedExprs.[computedErq.Expr].Manikin, 
                                      Some evaluatedExprs.[computedErq.Expr],
                                      0, 0
                | None ->
                    // if none, look if there is a group of request for the same expression whose requestors are all known.
                    let erqsByExpr = evalRequests |> List.groupBy (fun erq -> erq.Expr)                              
                    let _, erqsForExpr = erqsByExpr |> List.find (fun (expr, rs) -> 
                        rs |> List.sumBy (fun r -> r.Multiplicity) = exprOccurrences expr)
                    let multiplicity = erqsForExpr |> List.sumBy (fun r -> r.Multiplicity)
                    let requestors = erqsForExpr |> List.length

                    // If a request from the group has a specified storage target, process it first.
                    match List.tryFind (fun erq -> erq.ReqManikin <> None) erqsForExpr with
                    | Some erqWithStorage -> 
                        erqWithStorage, erqWithStorage.ReqManikin, None, multiplicity, requestors
                    | None -> 
                        // Otherwise process any (the first) request from the group.
                        erqsForExpr.[0], None, None, multiplicity, requestors
        
            /// true, if result is processed by multiple requestors
            let erqResultShared = erqRequestors > 1

            /// stores the evaluation result and executes Afterwards functions of the requestor
            let completeEvalRequest result =
                evaluatedExprs <- evaluatedExprs |> Map.add erqToProcess.Expr result
                erqToProcess.OnCompletion result

            match erqResult with
            | Some result ->
                // expr is already evaluated
                completeEvalRequest result
            | None ->
                // emit exec unit to evaluate expression
                match erqToProcess.Expr with
                | UExpr(op, typ, shp, srcs) as erqExpr ->
                    let nSrc = List.length srcs
                    let mutable subreqResults : Map<UExprT, EvalResultT option> = Map.empty

                    let onMaybeCompleted () =
                        if List.forall (fun s -> Map.containsKey s subreqResults) srcs then  
                            // continuation, when eval requests for all sources have been processed                     
                            let subres = Map.map (fun k v -> Option.get v) subreqResults

                            // determine our definitive target storage
                            let srcViews, srcShared, srcExeUnitIds = 
                                srcs 
                                |> List.map (fun s -> subres.[s].Manikin, subres.[s].Shared, subres.[s].ExecUnitId) 
                                |> List.unzip3
                            let trgtView, trgtShared =
                                gen.TrgtGivenSrcs {MemAllocator=newMemory
                                                   TargetType=typ
                                                   TargetShape=numShapeOf erqExpr
                                                   TargetRequest=erqTarget
                                                   Op=op
                                                   Srcs=List.zip srcViews srcShared}
                            let trgtShared = trgtShared || erqResultShared

                            // generate execution items
                            let items = 
                                gen.ExecItemsForOp {MemAllocator=newMemory
                                                    Target=trgtView
                                                    Op=op
                                                    Srcs=List.zip srcViews srcShared
                                                    SubmitInitItems=submitInitItems}
                                @ if Trace.isActive () then 
                                    gen.TraceItemsForExpr {MemAllocator=newMemory
                                                           Target=trgtView
                                                           Expr=erqExpr}
                                  else []

                            // emit execution unit 
                            let eu = {
                                Id         = newExecUnitId()
                                Items      = items
                                DependsOn  = srcExeUnitIds
                                Expr       = erqExpr
                                Manikins   = trgtView :: srcViews
                                RerunAfter = []
                            }                                    
                            submitExecUnit eu
                           
                            completeEvalRequest {ExecUnitId=eu.Id; Manikin=trgtView; Shared=trgtShared}

                    // submit eval requests from sources
                    if List.isEmpty srcs then onMaybeCompleted ()
                    else
                        let srcReqStorages = 
                            gen.SrcReqs {TargetShape=numShapeOf erqExpr
                                         TargetRequest=erqTarget
                                         Op=op
                                         SrcShapes=List.map numShapeOf srcs}
                        for src, srcReqStorage in List.zip srcs srcReqStorages do
                            submitEvalRequest src erqMultiplicity srcReqStorage (fun res ->
                                subreqResults <- subreqResults |> Map.add src (Some res)
                                onMaybeCompleted())     

            // remove eval request        
            evalRequests <- evalRequests |> List.filter (fun erq -> erq.Id <> erqToProcess.Id)

        // create initial evaluation request
        let mutable exprRes = None
        submitEvalRequest expr 1 None (fun res -> exprRes <- Some res)

        // processing loop
        while not (List.isEmpty evalRequests) do
            processEvalRequest ()

        // post-process execUnits
        if Compiler.Cuda.Debug.TraceCompile then printfn "Adding StoreToVar dependencies..."
        let execUnits =
            execUnits
            |> List.map (fun eu ->
                match eu.Expr with
                | UExpr(UUnaryOp (StoreToVar storeVs), _, _, _) ->
                    // For every StoreToVar operation:
                    // Find all EUs that read from the variable's memory.
                    let varMem = MemExternal storeVs
                    let varAccessIds =
                        execUnits
                        |> List.filter (fun ceu ->
                            let readsVar = 
                                match ceu.Expr with
                                | UExpr(ULeafOp (Var readVs), _, _, _) when readVs = storeVs -> true
                                | _ -> false
                            let accessesMemory = 
                                ceu.Manikins 
                                |> List.exists (fun m -> ArrayNDManikin.storage m = varMem)
                            ceu <> eu && (readsVar || accessesMemory))
                        |> List.map (fun eu -> eu.Id)
                        
                    // Add their ids to the StoreToVar EU dependencies, so that 
                    // they are executed before the original variable value 
                    // gets overwritten.
                    {eu with DependsOn = eu.DependsOn @ varAccessIds}                
                | _ -> eu
            )

        //Microsoft.VisualStudio.Profiler.DataCollection.StartProfile (Microsoft.VisualStudio.Profiler.ProfileLevel.Process, Microsoft.VisualStudio.Profiler.DataCollection.CurrentId) |> ignore

        #if !DISABLE_RERUN_AFTER
        // Build RerunAfter dependencies.
        if Compiler.Cuda.Debug.TraceCompile then printfn "Building RerunAfter dependencies..."
        let sw = Stopwatch.StartNew()
        //let execUnits = buildRerunAfterOld execUnits exprRes
        let execUnits = buildRerunAfter execUnits 
        if Compiler.Cuda.Debug.Timing then
            printfn "Building RerunAfter dependencies took %A" sw.Elapsed
        #endif // !DISABLE_RERUN_AFTER

        {
            Expr = expr
            ExecUnits = execUnits
            Result = exprRes.Value
            MemAllocs = memAllocs
            InitItems = initItems
        }


