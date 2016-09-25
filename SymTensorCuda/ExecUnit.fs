namespace SymTensor.Compiler

open System
open System.Diagnostics
open System.Collections.Generic

open Basics
open ArrayNDNS
open SymTensor
open UExprTypes


[<AutoOpen>]
module ExecUnitsTypes = 

    /// Id of an ExecUnitT
    type ExecUnitIdT = int

    /// a group of commands that must be executed sequentially
    type ExecUnitT<'e> = {
        Id:           ExecUnitIdT 
        DependsOn:    ExecUnitIdT list
        Items:        'e list
        Expr:         UExprT
        Manikins:     ArrayNDManikinT list
        RerunAfter:   ExecUnitIdT list
    }

    /// a channel id
    type ChannelIdT = string

    /// id of default channel
    let dfltChId = "#"

    /// manikins representing the data in each channel
    type ChannelManikinsT = Map<ChannelIdT, ArrayNDManikinT>

    /// manikins representing the data in each channel and flag if it is shared
    type ChannelManikinsAndSharedT = Map<ChannelIdT, ArrayNDManikinT * bool>

    /// requests for manikins representing the data in each channel
    type ChannelReqsT = Map<ChannelIdT, ArrayNDManikinT option>

    /// result of an evaluation request
    type EvalResultT = {
        ExecUnitId:     ExecUnitIdT 
        Channels:       ChannelManikinsAndSharedT
    }

    /// an evaluation request
    type EvalReqT = {
        Id:             int 
        Expr:           UExprT
        Multiplicity:   int
        ChannelReqs:    ChannelReqsT
        OnCompletion:   EvalResultT -> unit
    }

    type MemAllocatorT = TypeNameT -> int -> MemAllocKindT -> MemManikinT

    type ExecItemsForOpArgs<'e> = {
        MemAllocator:       MemAllocatorT
        Target:             ChannelManikinsT
        Op:                 UOpT
        Metadata:           UMetadata
        Srcs:               ChannelManikinsAndSharedT list
        SubmitInitItems:    'e list -> unit
    }

    type TraceItemsForExprArgs = {
        MemAllocator:       MemAllocatorT
        Target:             ChannelManikinsT
        Expr:               UExprT
    }

    type TrgtGivenSrcsArgs = {
        MemAllocator:       MemAllocatorT
        TargetRequest:      ChannelReqsT
        Op:                 UOpT
        Metadata:           UMetadata
        Srcs:               ChannelManikinsAndSharedT list    
    }

    type SrcReqsArgs = {
        TargetShape:        NShapeSpecT
        TargetRequest:      ChannelReqsT
        Op:                 UOpT
        SrcShapes:          NShapeSpecT list 
    }

    /// record containing functions called by the ExecUnitT generator
    type ExecUnitsGeneratorT<'e> = {
        ExecItemsForOp:         ExecItemsForOpArgs<'e> -> 'e list
        TraceItemsForExpr:      TraceItemsForExprArgs -> 'e list
        TrgtGivenSrcs:          TrgtGivenSrcsArgs -> ChannelManikinsAndSharedT
        SrcReqs:                SrcReqsArgs -> ChannelReqsT list
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

    type private ExecUnitList<'e> = ResizeArray<ExecUnitT<'e>>

    /// a collection of ExecUnits
    type Collection<'e> (eus: ExecUnitT<'e> list) =     
              
        do if Compiler.Cuda.Debug.TraceCompile then
            printfn "Creating ExecUnit collection..."
        let sw = Stopwatch.StartNew()

        let byIdMap = Dictionary<ExecUnitIdT, ExecUnitT<'e>> ()
        do for eu in eus do
            byIdMap.Add (eu.Id, eu) |> ignore

        let dependants = Dictionary<ExecUnitIdT, ExecUnitList<'e>> ()
        do
            for eu in eus do
                dependants.Add (eu.Id, ResizeArray<ExecUnitT<'e>> ())
            for eu in eus do
                for d in eu.DependsOn do
                    dependants.[d].Add eu

        let sortedByDep = ResizeArray<ExecUnitT<'e>> ()
        do
            let satisfied = HashSet<ExecUnitIdT> ()
            let rec addWithDependencies (eu: ExecUnitT<'e>) =
                if not (satisfied.Contains eu.Id) then
                    for dep in eu.DependsOn do
                        addWithDependencies byIdMap.[dep]
                    sortedByDep.Add eu
                    satisfied.Add eu.Id |> ignore
            for eu in eus do
                addWithDependencies eu

        let storesByVar =
            let build = Dictionary<VarSpecT, List<ExecUnitT<_>>> ()
            for eu in eus do
                match eu.Expr with
                | UExpr (UUnaryOp (Expr.StoreToVar vs), _, _) ->
                    if not (build.ContainsKey vs) then build.[vs] <- List ()
                    build.[vs].Add eu
                | _ -> ()
            build         
            
        do if Compiler.Cuda.Debug.Timing then
            printfn "Creating ExecUnit collection took %A" sw.Elapsed

        /// list of all execution unit contained in this collection
        member this.ExecUnits = eus

        /// execution unit by id
        member this.ById id = byIdMap.[id]

        /// all ExecUnits that depend directly on eu
        member this.DependantsOf (eu: ExecUnitT<'e>) = dependants.[eu.Id].AsReadOnly () :> seq<_>

        /// all ExecUnits that eu directly depends on
        member this.DependsOn (eu: ExecUnitT<'e>) = eu.DependsOn |> List.map this.ById

        /// a list of ExecUnits in this collection so that an ExecUnit comes after all ExecUnits it depends on
        member this.SortedByDep = sortedByDep |> List.ofSeq
       
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
                | UExpr(ULeafOp (Expr.Var readVs), _, _) ->
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

        if SymTensor.Compiler.Cuda.Debug.TraceCompile then
            printfn "UExpr contains %d unique ops" (UExpr.countUniqueOps expr)

        // number of occurrences of subexpressions
        let exprOccurrences = UExpr.subExprOccurrences expr

        // calculates the numeric shape
        let numShapeOf expr = UExpr.shapeOf expr |> ShapeSpec.eval 

        // execution units
        let execUnits = ResizeArray<ExecUnitT<'e>>()
        let mutable execUnitIdCnt = 0
        let newExecUnitId () =
            execUnitIdCnt <- execUnitIdCnt + 1
            execUnitIdCnt
        let submitExecUnit eu =
            execUnits.Add eu

        let initItems = ResizeArray<'e>()
        let submitInitItems (ii: 'e list) =
            initItems.AddRange ii

        // storage space
        let memAllocs = ResizeArray<MemAllocManikinT>()
        let newMemory typ elements kind = 
            let mem = {Id=memAllocs.Count; TypeName=typ; Elements=elements; Kind=kind}
            memAllocs.Add mem
            MemAlloc mem

        // evaluation request
        let evalReqsByExpr = Dictionary<UExprT, ResizeArray<EvalReqT>>(HashIdentity.Reference)
        let evalReqMultiplicities = Dictionary<UExprT, int>(HashIdentity.Reference)
        let exprsWithReqMultiplicity = Queue<UExprT>()
        let mutable evalReqCnt = 0
        let submitEvalRequest expr multiplicity storage onCompletion =
            evalReqCnt <- evalReqCnt + 1
            let evalReq = {Id=evalReqCnt; Expr=expr; Multiplicity=multiplicity; 
                           ChannelReqs=storage; OnCompletion=onCompletion}

            if not (evalReqsByExpr.ContainsKey expr) then
                evalReqsByExpr.[expr] <- ResizeArray<EvalReqT>()
            evalReqsByExpr.[expr].Add evalReq

            if not (evalReqMultiplicities.ContainsKey expr) then
                evalReqMultiplicities.[expr] <- 0
            evalReqMultiplicities.[expr] <- evalReqMultiplicities.[expr] + evalReq.Multiplicity
            
            if evalReqMultiplicities.[expr] = exprOccurrences expr then
                exprsWithReqMultiplicity.Enqueue expr

        /// takes an evaluation request from the evaluation request queue and processes it
        let processEvalRequest () =   
            let erqExpr = exprsWithReqMultiplicity.Dequeue ()
            let erqsForExpr = evalReqsByExpr.[erqExpr] |> List.ofSeq
            
            // calculate how many requests are there and extract expression
            let erqMultiplicity = erqsForExpr |> List.sumBy (fun r -> r.Multiplicity)
            let erqRequestors = erqsForExpr |> List.length
            let erqResultShared = erqRequestors > 1

            // combine channel storage requets
            let erqsForExprByReqs = 
                erqsForExpr
                |> List.sortByDescending (fun erq ->
                    erq.ChannelReqs |> Map.toSeq |> Seq.filter (fun (_, cr) -> cr.IsSome) |> Seq.length)
            let erqChannelReqs : ChannelReqsT =
                (Map.empty, erqsForExprByReqs)
                ||> List.fold (fun reqsSoFar erq -> 
                    (reqsSoFar, erq.ChannelReqs)
                    ||> Map.fold (fun reqsSoFar channel req ->
                        match reqsSoFar |> Map.tryFind channel with
                        | Some (Some prvReq) -> reqsSoFar
                        | Some None | None -> reqsSoFar |> Map.add channel req))
                  
            // emit exec unit to evaluate expression
            let (UExpr(op, srcs, metadata)) = erqExpr
            let subreqResults = Dictionary<UExprT, EvalResultT>(HashIdentity.Reference)

            let onMaybeCompleted () =
                if srcs |> List.forall (fun s -> subreqResults.ContainsKey s) then  
                    // continuation, when eval requests for all sources have been processed                     

                    // determine our definitive target storage
                    let srcChannelsAndShared, srcExeUnitIds = 
                        srcs 
                        |> List.map (fun s -> subreqResults.[s].Channels, subreqResults.[s].ExecUnitId) 
                        |> List.unzip
                    let trgtChannelsAndShared =
                        gen.TrgtGivenSrcs {MemAllocator=newMemory
                                           TargetRequest=erqChannelReqs
                                           Op=op
                                           Metadata=metadata
                                           Srcs=srcChannelsAndShared}
                        |> Map.map (fun ch (manikin, shared) -> 
                                        manikin, shared || erqResultShared)

                    // build channel lists
                    let extractChannels (chsAndShared: ChannelManikinsAndSharedT) : ChannelManikinsT = 
                        chsAndShared |> Map.map (fun ch (manikin, shared) -> manikin) 

                    // generate execution items
                    let items = 
                        gen.ExecItemsForOp {MemAllocator=newMemory
                                            Target=extractChannels trgtChannelsAndShared
                                            Op=op
                                            Metadata=metadata
                                            Srcs=srcChannelsAndShared
                                            SubmitInitItems=submitInitItems}
                        @ if Trace.isActive () then 
                            gen.TraceItemsForExpr {MemAllocator=newMemory
                                                   Target=extractChannels trgtChannelsAndShared
                                                   Expr=erqExpr}
                            else []

                    // extract manikin from all channels
                    let srcManikins = 
                        srcChannelsAndShared
                        |> List.map (extractChannels >> Map.toList >> List.map snd)
                        |> List.concat
                    let trgtManikins = 
                        trgtChannelsAndShared |> extractChannels |> Map.toList |> List.map snd                        

                    // emit execution unit 
                    let eu = {
                        Id         = newExecUnitId()
                        Items      = items
                        DependsOn  = srcExeUnitIds
                        Expr       = erqExpr
                        Manikins   = trgtManikins @ srcManikins
                        RerunAfter = []
                    }                                    
                    submitExecUnit eu
                           
                    let result = {ExecUnitId=eu.Id; Channels=trgtChannelsAndShared}
                    for erq in erqsForExpr do
                        erq.OnCompletion result

            // submit eval requests from sources
            if List.isEmpty srcs then onMaybeCompleted ()
            else
                let srcReqStorages = 
                    gen.SrcReqs {TargetShape=numShapeOf erqExpr
                                 TargetRequest=erqChannelReqs
                                 Op=op
                                 SrcShapes=List.map numShapeOf srcs}
                for src, srcReqStorage in List.zip srcs srcReqStorages do
                    submitEvalRequest src erqMultiplicity srcReqStorage (fun res ->
                        subreqResults.[src] <- res
                        onMaybeCompleted())     


        // create initial evaluation request
        let mutable exprRes = None
        let trgtReq = Map.empty |> Map.add dfltChId None
        submitEvalRequest expr 1 trgtReq (fun res -> exprRes <- Some res)

        // processing loop
        let mutable uniqueProcessedRequests = 0
        while exprsWithReqMultiplicity.Count > 0 do
            processEvalRequest ()
            uniqueProcessedRequests <- uniqueProcessedRequests + 1
        let execUnits = List.ofSeq execUnits
        if Compiler.Cuda.Debug.TraceCompile then
            printfn "Processed %d unique evaluation requests and created %d execution units." 
                uniqueProcessedRequests execUnits.Length
        
        // build variable access and memory access tables
        let eusByReadVar = Dictionary<VarSpecT, HashSet<ExecUnitIdT>> ()
        let eusByAccessMem = Dictionary<MemManikinT, HashSet<ExecUnitIdT>> ()
        for eu in execUnits do
            match eu.Expr with
            | UExpr(ULeafOp (Expr.Var vs), _, _) -> 
                if not (eusByReadVar.ContainsKey vs) then
                    eusByReadVar.[vs] <- HashSet<ExecUnitIdT> ()
                eusByReadVar.[vs].Add eu.Id |> ignore
            | _ -> ()            
            for m in eu.Manikins do
                let mem = ArrayNDManikin.storage m
                if not (eusByAccessMem.ContainsKey mem) then
                    eusByAccessMem.[mem] <- HashSet<ExecUnitIdT> ()
                eusByAccessMem.[mem].Add eu.Id |> ignore

        // add extra dependencies to ExecUnits that execute a StoreToVar operation
        let sw = Stopwatch.StartNew()
        if Compiler.Cuda.Debug.TraceCompile then printfn "Adding StoreToVar dependencies..."
        let execUnits =
            execUnits
            |> List.map (fun eu ->
                match eu.Expr with
                | UExpr(UUnaryOp (Expr.StoreToVar storeVs), _, _) ->
                    // For every StoreToVar operation:
                    // Find all EUs that read from the variable's memory.
                    let readVarEus =
                        match eusByReadVar.TryFind storeVs with
                        | Some eus -> eus |> List.ofSeq
                        | None -> []
                    let accessMemEus =
                        match eusByAccessMem.TryFind (MemExternal storeVs) with
                        | Some eus -> eus |> List.ofSeq
                        | None -> []
                        
                    // Add their ids to the StoreToVar EU dependencies, so that 
                    // they are executed before the original variable value 
                    // gets overwritten.
                    let varAccessEus =
                        readVarEus @ accessMemEus
                        |> List.filter ((<>) eu.Id)
                    {eu with DependsOn = eu.DependsOn @ varAccessEus}                
                | _ -> eu
            )
        if Compiler.Cuda.Debug.Timing then
            printfn "Adding StoreToVar dependencies took %A" sw.Elapsed

        // Build RerunAfter dependencies.
        #if !DISABLE_RERUN_AFTER
        if Compiler.Cuda.Debug.TraceCompile then printfn "Building RerunAfter dependencies..."
        let sw = Stopwatch.StartNew()
        let execUnits = buildRerunAfter execUnits 
        if Compiler.Cuda.Debug.Timing then
            printfn "Building RerunAfter dependencies took %A" sw.Elapsed
        #endif 

        {
            Expr = expr
            ExecUnits = execUnits
            Result = exprRes.Value
            MemAllocs = memAllocs |> List.ofSeq
            InitItems = initItems |> List.ofSeq
        }


