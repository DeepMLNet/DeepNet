namespace SymTensor.Compiler

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

    /// generator function record
    type ExecUnitsGeneratorT<'e> = {ExecItemsForOp: (TypeNameT -> int -> MemManikinT) -> ArrayNDManikinT -> UOpT -> 
                                                    ArrayNDManikinT list -> 'e list;
                                    WarmupExecItemsForOp: (TypeNameT -> int -> MemManikinT) -> ArrayNDManikinT -> UOpT -> 
                                                          ArrayNDManikinT list -> 'e list;
                                    ExecItemsForTrace: (TypeNameT -> int -> MemManikinT) -> ArrayNDManikinT -> 
                                                       UExprT -> 'e list;
                                    TrgtGivenSrc: (TypeNameT -> int -> MemManikinT) -> TypeNameT -> NShapeSpecT -> 
                                                  ArrayNDManikinT option -> UOpT -> ArrayNDManikinT list -> bool list -> 
                                                  ArrayNDManikinT * bool;
                                    SrcReqsGivenTrgt: NShapeSpecT -> ArrayNDManikinT option -> UOpT -> NShapeSpecT list -> 
                                                      ArrayNDManikinT option list;}

module ExecUnit =


    /// a collection of ExecUnits
    type Collection<'e> (eus: ExecUnitT<'e> seq) =
        
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


        //member this.DependantsOf (eu: ExecUnitT<'e>) = dependants.[eu.Id].AsReadOnly ()

        /// execution unit by id
        member this.ById id = byIdMap.[id]

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


    /// generates execution units that will evaluate the given unified expression
    let exprToExecUnits gen (expr: UExprT) =
        // number of occurrences of subexpressions
        let exprOccurrences = UExpr.subExprOccurrences expr

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

        let mutable warmupItems = []
        let submitWarmupItems wi =
            warmupItems <- wi @ warmupItems

        // storage space
        let mutable memAllocIdCnt = 0
        let mutable memAllocs = []
        let newMemory typ elements = 
            let mem = {Id=(List.length memAllocs); TypeName=typ; Elements=elements}
            memAllocs <- mem :: memAllocs
            MemAlloc mem

        // evaluation requestion
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
                                gen.TrgtGivenSrc newMemory typ (numShapeOf erqExpr) erqTarget op srcViews srcShared
                            let trgtShared = trgtShared || erqResultShared

                            // generate execution items
                            let items = 
                                gen.ExecItemsForOp newMemory trgtView op srcViews  @
                                if Trace.isActive () then gen.ExecItemsForTrace newMemory trgtView erqExpr
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

                            // handle warmup items
                            submitWarmupItems (gen.WarmupExecItemsForOp newMemory trgtView op srcViews)
                            
                            completeEvalRequest {ExecUnitId=eu.Id; Manikin=trgtView; Shared=trgtShared}

                    // submit eval requests from sources
                    if List.isEmpty srcs then onMaybeCompleted ()
                    else
                        let srcReqStorages = 
                            gen.SrcReqsGivenTrgt (numShapeOf erqExpr) erqTarget op (List.map numShapeOf srcs)                    
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

        // Build RerunAfter dependencies.
        let coll = Collection execUnits
        let execUnits = coll.SortedByDep
        let lastEu = 
            execUnits 
            |> List.find (fun eu -> eu.Id = exprRes.Value.ExecUnitId)           
        let emptyRerunAfter =
            seq {for eu in execUnits -> eu.Id, []} |> Map.ofSeq
        let rerunAfter =            
            (emptyRerunAfter, execUnits)
            ||> List.fold (fun (rerunAfterSoFar: Map<ExecUnitIdT, ExecUnitIdT list>) eu ->
                /// true if a reruns after b. (either by having b in its RerunAfter list or by 
                /// having a predecessor that has b in its RerunAfter list)
                let rerunsAfter (a: ExecUnitT<'e>) (b: ExecUnitT<'e>) = 
                    if rerunAfterSoFar.[a.Id] |> List.contains b.Id then true
                    else
                        coll.AllPredecessorsOf a
                        |> Seq.exists (fun eu ->
                            eu.RerunAfter |> List.contains b.Id)

                // An ExecUnit may not be rerun while the storage it uses is still in use by the previous invocation.
                let rerunAfter = seq {
                    // For each storage:
                    for m in eu.Manikins do
                        // Starting from the result, find the last node in each tree branch that accesses the storage.
                        // We may only run again, after these ExecUnits.
                        yield! coll.LastStorageAccess m.Storage lastEu 

                    // For each variable read:
                    match eu.Expr with
                    | UExpr(ULeafOp (Var readVs), _, _, _) ->
                        // Find all StoreToVars to the same variable operations.
                        // We may only run again, after the previous variable write has been completed.
                        for ceu in execUnits do
                            match ceu.Expr with
                            | UExpr(UUnaryOp (StoreToVar storeVs), _, _, _) when storeVs = readVs -> yield ceu
                            | _ -> ()
                    | _ -> ()
                }

                // Filter redundant RerunAfter dependencies.
                let rerunAfterIds =
                    rerunAfter
                    |> Seq.filter (fun ra ->
                        // We can omit a node from our RerunAfter list, if
                        // - we depend on the node, because then we are run afterwards anyway,
                        let euDependsOnRa = coll.IsSuccessorOf eu ra 
                        // - any of our predecessors has the node already in their RerunAfter list.
                        let euDependsOnNodeRerunningAfterRa = rerunsAfter eu ra 
                        // - we, or any of our predecessors, have a succesor of the node already in our RerunAfter list.
                        let allRerunAfterIds =
                            coll.AllPredecessorsOf eu 
                            |> Seq.map (fun peu -> rerunAfterSoFar.[peu.Id])
                            |> Seq.concat
                            |> Seq.append (rerunAfter |> Seq.map (fun eu -> eu.Id))
                            |> Set.ofSeq
                        let successorsOfRaIds =
                            coll.AllSuccessorsOf ra 
                            |> Seq.map (fun eu -> eu.Id)
                            |> Set.ofSeq
                        let euRerunsAfterSuccessorOfRa =
                            Set.intersect allRerunAfterIds successorsOfRaIds
                            |> Set.isEmpty
                            |> not
                        // combine
                        not (euDependsOnRa || euDependsOnNodeRerunningAfterRa || euRerunsAfterSuccessorOfRa)
                        )
                    |> Seq.map (fun eu -> eu.Id)
                    |> Seq.toList

                rerunAfterSoFar |> Map.add eu.Id rerunAfterIds
            )
        let execUnits =
            execUnits
            |> List.map (fun eu -> {eu with RerunAfter=rerunAfter.[eu.Id]})

        execUnits, exprRes, memAllocs, warmupItems


