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
        View:           ArrayNDManikinT; 
        Shared:         bool;
    }

    /// an evaluation request
    type EvalReqT = {
        Id:             int; 
        Expr:           UExprT; 
        Multiplicity:   int; 
        View:           ArrayNDManikinT option; 
        OnCompletion:   EvalResultT -> unit
    }

    /// generator function record
    type ExecUnitsGeneratorT<'e> = {ExecItemsForOp: (TypeNameT -> int -> MemManikinT) -> ArrayNDManikinT -> UOpT -> 
                                                    ArrayNDManikinT list -> 'e list;
                                    TrgtGivenSrc: (TypeNameT -> int -> MemManikinT) -> TypeNameT -> NShapeSpecT -> 
                                                  ArrayNDManikinT option -> UOpT -> ArrayNDManikinT list -> bool list -> 
                                                  ArrayNDManikinT * bool;
                                    SrcReqsGivenTrgt: NShapeSpecT -> ArrayNDManikinT option -> UOpT -> NShapeSpecT list -> 
                                                      ArrayNDManikinT option list;}

module ExecUnit =

    /// gets an ExecUnit by its id
    let byId id (eus: ExecUnitT<_> list) =
        eus |> List.find (fun eu -> eu.Id = id)

    /// sorts a list of ExecUnits so that an ExecUnit comes after all ExecUnits it depends on
    let sortByDep (unsortedEus: ExecUnitT<'e> list) =
        let unsorted = LinkedList(unsortedEus)
        let sorted = List<ExecUnitT<'e>>(List.length unsortedEus)
        
        while unsorted.Count > 0 do           
            let toAdd =
                unsorted
                |> Seq.filter (fun eu -> 
                    eu.DependsOn |> List.forall (fun depId ->
                        sorted |> Seq.exists (fun seu -> seu.Id = depId)))
            
            for eu in toAdd do
                sorted.Add eu |> ignore
                unsorted.Remove eu |> ignore

        sorted |> Seq.toList

    /// true if a is a successor of b. 
    let rec isSuccessorOf (a: ExecUnitT<_>) (b: ExecUnitT<_>) eus =
        if b.DependsOn |> List.contains a.Id then true
        else a.DependsOn |> List.exists (fun euId -> 
            let eu = byId euId eus 
            isSuccessorOf eu b eus)

    /// returns all successors of a
    let rec allSuccessorsOf (a: ExecUnitT<_>) eus = seq {
        for eu in eus do
            if eu.DependsOn |> List.contains a.Id then 
                yield eu
                yield! allSuccessorsOf eu eus
    }

    /// returns all predecessors of a
    let rec allPredecessorsOf (a: ExecUnitT<_>) eus = seq {
        for deuId in a.DependsOn do
            let deu = byId deuId eus
            yield deu
            yield! allPredecessorsOf deu eus
    }

    /// true if a reruns after b.
    let rec rerunsAfter (a: ExecUnitT<_>) (b: ExecUnitT<_>) eus =
        if a.RerunAfter |> List.contains b.Id then true
        else a.DependsOn |> List.exists (fun euId -> 
            let eu = byId euId eus 
            rerunsAfter eu b eus)

    /// Return all EUs above or equal to "eu" that access "storage" for the last time.
    let rec lastStorageAccess storage (eu: ExecUnitT<_>) eus = seq {
        let isAccessing =
            eu.Manikins
            |> List.exists (fun m -> m.Storage = storage)

        if isAccessing then yield eu
        else
            for deuId in eu.DependsOn do
                yield! lastStorageAccess storage (byId deuId eus) eus
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
                             View=storage; OnCompletion=onCompletion} :: evalRequests

        // evaluated requests
        let mutable evaluatedExprs : Map<UExprT, EvalResultT> = Map.empty

        /// takes an evaluation request from the evaluation request queue and processes it
        let processEvalRequest () =   
            // find a request to process and target storage
            let erqToProcess, erqTarget, erqResult, erqMultiplicity, erqRequestors =
                // First, look if there are any expressions which are already computed.
                match evalRequests |> List.tryFind (fun erq -> evaluatedExprs |> Map.containsKey erq.Expr) with
                | Some computedErq -> computedErq, 
                                      Some evaluatedExprs.[computedErq.Expr].View, 
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
                    match List.tryFind (fun erq -> erq.View <> None) erqsForExpr with
                    | Some erqWithStorage -> 
                        erqWithStorage, erqWithStorage.View, None, multiplicity, requestors
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
                                |> List.map (fun s -> subres.[s].View, subres.[s].Shared, subres.[s].ExecUnitId) 
                                |> List.unzip3
                            let trgtView, trgtShared =
                                gen.TrgtGivenSrc newMemory typ (numShapeOf erqExpr) erqTarget op srcViews srcShared
                            let trgtShared = trgtShared || erqResultShared

                            // emit execution unit 
                            let eu = {
                                Id         = newExecUnitId();
                                Items      = gen.ExecItemsForOp newMemory trgtView op srcViews;
                                DependsOn  = srcExeUnitIds;
                                Expr       = erqExpr;
                                Manikins   = trgtView::srcViews;
                                RerunAfter = [];
                            }                                    
                            submitExecUnit eu

                            completeEvalRequest {ExecUnitId=eu.Id; View=trgtView; Shared=trgtShared}

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
        let execUnits = sortByDep execUnits
        let lastEu = 
            execUnits 
            |> List.find (fun eu -> eu.Id = exprRes.Value.ExecUnitId)
        let rec buildRerunAfter eus processedEUs =
            match eus with
            | eu::eus ->
                // An ExecUnit may not be rerun while the storage it uses is still in use by the previous invocation.
                let rerunAfter = seq {
                    // For each storage:
                    for m in eu.Manikins do
                        // Starting from the result, find the last node in each tree branch that accesses the storage.
                        // We may only run again, after these ExecUnits.
                        yield! lastStorageAccess m.Storage lastEu execUnits

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
                        let euDependsOnRa = isSuccessorOf eu ra execUnits
                        // - any of our predecessors has the node already in their RerunAfter list.
                        let euDependsOnNodeRerunningAfterRa = rerunsAfter eu ra processedEUs
                        // - we, or any of our predecessors, have a succesor of the node already in our RerunAfter list.
                        let allRerunAfterIds =
                            allPredecessorsOf eu processedEUs
                            |> Seq.map (fun peu -> peu.RerunAfter)
                            |> Seq.concat
                            |> Seq.append (rerunAfter |> Seq.map (fun eu -> eu.Id))
                            |> Set.ofSeq
                        let successorsOfRaIds =
                            allSuccessorsOf ra execUnits
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

                buildRerunAfter eus ({eu with RerunAfter=rerunAfterIds} :: processedEUs)
            | [] -> List.rev processedEUs
        let execUnits =
            buildRerunAfter execUnits []      

        execUnits, exprRes, memAllocs


