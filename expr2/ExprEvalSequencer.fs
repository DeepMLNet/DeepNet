module ExprEvalSequencer

open Util
open Shape
open Op


type StreamT = int
type EventT = {EventObjectId: int; CorrelationId: int; SendingExecUnitId: int}

type StrideT = int list
type StorageT = {Name: string; Elements: int}

let canWorkInPlace unaryOp = 
    match unaryOp with
    // unary elementwise
    | Negate -> true
    | Log -> true
    | Exp -> true
    // reductions
    | Sum -> false
    | SumAxis _ -> false
    // shape operations
    | Reshape _ -> true
    | Broadcast _ -> true
    | SwapDim _ -> true
    // misc
    | Annotated _ -> true
    
let canWorkInFirstPlace binaryOp = 
    match binaryOp with
    // binary elementwise
    | Add -> true
    | Substract -> true
    | Multiply -> true
    | Divide -> true
    | Power -> true
    // matrix/tensor operations
    | Dot -> false
    | TensorProduct -> false

let canWorkInSecondPlace binaryOp = canWorkInFirstPlace binaryOp

type EventPlaceHolderT = EventT option ref

type ExeOpT =
    | ExecLeaf of StorageT * LeafOpT
    | ExecUnary of StorageT * UnaryOpT * StorageT
    | ExecBinary of StorageT * BinaryOpT * StorageT * StorageT
    | WaitOnEvent of EventT
    | EmitEvent of EventPlaceHolderT
    | ExecUnitStartInfo of string
    | ExecUnitEndInfo

type ExecUnitIdT = int
type ExecUnitT = {Id: ExecUnitIdT; DependsOn: ExecUnitIdT list; Items: ExeOpT list; }

type EvalResultT = {ExecUnitId: ExecUnitIdT; Storage: StorageT; Shared: bool}
type EvalReqT = {Id: int; Expr: ExprT; Multiplicity: int; Storage: StorageT option; OnCompletion: EvalResultT -> unit}

let exprToExecUnits (sizeSymbolEnv: SymbolEnvT) (expr: ExprT) =
    // number of occurrences of subexpressions
    let exprOccurrences = subExprOccurrences expr

    // calculates the numeric shape
    let numShapeOf expr = shapeOf expr |> ShapeSpec.eval sizeSymbolEnv

    // execution units
    let mutable execUnits = []
    let mutable execUnitIdCnt = 0
    let newExecUnit () =
        execUnitIdCnt <- execUnitIdCnt + 1
        {Id=execUnitIdCnt; DependsOn=[]; Items=[];}
    let submitExecUnit eu =
        execUnits <- eu :: execUnits

    // storage space
    let mutable storageIdCnt = 0
    let newStorage shape =
        storageIdCnt <- storageIdCnt + 1
        {Name=sprintf "s%d" storageIdCnt; Elements=List.fold (*) 1 shape}

    // evaluation requestion
    let mutable evalRequests : EvalReqT list = []
    let mutable evalRequestIdCnt = 0
    let submitEvalRequest expr multiplicity storage onCompletion =
        evalRequestIdCnt <- evalRequestIdCnt + 1
        evalRequests <- {Id=evalRequestIdCnt; Expr=expr; Multiplicity=multiplicity; 
                         Storage=storage; OnCompletion=onCompletion} :: evalRequests

    // evaluated requests
    let mutable evaluatedExprs : Map<ExprT, EvalResultT> = Map.empty

    /// takes an evaluation request from the evaluation request queue and processes it
    let processEvalRequest () =   
        // find a request to process and target storage
        let erqToProcess, erqStorage, erqResult, erqMultiplicity, erqRequestors =
            // First, look if there are any expressions which are already computed.
            match evalRequests |> List.tryFind (fun erq -> evaluatedExprs |> Map.containsKey erq.Expr) with
            | Some computedErq -> computedErq, 
                                  Some evaluatedExprs.[computedErq.Expr].Storage, 
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
                match List.tryFind (fun erq -> erq.Storage <> None) erqsForExpr with
                | Some erqWithStorage -> 
                    erqWithStorage, erqWithStorage.Storage, None, multiplicity, requestors
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
            | Leaf(op) ->
                // If no desired storage has been specified, we allocate a new one for this leaf.
                let targetStorage = 
                    match erqStorage with
                    | Some s -> s
                    | None -> newStorage (numShapeOf erqToProcess.Expr)
                let targetShared = erqResultShared

                // emit execution unit 
                let eu = {newExecUnit() with Items=[ExecLeaf(targetStorage, op)]}
                submitExecUnit eu

                completeEvalRequest {ExecUnitId=eu.Id; Storage=targetStorage; Shared=targetShared}
            | Unary(op, aExpr) -> 
                // request aExpr to be evaluated directly into our storage, if we can work inplace.
                let aReqStorage = if canWorkInPlace op then erqStorage else None

                submitEvalRequest aExpr erqMultiplicity aReqStorage 
                    (fun aRes ->
                        // determine our definitive storage
                        let targetStorage =
                            match erqStorage with
                            | Some s -> s
                            | None when canWorkInPlace op && not aRes.Shared -> aRes.Storage
                            | None -> newStorage (numShapeOf erqToProcess.Expr)                               
                        let targetShared =
                            (if targetStorage = aRes.Storage then aRes.Shared else false) ||
                            erqResultShared

                        // emit execution unit 
                        let eu = {newExecUnit() with Items=[ExecUnary(targetStorage, op, aRes.Storage)];
                                                     DependsOn=[aRes.ExecUnitId]}                                    
                        submitExecUnit eu

                        completeEvalRequest {ExecUnitId=eu.Id; Storage=targetStorage; Shared=targetShared}
                    )                                                                       
            | Binary(op, aExpr, bExpr) ->
                // request aExpr or bExpr to be evaluated directly into our storage, if we can work inplace.
                let aReqStorage, bReqStorage = 
                    match canWorkInFirstPlace op, canWorkInSecondPlace op with
                    | true, _ -> erqStorage, None
                    | _, true -> None, erqStorage
                    | false, false -> None, None

                // callback when aExpr and bExpr requests have been evaluated
                let mutable aRes = None
                let mutable bRes = None
                let onMaybeCompleted () =
                    match aRes, bRes with
                    | Some aRes, Some bRes ->
                        // determine our definitive storage
                        let targetStorage =
                            match erqStorage with
                            | Some s -> s
                            | None when canWorkInFirstPlace op && not aRes.Shared -> aRes.Storage
                            | None when canWorkInSecondPlace op && not bRes.Shared -> bRes.Storage
                            | None -> newStorage (numShapeOf erqToProcess.Expr)
                        let targetShared = 
                            (if targetStorage = aRes.Storage then aRes.Shared else false) ||
                            (if targetStorage = bRes.Storage then bRes.Shared else false) ||
                            erqResultShared

                        // emit execution unit 
                        let eu = {newExecUnit() with Items=[ExecBinary(targetStorage, op, aRes.Storage, bRes.Storage)];
                                                     DependsOn=[aRes.ExecUnitId; bRes.ExecUnitId]}
                        submitExecUnit eu

                        completeEvalRequest {ExecUnitId=eu.Id; Storage=targetStorage; Shared=targetShared}
                    | _ -> ()    
                    
                submitEvalRequest aExpr erqMultiplicity aReqStorage (fun res -> aRes <- Some res; onMaybeCompleted())
                submitEvalRequest bExpr erqMultiplicity bReqStorage (fun res -> bRes <- Some res; onMaybeCompleted())
        
        // remove eval request        
        evalRequests <- evalRequests |> List.filter (fun erq -> erq.Id <> erqToProcess.Id)

    // create initial evaluation request
    let mutable exprRes = None
    submitEvalRequest expr 1 None (fun res -> exprRes <- Some res)

    // processing loop
    while not (List.isEmpty evalRequests) do
        processEvalRequest ()

    execUnits, exprRes


type MutableList<'a> = System.Collections.Generic.List<'a>
type StreamSeqT = List<ExeOpT>

/// converts execution units to stream commands
let execUnitsToStreamCommands (execUnits: ExecUnitT list) =
    /// event counter
    let mutable eventObjectCnt = 0
    /// correlation counter
    let mutable correlationCnt = 0
    /// all allocated streams
    let streams = new MutableList<StreamSeqT>()
    /// stream used by an ExecUnit        
    let mutable streamOfUnit : Map<ExecUnitIdT, StreamT> = Map.empty
    /// event emitted by an ExecUnit when its execution is finished
    let mutable eventOfUnit : Map<ExecUnitIdT, EventT> = Map.empty
    let mutable eventPlaceHolders : Map<ExecUnitIdT, EventPlaceHolderT> = Map.empty
    /// ExecUnits that still need to be processed
    let mutable execUnitsToProcess = execUnits

    /// create a new event object id
    let newEventObjectId() =
        eventObjectCnt <- eventObjectCnt + 1
        eventObjectCnt

    /// creates a new correlation id
    let newCorrelationId() =
        correlationCnt <- correlationCnt + 1
        correlationCnt

    /// creates a new stream
    let newStream () =
        streams.Add([])
        //if streams.Count - 1 = 2 then failwith "why are you allocating me?"
        streams.Count - 1

    /// length of a stream
    let streamLength (s: int) =
        List.length streams.[s]

    /// emits an ExeOp to the given streams
    let emitToStream s (exeOp: ExeOpT) =
        streams.[s] <- streams.[s] @ [exeOp]

    /// get the ExecUnit with the given id
    let execUnitById (execUnitId: ExecUnitIdT) =
        execUnits |> List.find (fun eu -> eu.Id = execUnitId)

    /// gets the ExecUnits that depend on the specified unit
    let dependants (execUnitId: ExecUnitIdT) =
        execUnits |> List.filter (fun eu -> eu.DependsOn |> List.contains execUnitId)

    /// gets the ExecUnits on which the specified unit depends
    let dependsOn (execUnitId: ExecUnitIdT) =
        (execUnitById execUnitId).DependsOn

    /// true if all ExecUnits that eu depends on have already been processed
    let dependsSatisfied eu =
        eu.DependsOn |> List.forall (fun id -> not (List.exists (fun (eutp: ExecUnitT) -> eutp.Id = id) execUnitsToProcess))

    /// stream used by ExecUnit with Id euId
    let tryGetStreamOfExecUnitId euId =
        if streamOfUnit |> Map.containsKey euId then Some streamOfUnit.[euId]
        else None

    /// all streams that are used by the successors of an ExecUnit
    let rec usedStreamsOfAndBelowExecUnit (eu: ExecUnitT) = 
        eu.Id |> dependants |> List.fold (fun ustrs subEu -> 
            match tryGetStreamOfExecUnitId subEu.Id with
            | Some us -> us::ustrs
            | None -> ustrs) 
            (match tryGetStreamOfExecUnitId eu.Id with
            | Some us -> [us]
            | None -> [])

    /// all streams that can currently be reused safely below an ExecUnit
    let rec availableStreamsBelowExecUnit (eu: ExecUnitT) =
        // streams already avaiable from the ExecUnits we depend on
        let availFromAbove = 
            eu.DependsOn |> List.map (execUnitById >> availableStreamsBelowExecUnit) |> List.concat |> Set.ofList
        // streams that end with the nodes that we depend on
        let endingHere = 
            eu.DependsOn 
            |> List.filter (fun pid -> 
                pid |> dependants |> List.exists (fun dep -> 
                    tryGetStreamOfExecUnitId dep.Id = Some streamOfUnit.[pid]) |> not)
            |> List.map (fun pid -> streamOfUnit.[pid]) 
            |> Set.ofList
        // my stream
        let myStream = Set.singleton streamOfUnit.[eu.Id]
        // streams that are used by nodes that depend on us
        let usedBelow = 
            eu.Id |> dependants |> List.map usedStreamsOfAndBelowExecUnit |> List.concat |> Set.ofList
            
        (availFromAbove + endingHere + myStream) - usedBelow |> Set.toList

    /// returns true if sucId is a successor (thus dependent on) pId
    let rec isSuccessorOf pId sucId =
        if pId = sucId then true
        else pId |> dependants |> List.exists (fun d -> isSuccessorOf d.Id sucId) 

    /// find reuseable event objects for the given ExecUnit
    let tryFindAvailableEventObjectsFor (eu: ExecUnitIdT)  =
        let rec findRec peu =
            match eventOfUnit |> Map.tryFind peu with
            | Some {EventObjectId=eObjId} when 
                    peu |> dependants |> List.forall (fun d -> isSuccessorOf d.Id eu) 
                -> Some eObjId
            | _ -> 
                peu |> dependsOn |> List.tryPick findRec            
        findRec eu

    // generate streams
    while not execUnitsToProcess.IsEmpty do
        // find an execution unit that has all dependencies satisfied
        let eu = execUnitsToProcess |> List.find dependsSatisfied

        /// all streams that are reuseable below the units we depend on
        let availStreams = 
            eu.DependsOn |> List.map (execUnitById >> availableStreamsBelowExecUnit) |> List.concat |> Set.ofList
        /// all streams of the units we depend on
        let streamTakeOverCands = 
            eu.DependsOn |> List.map (fun pId -> streamOfUnit.[pId]) |> Set.ofList
        /// all streams of the units we depend on, that have not been reused by our siblings or their successors
        let streamsNotYetTakenOver = Set.intersect availStreams streamTakeOverCands

        // If we can take over a stream, then take over the one with least commands in it
        // Otherwise, use the first available reusable stream or create a new one.
        let euStream = match streamsNotYetTakenOver |> Set.toList with
                       | _::_ as cs -> cs |> List.minBy streamLength
                       | [] -> match availStreams |> Set.toList with
                               | s::_ -> s
                               | [] -> newStream ()

        // store stream
        streamOfUnit <- streamOfUnit |> Map.add eu.Id euStream
               
        // our stream needs to wait on the results of the streams we depend on
        for endingUnitId in eu.DependsOn |> List.filter (fun pId -> streamOfUnit.[pId] <> euStream) do            
            match eventOfUnit |> Map.tryFind endingUnitId with
            | Some evt ->
                // wait on already emitted event, if possible
                WaitOnEvent evt |> emitToStream euStream
            | None ->
                // if possible, reuse an existing event object
                let evtObjId =
                    match tryFindAvailableEventObjectsFor endingUnitId with
                    | Some id -> id
                    | None -> newEventObjectId()
                let evt = {EventObjectId=evtObjId; CorrelationId=newCorrelationId(); SendingExecUnitId=endingUnitId}
                eventOfUnit <- eventOfUnit |> Map.add endingUnitId evt
                
                // fill in event placeholder and wait for event
                eventPlaceHolders.[endingUnitId] := Some evt
                WaitOnEvent evt |> emitToStream euStream

        // emit our instructions
        //sprintf "%A" eu |> ExecUnitStartInfo |> emitToStream euStream
        for cmd in eu.Items do
            cmd |> emitToStream euStream

        // emit an event placeholder to allow for synchronization
        let evtPh = ref None
        eventPlaceHolders <- eventPlaceHolders |> Map.add eu.Id evtPh
        EmitEvent evtPh |> emitToStream streamOfUnit.[eu.Id]

        // remove from queue
        execUnitsToProcess <- execUnitsToProcess |> List.withoutValue eu

    streams 
        |> Seq.map (fun stream -> 
            stream |> List.filter (fun op -> 
                match op with
                | EmitEvent re when !re = None -> false
                | _ -> true))
        |> Seq.toList
       


// TODO: variable memory allocation
