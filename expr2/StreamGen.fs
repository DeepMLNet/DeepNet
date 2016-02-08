module StreamGen

open Util
open ExecUnitsGen

/// a compute stream
type StreamT = int

/// an event that can be used for synchronization
type EventT = {EventObjectId: int; CorrelationId: int; SendingExecUnitId: int}

/// a placeholder (reference to) an event
type EventPlaceHolderT = EventT option ref

/// a command executed on a stream
type ExeOpT<'e> =
    | Perform of 'e
    | WaitOnEvent of EventT
    | EmitEvent of EventPlaceHolderT
    | ExecUnitStartInfo of string
    | ExecUnitEndInfo

/// a sequence of commands executed on a stream
type StreamSeqT<'e> = List<ExeOpT<'e>>

/// converts execution units to stream commands
let execUnitsToStreamCommands (execUnits: ExecUnitT<'e> list) : (StreamSeqT<'e> list * int) =
    /// event counter
    let mutable eventObjectCnt = 0
    /// correlation counter
    let mutable correlationCnt = 0
    /// all allocated streams
    let streams = new ResizeArray<StreamSeqT<'e>>()
    /// stream used by an ExecUnit        
    let mutable streamOfUnit : Map<ExecUnitIdT, StreamT> = Map.empty
    /// event emitted by an ExecUnit when its execution is finished
    let mutable eventOfUnit : Map<ExecUnitIdT, EventT> = Map.empty
    /// placeholder for an event
    let mutable eventPlaceHolders : Map<ExecUnitIdT, EventPlaceHolderT> = Map.empty
    /// ExecUnits that still need to be processed
    let mutable execUnitsToProcess = execUnits

    /// create a new event object id
    let newEventObjectId() =
        eventObjectCnt <- eventObjectCnt + 1
        eventObjectCnt - 1

    /// creates a new correlation id
    let newCorrelationId() =
        correlationCnt <- correlationCnt + 1
        correlationCnt - 1

    /// creates a new stream
    let newStream () =
        streams.Add([])
        streams.Count - 1

    /// length of a stream
    let streamLength (s: int) =
        List.length streams.[s]

    /// emits an ExeOp to the given streams
    let emitToStream s (exeOp: ExeOpT<'e>) =
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
        eu.DependsOn 
        |> List.forall (fun id -> 
            not (List.exists (fun (eutp: ExecUnitT<'e>) -> eutp.Id = id) execUnitsToProcess))

    /// stream used by ExecUnit with Id euId
    let tryGetStreamOfExecUnitId euId =
        if streamOfUnit |> Map.containsKey euId then Some streamOfUnit.[euId]
        else None

    /// all streams that are used by the successors of an ExecUnit
    let rec usedStreamsOfAndBelowExecUnit (eu: ExecUnitT<'e>) = 
        eu.Id |> dependants |> List.fold (fun ustrs subEu -> 
            match tryGetStreamOfExecUnitId subEu.Id with
            | Some us -> us::ustrs
            | None -> ustrs) 
            (match tryGetStreamOfExecUnitId eu.Id with
            | Some us -> [us]
            | None -> [])

    /// all streams that can currently be reused safely below an ExecUnit
    let rec availableStreamsBelowExecUnit (eu: ExecUnitT<'e>) =
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
        for cmd in eu.Items do
            Perform cmd |> emitToStream euStream

        // emit an event placeholder to allow for synchronization
        let evtPh = ref None
        eventPlaceHolders <- eventPlaceHolders |> Map.add eu.Id evtPh
        EmitEvent evtPh |> emitToStream streamOfUnit.[eu.Id]

        // remove from queue
        execUnitsToProcess <- execUnitsToProcess |> List.withoutValue eu

    // remove empty EmitEvent placeholders
    let streams = 
        streams 
        |> Seq.map (fun stream -> 
            stream |> List.filter (fun op -> 
                match op with
                | EmitEvent re when !re = None -> false
                | _ -> true))
        |> Seq.toList
       
    streams, eventObjectCnt


