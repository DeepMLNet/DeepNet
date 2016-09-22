namespace SymTensor.Compiler.Cuda

open System.Diagnostics
open System.Collections.Generic

open Basics
open SymTensor
open SymTensor.Compiler


[<AutoOpen>]
module CudaCmdTypes =

    /// a placeholder (reference to) an event
    type EventPlaceHolderT = EventT option ref

    /// a command executed on a stream
    type CudaCmdT<'e> =
        | Perform                   of 'e
        | WaitOnEvent               of EventT
        | EmitEvent                 of EventPlaceHolderT
        | WaitOnRerunEvent          of EventT
        | EmitRerunEvent            of EventPlaceHolderT
        | RerunSatisfied            of ExecUnitIdT
        | ExecUnitStart             of ExecUnitIdT
        | ExecUnitEnd               of ExecUnitIdT

    /// a sequence of commands executed on a stream
    type StreamCmdsT<'e> = CudaCmdT<'e> list


module CudaStreamSeq =

    type private StreamReuse = {
        Stream:                     StreamT
        mutable StreamAvailable:    bool
        CandidateUnits:             Set<ExecUnitIdT>
    }

    type private EventRelationship = {
        Emitter:                    ExecUnitIdT
        Waiter:                     ExecUnitIdT
    }

    type private EventObjectReuse = {
        EmittedEvent:               EventPlaceHolderT
        mutable Available:          bool
        Waiters:                    HashSet<ExecUnitIdT>
        WaitedUpon:                 Set<EventRelationship>
        CandidateUnits:             Set<ExecUnitIdT>  
    }

    /// converts execution units to stream commands
    let execUnitsToStreams (execUnits: ExecUnitT<'e> list) : (StreamCmdsT<'e> list * int) =
        /// collection of ExecUnits we have to process
        let coll = ExecUnit.Collection execUnits

        /// event counter
        let mutable eventObjectCnt = 0
        /// correlation counter
        let mutable correlationCnt = 0
        /// all allocated streams
        let streams = ResizeArray<ResizeArray<CudaCmdT<'e>>> ()
        /// stream used by an ExecUnit        
        let mutable streamOfUnit = Dictionary<ExecUnitIdT, StreamT> ()
        /// event emitted by an ExecUnit when its execution is finished
        let mutable eventOfUnit = Dictionary<ExecUnitIdT, EventT> ()
        /// placeholder for an event
        let mutable eventPlaceHolders = Dictionary<ExecUnitIdT, EventPlaceHolderT> ()
        /// rerun events
        let rerunEvent = Dictionary<ExecUnitIdT, EventPlaceHolderT> ()

        let streamReuse = Dictionary<ExecUnitIdT, StreamReuse> ()
        let eventObjectReuse = Dictionary<ExecUnitIdT, EventObjectReuse> ()

        /// ExecUnits that still need to be processed
        let execUnitsToProcess = Queue (execUnits)
        /// ExecUnitIds that have already been processed
        let processedExecUnitIds = HashSet<ExecUnitIdT> ()

        /// create a new event object id
        let newEventObjectId () =
            eventObjectCnt <- eventObjectCnt + 1
            eventObjectCnt - 1

        /// creates a new correlation id
        let newCorrelationId () =
            correlationCnt <- correlationCnt + 1
            correlationCnt - 1

        /// creates a new stream
        let newStream () =
            streams.Add (ResizeArray<CudaCmdT<'e>> ())
            streams.Count - 1

        /// length of a stream
        let streamLength (s: int) =
            streams.[s].Count

        /// emits an ExeOp to the given streams
        let emitToStream s (exeOp: CudaCmdT<'e>) =
            streams.[s].Add exeOp

        /// true if all ExecUnits that eu depends on have already been processed
        let dependsSatisfied eu =
            eu.DependsOn 
            |> List.forall (fun depId -> processedExecUnitIds.Contains depId)

        /// all RerunAfter that are satisfied on the specified stream,
        /// optionally up to the given ExecUnitId
        let rerunsSatisfiedOnStream strmId untilExecUnitId =           
            streams.[strmId]
            |> Seq.takeWhile (fun cmd ->
                match cmd, untilExecUnitId with
                | ExecUnitEnd id, Some stopId when stopId = id -> false
                | _ -> true)
            |> Seq.choose (fun cmd ->
                match cmd with
                | RerunSatisfied id -> Some id
                | _ -> None)       

        /// all streams that are used by the successors of ExecUnit "eu"
        let rec usedStreamsOfExecUnitAndItsSuccessors (eu: ExecUnitT<'e>) = 
            seq {
                for seu in Seq.append (Seq.singleton eu) (coll.AllSuccessorsOf eu) do
                    if streamOfUnit.ContainsKey seu.Id then 
                        yield streamOfUnit.[seu.Id]
            } |> Seq.cache

        /// all streams that can currently be reused below ExecUnit "eu"
        let availableStreamsBelowExecUnit (eu: ExecUnitT<'e>) =
            let rec build (cache: Map<ExecUnitIdT, StreamT seq>) eu = seq {
                // extend usedStreamsOfExecUnitAndItsSuccessors cache by our dependants
                let mutable cache = cache
                for dep in coll.DependantsOf eu do
                    if not (cache |> Map.containsKey dep.Id) then
                        cache <- cache |> Map.add dep.Id (usedStreamsOfExecUnitAndItsSuccessors dep)

                // streams that are used by our successors
                let usedBySuccessors = 
                    coll.DependantsOf eu
                    |> Seq.collect (fun dep -> cache.[dep.Id])

                // check if eu's stream is available
                let euStream = streamOfUnit.[eu.Id]
                if not (usedBySuccessors |> Seq.contains euStream) then
                    yield euStream

                // extend usedStreamsOfExecUnitAndItsSuccessors cache by ourselves
                let usedByEuAndSuccessors =
                    Seq.append (Seq.singleton euStream) usedBySuccessors
                cache <- cache |> Map.add eu.Id usedByEuAndSuccessors

                // yield streams available from above
                for parent in coll.DependsOn eu do
                    yield! build cache parent 
            }

            build (Map [eu.Id, Seq.empty]) eu

        /// Find an event object id that can be used by the given ExecUnit "eu".
        /// This can either be an existing event object id or a newly generated one.
        let findAvailableEventObjectIdFor (eu: ExecUnitT<'e>)  =           
            let rec tryFindReuseable (candEmitter: ExecUnitT<'e>) =          
                match eventOfUnit.TryFind candEmitter.Id with
                | Some candEvt when
                    // An event "candEvt" emitted by "candEmitter" is reusable by "eu", if ...
                    // 1. eu depends on all dependants of the event emitter and 
                    coll.DependantsOf candEmitter
                    |> Seq.forall (fun depOfEmitter -> coll.IsSuccessorOf eu depOfEmitter)
                    &&
                    // 2. the event has not been already reused by a successor of the event emitter.
                    coll.AllSuccessorsOf candEmitter
                    |> Seq.forall (fun succOfEmitter -> 
                        match eventOfUnit.TryFind succOfEmitter.Id with
                        | Some succOfEmitterEvt when 
                            succOfEmitterEvt.EventObjectId = candEvt.EventObjectId -> false
                        | _ -> true)
                    -> Some candEvt.EventObjectId
                | _ ->
                    // Try the units above "candEmitter" if "candEvt" cannot be used.
                    coll.DependsOn candEmitter |> Seq.tryPick tryFindReuseable

            // if possible, reuse an existing event object
            match coll.DependsOn eu |> Seq.tryPick tryFindReuseable with
            | Some id -> id
            | None -> newEventObjectId ()

        /// gets the rerun event for waiting on the ExecUnit with id rraId
        let getRerunEvent (rraId: ExecUnitIdT) =
            match !rerunEvent.[rraId] with
            | Some evt -> evt
            | None ->
                let evt = {
                    EventObjectId=newEventObjectId()
                    CorrelationId=newCorrelationId() 
                    EmittingExecUnitId=rraId
                }               
                rerunEvent.[rraId] := Some evt
                evt

        // create rerun event placeholders
        for eu in execUnits do
            rerunEvent.Add (eu.Id, ref None)           

        // generate streams
        while execUnitsToProcess.Count > 0 do
            // find an execution unit that has all dependencies satisfied
            let eu = execUnitsToProcess.Dequeue()
            if dependsSatisfied eu then
                // depends satisfied, process execution unit

                #if OPTIMIZED_STREAM_ASSIGNMENT

                // Check if an existing stream can be reused for this execution unit.
                let streamToReuse =
                    // Try to take over the stream of a unit we depend on directly.
                    let streamTakeOverCands = 
                        coll.DependsOn eu
                        |> List.choose (fun deu -> 
                            match streamReuse.[deu.Id] with
                            | {StreamAvailable=true} as sr -> Some sr
                            | {StreamAvailable=false} -> None)

                    if not (List.isEmpty streamTakeOverCands) then
                        // Take over the stream with least commands in it.
                        Some (streamTakeOverCands |> List.minBy (fun sr -> streamLength sr.Stream))
                    else
                        // No stream available from a unit we depend on directly.                       
                        // Build list of streams that can be reused.
                        let reuseCands = seq {
                            for deu in coll.DependsOn eu do
                                for ceuId in streamReuse.[deu.Id].CandidateUnits do
                                    match streamReuse.[ceuId] with
                                    | {StreamAvailable=true} as sr -> yield sr
                                    | {StreamAvailable=false} -> ()
                        }
                        // Use the first available reusable stream, if any is available.
                        Seq.tryHead reuseCands 

                // Determine stream of this execution unit.
                let euStream =
                    match streamToReuse with
                    | Some sr -> 
                        sr.StreamAvailable <- false
                        sr.Stream
                    | None -> newStream ()                                  

                // Build stream reuse information for this execution unit.
                let candidateReuseUnits = [
                    for deu in coll.DependsOn eu do
                        let sr = streamReuse.[deu.Id]
                        if sr.StreamAvailable then yield deu.Id
                        for ceuId in sr.CandidateUnits do
                            if streamReuse.[ceuId].StreamAvailable then yield ceuId
                ]
                streamReuse.[eu.Id] <- {
                    Stream          = euStream
                    StreamAvailable = true
                    CandidateUnits  = candidateReuseUnits |> Set.ofList 
                }
                 
                #else

                /// all streams that are reusable below the units we depend on            
                let availStreams =
                    coll.DependsOn eu 
                    |> Seq.collect availableStreamsBelowExecUnit
                    |> Seq.cache
                /// all streams of the units we directly depend on, that are reusable below the units we depend on            
                let streamTakeOverCands = 
                    eu.DependsOn 
                    |> Seq.map (fun pId -> streamOfUnit.[pId]) 
                    |> Seq.filter (fun strm -> availStreams |> Seq.contains strm)

                // If we can take over a stream, then take over the one with least commands in it.
                // Otherwise, use the first available reusable stream or create a new one.
                let euStream = 
                    if Seq.isEmpty streamTakeOverCands then 
                        if Seq.isEmpty availStreams then newStream ()
                        else Seq.head availStreams
                    else streamTakeOverCands |> Seq.minBy streamLength

                #endif

                // store stream
                streamOfUnit.Add (eu.Id, euStream)

                // emit ExecUnit start marker
                ExecUnitStart eu.Id |> emitToStream euStream
               
                // our stream needs to wait on the results of the streams we depend on
                let euWaitingUponDirectly = ResizeArray<EventRelationship> ()
                let endingUnits = eu.DependsOn |> Seq.filter (fun pId -> streamOfUnit.[pId] <> euStream)
                for endingUnitId in endingUnits do            
                    match eventOfUnit.TryFind endingUnitId with
                    | Some evt ->
                        // wait on already emitted event, if possible
                        WaitOnEvent evt |> emitToStream euStream
                    | None ->
                        #if OPTIMIZED_EVENT_ASSIGNMENT
                        // find a reuseable event object or create a new one
                        let evtObjId =
                            let eor = eventObjectReuse.[endingUnitId]
                            let reuseable = seq {
                                // for all canididate units:
                                for ceuId in eor.CandidateUnits do
                                    // check that candidate unit has all its dependants processed
                                    let ceu = coll.ById ceuId
                                    let dependantsProcessed =
                                        coll.DependantsOf ceu   
                                        |> Seq.forall (fun deu -> processedExecUnitIds.Contains deu.Id)
                                    if dependantsProcessed then
                                        // check that candidate unit has available event 
                                        let ceor = eventObjectReuse.[ceuId]
                                        if Option.isSome !ceor.EmittedEvent && ceor.Available then
                                            // check that eu waits upon all waiters of the candidate unit
                                            let needToWaitUpon = 
                                                ceor.Waiters
                                                |> Seq.map (fun waiterId -> {Emitter=ceuId; Waiter=waiterId})
                                                |> Set.ofSeq
                                            if Set.isSuperset eor.WaitedUpon needToWaitUpon then
                                                yield ceor
                            }
                            match Seq.tryHead reuseable with
                            | Some eor ->
                                eor.Available <- false
                                (!eor.EmittedEvent).Value.EventObjectId
                            | None -> newEventObjectId ()
                        #else
                        let evtObjId = findAvailableEventObjectIdFor (coll.ById endingUnitId)               
                        #endif

                        // assign an event to the last ExecUnit of the ending stream
                        let evt = {EventObjectId=evtObjId; CorrelationId=newCorrelationId(); EmittingExecUnitId=endingUnitId}
                        eventOfUnit.Add (endingUnitId, evt)
                
                        // fill in event placeholder and wait for event
                        eventPlaceHolders.[endingUnitId] := Some evt
                        WaitOnEvent evt |> emitToStream euStream

                    // add this execution unit to the waiters list of units we wait upon
                    eventObjectReuse.[endingUnitId].Waiters.Add eu.Id |> ignore

                    // add to list of units this units waits upon directly
                    euWaitingUponDirectly.Add {
                        Emitter = endingUnitId
                        Waiter  = eu.Id
                    }                    

                // find out which RerunAfter constraints are not yet satisfied on this stream
                let rerunsSatisfiedByDeps =
                    endingUnits
                    |> Seq.collect (fun pId -> rerunsSatisfiedOnStream streamOfUnit.[pId] (Some pId))
                    |> Set.ofSeq
                let rerunsSatisfiedOnStream = rerunsSatisfiedOnStream euStream None |> Set.ofSeq
                let rerunsRequired = eu.RerunAfter |> Set.ofList
                let rerunsMissing = rerunsRequired - rerunsSatisfiedByDeps - rerunsSatisfiedOnStream |> Set.toList

                // wait until missing RerunAfter constraints are satisfied
                for missingRraId in rerunsMissing do
                    WaitOnRerunEvent (getRerunEvent missingRraId) |> emitToStream euStream

                // emit that the missing RerunAfter constraints are now satisfied in this stream
                let rerunsUnmarkedOnStream = (rerunsSatisfiedByDeps + rerunsRequired) - rerunsSatisfiedOnStream
                for rrm in rerunsUnmarkedOnStream do
                    RerunSatisfied rrm |> emitToStream euStream

                // emit our instructions
                for cmd in eu.Items do
                    Perform cmd |> emitToStream euStream

                // emit an event placeholder to allow for synchronization
                let evtPh = ref None
                eventPlaceHolders.Add (eu.Id, evtPh)
                EmitEvent evtPh |> emitToStream euStream

                // build event reuse information 
                let waitingUpon = [
                    yield! euWaitingUponDirectly
                    for deu in coll.DependsOn eu do
                        let deor = eventObjectReuse.[deu.Id]
                        for er in deor.WaitedUpon do
                            if eventObjectReuse.[er.Emitter].Available then
                                yield er
                ]
                eventObjectReuse.[eu.Id] <- {
                    EmittedEvent    = evtPh
                    Available       = true
                    Waiters         = HashSet<ExecUnitIdT> ()
                    WaitedUpon      = waitingUpon |> Set.ofList
                    CandidateUnits  = waitingUpon |> List.map (fun er -> er.Emitter) |> Set.ofList
                }

                // emit rerun event
                EmitRerunEvent rerunEvent.[eu.Id] |> emitToStream euStream

                // emit ExecUnit end marker
                ExecUnitEnd eu.Id |> emitToStream euStream

                // mark as processed
                processedExecUnitIds.Add (eu.Id) |> ignore

            else
                // depends not satisifed, put at back of queue
                execUnitsToProcess.Enqueue eu

        // remove empty EmitEvent and EmitRerunEvent placeholders
        let streams = 
            streams 
            |> Seq.map (
                Seq.filter (fun op -> 
                    match op with
                    | EmitEvent re when !re = None -> false
                    | EmitRerunEvent re when !re = None -> false
                    | _ -> true)
                >> Seq.toList)
            |> Seq.toList
       
        streams, eventObjectCnt


