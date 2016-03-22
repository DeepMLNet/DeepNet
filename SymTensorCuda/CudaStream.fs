namespace SymTensor.Compiler.Cuda

open Basics
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
        | WaitOnRerunEvent          of EventPlaceHolderT
        | EmitRerunEvent            of EventT
        | RerunSatisfied            of ExecUnitIdT
        | ExecUnitStart             of ExecUnitIdT
        | ExecUnitEnd               of ExecUnitIdT

    /// a sequence of commands executed on a stream
    type StreamCmdsT<'e> = List<CudaCmdT<'e>>


module CudaStreamSeq =

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
        let mutable streamOfUnit : Map<ExecUnitIdT, StreamT> = Map.empty
        /// event emitted by an ExecUnit when its execution is finished
        let mutable eventOfUnit : Map<ExecUnitIdT, EventT> = Map.empty
        /// placeholder for an event
        let mutable eventPlaceHolders : Map<ExecUnitIdT, EventPlaceHolderT> = Map.empty
        /// event placeholders that are waiting for the RerunAfter event of the ExecUnit
        let mutable rerunEventWaiters : Map<ExecUnitIdT, ResizeArray<EventPlaceHolderT>> = Map.empty
        /// ExecUnits that still need to be processed
        let mutable execUnitsToProcess = execUnits
        /// ExecUnitIds that have already been processed
        let mutable processedExecUnitIds = Set.empty

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
            streams.Add(ResizeArray<CudaCmdT<'e>> ())
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
            |> Seq.forall (fun depId -> Set.contains depId processedExecUnitIds)

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
                    match streamOfUnit |> Map.tryFind seu.Id with
                    | Some s -> yield s
                    | None -> ()
            } |> Seq.cache

        /// all streams that can currently be reused below ExecUnit "eu"
        let rec availableStreamsBelowExecUnit (cache: Map<ExecUnitIdT, StreamT seq>) eu = seq {
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
                yield! availableStreamsBelowExecUnit cache parent 
        }

        /// Find an event object id that can be used by the given ExecUnit "eu".
        /// This can either be an existing event object id or a newly generated one.
        let findAvailableEventObjectIdFor (eu: ExecUnitT<'e>)  =           
            let rec tryFindReuseable (candEmitter: ExecUnitT<'e>) =               
                match eventOfUnit |> Map.tryFind candEmitter.Id with
                | Some candEvt when
                    // An event "candEvt" emitted by "candEmitter" is reuseable by "eu", if ...
                    // 1. eu depends on all dependants of the event emitter and 
                    coll.DependantsOf candEmitter
                    |> Seq.forall (fun depOfEmitter -> coll.IsSuccessorOf eu depOfEmitter)
                    &&
                    // 2. the event has not been already reused by a successor of the event emitter.
                    coll.AllSuccessorsOf candEmitter
                    |> Seq.forall (fun succOfEmitter -> 
                        match eventOfUnit |> Map.tryFind succOfEmitter.Id with
                        | Some succOfEmitterEvt when succOfEmitterEvt = candEvt -> false
                        | _ -> true)
                    -> Some candEvt.EventObjectId
                | _ ->
                    // Try the units above "candEmitter" if "candEvt" cannot be used.
                    coll.DependsOn candEmitter |> Seq.tryPick tryFindReuseable

            // if possible, reuse an existing event object
            match coll.DependsOn eu |> Seq.tryPick tryFindReuseable with
            | Some id -> id
            | None -> newEventObjectId ()

        // generate streams
        while not execUnitsToProcess.IsEmpty do
            // find an execution unit that has all dependencies satisfied
            let eu = execUnitsToProcess |> List.find dependsSatisfied

            /// all streams that are reuseable below the units we depend on            
            let availStreams =
                coll.DependsOn eu 
                |> Seq.collect (availableStreamsBelowExecUnit (Map.empty |> Map.add eu.Id Seq.empty))
                |> Seq.cache
            /// all streams of the units we directly depend on, that are reuseable below the units we depend on            
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

            // store stream
            streamOfUnit <- streamOfUnit |> Map.add eu.Id euStream

            // emit ExecUnit start marker
            ExecUnitStart eu.Id |> emitToStream euStream
               
            // our stream needs to wait on the results of the streams we depend on
            let endingUnits = eu.DependsOn |> Seq.filter (fun pId -> streamOfUnit.[pId] <> euStream)
            for endingUnitId in endingUnits do            
                match eventOfUnit |> Map.tryFind endingUnitId with
                | Some evt ->
                    // wait on already emitted event, if possible
                    WaitOnEvent evt |> emitToStream euStream
                | None ->
                    // assign an event (either new or reuseable) to the last ExecUnit of the ending stream
                    let evtObjId = findAvailableEventObjectIdFor (coll.ById endingUnitId)
                    let evt = {EventObjectId=evtObjId; CorrelationId=newCorrelationId(); EmittingExecUnitId=endingUnitId}
                    eventOfUnit <- eventOfUnit |> Map.add endingUnitId evt
                
                    // fill in event placeholder and wait for event
                    eventPlaceHolders.[endingUnitId] := Some evt
                    WaitOnEvent evt |> emitToStream euStream

            // find out which RerunAfter constraints are not yet satisfied on this stream
            let rerunsSatisfiedByDeps =
                endingUnits
                |> Seq.collect (fun pId -> rerunsSatisfiedOnStream streamOfUnit.[pId] (Some pId))
                |> Set.ofSeq
            let rerunsSatisfiedOnStream = rerunsSatisfiedOnStream euStream None |> Set.ofSeq
            let rerunsRequired = eu.RerunAfter |> Set.ofList
            let rerunsMissing = rerunsRequired - rerunsSatisfiedByDeps - rerunsSatisfiedOnStream |> Set.toList

            // wait until missing RerunAfter constraints are satisfied
            if not (List.isEmpty rerunsMissing) then
                // emit an placeholder WaitOnRerunEvent 
                let evtPh = ref None
                WaitOnRerunEvent evtPh |> emitToStream euStream
                // add ourselves to the list of event waiters of the ExecUnitTs we are allowed to rerun after
                for rraId in eu.RerunAfter do
                    match rerunEventWaiters |> Map.tryFind rraId with
                    | Some waiters -> waiters.Add evtPh
                    | None ->
                        let waiters = ResizeArray<EventPlaceHolderT>()
                        waiters.Add evtPh
                        rerunEventWaiters <- rerunEventWaiters |> Map.add rraId waiters

            // emit that the missing RerunAfter constraints are now satisfied in this stream
            let rerunsUnmarkedOnStream = (rerunsSatisfiedByDeps + rerunsRequired) - rerunsSatisfiedOnStream
            for rrm in rerunsUnmarkedOnStream do
                RerunSatisfied rrm |> emitToStream euStream

            // emit our instructions
            for cmd in eu.Items do
                Perform cmd |> emitToStream euStream

            // emit an event placeholder to allow for synchronization
            let evtPh = ref None
            eventPlaceHolders <- eventPlaceHolders |> Map.add eu.Id evtPh
            EmitEvent evtPh |> emitToStream euStream

            // check if we need to emit a rerun event
            match rerunEventWaiters |> Map.tryFind eu.Id with
            | Some waiters ->
                // create new event and emit it
                let evt = {EventObjectId=newEventObjectId(); 
                           CorrelationId=newCorrelationId(); 
                           EmittingExecUnitId=eu.Id}
                EmitRerunEvent evt |> emitToStream euStream

                // fill event into all waiters
                for evtPh in waiters do
                    evtPh := Some evt
            | None -> ()

            // emit ExecUnit end marker
            ExecUnitEnd eu.Id |> emitToStream euStream

            // remove from queue
            execUnitsToProcess <- execUnitsToProcess |> List.withoutValue eu
            processedExecUnitIds <- processedExecUnitIds |> Set.add eu.Id

        // remove empty EmitEvent placeholders
        let streams = 
            streams 
            |> Seq.map (
                Seq.filter (fun op -> 
                    match op with
                    | EmitEvent re when !re = None -> false
                    | _ -> true)
                >> Seq.toList)
            |> Seq.toList
       
        streams, eventObjectCnt


