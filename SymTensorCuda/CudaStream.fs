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
        /// event counter
        let mutable eventObjectCnt = 0
        /// correlation counter
        let mutable correlationCnt = 0
        /// all allocated streams
        let streams = new ResizeArray<StreamCmdsT<'e>>()
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
        let emitToStream s (exeOp: CudaCmdT<'e>) =
            streams.[s] <- streams.[s] @ [exeOp]

        let rerunsSatisfiedOnStream s untilExecUnitId =           
            let rec build strm = seq {               
                match strm, untilExecUnitId with
                | RerunSatisfied id :: rstrm, _ -> 
                    yield id
                    yield! build rstrm 
                | ExecUnitEnd id :: _, Some stopId when stopId = id -> ()
                | _ :: rstrm, _ -> yield! build rstrm
                | [], _ -> ()
            }
            build streams.[s]

        /// get the ExecUnit with the given id
        let execUnitById (execUnitId: ExecUnitIdT) =
            execUnits |> List.find (fun eu -> eu.Id = execUnitId)

        /// gets the ExecUnits that depend on the specified unit
        let dependants (execUnitId: ExecUnitIdT) =
            execUnits |> List.filter (fun eu -> eu.DependsOn |> List.contains execUnitId)

        /// gets all ExecUnits that are successors of the specified unit
        let rec successors (execUnitId: ExecUnitIdT) = seq {
            for d in dependants execUnitId do
                yield d
                yield! successors d.Id
        }        

        /// gets the ExecUnits on which the specified unit depends
        let dependsOn (execUnitId: ExecUnitIdT) =
            (execUnitById execUnitId).DependsOn

        /// true if all ExecUnits that eu depends on have already been processed
        let dependsSatisfied eu =
            eu.DependsOn 
            |> List.forall (fun id -> 
                not (List.exists (fun (eutp: ExecUnitT<'e>) -> eutp.Id = id) execUnitsToProcess))

        /// all streams that are used by the successors of an ExecUnit
        let rec usedStreamsOfAndBelowExecUnit (eu: ExecUnitT<'e>) = seq {
            for seu in Seq.append (Seq.singleton eu) (successors eu.Id) do
                match streamOfUnit |> Map.tryFind seu.Id with
                | Some s -> yield s
                | None -> ()
        }

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
                        streamOfUnit |> Map.tryFind dep.Id = Some streamOfUnit.[pid]) |> not)
                |> List.map (fun pid -> streamOfUnit.[pid]) 
                |> Set.ofList
            // my stream
            let myStream = Set.singleton streamOfUnit.[eu.Id]
            // streams that are used by nodes that depend on us
            let usedBelow = 
                eu.Id |> dependants |> List.map usedStreamsOfAndBelowExecUnit |> Seq.concat |> Set.ofSeq
            
            (availFromAbove + endingHere + myStream) - usedBelow |> Set.toList

        /// returns true if sucId is a successor (thus dependent on) pId
        let rec isSuccessorOf pId sucId =
            if pId = sucId then true
            else pId |> dependants |> List.exists (fun d -> isSuccessorOf d.Id sucId) 

        /// all streams that are used by the successors of an ExecUnit
        let rec usedEventsOfAndBelowExecUnit (eu: ExecUnitT<'e>) = seq {
            for seu in Seq.append (Seq.singleton eu) (successors eu.Id) do
                match eventOfUnit |> Map.tryFind seu.Id with
                | Some evt -> yield evt
                | None -> ()
                for seud in seu.DependsOn do
                    match eventOfUnit |> Map.tryFind seud with
                    | Some evt -> yield evt
                    | None -> ()
        }

        /// find reuseable event objects for the given ExecUnit
        let rec tryFindAvailableEventObjectFor (eu: ExecUnitT<'e>)  =
            // Event is reuseable by eu if eu depends on all dependants of the event emitter
            // and the event has not been reused by a successor of the event emitter.
            let rec findRec peuId =
                match eventOfUnit |> Map.tryFind peuId with
                | Some {EventObjectId=eObjId} as sevt when 
                        peuId 
                        |> dependants 
                        |> List.forall (fun d -> isSuccessorOf d.Id eu.Id) 
                        &&
                        not(peuId 
                            |> dependants 
                            |> List.exists (usedEventsOfAndBelowExecUnit >> Seq.contains sevt.Value))
                    -> Some eObjId
                | _ -> 
                    peuId |> dependsOn |> List.tryPick findRec            

            findRec eu.Id

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

            // If we can take over a stream, then take over the one with least commands in it.
            // Otherwise, use the first available reusable stream or create a new one.
            let euStream = match streamsNotYetTakenOver |> Set.toList with
                           | _::_ as cs -> cs |> List.minBy streamLength
                           | [] -> match availStreams |> Set.toList with
                                   | s::_ -> s
                                   | [] -> newStream ()

            // store stream
            streamOfUnit <- streamOfUnit |> Map.add eu.Id euStream

            // emit ExecUnit start marker
            ExecUnitStart eu.Id |> emitToStream euStream
               
            // our stream needs to wait on the results of the streams we depend on
            let endingUnits = eu.DependsOn |> List.filter (fun pId -> streamOfUnit.[pId] <> euStream)
            for endingUnitId in endingUnits do            
                match eventOfUnit |> Map.tryFind endingUnitId with
                | Some evt ->
                    // wait on already emitted event, if possible
                    WaitOnEvent evt |> emitToStream euStream
                | None ->
                    // if possible, reuse an existing event object
                    let evtObjId =
                        match tryFindAvailableEventObjectFor (execUnitById endingUnitId) with
                        | Some id -> id
                        | None -> newEventObjectId()
                    let evt = {EventObjectId=evtObjId; CorrelationId=newCorrelationId(); EmittingExecUnitId=endingUnitId}
                    eventOfUnit <- eventOfUnit |> Map.add endingUnitId evt
                
                    // fill in event placeholder and wait for event
                    eventPlaceHolders.[endingUnitId] := Some evt
                    WaitOnEvent evt |> emitToStream euStream

            // find out which RerunAfter constraints are not yet satisfied on this stream
            let rerunsSatisfiedByDeps =
                endingUnits
                |> List.map (fun pId ->
                    rerunsSatisfiedOnStream streamOfUnit.[pId] (Some pId))
                |> Seq.concat
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


