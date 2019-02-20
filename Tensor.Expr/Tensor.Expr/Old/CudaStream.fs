namespace Tensor.Expr.Compiler.Cuda

open System.Diagnostics
open System.Collections.Generic

open Tensor.Utils
open DeepNet.Utils

open Tensor.Expr
open Tensor.Expr.Compiler


[<AutoOpen>]
module CudaCmdTypes =

    /// a placeholder (reference to) an event
    type EventPlaceHolderT = EventT option ref

    /// a command executed on a stream
    type CudaCmdT =
        | Perform                   of CudaExecItemT
        | WaitOnEvent               of EventT
        | EmitEvent                 of EventPlaceHolderT
        | WaitOnRerunEvent          of EventT
        | EmitRerunEvent            of EventPlaceHolderT
        | RerunSatisfied            of ExecUnitIdT
        | ExecUnitStart             of ExecUnitIdT
        | ExecUnitEnd               of ExecUnitIdT

    /// a sequence of commands executed on a stream
    type StreamCmdsT = CudaCmdT list


module CudaStreamSeq =

    type private StreamInfoT = {
        Stream:                     StreamT
        mutable Available:          bool
        ReuseCandidateUnits:        Set<ExecUnitIdT>
    }

    type private EventRelationshipT = {
        Emitter:                    ExecUnitIdT
        Waiter:                     ExecUnitIdT
    }

    type private EventInfoT = {
        EmittedEvent:               EventPlaceHolderT
        mutable Available:          bool
        Waiters:                    HashSet<ExecUnitIdT>
        WaitedUpon:                 Set<EventRelationshipT>
        ReuseCandidateUnits:        Set<ExecUnitIdT>  
    }

    /// converts execution units to stream commands
    let execUnitsToStreams (execUnits: ExecUnitT list) : (StreamCmdsT list * int) =
        /// collection of ExecUnits we have to process
        let coll = ExecUnit.Collection execUnits

        /// event counter
        let mutable eventObjectCnt = 0
        /// correlation counter
        let mutable correlationCnt = 0
        /// all allocated streams
        let streams = ResizeArray<ResizeArray<CudaCmdT>> ()
        /// rerun events
        let rerunEvent = Dictionary<ExecUnitIdT, EventPlaceHolderT> ()
        /// stream reuse information
        let streamInfo = Dictionary<ExecUnitIdT, StreamInfoT> ()
        /// event reuse information
        let eventInfo = Dictionary<ExecUnitIdT, EventInfoT> ()
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
            streams.Add (ResizeArray<CudaCmdT> ())
            streams.Count - 1

        /// length of a stream
        let streamLength (s: int) = streams.[s].Count

        /// emits an ExeOp to the given streams
        let emitToStream s (exeOp: CudaCmdT) =
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

                // Check if an existing stream can be reused for this execution unit.
                let streamToReuse =
                    // Try to take over the stream of a unit we depend on directly.
                    let streamTakeOverCands = 
                        coll.DependsOn eu
                        |> List.choose (fun deu -> 
                            match streamInfo.[deu.Id] with
                            | {Available=true} as sr -> Some sr
                            | {Available=false} -> None)

                    if not (List.isEmpty streamTakeOverCands) then
                        // Take over the stream with least commands in it.
                        Some (streamTakeOverCands |> List.minBy (fun sr -> streamLength sr.Stream))
                    else
                        // No stream available from a unit we depend on directly.                       
                        // Build list of streams that can be reused.
                        let reuseCands = seq {
                            for deu in coll.DependsOn eu do
                                for ceuId in streamInfo.[deu.Id].ReuseCandidateUnits do
                                    match streamInfo.[ceuId] with
                                    | {Available=true} as sr -> yield sr
                                    | {Available=false} -> ()
                        }
                        // Use the first available reusable stream, if any is available.
                        Seq.tryHead reuseCands 

                // Determine stream of this execution unit.
                let euStream =
                    match streamToReuse with
                    | Some sr -> 
                        sr.Available <- false
                        sr.Stream
                    | None -> newStream ()                                  

                // Build stream reuse information for this execution unit.
                let candidateReuseUnits = [
                    for deu in coll.DependsOn eu do
                        let sr = streamInfo.[deu.Id]
                        if sr.Available then yield deu.Id
                        for ceuId in sr.ReuseCandidateUnits do
                            if streamInfo.[ceuId].Available then yield ceuId
                ]
                streamInfo.[eu.Id] <- {
                    Stream          = euStream
                    Available       = true
                    ReuseCandidateUnits  = candidateReuseUnits |> Set.ofList 
                }
                 
                // emit ExecUnit start marker
                ExecUnitStart eu.Id |> emitToStream euStream
               
                // our stream needs to wait on the results of the streams we depend on
                let euWaitingUponDirectly = ResizeArray<EventRelationshipT> ()
                let endingUnits = eu.DependsOn |> Seq.filter (fun pId -> streamInfo.[pId].Stream <> euStream)
                for endingUnitId in endingUnits do    
                    let eor = eventInfo.[endingUnitId]                        
                    match !eor.EmittedEvent with
                    | Some evt ->
                        // wait on already emitted event, if possible
                        WaitOnEvent evt |> emitToStream euStream
                    | None ->
                        // find a reuseable event object or create a new one
                        let evtObjId =                            
                            let reuseable = seq {
                                // for all canididate units:
                                for ceuId in eor.ReuseCandidateUnits do
                                    // check that candidate unit has all its dependants processed
                                    let ceu = coll.ById ceuId
                                    let dependantsProcessed =
                                        coll.DependantsOf ceu   
                                        |> Seq.forall (fun deu -> processedExecUnitIds.Contains deu.Id)
                                    if dependantsProcessed then
                                        // check that candidate unit has available event 
                                        let ceor = eventInfo.[ceuId]
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

                        // assign an event to the last ExecUnit of the ending stream
                        let evt = {EventObjectId=evtObjId; CorrelationId=newCorrelationId(); EmittingExecUnitId=endingUnitId}
                
                        // fill in event placeholder and wait for event
                        eor.EmittedEvent := Some evt
                        WaitOnEvent evt |> emitToStream euStream

                    // add this execution unit to the waiters list of units we wait upon
                    eor.Waiters.Add eu.Id |> ignore

                    // add to list of units this units waits upon directly
                    euWaitingUponDirectly.Add {
                        Emitter = endingUnitId
                        Waiter  = eu.Id
                    }                    

                // find out which RerunAfter constraints are not yet satisfied on this stream
                let rerunsSatisfiedByDeps =
                    endingUnits
                    |> Seq.collect (fun pId -> rerunsSatisfiedOnStream streamInfo.[pId].Stream (Some pId))
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
                    Perform (cmd :?> CudaExecItemT) |> emitToStream euStream

                // emit an event placeholder to allow for synchronization
                let evtPh = ref None
                EmitEvent evtPh |> emitToStream euStream

                // build event reuse information 
                let waitingUpon = [
                    yield! euWaitingUponDirectly
                    for deu in coll.DependsOn eu do
                        let deor = eventInfo.[deu.Id]
                        for er in deor.WaitedUpon do
                            if eventInfo.[er.Emitter].Available then
                                yield er
                ]
                eventInfo.[eu.Id] <- {
                    EmittedEvent    = evtPh
                    Available       = true
                    Waiters         = HashSet<ExecUnitIdT> ()
                    WaitedUpon      = waitingUpon |> Set.ofList
                    ReuseCandidateUnits  = waitingUpon |> List.map (fun er -> er.Emitter) |> Set.ofList
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


