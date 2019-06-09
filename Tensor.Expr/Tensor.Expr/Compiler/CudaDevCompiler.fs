module Tensor.Expr.Compiler.Cuda

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Expr.Base
open Tensor.Expr.Compiler
open Tensor.Cuda


/// A stub for a CUDA stream.
type Stream = Stream of int

/// A stub for a CUDA event.
type Event = Event of int


type ActionCudaData = {
    /// CUDA stream to execute on.
    Stream: Stream
    /// CUDA events to wait upon before executing.
    WaitEvents: Event list
    /// CUDA events to emit after execution.
    EmitEvent: Event option
} with
    interface IActionDeviceData



/// Assigns a CUDA stream to each action group.
/// Streams are assigned for maximum possible parallelism and reused if possible.
let assignStreams (dev: TensorCudaDevice) (group: BaseExprGroup) (actionGroups: Map<BaseExpr, ActionNode>) = 
    let availStreams = Queue<Stream> ()
    let mutable streamCount = 0

    /// Gets an available stream or allocates a new one.
    let acquireStream () =
        if availStreams.Count > 0 then
            availStreams.Dequeue()
        else
            let stream = Stream streamCount
            streamCount <- streamCount + 1
            stream

    /// Marks the specified stream as available.
    let releaseStream stream =
        availStreams.Enqueue stream

    let actionGroups = Dictionary<BaseExpr, ActionNode> ()

    /// True if the stream of the expression has been reused by one of its dependants.
    let streamReused = Dictionary<BaseExpr, bool> ()

    let processExpr expr =
        let actGrp = actionGroups.[expr]
        match actGrp.Dev with
        | :? TensorCudaDevice as aDev when aDev = dev ->

            // Try to take over stream from an arguments.
            let argStream =
                actGrp.DependsOn
                |> Seq.tryPick (fun arg ->
                    match arg.DevData with
                    | Some (:? ActionCudaData as devData) when not streamReused.[arg.Expr.Value] ->
                        streamReused.[arg.Expr.Value] <- true
                        Some devData.Stream
                    | _ -> None)
                
            // Use any available stream, if takeover was not possible.
            let stream =
                match argStream with
                | Some stream -> stream
                | None -> acquireStream ()
            streamReused.[expr] <- false

            // Store CUDA-specifc data to action group.
            let devData = {
                Stream = stream
                WaitEvents = []
                EmitEvent = None
            }     
            actionGroups.[expr] <-
                {actGrp with DevData=Some (devData :> IActionDeviceData)}
        | _ -> ()
            
    let allDepsProcessed expr =
        match actionGroups.[expr].DevData with
        | Some (:? ActionCudaData as devData) ->
            // Release the stream, if it has not been used by any dependant.
            if not streamReused.[expr] then
                releaseStream devData.Stream
        | _ -> ()

    group.Iter (processExpr, allDepsOfExprEvaled=allDepsProcessed)
    Map.ofDictionary actionGroups



/// Event holder for event reuse in `assignSyncEvents`.
[<ReferenceEquality>]
type private EventHolder = {
    Event: Event
    NecessaryDependingOn: BaseExpr list
    mutable Available: bool
}

/// Assigns CUDA events to action groups to make dependants on other streams
/// wait until the execution of the dependee is completed.
let assignSyncEvents (dev: TensorCudaDevice) (group: BaseExprGroup) (actionGroups: Map<BaseExpr, ActionNode>) =   
    let actionGroups = Dictionary actionGroups
    let depending = BaseExprTransitiveDependencies ()

    let mutable eventCount = 0
    let reuseableEvents = Dictionary<BaseExpr, EventHolder list> ()

    /// Propagates reuseable events from the arguments of an expression
    /// to the expression itself.
    let propagateReuseableEvents (expr: BaseExpr) =
        reuseableEvents.[expr] <-
            expr.Args
            |> Map.toSeq
            |> Seq.collect (fun (arg, argExprCh) ->
                reuseableEvents.[argExprCh.Expr]
                |> Seq.filter (fun eh -> eh.Available))
            |> List.ofSeq                

    /// Adds a reuseable event to an expression.
    let addReuseableEvent (expr: BaseExpr) event =
        let eh = {
            Event = event
            Available = true
            NecessaryDependingOn = group.Dependants expr |> List.ofSeq
        }
        reuseableEvents.[expr] <- eh :: reuseableEvents.[expr]

    /// Acquires an event for the specified expression.
    /// First, reusing an event is tried.
    /// If not possible, a new event is allocated.
    let acquireEvent (expr: BaseExpr) =
        let reuseable =
            reuseableEvents.[expr]
            |> Seq.filter (fun eh -> eh.Available)
            |> Seq.filter (fun eh ->
                eh.NecessaryDependingOn
                |> Seq.filter (fun dep -> dep <> expr)
                |> Seq.forall depending.[expr].Contains)
        match Seq.tryHead reuseable with
        | Some reusedHolder ->
            reusedHolder.Available <- false
            reusedHolder.Event
        | None ->
            let event = Event eventCount
            eventCount <- eventCount + 1
            event

    /// Processes an expression.
    let processExpr (expr: BaseExpr) =   

        // Update transitive dependencies.
        depending.Process expr

        // Propagate reuseable events from arguments.
        propagateReuseableEvents expr

        let actGrp = actionGroups.[expr]
        match actGrp.DevData with
        | Some (:? ActionCudaData as devData) ->

            // Check if this expression's dependants will need to synchronize on this expression.
            let syncNecessary =
                group.Dependants expr
                |> Seq.exists (fun depExpr ->
                    match actionGroups.[depExpr].DevData with
                    | Some (:? ActionCudaData as depDevData) when depDevData.Stream <> devData.Stream ->
                        true
                    | _ -> false)

            // If so, allocate an event for emitting after this expression has been evaluated.
            let emitEvent =
                if syncNecessary then 
                    let event = acquireEvent expr
                    addReuseableEvent expr event
                    Some event
                else None

            // Create list of events we have to wait upon for synchronization.
            let waitEvents =
                expr.Args
                |> Map.toSeq
                |> Seq.choose (fun (_arg, argExprCh) ->
                    match actionGroups.[argExprCh.Expr].DevData with
                    | Some (:? ActionCudaData as argDevData) when argDevData.Stream <> devData.Stream ->
                        Some argDevData.EmitEvent.Value
                    | _ -> None)
                |> List.ofSeq

            // Update device data in action group.
            let devData = {
                devData with
                    EmitEvent = emitEvent
                    WaitEvents = waitEvents
            }
            actionGroups.[expr] <- {actGrp with DevData = Some (devData :> IActionDeviceData)}
        | _ -> ()

    /// Called after all dependencies of an expression have been processed.
    let allDepsProcessed (expr: BaseExpr) =
        // Clear dependency set of expression to save memory.
        depending.Remove expr 

    group.Iter (processExpr, allDepsOfExprEvaled=allDepsProcessed)
    Map.ofDictionary actionGroups


