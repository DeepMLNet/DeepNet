namespace Tensor.Expr.Ops

open System
open System.Threading

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Backend



[<RequireQualifiedAccess>]
type internal RuntimeStorageTrack =
    | Untracked
    | Tracked of HashSet<ActionGroup>
   

type BaseExprWorkspace (recipe: CompileResult, execThreadCount: int) =

    let mutable _disposed = false

    let allocStorage (stubs: AllocStub list) =
        stubs
        |> List.map (fun stub ->
            stub, stub.Dev.CreateUntyped stub.TypeName.Type stub.Size)
        |> dict

    let disposeStorage (storages: IDictionary<AllocStub, ITensorStorage>) =
        for KeyValue (stub, stor) in storages do
            match stor with
            | :? IDisposable as d -> d.Dispose()
            | _ -> ()

    let storages = allocStorage recipe.Allocs

    /// Iterates over a set of action groups. 
    let iter (fn: ActionGroup -> unit) (actGrps: ActionGroup list) =
        /// Action groups missing for execution of an execution group.
        let missing = Dictionary<ActionGroup, HashSet<ActionGroup>> ()
        /// Action groups that can be executed.
        let ready = ConcurrentQueue<ActionGroup> ()

        /// Initialize list of missing action groups.
        for actGrp in actGrps do
            missing.[actGrp] <- HashSet actGrp.DependsOn
            if actGrp.DependsOn.Count = 0 then
                ready.Enqueue actGrp
              
        /// Stop notification.
        let mutable stop = false
        /// Number of action groups remaining to execute.
        let mutable remaining = actGrps.Length
        /// Lock for number of remaining action groups.
        let remainingLock = obj ()

        /// Event to notify workers of new available work.
        let workEvent = Array.init execThreadCount (fun _ -> new AutoResetEvent (true))
        /// Notifies all workers that new work is available.
        let notifyAll () =
            for evt in workEvent do
                evt.Set() |> ignore
        
        /// Worker thread function.
        let workerFn (threadId: int) =
            while not stop do                
                // Wait for new work to become available.
                workEvent.[threadId].WaitOne() |> ignore

                // Process work until no work is available.
                let mutable hasWork = true
                while hasWork do
                    match ready.TryDequeue () with
                    | Some actGrp ->
                        // Execute action group.
                        fn actGrp
                                                                   
                        // Notify dependants that result is available.
                        for dep in actGrp.Dependants do
                            let missingForDep = missing.[dep]
                            let noMissingDeps = 
                                lock missingForDep (fun () ->
                                    missingForDep.Remove actGrp |> ignore
                                    missingForDep.Count = 0)
                            if noMissingDeps then
                                ready.Enqueue dep
                                notifyAll ()

                        // Check if all action groups have been executed.
                        stop <-
                            lock remainingLock (fun () ->
                                remaining <- remaining - 1
                                remaining = 0)  
                        if stop then
                            notifyAll ()

                        hasWork <- true
                    | None ->
                        hasWork <- false

        // Start worker threads.
        let workers =
            Array.init execThreadCount (fun threadId ->
                Thread (fun () -> workerFn threadId))
        for worker in workers do
            worker.Start ()

        // Wait until work is finished.
        for worker in workers do
            worker.Join ()

        // Clean up events.
        for evt in workEvent do
            (evt :> IDisposable).Dispose ()


    /// Executes a set of action groups.
    let execute (varValues: Map<VarName, ITensor>) (actGrps: ActionGroup list) =

        /// Tensor values for ops that return run-time values.
        let rtStubVals = ConcurrentDictionary<TensorStub, ITensor> () 
        /// Tracking of users of the storage of run-time values.
        let rtStorageTrack = ConcurrentDictionary<ITensorStorage, RuntimeStorageTrack> ()
        
        /// Gets the tensor for the specified tensor stub.
        let tensorForStub (stub: TensorStub) : ITensor =
            if stub.IsRuntime then
                // Lookup value for runtime tensor stub.
                rtStubVals.TryFind stub
                |> Option.defaultWith (fun () ->
                    failwithf "Value for runtime tensor stub %A is unknown." stub)  
            else
                // Construct value for compile-time tensor stub.
                let storage =
                    match stub.Storage with
                    | StorageStub.Allocated allocStub ->
                        storages.TryFind allocStub 
                        |> Option.defaultWith (fun () ->
                            failwithf "Storage for tensor stub %A with allocated storage is unknown." stub)                   
                    | StorageStub.Fixed storage -> storage
                    | StorageStub.Runtime _ -> 
                        failwith "Runtime storage was specified for compile-time tensor stub."
                Tensor.NewOfType (stub.Layout.Value, storage)
                    

        let execData: ExecuteData = {
            StubValue = tensorForStub
        }

        let execFn (actGrp: ActionGroup) =
            // Execute op.
            let result = actGrp.Action.Execute execData

            // Handle run-time channel data.
            for KeyValue(ch, chValue) in result.RuntimeChValues do
                match actGrp.ChStubs |> Map.tryFind ch with
                | Some stub ->
                    // Check that channel has a runtime stub.
                    if not stub.IsRuntime then
                        failwithf "Action group %A returned a runtime value for non-runtime channel %A."
                            actGrp ch

                    // Check that returned value matches layout, if stub specifies one.
                    match stub.Layout with
                    | Some layout when layout <> chValue.Value.Layout ->
                        failwithf "Value for tensor stub %A has incompatiable layout %A."
                            stub chValue.Value.Layout
                    | _ -> ()

                    // Check that shape, data type and device match stub.
                    if stub.Shape <> chValue.Value.Shape then
                        failwithf "Value for tensor stub %A has wrong shape %A." stub chValue.Value.Shape
                    if stub.TypeName.Type <> chValue.Value.DataType then
                        failwithf "Value for tensor stub %A has wrong data type %A." stub chValue.Value.DataType
                    if stub.Dev <> chValue.Value.Dev then
                        failwithf "Value for tensor stub %A has wrong device %A." stub chValue.Value.Dev

                    // Store value.
                    rtStubVals.[stub] <- chValue.Value

                    // Set up tracking, if op allocated storage for its result.
                    let storage = chValue.Value.Storage
                    if chValue.Temporary then
                        // Create tracker, if value is marked as temporary.
                        let tracker = RuntimeStorageTrack.Tracked (HashSet<_> ())
                        let prevTracker = rtStorageTrack.GetOrAdd (storage, tracker)
                        match prevTracker with
                        | RuntimeStorageTrack.Untracked ->
                            failwithf "Action group %A tried marking non-new storage %A of channel %A as temporary."
                                actGrp storage ch
                        | RuntimeStorageTrack.Tracked _ -> ()
                    else
                        // Mark value's storage as untracked, if it is not temporary.
                        rtStorageTrack.TryAdd (storage, RuntimeStorageTrack.Untracked) |> ignore

                    // Perform tracking.
                    match rtStorageTrack.TryGetValue storage with
                    | true, (RuntimeStorageTrack.Tracked users) ->
                        let unused = 
                            lock users (fun () ->
                                // Add channel users to storage users.
                                for chUser in actGrp.RuntimeChStubUsers.[ch] do
                                    users.Add chUser |> ignore
                                // Remove executed action group.
                                // TODO: This does not work.
                                // Removal must be performed over arguments not channels.
                                users.Remove actGrp |> ignore
                                // Check if empty.
                                users.Count = 0)
                        
                        // Dispose storage if it is not needed anymore.
                        if unused then
                            rtStorageTrack.TryRemove storage |> ignore
                            match storage with
                            | :? IDisposable as disp -> disp.Dispose ()
                            | _ -> ()
                    | _ -> ()

                | None ->
                    failwithf "Action group %A returned a runtime value for a non-existant channel %A."
                        actGrp ch

            ()

        iter execFn actGrps
        


    interface IDisposable with
        member this.Dispose () =
            if not _disposed then 
                disposeStorage storages
                _disposed <- true



    // Tensor is made disposable, so what next?
    // => Create tensors for the TensorStubs.
