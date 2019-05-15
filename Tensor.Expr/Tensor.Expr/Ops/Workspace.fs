namespace Tensor.Expr.Ops

open System
open System.Threading

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Backend



/// Tracking of usage of a runtime stub value.
[<RequireQualifiedAccess>]
type internal RuntimeStorageTrack =
    /// Value is not tracked, i.e. not disposed when no longer needed.
    | Untracked
    /// Value is used by the set of action groups and will be disposed
    /// when all dependants of these action groups have been executed.
    | Tracked of HashSet<ActionGroup>
   

/// Storage for runtime stub values.
type internal RuntimeValues () =

    /// Tensor values for ops that return run-time values.
    let stubVals = ConcurrentDictionary<TensorStub, ITensor> () 
    /// Tracking of users of the storage of run-time values.
    let storageTrack = ConcurrentDictionary<ITensorStorage, RuntimeStorageTrack> ()    

    let checkRuntime (stub: TensorStub) =
        if not stub.IsRuntime then
            failwithf "Tensor stub %A is not run-time." stub

    /// Returns the value for a runtime tensor stub.
    member __.Value (stub: TensorStub) =
        checkRuntime stub
        stubVals.TryFind stub
        |> Option.defaultWith (fun () ->
            failwithf "Value for runtime tensor stub %A is unknown." stub)        
            
    /// Adds a value for a runtime tensor stub and starts tracking its storage, 
    /// if the value is temporary.
    member __.AddValue (stub: TensorStub) (chValue: RuntimeChValue) (owner: ActionGroup) =
        checkRuntime stub

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
        stubVals.[stub] <- chValue.Value

        // Set up tracking, if op allocated storage for its result.
        let storage = chValue.Value.Storage
        if chValue.Temporary then
            // Create tracker, if value is marked as temporary.
            let tracker = RuntimeStorageTrack.Tracked (HashSet<_> ())
            let prevTracker = storageTrack.GetOrAdd (storage, tracker)
            match prevTracker with
            | RuntimeStorageTrack.Untracked ->
                failwithf "It was tried to mark already known non-temporary storage %A as temporary." storage 
            | RuntimeStorageTrack.Tracked _ -> ()
        else
            // Mark value's storage as untracked, if it is not temporary.
            storageTrack.TryAdd (storage, RuntimeStorageTrack.Untracked) |> ignore

        // Add action group as owner if its runtime value.
        match storageTrack.TryGetValue storage with
        | true, (RuntimeStorageTrack.Tracked users) ->
            lock users (fun () -> 
                users.Add owner |> ignore)  
        | _ -> ()        

    /// Notifies the value storage that the value for the specified stub is no longer needed
    /// by the action group, i.e. when all its dependants have been executed.
    member __.DoneWithValue (stub: TensorStub) (actGrp: ActionGroup) =
        checkRuntime stub

        let rtStubValue = stubVals.[stub]
        let rtStorage = rtStubValue.Storage

        match storageTrack.TryGetValue rtStorage with
        | true, (RuntimeStorageTrack.Tracked users) ->
            // Remove action group from users of storage.
            let unused =
                lock users (fun () ->
                    users.Remove actGrp |> ignore
                    users.Count = 0)

            // Dispose storage if it is not needed anymore.
            if unused then
                storageTrack.TryRemove rtStorage |> ignore
                match rtStorage with
                | :? IDisposable as disp -> disp.Dispose ()
                | _ -> ()
        | _ -> ()

        // Remove value.
        stubVals.TryRemove stub |> ignore



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

    /// Allocated storages for allocation stubs.
    let storages = allocStorage recipe.Allocs

    /// Iterates over a set of action groups. 
    let iter (fn: ActionGroup -> unit) (allDepsExecedFn: ActionGroup -> unit) (actGrps: ActionGroup list) =
        /// Action groups missing for execution of an execution group.
        let missing = Dictionary<ActionGroup, HashSet<ActionGroup>> ()
        /// Dependants of an action group that have not yet beed executed.
        let unexecedDeps = Dictionary<ActionGroup, HashSet<ActionGroup>> ()
        /// Action groups that can be executed.
        let ready = ConcurrentQueue<ActionGroup> ()

        /// Initialize list of missing action groups.
        for actGrp in actGrps do
            missing.[actGrp] <- HashSet actGrp.DependsOn
            unexecedDeps.[actGrp] <- HashSet actGrp.Dependants
            if actGrp.DependsOn.Count = 0 then
                ready.Enqueue actGrp
              
        /// Stops all worker threads if true.
        let mutable stop = false
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
                        // Execute action group, if it is not the finish marker.
                        match actGrp.Action with
                        | :? FinishAction -> ()
                        | _ -> fn actGrp
                                        
                        // Remove action group from list of not executed dependants
                        // and check if all its dependants have been executed.
                        for dep in actGrp.DependsOn do
                            let unexeced = unexecedDeps.[dep]
                            let allDepsExeced =
                                lock unexeced (fun () ->
                                    unexeced.Remove actGrp |> ignore
                                    unexeced.Count = 0) 
                            if allDepsExeced then
                                allDepsExecedFn actGrp

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

                        // Stop execution if finish action group is reached.
                        match actGrp.Action with
                        | :? FinishAction -> 
                            stop <- true
                            notifyAll ()
                        | _ -> ()

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
        let rtValues = RuntimeValues ()

        /// Gets the tensor for the specified tensor stub.
        let tensorForStub (stub: TensorStub) : ITensor =
            if stub.IsRuntime then
                // Lookup value for runtime tensor stub.
                rtValues.Value stub
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

        let exec (actGrp: ActionGroup) =
            // Execute op.
            let result = actGrp.Action.Execute execData

            // Store run-time channel values.
            let notProcessed = 
                actGrp.ChStubs
                |> Map.toSeq
                |> Seq.filter (fun (ch, chStub) -> chStub.IsRuntime)
                |> Seq.map fst
                |> HashSet
            for KeyValue(ch, chValue) in result.RuntimeChValues do
                let chStub =
                    actGrp.ChStubs 
                    |> Map.tryFind ch
                    |> Option.defaultWith (fun () ->
                        failwithf "Action group %A returned a runtime value for a non-existant channel %A." actGrp ch)
                rtValues.AddValue chStub chValue actGrp
                notProcessed.Remove ch |> ignore
            if notProcessed.Count > 0 then
                failwithf "Action group %A did not return runtime values for channels %A." actGrp notProcessed

        let allDepsExeced (actGrp: ActionGroup) = 
            // Notify storage that runtime channel values of action group are no longer needed.
            for KeyValue(_ch, chStub) in actGrp.ChStubs do
                if chStub.IsRuntime then
                    rtValues.DoneWithValue chStub actGrp

        iter exec allDepsExeced actGrps
        

    interface IDisposable with
        member this.Dispose () =
            if not _disposed then 
                disposeStorage storages
                _disposed <- true

