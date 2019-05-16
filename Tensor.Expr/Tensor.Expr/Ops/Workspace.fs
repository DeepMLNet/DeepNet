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
    member this.Value (stub: TensorStub) =
        checkRuntime stub
        stubVals.TryFind stub
        |> Option.defaultWith (fun () ->
            failwithf "Value for runtime tensor stub %A is unknown." stub)    

            
    /// Returns true if the runtime tensor stub has a value.
    member this.HasValue (stub: TensorStub) =
        checkRuntime stub
        stubVals.ContainsKey stub

            
    /// Adds a value for a runtime tensor stub and starts tracking its storage, 
    /// if the value is temporary.
    member this.AddValue (stub: TensorStub) (chValue: ITensor) =
        checkRuntime stub

        // Check that returned value matches layout, if stub specifies one.
        match stub.Layout with
        | Some layout when layout <> chValue.Layout ->
            failwithf "Value for tensor stub %A has incompatiable layout %A."
                stub chValue.Layout
        | _ -> ()

        // Check that shape, data type and device match stub.
        if stub.Shape <> chValue.Shape then
            failwithf "Value for tensor stub %A has wrong shape %A." stub chValue.Shape
        if stub.TypeName.Type <> chValue.DataType then
            failwithf "Value for tensor stub %A has wrong data type %A." stub chValue.DataType
        if stub.Dev <> chValue.Dev then
            failwithf "Value for tensor stub %A has wrong device %A." stub chValue.Dev

        // Store value.
        let oldChValue = stubVals.GetOrAdd (stub, chValue)
        if oldChValue <> chValue then
            failwithf "Value for tensor stub %A cannot be changed from %A to %A." 
                stub oldChValue chValue

        // Set up tracking, if op allocated storage for its result.
        match stub.Storage with
        | StorageStub.Temporary _ ->
            // Create tracker, if value is marked as temporary.
            let tracker = RuntimeStorageTrack.Tracked (HashSet<_> ())
            let prevTracker = storageTrack.GetOrAdd (chValue.Storage, tracker)
            match prevTracker with
            | RuntimeStorageTrack.Untracked ->
                failwithf "Already known external storage %A cannot be returned as temporary." 
                    chValue.Storage
            | RuntimeStorageTrack.Tracked _ -> ()
        | StorageStub.External _ ->
            // Mark value's storage as untracked, if it is not temporary.
            storageTrack.TryAdd (chValue.Storage, RuntimeStorageTrack.Untracked) |> ignore
        | _ -> failwith "Unknown storage stub for runtime tensor stub."


    /// Adds the action group as user of the runtime value of the tensor stub.
    member this.AddUser (stub: TensorStub) (owner: ActionGroup) =
        let chValue = this.Value stub 
        match storageTrack.TryGetValue chValue.Storage with
        | true, (RuntimeStorageTrack.Tracked users) ->
            lock users (fun () -> 
                users.Add owner |> ignore)  
        | _ -> ()           
        

    /// Notifies the value storage that the value for the specified stub is no longer needed
    /// by the action group, i.e. when all its dependants have been executed.
    member this.DoneWithValue (stub: TensorStub) (actGrp: ActionGroup) =
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



type BaseExprWorkspace (recipe: CompileResult) =
    let mutable _disposed = false
    let checkDisposed () =
        if _disposed then
            raise (ObjectDisposedException ("BaseExprWorkSpace"))

    /// Allocate and return storage for the specified storage stubs.
    let allocStorage (stubs: AllocStub list) =
        stubs
        |> List.map (fun stub ->
            stub, stub.Dev.CreateUntyped stub.TypeName.Type stub.Size)
        |> dict

    /// Disposes the storage.
    let disposeStorage (storages: IDictionary<AllocStub, ITensorStorage>) =
        for KeyValue (_stub, stor) in storages do
            match stor with
            | :? IDisposable as d -> d.Dispose()
            | _ -> ()

    /// Allocated storages for allocation stubs.
    let storages = allocStorage recipe.Allocs

    /// Iterates over a set of action groups. 
    let iter (fn: ActionGroup -> unit) (allDepsExecedFn: ActionGroup -> unit) 
            (execThreadCount: int option) (actGrps: ActionGroup list) =

        let execThreadCount = defaultArg execThreadCount Environment.ProcessorCount
        if execThreadCount < 1 then 
            failwith "Execution thread count must be at least 1."

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
                        // Check for finish marker.
                        match actGrp.Action with
                        | :? FinishAction -> 
                            // Stop execution.
                            stop <- true
                            hasWork <- false
                            notifyAll ()
                        | _ -> 
                            // Execute op.
                            fn actGrp
                                        
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
    let execute (execEnv: ExecuteEnv) (actGrps: ActionGroup list) =
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
                    | StorageStub.Temporary _ 
                    | StorageStub.External _ -> 
                        failwith "Runtime storage was specified for compile-time tensor stub."
                Tensor.NewOfType (stub.Layout.Value, storage)
                    
        /// Execution data.
        let execData: ExecuteData = {
            Env = execEnv
            StubValue = tensorForStub
        }

        /// Executes the action group.
        let exec (actGrp: ActionGroup) =
            // Execute op.
            let result = actGrp.Action.Execute execData

            // Store run-time channel values.
            let notReturned = 
                actGrp.ChStubs
                |> Map.toSeq
                |> Seq.filter (fun (_ch, chStub) -> chStub.IsRuntime)
                |> Seq.map fst
                |> HashSet
            for KeyValue(ch, chValue) in result.RuntimeChValues do
                let chStub =
                    actGrp.ChStubs 
                    |> Map.tryFind ch
                    |> Option.defaultWith (fun () ->
                        failwithf "Action group %A returned a runtime value for a non-existant channel %A." actGrp ch)
                rtValues.AddValue chStub chValue 
                rtValues.AddUser chStub actGrp
                notReturned.Remove ch |> ignore
            for ch in notReturned do
                let chStub = actGrp.ChStubs.[ch]
                if not (rtValues.HasValue chStub) then
                    failwithf "Action group %A did not return a runtime value for channel %A which has no value so far." 
                        actGrp ch

        /// Called when all dependencies of the specified action group have been executed.
        let allDepsExeced (actGrp: ActionGroup) = 
            // Notify storage that runtime channel values of action group are no longer needed.
            for KeyValue(_ch, chStub) in actGrp.ChStubs do
                if chStub.IsRuntime then
                    rtValues.DoneWithValue chStub actGrp

        // Perform parallel execution of action groups.
        iter exec allDepsExeced execEnv.ThreadCount actGrps

        // Get result values.
        // These may point into allocated storage or variables.
        let resultVals =
            recipe.ResultStubs
            |> Map.map (fun _exprCh stub -> 
                let value = tensorForStub stub
                if execEnv.RawResults then value
                else ITensor.copy value)
        resultVals


    /// Executes the workspace and returns the values of the root expressions.
    member this.Execute (env: ExecuteEnv) : Map<BaseExprCh, ITensor> =
        checkDisposed ()
        execute env recipe.ActionGroups


    interface IDisposable with
        member this.Dispose () =
            if not _disposed then 
                disposeStorage storages
                _disposed <- true

