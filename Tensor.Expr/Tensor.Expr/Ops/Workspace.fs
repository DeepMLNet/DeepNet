namespace Tensor.Expr.Ops

open System
open System.Threading

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Backend



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

        /// Tensor values for ops that returned dynamic values.
        let dynStubVals = ConcurrentDictionary<TensorStub, ITensor> () 
        /// Users of dynamic tensor storages.
        let dynStubUsers = ConcurrentDictionary<ITensorStorage, HashSet<ActionGroup>> ()


        /// Gets the tensor for the specified tensor stub.
        let tensorForStub (stub: TensorStub) : ITensor =
            // Construct value for tensor stub.
            let value =
                match stub.Storage with
                | StorageStub.Dynamic ->
                    dynStubVals.TryFind stub
                    |> Option.defaultWith (fun () ->
                        failwithf "Value for tensor stub %A with dynamic storage is unknown." stub)
                | StorageStub.Allocated allocStub ->
                    let storage =
                        storages.TryFind allocStub 
                        |> Option.defaultWith (fun () ->
                            failwithf "Storage for tensor stub %A with allocated storage is unknown." stub)
                    match stub.Layout with
                    | Some layout -> Tensor.NewOfType (layout, storage)
                    | None ->
                        dynStubVals.TryFind stub
                        |> Option.defaultWith (fun () ->
                            failwithf "Value for tensor stub %A with dynamic layout is unknown." stub)                        
                | StorageStub.VarStorage varName ->
                    // Lookup variable from variable storage.
                    match varValues |> Map.tryFind varName with
                    | Some value -> value
                    | None -> failwithf "Variable for tensor stub %A was not specified." stub
                | StorageStub.Fixed storage ->
                    match stub.Layout with
                    | Some layout -> Tensor.NewOfType (layout, storage)
                    | None ->
                        dynStubVals.TryFind stub
                        |> Option.defaultWith (fun () ->
                            failwithf "Value for tensor stub %A with dynamic layout is unknown." stub)                     

            // Check that layout matches, if the stub specified one.
            match stub.OffsetStride with
            | Some (offset, stride) when offset <> value.Layout.Offset || 
                                         stride <> value.Layout.Stride ->
                failwithf "Value for tensor stub %A with layout %A has incompatiable layout %A."
                    stub stub.Layout.Value value.Layout
            | _ -> ()

            value


        let execData: ExecuteData = {
            StubValue = tensorForStub
        }

        let execFn (actGrp: ActionGroup) =
            let result = actGrp.Action.Execute execData

            // Now handle the dynamic channel data.
            for KeyValue(ch, value) in result.DynamicChValues do
                match actGrp.ChStubs |> Map.tryFind ch with
                | Some stub ->
                    // Check that channel is expected to be dynamic.
                    match stub.OffsetStride, stub.Storage with
                    | None, _ -> ()
                    | _, StorageStub.Dynamic -> ()
                    | _ -> 
                        failwithf "Action group %A returned a dynamic value for non-dynamic channel %A."
                            actGrp ch

                    // Store value.
                    dynStubVals.[stub] <- value

                    // Mark it as unused.
                    // How does that work multi-threaded?
                    // Have to use concurrent dictionaries for that to work.
                    // Yes, sounds good.
                    // Problem is yet that tensor stubs are compared using structured equality.



                | None ->
                    failwithf "Action group %A returned a dynamic value for a non-existant channel %A."
                        actGrp ch

            ()

        iter execFn actGrps
        
    // should variables be replaceable from run to run?
    // Yes, why not?



    interface IDisposable with
        member this.Dispose () =
            if not _disposed then 
                disposeStorage storages
                _disposed <- true



    // Tensor is made disposable, so what next?
    // => Create tensors for the TensorStubs.
