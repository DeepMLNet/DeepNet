namespace Tensor.Expr.Ops

open System

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Backend



type BaseExprWorkspace (recipe: CompileResult) =

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

    let tensorForStub (stub: TensorStub) =
        // Also what to do when OffsetStride is unknown?
        // So how to do the propagation actually?
        // Obviously we chose to work with static and dynamic tensors.
        // So?
        // How to perform the evaluation?
        // Well, run the action groups.
        // So what can happen is that we get dynamic tensors returned from an action.
        // These dynamic tensors must somehow be stored?
        // We can store them as a result of an expression.
        // Yes, but how to attach them to the arguments?
        // Not necessarily a problem.
        // Try to write the execution code and see what happens.
        ()

    let execute (actGrps: ActionGroup list) =
        ()
        // So what do we want to support?
        // Best execution model is a step-by-step model, where we
        // execute all action groups that become available, until
        // no execution is possible anymore, due to external factors.
        // But this is a bit conflicting with the requirement of
        // overwritable args.
        // Also how to determine of part of the graph can be preexecuted?
        // Perhaps preexecution is not a good idea and should not be done?
        // Yes.
        // We could replace this idea by using the interpreted evaluator to
        // evaluate parts of the graph and replace them by Data.
        // So, pre-execution support is killed.
        // Do we want to have step-wise execution?
        // I.e. run this whole thing step-by-step for some reasons?
        // Useful for debuggers maybe?
        // maybe, but we have trace for that, so maybe don't worry about that for now.
        // Okay, so how to do this parallel execution?
        // Well...
        // That was not straight-forward with CUDA.
        // For host we just use dispatch the actions to worker threads?
        // 

    interface IDisposable with
        member this.Dispose () =
            if not _disposed then 
                disposeStorage storages
                _disposed <- true



    // Tensor is made disposable, so what next?
    // => Create tensors for the TensorStubs.
