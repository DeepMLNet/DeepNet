namespace Tensor.Expr.Ops

open DeepNet.Utils
open Tensor
open Tensor.Expr
open Tensor.Cuda



type ActionCudaData = {
    /// CUDA stream to execute on.
    Stream: int
    /// True, if a dependant has taken over the stream.
    mutable StreamUsedByDependant: bool
    /// CUDA events to wait upon before executing.
    WaitEvents: int list
    /// CUDA events to emit after execution.
    EmitEvents: int list
} with
    interface IActionDeviceData


type CudaDevHandler (dev: TensorCudaDevice) =

    // Type is instantiated for one compilation session.
    // 

    let availStreams = ResizeArray<int> ()

    interface IDevHandler with
        member __.ProcessActionGroup actGrp =
            match actGrp.Dev with
            | :? TensorCudaDevice as aDev when aDev = dev ->
                // Need to assign stream. 
                // Try to take over stream from argument.
                // Need to check if this is last dependant of a previously processed action group.
                // If so, free streams that were not taken over.
                // 
                failwith "TODO"
            | _ -> None
            