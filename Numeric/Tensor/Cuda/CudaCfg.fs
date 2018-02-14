namespace Tensor.Cuda

open System
open System.Threading

open ManagedCuda
open ManagedCuda.BasicTypes



/// CUDA backend configuration
module Cfg = 

    let mutable FastKernelMath = false
    let mutable RestrictKernels = false
    let mutable DebugCompile = false
    let mutable DisableKernelCache = false


/// CUDA backend configuration
type Cfg () = 

    static let stream = new ThreadLocal<CUstream> (fun () -> CUstream.NullStream)
    static let stacktrace = new ThreadLocal<bool> (fun () -> false)

    /// The CUDA stream to execute CUDA operations on.
    /// This setting is local to the calling thread and defaults to the null stream.
    static member Stream
        with get() = stream.Value
        and set(v) = stream.Value <- v

    /// If set to true, CUDA operations produce an acurate stack trace
    /// when an error is encountered. However, this affects performance,
    /// even if no error occurs.
    /// This setting is local to the calling thread and defaults to false.
    static member Stacktrace
        with get() = stacktrace.Value
        and set(v) = stacktrace.Value <- v
        