namespace Tensor.Cuda

open System
open System.Threading

open ManagedCuda
open ManagedCuda.BasicTypes



/// CUDA backend configuration
type Cfg private () = 

    static let stream = new ThreadLocal<CUstream> (fun () -> CUstream.NullStream)
    static let stacktrace = new ThreadLocal<bool> (fun () -> false)
    static let fastKernelMath = new ThreadLocal<bool> (fun() -> false)
    static let restrictKernels = new ThreadLocal<bool> (fun() -> false)
    static let debugCompile = new ThreadLocal<bool> (fun() -> false)
    static let disableKernelCache = new ThreadLocal<bool> (fun() -> false)

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
        
    /// If set to true, CUDA uses fast math functions with lower accuracy.
    /// This setting is local to the calling thread and defaults to false.
    static member FastKernelMath 
        with get() = fastKernelMath.Value
        and set(v) = fastKernelMath.Value <- v

    /// If set to true, all arguments are passed as restriced to CUDA kernels (experimental).
    /// This setting is local to the calling thread and defaults to false.
    static member RestrictKernels 
        with get() = restrictKernels.Value
        and set(v) = restrictKernels.Value <- v

    /// If set to true, all CUDA kernels are compiled with debug flags.
    /// This setting is local to the calling thread and defaults to false.
    static member DebugCompile 
        with get() = debugCompile.Value
        and set(v) = debugCompile.Value <- v
        
    /// If set to true, the CUDA kernel cache is disabled.
    /// This setting is local to the calling thread and defaults to false.
    static member DisableKernelCache 
        with get() = disableKernelCache.Value
        and set(v) = disableKernelCache.Value <- v
        
     