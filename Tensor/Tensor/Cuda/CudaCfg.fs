namespace Tensor.Cuda

open System
open System.Threading

open ManagedCuda
open ManagedCuda.BasicTypes



/// <summary>Options for configuring operations performed on CUDA tensors.</summary>
/// <remarks>All settings are local to the calling thread.</remarks>
/// <seealso cref="CudaTensor"/>
type Cfg private () = 

    static let stream = new ThreadLocal<CUstream> (fun () -> CUstream.NullStream)
    static let stacktrace = new ThreadLocal<bool> (fun () -> false)
    static let fastKernelMath = new ThreadLocal<bool> (fun() -> false)
    static let restrictKernels = new ThreadLocal<bool> (fun() -> false)
    static let debugCompile = new ThreadLocal<bool> (fun() -> false)
    static let disableKernelCache = new ThreadLocal<bool> (fun() -> false)

    /// <summary>The CUDA stream to execute CUDA operations on.</summary>
    /// <remarks>This setting is local to the calling thread and defaults to the null stream.</remarks>
    static member Stream
        with get() = stream.Value
        and set(v) = stream.Value <- v

    /// <summary>If set to true, CUDA operations produce an acurate stack trace when an error is encountered.</summary>
    /// <remarks>
    /// <para>Setting this to <c>true</c> affects performance, even if no error occurs.</para>
    /// <para>This setting is local to the calling thread and defaults to <c>false</c>.</para>
    /// </remarks>
    static member Stacktrace
        with get() = stacktrace.Value
        and set(v) = stacktrace.Value <- v
        
    /// <summary>If set to true, CUDA uses fast math functions with lower accuracy.</summary>
    /// <remarks>This setting is local to the calling thread and defaults to <c>false</c>.</remarks>
    static member FastKernelMath 
        with get() = fastKernelMath.Value
        and set(v) = fastKernelMath.Value <- v

    /// <summary>If set to true, all arguments are passed as restriced to CUDA kernels (experimental).</summary>
    /// <remarks>This setting is local to the calling thread and defaults to <c>false</c>.</remarks>
    static member RestrictKernels 
        with get() = restrictKernels.Value
        and set(v) = restrictKernels.Value <- v

    /// <summary>If set to true, all CUDA kernels are compiled with debug flags.</summary>
    /// <remarks>This setting is local to the calling thread and defaults to <c>false</c>.</remarks>
    static member DebugCompile 
        with get() = debugCompile.Value
        and set(v) = debugCompile.Value <- v
        
    /// <summary>If set to true, the CUDA kernel cache is disabled.</summary>
    /// <remarks>This setting is local to the calling thread and defaults to <c>false</c>.</remarks>
    static member DisableKernelCache 
        with get() = disableKernelCache.Value
        and set(v) = disableKernelCache.Value <- v
        
     