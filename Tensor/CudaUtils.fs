namespace Tensor.Utils

open ManagedCuda
open ManagedCuda.BasicTypes


[<AutoOpen>]
module CudaSupTypes =

    /// dimensionality of parallel work to perform (x, y, z)
    type WorkDimT = int64 * int64 * int64

    /// CUDA block dimension (x, y, z)
    type BlockDimT = int * int * int

    /// CUDA grid dimension (x, y, z)
    type GridDimT = int * int * int

    /// CUDA launch dimension
    type LaunchDimT = {
        Block: BlockDimT
        Grid:  GridDimT
    }


module CudaSup =

    /// convert block/grid dimension (x, y, z) to VectorTypes.dim3
    let toDim3 d =
        let (x: int), y, z = d
        VectorTypes.dim3 (x, y, z)

    /// CUDA context
    let context = 
        let cudaCntxt = 
            try
                new CudaContext(createNew=false)
            with e ->
                printfn "Cannot create CUDA context: %s" e.Message
                failwith "Cannot creat CUDA context"
                exit 10

        cudaCntxt

    let deviceInfo =
        context.GetDeviceInfo()

    let maxBlockDim =
        int deviceInfo.MaxBlockDim.x, int deviceInfo.MaxBlockDim.y, int deviceInfo.MaxBlockDim.z

    let maxGridDim =
        int deviceInfo.MaxGridDim.x, int deviceInfo.MaxGridDim.y, int deviceInfo.MaxGridDim.z
    
    let nvccArch =
        sprintf "compute_%d%d" deviceInfo.ComputeCapability.Major deviceInfo.ComputeCapability.Minor

    let nvccCode =
        sprintf "sm_%d%d" deviceInfo.ComputeCapability.Major deviceInfo.ComputeCapability.Minor

    let printInfo () =
        let di = deviceInfo
        printfn "CUDA device:                                         %s" di.DeviceName
        printfn "CUDA driver version:                                 %A" di.DriverVersion
        printfn "CUDA device global memory:                           %A bytes" di.TotalGlobalMemory
        printfn "CUDA device free memory:                             %A bytes" (context.GetFreeDeviceMemorySize())
        printfn "CUDA device compute capability:                      %A" di.ComputeCapability
        printfn "CUDA device maximum block size:                      %A" di.MaxThreadsPerBlock                       
        printfn "CUDA device maximum block dimensions:                %A" di.MaxBlockDim
        printfn "CUDA device maximum grid dimensions:                 %A" di.MaxGridDim    
        printfn "CUDA device async engine count:                      %d" di.AsyncEngineCount
        printfn "CUDA device can execute kernels concurrently:        %A" di.ConcurrentKernels
        printfn "CUDA device can overlap kernels and memory transfer: %A" di.GpuOverlap

    let printDevice () =
        let di = deviceInfo
        printfn "Using CUDA device \"%s\" with %d multiprocessors @ %.2f GHz and %d MB memory." 
            di.DeviceName di.MultiProcessorCount 
            (float di.ClockRate / 10.0**6.0) (int64 di.TotalGlobalMemory / pown 2L 20)

    // CUDA BLAS handle
    let blas =
        new CudaBlas.CudaBlas()

    /// Ensures that CUDA is initialized. Multiple calls are allowed and have no effect.
    let init () =       
        // make a dummy call on the context to ensure that it is created
        context.GetSharedMemConfig() |> ignore

    /// shutsdown CUDA (necessary for correct profiler results)  
    let shutdown () =
        context.Synchronize ()
        CudaContext.ProfilerStop ()
        context.Synchronize ()
        blas.Dispose ()
        context.Dispose ()

    /// Checks that the thread's current CUDA context is the CUDA context that was active
    /// or created while this module was initialized.
    let checkContext () =
        let ctx = ref (CUcontext ())
        if DriverAPINativeMethods.ContextManagement.cuCtxGetCurrent (ctx) <> CUResult.Success then
            failwith "cuCtxGetCurrent failed"
        if context.Context <> (!ctx) then
            failwithf "Current CUDA context %A does not match library initialization CUDA context %A"
                (!ctx).Pointer context.Context.Pointer

    /// Sets the thread's current CUDA context to the CUDA context that was active
    /// or created while this module was initialized.
    let setContext () =
        context.SetCurrent ()

    /// Is equivalent to int64 (ceil(float a / float b)).
    let divCeil a b = 
        if a % b = 0L then a / b 
        else a / b + 1L

    /// Computes CUDA launch dimensions from work dimensions and maximum block size.
    /// It is possible that the calculated launch dimensions will be smaller than the
    /// specified work dimensions, since the maximum block and grid sizes are limited.
    let computeLaunchDim (workDim: WorkDimT) maxBlockSize =
        let (./) a b = divCeil a b

        let wx, wy, wz = workDim
        let mbx, mby, mbz = maxBlockDim
        let mbx, mby, mbz = int64 mbx, int64 mby, int64 mbz
        let mgx, mgy, mgz = maxGridDim
        let mgx, mgy, mgz = int64 mgx, int64 mgy, int64 mgz
        let maxBlockSize = int64 maxBlockSize

        let bx = min mbx (min wx maxBlockSize)
        let by = min mby (min wy (maxBlockSize / bx))
        let bz = min mbz (min wz (maxBlockSize / (bx * by)))

        let gx = min mgx (wx ./ bx)
        let gy = min mgy (wy ./ by)
        let gz = min mgz (wz ./ bz)

        assert (if wx = 1L then bx = 1L && gx = 1L else true)
        assert (if wy = 1L then by = 1L && gy = 1L else true)
        assert (if wz = 1L then bz = 1L && gz = 1L else true)

        let mv = int64 Microsoft.FSharp.Core.int32.MaxValue
        assert (bx <= mv && by <= mv && bz <= mv)
        assert (gx <= mv && gy <= mv && gz <= mv)

        {Block = int32 bx, int32 by, int32 bz; Grid = int32 gx, int32 gy, int32 gz;}

    /// call implicit type conversation
    let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x) 

    /// gets device pointer as IntPtr
    let getIntPtr (cuPtr: CUdeviceptr) : System.IntPtr =
        !> cuPtr.Pointer





