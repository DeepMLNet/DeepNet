namespace Basics.Cuda

open ManagedCuda


[<AutoOpen>]
module CudaSupTypes =

    /// dimensionality of parallel work to perform
    type WorkDimT = int * int * int

    /// CUDA block dimension
    type BlockDimT = int * int * int

    /// CUDA grid dimension
    type GridDimT = int * int * int

    /// CUDA launch dimension
    type LaunchDimT = {Block: BlockDimT; Grid: GridDimT;}


module CudaSup =

    /// convert block/grid dimension to VectorTypes.dim3
    let toDim3 d =
        let (x: int), y, z = d
        VectorTypes.dim3(x, y, z)

    /// CUDA context
    let context = 
        let cudaCntxt = 
            try
                new CudaContext(createNew=false)
            with
            e ->
                printfn "Cannot create CUDA context: %s" e.Message
                exit 10

        cudaCntxt

    let deviceInfo =
        context.GetDeviceInfo()

    let maxBlockDim =
        int deviceInfo.MaxBlockDim.x, int deviceInfo.MaxBlockDim.y, int deviceInfo.MaxBlockDim.z

    let maxGridDim =
        int deviceInfo.MaxGridDim.x, int deviceInfo.MaxGridDim.y, int deviceInfo.MaxGridDim.z
    
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


    // CUDA BLAS handle
    let blas =
        new CudaBlas.CudaBlas()

    /// shutsdown CUDA (necessary for correct profiler results)  
    let shutdown () =
        context.Synchronize ()
        blas.Dispose ()
        CudaContext.ProfilerStop ()
        context.Dispose ()


    /// Computes CUDA launch dimensions from work dimensions and maximum block size.
    /// It is possible that the calculated launch dimensions will be smaller than the
    /// specified work dimensions, since the maximum block and grid sizes are limited.
    let computeLaunchDim (workDim: WorkDimT) maxBlockSize =
        let wx, wy, wz = workDim
        let mbx, mby, mbz = maxBlockDim
        let mgx, mgy, mgz = maxGridDim

        let bx = min mbx (min wx maxBlockSize)
        let by = min mby (min wy (maxBlockSize / bx))
        let bz = min mbz (min wz (maxBlockSize / (bx * by)))

        let gx = min mgx (wx / bx + 1)
        let gy = min mgy (wy / by + 1)
        let gz = min mgz (wz / bz + 1)

        {Block = bx, by, bz; Grid = gx, gy, gz;}






