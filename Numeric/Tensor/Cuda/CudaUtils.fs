namespace Tensor.Cuda

open System
open System.Runtime
open System.Threading
open System.Collections.Concurrent

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor.Utils


/// out of CUDA memory
exception OutOfCudaMemory of msg:string with override __.Message = __.msg

/// generic CUDA error
exception CudaError of msg:string with override __.Message = __.msg


/// Cuda support types functions
module internal Cuda =

    /// dimensionality of parallel work to perform (x, y, z)
    type WorkDim = int64 * int64 * int64

    /// CUDA block dimension (x, y, z)
    type BlockDim = int * int * int

    /// CUDA grid dimension (x, y, z)
    type GridDim = int * int * int

    /// CUDA launch dimension
    type LaunchDim = {
        Block: BlockDim
        Grid:  GridDim
    }

    /// convert block/grid dimension (x, y, z) to VectorTypes.dim3
    let toDim3 d =
        let (x: int), y, z = d
        VectorTypes.dim3 (x, y, z)

    /// CUDA context
    let context = 
        try new CudaContext(createNew=false)
        with e ->
            let msg = sprintf "Cannot create CUDA context: %s" e.Message
            raise (CudaError msg)        

    // CUDA BLAS handle
    let blas =
        new CudaBlas.CudaBlas()

    /// CUDA device info
    let deviceInfo =
        context.GetDeviceInfo()

    /// CUDA maximum block dimension
    let maxBlockDim : BlockDim =
        int deviceInfo.MaxBlockDim.x, int deviceInfo.MaxBlockDim.y, int deviceInfo.MaxBlockDim.z

    /// CUDA maximum grid dimension
    let maxGridDim : GridDim =
        int deviceInfo.MaxGridDim.x, int deviceInfo.MaxGridDim.y, int deviceInfo.MaxGridDim.z
    
    /// nvcc arch code
    let nvccArch =
        sprintf "compute_%d%d" deviceInfo.ComputeCapability.Major deviceInfo.ComputeCapability.Minor

    /// nvcc sm code
    let nvccCode =
        sprintf "sm_%d%d" deviceInfo.ComputeCapability.Major deviceInfo.ComputeCapability.Minor

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
    let computeLaunchDim (workDim: WorkDim) maxBlockSize =
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

    /// gets device pointer as IntPtr
    let getIntPtr (cuPtr: CUdeviceptr) : System.IntPtr =
        let inline (!>) (x:^a) : ^b = 
            ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x) 
        !> cuPtr.Pointer
   
    // callback support
    type private CallbackFn = unit -> unit
    let private callbackQueue = ConcurrentQueue<CudaEvent * CallbackFn> ()
    let private callbackEvent = new AutoResetEvent(false)
    let private callbackThreadFn () =
        setContext ()
        while true do
            let rec processCallbacks() =
                match callbackQueue.TryDequeue() with
                | Some (event, callbackFn) ->
                    event.Synchronize ()
                    event.Dispose()
                    try callbackFn()
                    with e -> 
                        printfn "CUDA callback failed with exception:\n%s" (e.ToString())
                        exit -100
                    processCallbacks()
                | None -> ()
            processCallbacks()
            callbackEvent.WaitOne() |> ignore
    let private callbackThread = new Thread(callbackThreadFn, IsBackground=true)
    do callbackThread.Start()

    /// Places a callback function on a CUDA stream.
    /// The function is executed on a global callback thread and is allowed to make CUDA calls.
    /// The thread's CUDA context has been set to the libraries CUDA context.
    /// The CUDA stream continues execution while the callback function is being invoked.
    /// The callback can be blocked by waiting for other callbacks.
    let callback (cuStream: CUstream) (fn: unit -> unit) =
        let event = new CudaEvent (CUEventFlags.BlockingSync ||| CUEventFlags.DisableTiming)
        event.Record(cuStream)
        callbackQueue.Enqueue((event, fn))
        callbackEvent.Set() |> ignore

    /// Places a callback function on a CUDA stream.
    /// The function is executed on a thread-pool thread and is allowed to make CUDA calls.
    /// This function is less efficient than Cuda.callback.
    let callbackWithResult (cuStream: CUstream) (fn: CUResult -> unit) =
        let threadPoolCallback (result: obj) =
            fn (unbox result)
        let cudaCallback (strm: CUstream) (result: CUResult) (userData: nativeint) =
            ThreadPool.QueueUserWorkItem (WaitCallback threadPoolCallback, box result) |> ignore
        use stream = new CudaStream (cuStream)
        stream.AddCallback (CUstreamCallback cudaCallback, nativeint 0, CUStreamAddCallbackFlags.None)

    /// Keeps the given object alive (i.e. prevent it from being GCed) 
    /// until all operations that were queued on the given CUDA stream 
    /// up to now have been executed.
    let keepAlive (cuStream: CUstream) (x: obj) =
        callback cuStream (fun () -> GC.KeepAlive x)

    /// Keeps the given objects alive (i.e. prevent them from being GCed) 
    /// until all operations that were queued on the given CUDA stream 
    /// up to now have been executed.
    let keepAliveMany (cuStream: CUstream) (xs: obj list) =
        xs |> List.iter (keepAlive cuStream)

    /// minimum available CUDA memory before triggering GC
    let minAvailMem = 100000000L

    /// tries to obtain neededMem bytes of CUDA memory by invoking the GC if necessary
    let rec tryObtainMem neededMem retries = 
        let freeMem = context.GetFreeDeviceMemorySize() |> int64
        if freeMem < (neededMem + 10000L) || freeMem < minAvailMem then
            GCSettings.LargeObjectHeapCompactionMode <- GCLargeObjectHeapCompactionMode.CompactOnce
            GC.Collect (2, GCCollectionMode.Forced, true, true)
            GC.WaitForPendingFinalizers ()
            GC.WaitForFullGCComplete() |> ignore
            Thread.Yield() |> ignore
            if retries < 3 then
                Thread.Sleep(20)
            if retries > 0 then
                tryObtainMem neededMem (retries - 1)
        //printfn "CUDA has %d MB available" (freeMem / 1000000L)

    /// create a new CUDA device variable
    let newDevVar<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
            (elems: int64) = 

        let sizeInBytes = elems * sizeof64<'T>        
        tryObtainMem sizeInBytes 10

        try new CudaDeviceVariable<'T> (SizeT elems)
        with :? CudaException as e when e.CudaError = CUResult.ErrorOutOfMemory 
                                     || e.CudaError = CUResult.ErrorUnknown ->
            let msg = 
                sprintf "CUDA memory allocation of %d MB failed (%A)" 
                        (sizeInBytes / pown 2L 20) e.CudaError
            raise (OutOfCudaMemory msg)


