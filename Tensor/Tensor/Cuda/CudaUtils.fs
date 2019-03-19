namespace Tensor.Cuda

open System
open System.Runtime
open System.Threading
open System.Collections.Concurrent

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor.Utils
open DeepNet.Utils


/// Out of CUDA memory.
exception OutOfCudaMemoryException of msg:string with override __.Message = __.msg


/// Cuda support types functions
module internal Cuda =

    /// Pushes the specified CudaContext to the context stack
    /// and pops it off the stack when this object is disposed.
    type ContextGuard (ctx: CudaContext) =
        do ctx.PushContext()
        interface IDisposable with
            member this.Dispose() =
                ctx.PopContext()

    /// Pushes the specified CudaContext to the context stack
    /// and pops it off the stack when the guard object is disposed.
    let activate (ctx: CudaContext) =
        new ContextGuard(ctx)
        
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

    /// CUDA device info cache
    let private deviceInfos = ConditionalWeakTable<CudaContext, CudaDeviceProperties> ()

    /// CUDA device info
    let deviceInfo (ctx: CudaContext) =
        deviceInfos.GetValue (ctx, fun _ ->
            use _ctx = activate ctx
            ctx.GetDeviceInfo())    

    /// CUDA maximum block dimension
    let maxBlockDim (ctx: CudaContext) : BlockDim =
        let deviceInfo = deviceInfo ctx
        int deviceInfo.MaxBlockDim.x, int deviceInfo.MaxBlockDim.y, int deviceInfo.MaxBlockDim.z

    /// CUDA maximum grid dimension
    let maxGridDim (ctx: CudaContext) : GridDim =
        let deviceInfo = deviceInfo ctx
        int deviceInfo.MaxGridDim.x, int deviceInfo.MaxGridDim.y, int deviceInfo.MaxGridDim.z
    
    /// nvcc sm code
    let nvccCode (ctx: CudaContext) =
        let deviceInfo = deviceInfo ctx
        sprintf "sm_%d%d" deviceInfo.ComputeCapability.Major deviceInfo.ComputeCapability.Minor

    /// Is equivalent to int64 (ceil(float a / float b)).
    let divCeil a b = 
        if a % b = 0L then a / b 
        else a / b + 1L

    /// Computes CUDA launch dimensions from work dimensions and maximum block size.
    /// It is possible that the calculated launch dimensions will be smaller than the
    /// specified work dimensions, since the maximum block and grid sizes are limited.
    let computeLaunchDim (ctx: CudaContext) (workDim: WorkDim) maxBlockSize =
        let (./) a b = divCeil a b

        let wx, wy, wz = workDim
        let mbx, mby, mbz = maxBlockDim ctx
        let mbx, mby, mbz = int64 mbx, int64 mby, int64 mbz
        let mgx, mgy, mgz = maxGridDim ctx
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
   
    type private CallbackFn = unit -> unit
    type CallbackProcessor private (weakCtx: WeakReference<CudaContext>) =
        static let instances = ConditionalWeakTable<CudaContext, CallbackProcessor> ()

        let callbackQueue = ConcurrentQueue<CudaEvent * CallbackFn> ()
        let callbackEvent = new AutoResetEvent(false)

        let callbackThreadFn () =
            // process callbacks while context is alive
            let mutable ctx: CudaContext = null
            while weakCtx.TryGetTarget &ctx do
                ctx.SetCurrent()

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

                // releae our reference to context so that it may be garbage collected
                ctx <- null
                callbackEvent.WaitOne (1000) |> ignore

        let callbackThread = new Thread (callbackThreadFn, IsBackground=true)
        do callbackThread.Start()

        /// Enqueues the specified callback in the stream.
        member this.Enqueue (cuStream: CUstream) (fn: unit -> unit) =
            match weakCtx.TryGetTarget () with
            | true, ctx ->
                use _ctx = activate ctx
                let event = new CudaEvent (CUEventFlags.BlockingSync ||| CUEventFlags.DisableTiming)
                event.Record(cuStream)
                callbackQueue.Enqueue((event, fn))
                callbackEvent.Set() |> ignore
            | _ -> failwith "The CudaContext of this CallbackProcessor does not exist anymore." 

        /// Gets the callback processor for the specified CUDA context.
        static member forContext (ctx: CudaContext) =
            instances.GetValue (ctx, fun _ -> CallbackProcessor (WeakReference<_> ctx))

    /// Places a callback function on a CUDA stream.
    /// The function is executed on a global callback thread and is allowed to make CUDA calls.
    /// The thread's CUDA context has been set to the libraries CUDA context.
    /// The CUDA stream continues execution while the callback function is being invoked.
    /// The callback can be blocked by waiting for other callbacks.
    let callback (ctx: CudaContext) (cuStream: CUstream) (fn: unit -> unit) =
        let proc = CallbackProcessor.forContext ctx
        proc.Enqueue cuStream fn

    /// Places a callback function on a CUDA stream.
    /// The function is executed on a thread-pool thread and is allowed to make CUDA calls.
    /// This function is less efficient than Cuda.callback.
    //let callbackWithResult (cuStream: CUstream) (fn: CUResult -> unit) =
    //    let threadPoolCallback (result: obj) =
    //        fn (unbox result)
    //    let cudaCallback (strm: CUstream) (result: CUResult) (userData: nativeint) =
    //        ThreadPool.QueueUserWorkItem (WaitCallback threadPoolCallback, box result) |> ignore
    //    use stream = new CudaStream (cuStream)
    //    stream.AddCallback (CUstreamCallback cudaCallback, nativeint 0, CUStreamAddCallbackFlags.None)

    /// Keeps the given object alive (i.e. prevent it from being GCed) 
    /// until all operations that were queued on the given CUDA stream 
    /// up to now have been executed.
    let keepAlive (ctx: CudaContext) (cuStream: CUstream) (x: obj) =
        callback ctx cuStream (fun () -> GC.KeepAlive x)

    /// Keeps the given objects alive (i.e. prevent them from being GCed) 
    /// until all operations that were queued on the given CUDA stream 
    /// up to now have been executed.
    let keepAliveMany (ctx: CudaContext) (cuStream: CUstream) (xs: obj list) =
        xs |> List.iter (keepAlive ctx cuStream)

    /// minimum available CUDA memory before triggering GC
    let minAvailMem = 100000000L

    /// tries to obtain neededMem bytes of CUDA memory by invoking the GC if necessary
    let rec private tryObtainMem (context: CudaContext) neededMem retries = 
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
                tryObtainMem context neededMem (retries - 1)
        //printfn "CUDA has %d MB available" (freeMem / 1000000L)

    /// Create a new CUDA device variable in the current CUDA context.
    let newDevVar<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
            (ctx: CudaContext) (elems: int64) = 

        use _ctx = activate ctx

        let sizeInBytes = elems * sizeof64<'T>        
        tryObtainMem ctx sizeInBytes 10

        try new CudaDeviceVariable<'T> (SizeT elems)
        with :? CudaException as e when e.CudaError = CUResult.ErrorOutOfMemory 
                                     || e.CudaError = CUResult.ErrorUnknown ->
            let msg = 
                sprintf "CUDA memory allocation of %d MB failed (%A)." 
                        (sizeInBytes / pown 2L 20) e.CudaError
            raise (OutOfCudaMemoryException msg)

    ///// Checks that the thread's current CUDA context is the CUDA context that was active
    ///// or created while this module was initialized.
    //let checkContext () =
    //    let ctx = ref (CUcontext ())
    //    if DriverAPINativeMethods.ContextManagement.cuCtxGetCurrent (ctx) <> CUResult.Success then
    //        failwith "cuCtxGetCurrent failed"
    //    if context.Context <> (!ctx) then
    //        failwithf "Current CUDA context %A does not match library initialization CUDA context %A"
    //            (!ctx).Pointer context.Context.Pointer
                        

