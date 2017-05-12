namespace Tensor

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor.Utils
open Tensor.Cuda.Backend
open System


/// cannot register host memory with CUDA, maybe because it is not properly aligned
exception CannotCudaRegisterMemory of msg:string with override __.Message = __.msg

/// out of CUDA memory
exception OutOfCudaMemory of msg:string with override __.Message = __.msg

/// generic CUDA error
exception CudaError of msg:string with override __.Message = __.msg


/// CUDA helpers
module private CudaHelpers =

    /// the CUDA context we are using
    let mutable context = None

    /// initializes the CUDA context if necessary
    let initContext () =
        match context with
        | Some _ -> ()
        | None ->
            try context <- Some (new CudaContext(createNew=false))
            with e ->
                let msg = sprintf "cannot create CUDA context: %s" e.Message
                raise (CudaError msg)

    /// create a new CUDA device variable
    let newDevVar<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
            (elems: int64) = 
        try new CudaDeviceVariable<'T> (SizeT elems)
        with :? CudaException as e when e.CudaError = CUResult.ErrorOutOfMemory 
                                     || e.CudaError = CUResult.ErrorUnknown ->
            let sizeInBytes = elems * sizeof64<'T>
            let msg = 
                sprintf "CUDA memory allocation of %d MB failed (%A)" 
                        (sizeInBytes / pown 2L 20) e.CudaError
            raise (OutOfCudaMemory msg)


/// CUDA registered memory support
module internal CudaRegMemSupport =

    /// synchronization lock
    let syncLock = obj ()

    /// registration count
    let registeredCount = new Dictionary<ITensorHostStorage, int>()

    /// master data registrations
    let dataRegistrations = new Dictionary<ITensorHostStorage, obj>()

    /// decreases reference count for page locked data
    let decrRefCount data  =
        registeredCount.[data] <- registeredCount.[data] - 1
        if registeredCount.[data] = 0 then
            dataRegistrations.Remove data |> ignore
            true
        else false


/// CUDA registered memory for fast data transfer.
/// Dispose to unregister memory with CUDA.
type CudaRegMemHnd internal (hostArray:  ITensorHostStorage, 
                             pinHnd:     PinnedMemory, 
                             cudaMem:    CudaRegisteredHostMemory<byte>) =
           
    let mutable disposed = false
    let checkDisposed () =
        if disposed then raise (ObjectDisposedException "CudaRegMemHnd")

    interface IDisposable with
        member this.Dispose() =          
            lock CudaRegMemSupport.syncLock (fun () ->
                if not disposed then 
                    if CudaRegMemSupport.decrRefCount hostArray then            
                        // unregister memory
                        try cudaMem.Unregister() 
                        with :? CudaException -> ()
                        // release cuda memory handle 
                        try cudaMem.Dispose()
                        with :? CudaException -> ()
                        // unpin managed memory
                        (pinHnd :> IDisposable).Dispose()
                disposed <- true)

    override this.Finalize () =
        (this :> IDisposable).Dispose()

    /// the data array
    member this.HostArray = 
        checkDisposed ()
        hostArray
    member internal this.HostArrayPriv = hostArray

    /// GC memory pin handle
    member this.PinHnd = 
        checkDisposed ()
        pinHnd
    member internal this.PinHndPriv = pinHnd

    /// pointer to data 
    member this.Ptr =
        this.PinHnd.Ptr

    /// the CudaRegisteredHostMemory
    member this.CudaRegisteredMemory = 
        checkDisposed ()
        cudaMem
    member internal this.CudaRegisteredMemoryPriv = cudaMem


/// Methods for locking a TensorHostStorage into memory and registering the memory with CUDA
/// for fast data transfers with GPU device.
module CudaRegMem =
    open CudaRegMemSupport

    /// get CudaRegMemHnd for already locked TensorHostStorage          
    let get data =      
        lock syncLock (fun () ->
            if not (dataRegistrations.ContainsKey data) then
                failwith "the specified TensorHostStorage is not registered with CUDA for fast data transfer" 
            registeredCount.[data] <- registeredCount.[data] + 1
            let dr = dataRegistrations.[data] :?> CudaRegMemHnd
            new CudaRegMemHnd(dr.HostArrayPriv, dr.PinHndPriv, dr.CudaRegisteredMemoryPriv)   
        )
        
    /// gets the CudaRegisteredMemory for already locked TensorHostStorage without 
    /// incrementing the reference count
    let getCudaRegisteredMemory data =
        lock syncLock (fun () ->
            if not (dataRegistrations.ContainsKey data) then
                failwith "the specified TensorHostStorage is not registered with CUDA for fast data transfer" 
            let dr = dataRegistrations.[data] :?> CudaRegMemHnd
            dr.CudaRegisteredMemory
        )            

    /// registers a TensorHostStorage (multiple registrations are okay) and returns the corresponding CudaRegMemHnd
    let register (data: ITensorHostStorage) = 
        lock syncLock (fun () ->
            if dataRegistrations.ContainsKey data then get data      
            else
                // pin managed memory so that address cannot change
                let pinHnd = data.Pin ()
                let dataAddr = pinHnd.Ptr
                let dataByteSize = data.DataSizeInBytes

                // construct cuda memory handle and register it
                let cudaMem = new CudaRegisteredHostMemory<byte> (dataAddr, SizeT dataByteSize)    
                try cudaMem.Register (BasicTypes.CUMemHostRegisterFlags.None)
                with :? CudaException as ex ->
                    if ex.CudaError = CUResult.ErrorInvalidValue then
                        // probably memory is not properly aligned
                        raise (CannotCudaRegisterMemory ex.Message)
                    else reraise ()

                // create handle object
                let dr = new CudaRegMemHnd(data, pinHnd, cudaMem)     
                dataRegistrations.[data] <- dr
                registeredCount.[data] <- 1
                dr
        )
              

/// type neutral interface to a CudaStorageT
type ITensorCudaStorage =
    abstract ByteData: CudaDeviceVariable<byte>
    abstract DataSize: int64
    abstract DataSizeInBytes: int64


/// Tensor storage on a CUDA device.
type TensorCudaStorage<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                    (data: CudaDeviceVariable<'T>) =

    new (nElems: int64) =
        CudaHelpers.initContext ()
        // CUDA cannot allocate memory of size zero
        let nElems = if nElems > 0L then nElems else 1L
        TensorCudaStorage<'T> (CudaHelpers.newDevVar nElems)
     
    /// data device variable
    member this.Data = data

    /// data size in elements
    member this.DataSize = int64 data.Size

    /// data size in bytes
    member this.DataSizeInBytes = int64 data.SizeInBytes

    /// data device variable as CudaDeviceVariable<byte>
    member this.ByteData =
        new CudaDeviceVariable<byte> (data.DevicePointer, data.SizeInBytes)        

    override this.Finalize() = 
        if data <> null then data.Dispose()

    /// data item access
    member this.Item 
        with get (addr: int64) =
            if typeof<'T> = typeof<bool> then
                let hostBuf : byte ref = ref 0uy
                this.ByteData.CopyToHost(hostBuf, SizeT (addr * sizeof64<byte>))
                !hostBuf <> 0uy |> box |> unbox
            else
                let hostBuf = ref (new 'T())
                this.Data.CopyToHost(hostBuf, SizeT (addr * sizeof64<'T>))
                !hostBuf
                
        and set (addr: int64) (value: 'T) = 
            if typeof<'T> = typeof<bool> then
                let byteVal = if (box value :?> bool) then 1uy else 0uy
                this.ByteData.CopyToDevice(byteVal, SizeT (addr * sizeof64<byte>))
            else
                this.Data.CopyToDevice(value, SizeT (addr * sizeof64<'T>))

    interface ITensorStorage<'T> with
        member this.Backend layout = 
            TensorCudaBackend<'T> (layout, this) :> ITensorBackend<_>
        member this.Device =
            TensorCudaDevice.Instance :> ITensorDevice

    interface ITensorCudaStorage with
        member this.ByteData = this.ByteData
        member this.DataSize = this.DataSize
        member this.DataSizeInBytes = this.DataSizeInBytes


and ITensorCudaBackend =
    abstract NativeTensor: NativeTensor

/// CUDA backend for tensors.
and TensorCudaBackend<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                    (layout: TensorLayout, storage: TensorCudaStorage<'T>) =

    let kernels = TensorKernels.Get (typeof<'T>, layout.NDims)

    let stream () = CUstream.NullStream

    let unsup op =
        let msg = 
            sprintf "the CUDA tensor backend currently does not support the %s operation" op
        raise (NotSupportedException msg)

    let callUnary fn trgt src1 : unit =
        let trgt, src1 = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt, src1)
        fn (stream(), trgt, src1)

    let callBinary fn trgt src1 src2 : unit =
        let trgt, src1, src2 = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt, src1, src2)
        fn (stream(), trgt, src1, src2)

    let callTenary fn trgt src1 src2 src3 : unit =
        let trgt, src1, src2, src3 = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt, src1, src2, src3)
        fn (stream(), trgt, src1, src2, src3)



    /// device pointer to first element of this tensor
    member this.DevicePtr : nativeint =
        Cuda.getIntPtr storage.Data.DevicePointer + nativeint (layout.Offset * sizeof64<'T>)        

    /// tensor information for native CUDA code
    member inline internal this.NativeTensor = {
        DataType    = typeof<'T>
        BasePtr     = storage.ByteData.DevicePointer |> Cuda.getIntPtr
        Offset      = layout.Offset
        Shape       = layout.Shape
        Stride      = layout.Stride
    }

    interface ITensorCudaBackend with
        member this.NativeTensor = this.NativeTensor

    /// gets NativeTensors for specified tensors
    static member internal GetNativeTensor (t: Tensor<'T>) =
        (t.Backend :?> TensorCudaBackend<'T>).NativeTensor

    /// gets NativeTensors for specified tensors
    static member internal GetNativeTensor (t: Tensor<'T>, a: Tensor<'TA>) =
        (t.Backend :?> TensorCudaBackend<'T>).NativeTensor, 
        (a.Backend :?> TensorCudaBackend<'TA>).NativeTensor 

    /// gets NativeTensors for specified tensors, optimized for elment-wise operations
    static member internal ElemwiseNativeTensor<'T> (t: Tensor<'T>) 
            : NativeTensor =
        (t.Backend :?> ITensorCudaBackend).NativeTensor

    /// gets NativeTensors for specified tensors, optimized for elment-wise operations
    static member internal ElemwiseNativeTensor<'T, 'TA> 
            (t: Tensor<'T>, a: Tensor<'TA>) : NativeTensor * NativeTensor =
        (t.Backend :?> ITensorCudaBackend).NativeTensor, 
        (a.Backend :?> ITensorCudaBackend).NativeTensor 

    /// gets NativeTensors for specified tensors, optimized for elment-wise operations
    static member internal ElemwiseNativeTensor<'T, 'TA, 'TB>  
            (t: Tensor<'T>, a: Tensor<'TA>, b: Tensor<'TB>) : NativeTensor * NativeTensor * NativeTensor =
        (t.Backend :?> ITensorCudaBackend).NativeTensor, 
        (a.Backend :?> ITensorCudaBackend).NativeTensor,
        (b.Backend :?> ITensorCudaBackend).NativeTensor

    /// gets NativeTensors for specified tensors, optimized for elment-wise operations
    static member internal ElemwiseNativeTensor<'T, 'TA, 'TB, 'TC>  
            (t: Tensor<'T>, a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>) 
            : NativeTensor * NativeTensor * NativeTensor * NativeTensor =
        (t.Backend :?> ITensorCudaBackend).NativeTensor, 
        (a.Backend :?> ITensorCudaBackend).NativeTensor,
        (b.Backend :?> ITensorCudaBackend).NativeTensor,
        (c.Backend :?> ITensorCudaBackend).NativeTensor


    interface ITensorBackend<'T> with

        member this.Item 
            with get idx = storage.[layout |> TensorLayout.addr (idx |> List.ofArray)]
            and set idx value = storage.[layout |> TensorLayout.addr (idx |> List.ofArray)] <- value

        member this.GetEnumerator() : System.Collections.IEnumerator = raise (System.NotImplementedException())
        member this.GetEnumerator() : System.Collections.Generic.IEnumerator<'T> = raise (System.NotImplementedException())


        member this.Transfer (trgt, src) =
            let regMem (storage: TensorHostStorage<'T>) = 
                try
                    let h = CudaRegMem.register storage
                    h :> IDisposable, h.Ptr
                with CannotCudaRegisterMemory _ -> 
                    let h = storage.Pin()
                    h :> IDisposable, h.Ptr

            let doTransfer (trgt: Tensor<'T>) (src: Tensor<'T>) = 
                match trgt.Storage, src.Storage with
                // transfer from host to CUDA
                | (:? TensorCudaStorage<'T> as trgtStorage), (:? TensorHostStorage<'T> as srcStorage) ->
                    let srcMemHnd, srcMemPtr = regMem srcStorage
                    // TODO: make it use stream! BUG
                    trgtStorage.Data.CopyToDevice (srcMemPtr, 
                                                   SizeT (sizeof64<'T> * src.Layout.Offset), 
                                                   SizeT (sizeof64<'T> * trgt.Layout.Offset),
                                                   SizeT (sizeof64<'T> * src.NElems))
                    srcMemHnd.Dispose()
                    true

                // transfer from CUDA to host
                | (:? TensorHostStorage<'T> as trgtStorage), (:? TensorCudaStorage<'T> as srcStorage) ->
                    let trgtMemHnd, trgtMemPtr = regMem trgtStorage
                    // TODO: make it use stream! BUG
                    srcStorage.Data.CopyToHost(trgtMemPtr, 
                                               SizeT (sizeof64<'T> * src.Layout.Offset), 
                                               SizeT (sizeof64<'T> * trgt.Layout.Offset), 
                                               SizeT (sizeof64<'T> * src.NElems))
                    trgtMemHnd.Dispose()
                    true
                | _ -> false

            // ensure that source is in row-major order
            let src =
                if TensorLayout.isC src.Layout then src
                else Tensor.copy (src, order=RowMajor)
               
            if TensorLayout.isC trgt.Layout then
                // target is in row-major order, do direct transfer
                doTransfer trgt src
            else
                // target is not in row-major order, transfer to temporary tensor
                // and copy into target
                let tmp = Tensor<'T> (trgt.Shape, trgt.Device, order=RowMajor)
                if doTransfer tmp src then
                    trgt.CopyFrom tmp
                    true
                else false

        member this.FillConst (value, trgt) = 
            let trgt = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt)
            kernels.FillConst (stream(), box value, trgt)

        member this.Copy(trgt, src) = 
            if TensorLayout.hasContiguousMemory trgt.Layout && 
               TensorLayout.hasContiguousMemory src.Layout &&
               trgt.Layout.Stride = src.Layout.Stride then
                // use CUDA memcpy for continous block of memory
                let trgtStorage = trgt.Storage :?> TensorCudaStorage<'T>
                let srcStorage = src.Storage :?> TensorCudaStorage<'T>
                trgtStorage.Data.AsyncCopyToDevice (srcStorage.Data, 
                                                    SizeT (sizeof64<'T> * src.Layout.Offset),
                                                    SizeT (sizeof64<'T> * trgt.Layout.Offset),
                                                    SizeT (sizeof64<'T> * src.NElems),
                                                    stream())
            else
                // call copy kernel
                let trgt, src = TensorCudaBackend<_>.GetNativeTensor (trgt, src)
                kernels.Copy(stream(), trgt, src)

        member this.Convert(trgt: Tensor<'T>, src: Tensor<'S>) =
            let convKernels = TensorConvertKernels.Get (typeof<'T>, typeof<'S>, trgt.NDims)
            callUnary convKernels.Convert trgt src

        member this.UnaryPlus(trgt, src1)   = callUnary kernels.UnaryPlus trgt src1
        member this.UnaryMinus(trgt, src1)  = callUnary kernels.UnaryMinus trgt src1
        member this.Abs(trgt, src1)         = callUnary kernels.Abs trgt src1
        member this.Sgn(trgt, src1)         = callUnary kernels.Sgn trgt src1
        member this.Log(trgt, src1)         = callUnary kernels.Log trgt src1
        member this.Log10(trgt, src1)       = callUnary kernels.Log10 trgt src1
        member this.Exp(trgt, src1)         = callUnary kernels.Exp trgt src1
        member this.Sin(trgt, src1)         = callUnary kernels.Sin trgt src1
        member this.Cos(trgt, src1)         = callUnary kernels.Cos trgt src1
        member this.Tan(trgt, src1)         = callUnary kernels.Tan trgt src1
        member this.Asin(trgt, src1)        = callUnary kernels.Asin trgt src1
        member this.Acos(trgt, src1)        = callUnary kernels.Acos trgt src1
        member this.Atan(trgt, src1)        = callUnary kernels.Atan trgt src1
        member this.Sinh(trgt, src1)        = callUnary kernels.Sinh trgt src1
        member this.Cosh(trgt, src1)        = callUnary kernels.Cosh trgt src1
        member this.Tanh(trgt, src1)        = callUnary kernels.Tanh trgt src1
        member this.Sqrt(trgt, src1)        = callUnary kernels.Sqrt trgt src1
        member this.Ceiling(trgt, src1)     = callUnary kernels.Ceiling trgt src1
        member this.Floor(trgt, src1)       = callUnary kernels.Floor trgt src1
        member this.Round(trgt, src1)       = callUnary kernels.Round trgt src1
        member this.Truncate(trgt, src1)    = callUnary kernels.Truncate trgt src1

        member this.Add(trgt, src1, src2)           = callBinary kernels.Add trgt src1 src2
        member this.Subtract(trgt, src1, src2)      = callBinary kernels.Subtract trgt src1 src2
        member this.Multiply(trgt, src1, src2)      = callBinary kernels.Multiply trgt src1 src2
        member this.Divide(trgt, src1, src2)        = callBinary kernels.Divide trgt src1 src2
        member this.Modulo(trgt, src1, src2)        = callBinary kernels.Modulo trgt src1 src2
        member this.Power(trgt, src1, src2)         = callBinary kernels.Power trgt src1 src2
        member this.MinElemwise(trgt, src1, src2)   = callBinary kernels.MinElemwise trgt src1 src2
        member this.MaxElemwise(trgt, src1, src2)   = callBinary kernels.MaxElemwise trgt src1 src2

        member this.IsFinite(trgt, src1)                = callUnary kernels.IsFinite trgt src1
        member this.Equal(trgt, src1, src2)             = callBinary kernels.Equal trgt src1 src2
        member this.NotEqual(trgt, src1, src2)          = callBinary kernels.NotEqual trgt src1 src2
        member this.Less(trgt, src1, src2)              = callBinary kernels.Less trgt src1 src2
        member this.LessOrEqual(trgt, src1, src2)       = callBinary kernels.LessOrEqual trgt src1 src2
        member this.Greater(trgt, src1, src2)           = callBinary kernels.Greater trgt src1 src2
        member this.GreaterOrEqual(trgt, src1, src2)    = callBinary kernels.GreaterOrEqual trgt src1 src2

        member this.IfThenElse(trgt, cond, ifTrue, ifFalse) = callTenary kernels.IfThenElse trgt cond ifTrue ifFalse

        member this.Negate(trgt, src1)      = callUnary kernels.Negate trgt src1
        member this.And(trgt, src1, src2)   = callBinary kernels.And trgt src1 src2
        member this.Or(trgt, src1, src2)    = callBinary kernels.Or trgt src1 src2
        member this.Xor(trgt, src1, src2)   = callBinary kernels.Xor trgt src1 src2

        member this.MinLastAxis(trgt, src1)     = callUnary kernels.MinLastAxis trgt src1
        member this.MaxLastAxis(trgt, src1)     = callUnary kernels.MaxLastAxis trgt src1
        member this.SumLastAxis(trgt, src1)     = callUnary kernels.SumLastAxis trgt src1
        member this.ProductLastAxis(trgt, src1) = callUnary kernels.ProductLastAxis trgt src1
        member this.AllLastAxis(trgt, src1)     = callUnary kernels.AllLastAxis trgt src1
        member this.AnyLastAxis(trgt, src1)     = callUnary kernels.AnyLastAxis trgt src1

        member this.ArgMinLastAxis(trgt, src1)  = callUnary kernels.ArgMinLastAxis trgt src1
        member this.ArgMaxLastAxis(trgt, src1)  = callUnary kernels.ArgMaxLastAxis trgt src1

        member this.Gather(trgt, srcIdxs, src) = 
            let gsKernels = TensorGatherScatterKernels.Get (typeof<'T>, trgt.NDims, src.NDims)
            let srcIdxs = {
                NDims = trgt.NDims
                Idxs  = srcIdxs |> List.map (Option.map (TensorCudaBackend<_>.GetNativeTensor))
            }
            let trgt, src = TensorCudaBackend<_>.GetNativeTensor (trgt, src)
            gsKernels.Gather (stream(), trgt, srcIdxs, src)

        member this.Scatter(trgt, trgtIdxs, src) = 
            let gsKernels = TensorGatherScatterKernels.Get (typeof<'T>, trgt.NDims, src.NDims)
            let trgtIdxs = {
                NDims = src.NDims
                Idxs  = trgtIdxs |> List.map (Option.map (TensorCudaBackend<_>.GetNativeTensor))
            }
            let trgt, src = TensorCudaBackend<_>.GetNativeTensor (trgt, src)
            kernels.FillConst (stream(), box (conv<'T> 0), trgt)
            gsKernels.Scatter (stream(), trgt, trgtIdxs, src)

        member this.VecVecDot (trgt, a, b) =
            ()
        member this.MatVecDot (trgt, a, b) =
            ()
        member this.MatMatDot (trgt, a, b) =
            ()
        member this.BatchedMatMatDot (trgt, a, b) =
            ()
        member this.BatchedInvert (trgt, src) =
            ()
        member this.SymmetricEigenDecomposition (part, trgtEigVals, trgtEigVec, src) =
            ()

        // unsupported for now on CUDA
        member this.Fill(fn, trgt, useThreads) = unsup "Fill"
        member this.FillIndexed(fn, trgt, useThreads) = unsup "FillIndexed"
        member this.Map(fn, trgt, src, useThreads) = unsup "Map"
        member this.Map2(fn, trgt, src1, src2, useThreads) = unsup "Map2"
        member this.MapIndexed(fn, trgt, src, useThreads) = unsup "MapIndexed"
        member this.MapIndexed2(fn, trgt, src1, src2, useThreads) = unsup "MapIndexed2"
        member this.FoldLastAxis(fn, initial, trgt, src, useThreads) = unsup "FoldLastAxis"
        member this.FoldLastAxisIndexed(fn, initial, trgt, src, useThreads) = unsup "FoldLastAxisIndexed"


/// Creates Tensors on a CUDA device.
and TensorCudaDevice private () =
    inherit BaseTensorDevice()
    static member Instance = TensorCudaDevice ()
    
    override this.Id = "Cuda"
    override this.Create nElems = 
        // we use reflection to drop the constraints on 'T 
        let ts = typedefof<TensorCudaStorage<_>>.MakeGenericType (typeof<'T>)
        Activator.CreateInstance(ts, nElems) :?> ITensorStorage<'T>
    override this.Zeroed = false


/// Tensor stored on CUDA device.
module CudaTensor =

    /// Tensor stored on CUDA device.
    let Dev = TensorCudaDevice.Instance :> ITensorDevice

    let transfer x = Tensor.transfer Dev x

    let zeros<'T> = Tensor.zeros<'T> Dev 

    let ones<'T> = Tensor.ones<'T> Dev

    let falses = Tensor.falses Dev

    let trues = Tensor.trues Dev

    let scalar<'T> = Tensor.scalar<'T> Dev

    let init<'T> = Tensor.init<'T> Dev

    let filled<'T> = Tensor.filled<'T> Dev

    let identity<'T> = Tensor.identity<'T> Dev

    let arange = Tensor.arange Dev

    let inline linspace start stop nElems = 
        Tensor.linspace Dev start stop nElems

    /// Creates a ITensor for the given pointer, allocation size in bytes, type and layout.
    let usingPtrAndType (ptr: CUdeviceptr) (sizeInBytes: SizeT) (typ: Type) (layout: TensorLayout) = 
        let devVarType = typedefof<CudaDeviceVariable<_>>.MakeGenericType [|typ|]
        let devVar = Activator.CreateInstance (devVarType, [|box ptr; box sizeInBytes|])

        let devStorType = typedefof<TensorCudaStorage<_>>.MakeGenericType [|typ|]
        let devStor = Activator.CreateInstance (devStorType, [|devVar|])

        let tensorType = typedefof<Tensor<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (tensorType, [|box layout; devStor|]) :?> ITensor

