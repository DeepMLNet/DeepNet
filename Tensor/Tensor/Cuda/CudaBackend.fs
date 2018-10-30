namespace Tensor.Cuda

open System
open System.Threading
open System.Runtime.InteropServices

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor
open Tensor.Backend
open Tensor.Host
open Tensor.Cuda.CudaBLASExtensions
open Tensor.Utils
open DeepNet.Utils



/// CUDA initialization helper.
/// We check that CUDA context access is possible before calling other CUDA functions to avoid
/// exceptions being thrown from static initializers.
module private CudaInit =

    /// the CUDA context we are using
    let mutable private initialized = false

    /// Checks if CUDA context access is possible.
    /// If not, a CudaError is thrown.
    let check () =
        if not initialized then
            // First try to obtain a CudaContext.
            try new CudaContext(createNew=false) |> ignore
            with e ->
                let msg = sprintf "Cannot create CUDA context: %s" e.Message
                raise (CudaException msg)
            // If this succeeds, initialize the Cuda module.
            Cuda.context.Context |> ignore
            Cuda.blas.CublasHandle |> ignore
            initialized <- true



/// type neutral interface to a CudaStorageT
type ITensorCudaStorage =
    abstract ByteData: CudaDeviceVariable<byte>
    abstract DataSize: int64
    abstract DataSizeInBytes: int64



/// Tensor storage on a CUDA device.
type TensorCudaStorage<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                    (data: CudaDeviceVariable<'T>) =

    new (nElems: int64) =
        CudaInit.check ()
        // CUDA cannot allocate memory of size zero
        let nElems = if nElems > 0L then nElems else 1L
        TensorCudaStorage<'T> (Cuda.newDevVar nElems)
     
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
        member this.Dev =
            TensorCudaDevice.Instance :> ITensorDevice

    interface ITensorCudaStorage with
        member this.ByteData = this.ByteData
        member this.DataSize = this.DataSize
        member this.DataSizeInBytes = this.DataSizeInBytes

    interface BLAS.IBLASStorage with
        member this.Pin () =
            let d = { new IDisposable with member this.Dispose() = () }
            d, Cuda.getIntPtr this.Data.DevicePointer



/// type-neutral interface to CUDA backend for tensors
and ITensorCudaBackend =
    abstract NativeTensor: NativeTensor



/// CUDA backend for tensors.
and TensorCudaBackend<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                    (layout: TensorLayout, storage: TensorCudaStorage<'T>) =

    let kernels = TensorKernels.Get (typeof<'T>, layout.NDims)
    let blasSupportKernels = BlasSupportKernels.Get ()

    let unsup op =
        let msg = 
            sprintf "the CUDA tensor backend currently does not support the %s operation" op
        raise (NotSupportedException msg)

    let callUnary fn trgt src1 : unit =
        let trgt, src1 = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt, src1)
        fn (Cfg.Stream, trgt, src1)
        Cuda.keepAliveMany Cfg.Stream [trgt; src1]

    let callBinary fn trgt src1 src2 : unit =
        let trgt, src1, src2 = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt, src1, src2)
        fn (Cfg.Stream, trgt, src1, src2)
        Cuda.keepAliveMany Cfg.Stream [trgt; src1; src2]

    let callTenary fn trgt src1 src2 src3 : unit =
        let trgt, src1, src2, src3 = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt, src1, src2, src3)
        fn (Cfg.Stream, trgt, src1, src2, src3)
        Cuda.keepAliveMany Cfg.Stream [trgt; src1; src2; src3]

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
        Storage     = storage
    }

    interface ITensorCudaBackend with
        member this.NativeTensor = this.NativeTensor

    /// gets NativeTensors for specified tensors
    static member internal GetNativeTensor (t: ITensorFrontend<'T>) =
        (t.Backend :?> TensorCudaBackend<'T>).NativeTensor

    /// gets NativeTensors for specified tensors
    static member internal GetNativeTensor (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>) =
        (t.Backend :?> TensorCudaBackend<'T>).NativeTensor, 
        (a.Backend :?> TensorCudaBackend<'TA>).NativeTensor 

    /// gets NativeTensors for specified tensors, optimized for elment-wise operations
    static member internal ElemwiseNativeTensor<'T> (t: ITensorFrontend<'T>) 
            : NativeTensor =
        (t.Backend :?> ITensorCudaBackend).NativeTensor

    /// gets NativeTensors for specified tensors, optimized for elment-wise operations
    static member internal ElemwiseNativeTensor<'T, 'TA> 
            (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>) : NativeTensor * NativeTensor =
        (t.Backend :?> ITensorCudaBackend).NativeTensor, 
        (a.Backend :?> ITensorCudaBackend).NativeTensor 

    /// gets NativeTensors for specified tensors, optimized for elment-wise operations
    static member internal ElemwiseNativeTensor<'T, 'TA, 'TB>  
            (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>, b: ITensorFrontend<'TB>) : NativeTensor * NativeTensor * NativeTensor =
        (t.Backend :?> ITensorCudaBackend).NativeTensor, 
        (a.Backend :?> ITensorCudaBackend).NativeTensor,
        (b.Backend :?> ITensorCudaBackend).NativeTensor

    /// gets NativeTensors for specified tensors, optimized for elment-wise operations
    static member internal ElemwiseNativeTensor<'T, 'TA, 'TB, 'TC>  
            (t: ITensorFrontend<'T>, a: ITensorFrontend<'TA>, b: ITensorFrontend<'TB>, c: ITensorFrontend<'TC>) 
            : NativeTensor * NativeTensor * NativeTensor * NativeTensor =
        (t.Backend :?> ITensorCudaBackend).NativeTensor, 
        (a.Backend :?> ITensorCudaBackend).NativeTensor,
        (b.Backend :?> ITensorCudaBackend).NativeTensor,
        (c.Backend :?> ITensorCudaBackend).NativeTensor


    interface ITensorBackend<'T> with

        member this.Item 
            with get idx = storage.[layout |> TensorLayout.addr (idx |> List.ofArray)]
            and set idx value = storage.[layout |> TensorLayout.addr (idx |> List.ofArray)] <- value
           
        member this.Transfer (trgt, src) =
            /// gets CUDA registered or pinned memory
            let getMem (storage: TensorHostStorage<'T>) = 
                try
                    let h = CudaRegMem.register storage
                    h :> IDisposable, h.Ptr
                with CannotCudaRegisterMemoryException _ -> 
                    let h = storage.Pin()
                    h :> IDisposable, h.Ptr

            let doTransfer (trgt: ITensorFrontend<'T>) (src: ITensorFrontend<'T>) = 
                match trgt.Storage, src.Storage with
                // transfer from host to CUDA
                | (:? TensorCudaStorage<'T> as trgtStorage), (:? TensorHostStorage<'T> as srcStorage) ->
                    let sizeInBytes = sizeof64<'T> * src.NElems
                    let srcMemHnd, srcMemPtr = getMem srcStorage
                    let trgtMemPtr = Cuda.getIntPtr trgtStorage.Data.DevicePointer

                    let srcMemOffsetPtr = srcMemPtr + nativeint (sizeof64<'T> * src.Layout.Offset)
                    let trgtMemOffsetPtr = trgtMemPtr + nativeint (sizeof64<'T> * trgt.Layout.Offset)
                    use srcRegMem = new CudaRegisteredHostMemory<byte> (srcMemOffsetPtr, SizeT sizeInBytes)
                    if Cfg.Stream = CUstream.NullStream then
                        srcRegMem.SynchronCopyToDevice (CUdeviceptr (SizeT trgtMemOffsetPtr))
                    else
                        srcRegMem.AsyncCopyToDevice(CUdeviceptr (SizeT trgtMemOffsetPtr), Cfg.Stream)
                   
                    Cuda.callback Cfg.Stream (fun () -> srcMemHnd.Dispose())
                    Cuda.keepAliveMany Cfg.Stream [trgtStorage; srcStorage]
                    true

                // transfer from CUDA to host
                | (:? TensorHostStorage<'T> as trgtStorage), (:? TensorCudaStorage<'T> as srcStorage) ->
                    let sizeInBytes = sizeof64<'T> * src.NElems
                    let trgtMemHnd, trgtMemPtr = getMem trgtStorage
                    let srcMemPtr = Cuda.getIntPtr srcStorage.Data.DevicePointer

                    let srcMemOffsetPtr = srcMemPtr + nativeint (sizeof64<'T> * src.Layout.Offset)
                    let trgtMemOffsetPtr = trgtMemPtr + nativeint (sizeof64<'T> * trgt.Layout.Offset)
                    use trgtRegMem = new CudaRegisteredHostMemory<byte> (trgtMemOffsetPtr, SizeT sizeInBytes)
                    if Cfg.Stream = CUstream.NullStream then
                        trgtRegMem.SynchronCopyToHost(CUdeviceptr (SizeT srcMemOffsetPtr))
                    else
                        trgtRegMem.AsyncCopyFromDevice(CUdeviceptr (SizeT srcMemOffsetPtr), Cfg.Stream)
                   
                    Cuda.callback Cfg.Stream (fun () -> trgtMemHnd.Dispose())
                    Cuda.keepAliveMany Cfg.Stream [trgtStorage; srcStorage]
                    true
                | _ -> false

            // ensure that source is in row-major order
            let src =
                if TensorLayout.isRowMajor src.Layout then src
                else src.Copy(order=RowMajor)
               
            if TensorLayout.isRowMajor trgt.Layout then
                // target is in row-major order, do direct transfer
                doTransfer trgt src
            else
                // target is not in row-major order, transfer to temporary tensor
                // and copy into target
                let tmp = Tensor<'T> (trgt.Shape, trgt.Dev, order=RowMajor)
                if doTransfer tmp src then
                    trgt.CopyFrom tmp
                    true
                else false

        member this.FillConst (value, trgt) = 
            let trgt = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt)
            kernels.FillConst (Cfg.Stream, box value, trgt)
            Cuda.keepAlive Cfg.Stream trgt

        member this.FillIncrementing (start, incr, trgt) = 
            let trgt = TensorCudaBackend<_>.ElemwiseNativeTensor (trgt)
            kernels.FillIncrementing (Cfg.Stream, box start, box incr, trgt)
            Cuda.keepAlive Cfg.Stream trgt            

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
                                                    Cfg.Stream)
                Cuda.keepAliveMany Cfg.Stream [trgtStorage; srcStorage]
            else
                // call copy kernel
                let trgt, src = TensorCudaBackend<_>.GetNativeTensor (trgt, src)
                kernels.Copy(Cfg.Stream, trgt, src)
                Cuda.keepAliveMany Cfg.Stream [trgt; src]

        member this.Convert(trgt: ITensorFrontend<'T>, src: ITensorFrontend<'S>) =
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
        member this.FindLastAxis(value, trgt, src1) = callUnary (kernels.FindLastAxis value) trgt src1        

        member this.Gather(trgt, srcIdxs, src) = 
            let gsKernels = TensorGatherScatterKernels.Get (typeof<'T>, trgt.NDims, src.NDims)
            let srcIdxs = {
                NDims = trgt.NDims
                Idxs  = srcIdxs |> List.map (Option.map (TensorCudaBackend<_>.GetNativeTensor))
            }
            let trgt, src = TensorCudaBackend<_>.GetNativeTensor (trgt, src)
            gsKernels.Gather (Cfg.Stream, trgt, srcIdxs, src)
            Cuda.keepAliveMany Cfg.Stream [trgt; srcIdxs; src]

        member this.Scatter(trgt, trgtIdxs, src) = 
            let gsKernels = TensorGatherScatterKernels.Get (typeof<'T>, trgt.NDims, src.NDims)
            let trgtIdxs = {
                NDims = src.NDims
                Idxs  = trgtIdxs |> List.map (Option.map (TensorCudaBackend<_>.GetNativeTensor))
            }
            let trgt, src = TensorCudaBackend<_>.GetNativeTensor (trgt, src)
            kernels.FillConst (Cfg.Stream, box (conv<'T> 0), trgt)
            gsKernels.Scatter (Cfg.Stream, trgt, trgtIdxs, src)
            Cuda.keepAliveMany Cfg.Stream [trgt; trgtIdxs; src]

        member this.VecVecDot (trgt, a, b) =
            use x = BLAS.GetVector (a, isSource=true, isTarget=false)
            use y = BLAS.GetVector (b, isSource=true, isTarget=false)
            use t = BLAS.GetScalar (trgt)
            CUBLAS.Invoke<'T, unit>
                (Cfg.Stream,
                 singleFn=(fun () -> Cuda.blas.Dot (x.CPtr<single>(a.NElems), x.CInc, 
                                                    y.CPtr(a.NElems), y.CInc, t.CPtr())),
                 doubleFn=(fun () -> Cuda.blas.Dot (x.CPtr<double>(a.NElems), x.CInc, 
                                                    y.CPtr(a.NElems), y.CInc, t.CPtr())))
            Cuda.keepAliveMany Cfg.Stream [x; y; t]

        member this.MatVecDot (trgt, a, b) =
            use a = BLAS.GetMatrix (a, isSource=true, isTarget=false, canTranspose=true)
            use x = BLAS.GetVector (b, isSource=true, isTarget=false)
            use y = BLAS.GetVector (trgt, isSource=false, isTarget=true)
            CUBLAS.Invoke<'T, unit>
                (Cfg.Stream,
                 singleFn=(fun () -> Cuda.blas.Gemv (a.CTrans, a.CRows, a.CCols, 1.0f,
                                                     a.CPtr(), a.CLd, x.CPtr(), x.CInc,
                                                     0.0f, y.CPtr(), y.CInc)),
                 doubleFn=(fun () -> Cuda.blas.Gemv (a.CTrans, a.CRows, a.CCols, 1.0,
                                                     a.CPtr(), a.CLd, x.CPtr(), x.CInc,
                                                     0.0, y.CPtr(), y.CInc)))  
            y.FetchResult()
            Cuda.keepAliveMany Cfg.Stream [a; x; y]

        member this.MatMatDot (trgt, a, b) =
            Cuda.blas.Stream <- Cfg.Stream
            use a = BLAS.GetMatrix (a, isSource=true, isTarget=false, canTranspose=true)
            use b = BLAS.GetMatrix (b, isSource=true, isTarget=false, canTranspose=true)
            use c = BLAS.GetMatrix (trgt, isSource=false, isTarget=true, canTranspose=false)
            CUBLAS.Invoke<'T, unit>
                (Cfg.Stream,
                 singleFn=(fun () -> Cuda.blas.Gemm (a.CTrans, b.CTrans, a.COpRows, b.COpCols, a.COpCols, 
                                                     1.0f, a.CPtr(), a.CLd, b.CPtr(), b.CLd,
                                                     0.0f, c.CPtr(), c.CLd)),
                 doubleFn=(fun () -> Cuda.blas.Gemm (a.CTrans, b.CTrans, a.COpRows, b.COpCols, a.COpCols, 
                                                     1.0, a.CPtr(), a.CLd, b.CPtr(), b.CLd,
                                                     0.0, c.CPtr(), c.CLd)))                                                      
            c.FetchResult()
            Cuda.keepAliveMany Cfg.Stream [a; b; c]

        member this.BatchedMatMatDot (trgt, a, b) =
            use a = BLAS.GetMatrix (a, isSource=true, isTarget=false, canTranspose=true)
            use b = BLAS.GetMatrix (b, isSource=true, isTarget=false, canTranspose=true)
            use c = BLAS.GetMatrix (trgt, isSource=false, isTarget=true, canTranspose=false)
            let aPtrs, aDispose = a.CPtrs Cfg.Stream
            let bPtrs, bDispose = b.CPtrs Cfg.Stream
            let cPtrs, cDispose = c.CPtrs Cfg.Stream
            CUBLAS.Invoke<'T, unit>
                (Cfg.Stream,
                 singleFn=(fun () -> Cuda.blas.GemmBatched (a.CTrans, b.CTrans, 
                                                            a.COpRows, b.COpCols, a.COpCols, 1.0f,
                                                            aPtrs, a.CLd, bPtrs, b.CLd,
                                                            0.0f, cPtrs, c.CLd,
                                                            a.CBatchSize)),
                 doubleFn=(fun () -> Cuda.blas.GemmBatched (a.CTrans, b.CTrans, 
                                                            a.COpRows, b.COpCols, a.COpCols, 1.0,
                                                            aPtrs, a.CLd, bPtrs, b.CLd,
                                                            0.0, cPtrs, c.CLd,
                                                            a.CBatchSize)))
            aDispose()
            bDispose()
            cDispose()
            c.FetchResult()
            Cuda.keepAliveMany Cfg.Stream [a; b; c]

        member this.BatchedInvert (trgt, src) =
            let size = trgt.Shape.[trgt.NDims-2]

            let lu = src.Copy(order=BLAS.MatrixOrder trgt.NDims)
            use a = BLAS.GetMatrix (lu, isSource=true, isTarget=false, canTranspose=false)
            use c = BLAS.GetMatrix (trgt, isSource=false, isTarget=true, canTranspose=false)
            let aPtrs, aDispose = a.CPtrs Cfg.Stream
            let cPtrs, cDispose = c.CPtrs Cfg.Stream

            // compute LU factorization
            let ipiv = new CudaDeviceVariable<int> (SizeT (size * a.BatchSize))
            let info = new CudaDeviceVariable<int> (SizeT a.BatchSize)
            CUBLAS.Invoke<'T, unit> 
                (Cfg.Stream,
                 singleFn=(fun () -> Cuda.blas.GetrfBatchedS (a.CRows, aPtrs, a.CLd, ipiv, info, a.CBatchSize)),
                 doubleFn=(fun () -> Cuda.blas.GetrfBatchedD (a.CRows, aPtrs, a.CLd, ipiv, info, a.CBatchSize)))
            if not (blasSupportKernels.CheckBlasInfo (Cfg.Stream, info)) then
                raise (SingularMatrixException "cannot invert singular matrix")

            // compute matrix inverse
            CUBLAS.Invoke<'T, unit> 
                (Cfg.Stream,
                 singleFn=(fun () -> Cuda.blas.GetriBatchedS (a.CRows, aPtrs, a.CLd, ipiv, cPtrs, c.CLd, info, a.CBatchSize)),
                 doubleFn=(fun () -> Cuda.blas.GetriBatchedD (a.CRows, aPtrs, a.CLd, ipiv, cPtrs, c.CLd, info, a.CBatchSize)))
            if not (blasSupportKernels.CheckBlasInfo (Cfg.Stream, info)) then
                raise (SingularMatrixException "cannot invert singular matrix")

            aDispose()
            cDispose()
            c.FetchResult()
            Cuda.keepAliveMany Cfg.Stream [a; c]
            Cuda.callback Cfg.Stream (fun () ->
                ipiv.Dispose()
                info.Dispose())                

        // unsupported for now on CUDA
        member this.BatchedSVD (trgtS, trgtUV, src) = unsup "BatchedSVD"
        member this.SymmetricEigenDecomposition (part, trgtEigVals, trgtEigVec, src) = unsup "SymmetricEigenDecomposition"
        member this.CountTrueLastAxis(trgt, src1) = unsup "CountTrueLastAxis"
        member this.MaskedGet(trgt, src, mask) = unsup "MaskedGet"
        member this.MaskedSet(trgt, mask, src) = unsup "MaskedSet"
        member this.TrueIndices(trgt, src) = unsup "TrueIndices"



/// Creates Tensors on a CUDA device.
and TensorCudaDevice private () =
    inherit BaseTensorDevice()
    static member Instance = TensorCudaDevice ()
    
    override this.Id = "Cuda"
    override this.Create nElems = 
        CudaInit.check ()
        // We use reflection to drop the constraints on 'T.
        let ts = typedefof<TensorCudaStorage<_>>.MakeGenericType (typeof<'T>)
        Activator.CreateInstance(ts, nElems) :?> ITensorStorage<'T>
    override this.Zeroed = false

