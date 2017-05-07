namespace Tensor

open ManagedCuda
open ManagedCuda.BasicTypes

open Basics
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
        member this.Id = "Cuda"
        member this.Backend layout = 
            TensorCudaBackend<'T> (layout, this) :> ITensorBackend<_>
        member this.Factory =
            TensorCudaStorageFactory.Instance :> ITensorStorageFactory

    interface ITensorCudaStorage with
        member this.ByteData = this.ByteData
        member this.DataSize = this.DataSize
        member this.DataSizeInBytes = this.DataSizeInBytes


/// CUDA backend for tensors.
and TensorCudaBackend<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                    (layout: TensorLayout, storage: TensorCudaStorage<'T>) =

    /// device pointer to first element of this tensor
    member this.DevicePtr : nativeint =
        CudaSup.getIntPtr storage.Data.DevicePointer + nativeint (layout.Offset * sizeof64<'T>)
        

    interface ITensorBackend<'T> with

        member this.Copy(trgt, src) = 
            let trgtStorage = trgt.Storage :?> TensorCudaStorage<'T>
            let srcStorage = src.Storage :?> TensorCudaStorage<'T>

            if TensorLayout.hasContiguousMemory trgt.Layout && 
               TensorLayout.hasContiguousMemory src.Layout &&
               trgt.Layout.Stride = src.Layout.Stride then
                // use fast CUDA memcpy
                trgtStorage.Data.CopyToDevice (srcStorage.Data, 
                                               SizeT (sizeof64<'T> * src.Layout.Offset),
                                               SizeT (sizeof64<'T> * trgt.Layout.Offset),
                                               SizeT (sizeof64<'T> * src.NElems))
            else
                // TODO: fix
                // use slow element by element copy over host
                printfn "using slow CUDA element by element copy"
                for idx in Tensor.allIdx trgt do
                    trgtStorage.[TensorLayout.addr idx trgt.Layout] <- 
                        srcStorage.[TensorLayout.addr idx src.Layout]

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
                    trgtStorage.Data.CopyToDevice (srcMemPtr, 
                                                   SizeT (sizeof64<'T> * src.Layout.Offset), 
                                                   SizeT (sizeof64<'T> * trgt.Layout.Offset),
                                                   SizeT (sizeof64<'T> * src.NElems))
                    srcMemHnd.Dispose()
                    true

                // transfer from CUDA to host
                | (:? TensorHostStorage<'T> as trgtStorage), (:? TensorCudaStorage<'T> as srcStorage) ->
                    let trgtMemHnd, trgtMemPtr = regMem trgtStorage
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
                let tmp = Tensor<'T> (trgt.Shape, trgt.Factory, order=RowMajor)
                if doTransfer tmp src then
                    trgt.CopyFrom tmp
                    true
                else false


        member this.ArgMaxLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.ArgMinLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.FoldLastAxis(fn, initial, trgt, src, useThreads) = raise (System.NotImplementedException())
        member this.FoldLastAxisIndexed(fn, initial, trgt, src, useThreads) = raise (System.NotImplementedException())
        member this.AllLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.AnyLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.GetEnumerator() : System.Collections.IEnumerator = raise (System.NotImplementedException())
        member this.GetEnumerator() : System.Collections.Generic.IEnumerator<'T> = raise (System.NotImplementedException())
        member this.MaxLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.MinLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.ProductLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.SumLastAxis(trgt, src1) = raise (System.NotImplementedException())
        member this.Gather(trgt, srcIdxs, src) = raise (System.NotImplementedException())
        member this.IfThenElse(trgt, cond, ifTrue, ifFalse) = raise (System.NotImplementedException())
        member this.Scatter(trgt, trgtIdxs, src) = raise (System.NotImplementedException())
        member this.Abs(trgt, src1) = raise (System.NotImplementedException())
        member this.Acos(trgt, src1) = raise (System.NotImplementedException())
        member this.Asin(trgt, src1) = raise (System.NotImplementedException())
        member this.Atan(trgt, src1) = raise (System.NotImplementedException())
        member this.Ceiling(trgt, src1) = raise (System.NotImplementedException())
        member this.Cos(trgt, src1) = raise (System.NotImplementedException())
        member this.Cosh(trgt, src1) = raise (System.NotImplementedException())
        member this.Exp(trgt, src1) = raise (System.NotImplementedException())
        member this.Floor(trgt, src1) = raise (System.NotImplementedException())
        member this.Log(trgt, src1) = raise (System.NotImplementedException())
        member this.Log10(trgt, src1) = raise (System.NotImplementedException())
        member this.Round(trgt, src1) = raise (System.NotImplementedException())
        member this.Sgn(trgt, src1) = raise (System.NotImplementedException())
        member this.Sin(trgt, src1) = raise (System.NotImplementedException())
        member this.Sinh(trgt, src1) = raise (System.NotImplementedException())
        member this.Sqrt(trgt, src1) = raise (System.NotImplementedException())
        member this.Tan(trgt, src1) = raise (System.NotImplementedException())
        member this.Tanh(trgt, src1) = raise (System.NotImplementedException())
        member this.Truncate(trgt, src1) = raise (System.NotImplementedException())
        member this.IsFinite(trgt, src1) = raise (System.NotImplementedException())
        member this.UnaryMinus(trgt, src1) = raise (System.NotImplementedException())
        member this.UnaryPlus(trgt, src1) = raise (System.NotImplementedException())
        member this.Negate(trgt, src1) = raise (System.NotImplementedException())
        member this.Fill(fn, trgt, useThreads) = raise (System.NotImplementedException())
        member this.FillIndexed(fn, trgt, useThreads) = raise (System.NotImplementedException())
        member this.Convert(trgt, src) = raise (System.NotImplementedException())
        member this.FillConst (value, trgt) = failwith "notimpl"
        member this.Map(fn, trgt, src, useThreads) = raise (System.NotImplementedException())
        member this.Map2(fn, trgt, src1, src2, useThreads) = raise (System.NotImplementedException())
        member this.MapIndexed(fn, trgt, src, useThreads) = raise (System.NotImplementedException())
        member this.MapIndexed2(fn, trgt, src1, src2, useThreads) = raise (System.NotImplementedException())
        member this.Add(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Subtract(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Multiply(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Divide(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Modulo(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Power(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Equal(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.NotEqual(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Less(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.LessOrEqual(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Greater(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.GreaterOrEqual(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.And(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Or(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Xor(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.MaxElemwise(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.MinElemwise(trgt, src1, src2) = raise (System.NotImplementedException())
        member this.Item 
            with get idx = storage.[layout |> TensorLayout.addr (idx |> List.ofArray)]
            and set idx value = storage.[layout |> TensorLayout.addr (idx |> List.ofArray)] <- value
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

            

and TensorCudaStorageFactory private () =
    static member Instance = TensorCudaStorageFactory ()

    interface ITensorStorageFactory with
        member this.Create nElems : ITensorStorage<'T> = 
            // we use reflection to drop the constraints on 'T 
            let ts = typedefof<TensorCudaStorage<_>>.MakeGenericType (typeof<'T>)
            Activator.CreateInstance(ts, nElems) :?> ITensorStorage<'T>
        member this.Zeroed = false


[<AutoOpen>]            
module CudaTensorTypes =
    /// Tensor stored on CUDA device.
    let DevCuda = TensorCudaStorageFactory.Instance


/// Tensor stored on CUDA device.
module CudaTensor =

    let transfer x = Tensor.transfer DevCuda x

    let zeros<'T> = Tensor.zeros<'T> DevCuda 

    let ones<'T> = Tensor.ones<'T> DevCuda

    let falses = Tensor.falses DevCuda

    let trues = Tensor.trues DevCuda

    let scalar<'T> = Tensor.scalar<'T> DevCuda

    let init<'T> = Tensor.init<'T> DevCuda

    let filled<'T> = Tensor.filled<'T> DevCuda

    let identity<'T> = Tensor.identity<'T> DevCuda

    let arange = Tensor.arange DevCuda

    let inline linspace start stop nElems = 
        Tensor.linspace DevCuda start stop nElems

    /// Creates a ITensor for the given pointer, allocation size in bytes, type and layout.
    let usingPtr (ptr: CUdeviceptr) (sizeInBytes: SizeT) (typ: Type) (layout: TensorLayout) = 
        let devVarType = typedefof<CudaDeviceVariable<_>>.MakeGenericType [|typ|]
        let devVar = Activator.CreateInstance (devVarType, [|box ptr; box sizeInBytes|])

        let devStorType = typedefof<TensorCudaStorage<_>>.MakeGenericType [|typ|]
        let devStor = Activator.CreateInstance (devStorType, [|devVar|])

        let tensorType = typedefof<Tensor<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (tensorType, [|box layout; devStor|]) :?> ITensor





