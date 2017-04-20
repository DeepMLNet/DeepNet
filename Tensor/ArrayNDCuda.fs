namespace ArrayNDNS

open System
open System.Reflection
open ManagedCuda
open ManagedCuda.BasicTypes

open Basics
open Basics.Cuda
open Tensor
open ArrayNDHost


module private CudaContext =
    let mutable context = None
    let init () =
        match context with
        | Some _ -> ()
        | None ->
            try
                context <- Some (new CudaContext(createNew=false))
            with e ->
                printfn "Cannot create CUDA context: %s" e.Message
                failwithf "Cannot create CUDA context: %s" e.Message

[<AutoOpen>]
module ArrayNDCudaTypes =

    /// variable stored on CUDA device
    let LocDev = ArrayLoc "CUDA"

    let (|LocDev|_|) arg =
        if arg = ArrayLoc "CUDA" then Some () else None

    /// type neutral interface to a CudaStorageT
    type ICudaStorage =
        abstract ByteData: CudaDeviceVariable<byte>

    /// create a new CUDA device variable
    let private newCudaDeviceVariable<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
            (elems: int64) = 
        try new CudaDeviceVariable<'T> (SizeT elems)
        with :? CudaException as e when e.CudaError = CUResult.ErrorOutOfMemory 
                                     || e.CudaError = CUResult.ErrorUnknown ->
            let sizeInBytes = elems * sizeof64<'T>
            failwithf "CUDA memory allocation of %d MB failed (%A)" 
                      (sizeInBytes / pown 2L 20) e.CudaError

    /// CUDA memory allocation
    type CudaStorageT<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                    (data: CudaDeviceVariable<'T>) =

        new (elems: int64) =
            CudaContext.init ()

            // CUDA cannot allocate memory of size zero
            let elems = if elems > 0L then elems else 1L
            CudaStorageT<'T> (newCudaDeviceVariable<'T> elems)

        member this.Data = data

        override this.Finalize() = 
            if data <> null then data.Dispose()

        interface ICudaStorage with
            member this.ByteData =
                new CudaDeviceVariable<byte> (data.DevicePointer, data.SizeInBytes)
        

    /// type-neutral interface to an ArrayNDCudaT
    type IArrayNDCudaT =
        inherit ITensor
        abstract Storage: ICudaStorage
        abstract ToHost: unit -> IArrayNDHostT

    /// an N-dimensional array with reshape and subview abilities stored in GPU device memory
    type ArrayNDCudaT<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                    (layout:    TensorLayout, 
                                     storage:   CudaStorageT<'T>) = 
        inherit Tensor<'T>(layout)
       
        let getElement (index: int64) =
            if typeof<'T> = typeof<bool> then
                let hostBuf : byte ref = ref 0uy
                (storage :> ICudaStorage).ByteData.CopyToHost(hostBuf, SizeT (index * sizeof64<byte>))
                !hostBuf <> 0uy |> box |> unbox
            else
                let hostBuf = ref (new 'T())
                storage.Data.CopyToHost(hostBuf, SizeT (index * sizeof64<'T>))
                !hostBuf

        let setElement (index: int64) (value: 'T) =
            if typeof<'T> = typeof<bool> then
                let byteVal = if (box value :?> bool) then 1uy else 0uy
                (storage :> ICudaStorage).ByteData.CopyToDevice(byteVal, SizeT (index * sizeof64<byte>))
            else
                storage.Data.CopyToDevice(value, SizeT (index * sizeof64<'T>))

        /// a new ArrayND stored on the GPU using newly allocated device memory
        new (layout: TensorLayout) =
            let elems = TensorLayout.nElems layout
            ArrayNDCudaT<'T> (layout, new CudaStorageT<'T> (elems))

        /// storage
        member this.Storage = storage

        override this.Location = LocDev

        override this.Item
            with get pos = getElement (TensorLayout.addr pos layout)
            and set pos value = 
                Tensor.doCheckFinite value            
                setElement (TensorLayout.addr pos layout) value 

        override this.NewOfSameType (layout: TensorLayout) = 
            ArrayNDCudaT<'T> (layout) :> Tensor<'T>

        override this.NewOfType<'N> (layout: TensorLayout) = 
            // drop constraint on 'N
            let aryType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|typeof<'N>|]
            Activator.CreateInstance (aryType, [|box layout|]) :?> Tensor<'N>

        override this.NewView (layout: TensorLayout) = 
            ArrayNDCudaT<'T> (layout, storage) :> Tensor<'T>

        member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =
            Tensor.view (this.ToRng allArgs) this
        member this.Item
            with get ([<System.ParamArray>] allArgs: obj []) = this.GetSlice (allArgs)
            and set (arg0: obj) (value: Tensor<'T>) = 
                this.SetSlice ([|arg0; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj) (value: Tensor<'T>) = 
                this.SetSlice ([|arg0; arg1; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj) (value: Tensor<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj) (value: Tensor<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj) (value: Tensor<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj) (value: Tensor<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj, arg6: obj) (value: Tensor<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; arg6; value :> obj|])

        static member (====) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) ==== b :?> ArrayNDCudaT<bool>
        static member (<<<<) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) <<<< b :?> ArrayNDCudaT<bool>
        static member (<<==) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) <<== b :?> ArrayNDCudaT<bool>
        static member (>>>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) >>>> b :?> ArrayNDCudaT<bool>
        static member (>>==) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) >>== b :?> ArrayNDCudaT<bool>            
        static member (<<>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) <<>> b :?> ArrayNDCudaT<bool>

        /// creates a new contiguous (row-major) ArrayNDCudaT in device memory of the given shape 
        static member NewC shp =
            ArrayNDCudaT<_> (TensorLayout.newC shp)    

        /// creates a new Fortran (column-major) ArrayNDCudaT in device memory of the given shape
        static member NewF shp =
            ArrayNDCudaT<_>(TensorLayout.newF shp)

        /// Copies a ArrayNDHostT into the specified ArrayNDCudaT.
        /// Both must of same shape. dst must also be contiguous and with offset zero.
        static member CopyIntoDev (dst: ArrayNDCudaT<'T>) (src: ArrayNDHostT<'T>) =
            if Tensor.shape dst <> Tensor.shape src then
                invalidArg "dst" "dst and src must be of same shape"
            if not (Tensor.isC dst) then
                invalidArg "dst" "dst must be contiguous"
            //printfn "CopyIntoDev: src: isC=%A  offset=%d" (ArrayND.isC src) (ArrayND.offset src)

            let src = Tensor.ensureC src 
            let srcMemHnd, srcMemPtr =
                try
                    let h = ArrayNDHostReg.lock src
                    h :> IDisposable, h.Ptr
                with CannotRegisterMemory -> 
                    let h = src.Pin()
                    h :> IDisposable, h.Ptr

            dst.Storage.Data.CopyToDevice(srcMemPtr, 
                                          SizeT (sizeof64<'T> * Tensor.offset src), 
                                          SizeT (sizeof64<'T> * Tensor.offset dst),
                                          SizeT (sizeof64<'T> * Tensor.nElems src))
            srcMemHnd.Dispose()

        /// Copies the specified ArrayNDHostT to the device
        static member OfHost (src: ArrayNDHostT<'T>) =
            let dst = ArrayNDCudaT<_>.NewC (Tensor.shape src)
            ArrayNDCudaT<_>.CopyIntoDev dst src
            dst

        /// Copies a ArrayNDCudaT into the specified ArrayNDHostT.
        /// Both must of same shape. dst must also be contiguous and with offset zero.
        static member CopyIntoHost (dst: ArrayNDHostT<'T>) (src: ArrayNDCudaT<'T>) =
            if Tensor.shape dst <> Tensor.shape src then
                invalidArg "dst" "dst and src must be of same shape"
            if not (Tensor.isC dst) then
                invalidArg "dst" "dst must be contiguous"

            let src = Tensor.ensureC src 
            let dstMemHnd, dstMemPtr =
                try
                    let h = ArrayNDHostReg.lock dst
                    h :> IDisposable, h.Ptr
                with CannotRegisterMemory -> 
                    let h = dst.Pin()
                    h :> IDisposable, h.Ptr

            //printfn "CopyIntoHost: src: isC=%A  offset=%d" (ArrayND.isC src) (ArrayND.offset src)
            //printfn "ArrayNDCuda.CopyIntoHost: srcBase=0x%x dstBase=0x%x bytes=%d" 
            //    (nativeint src.Storage.Data.DevicePointer.Pointer) dstMemPtr
            //    (sizeof<'T> * ArrayND.nElems src)

            src.Storage.Data.CopyToHost(dstMemPtr, 
                                        SizeT (sizeof64<'T> * Tensor.offset src), 
                                        SizeT (sizeof64<'T> * Tensor.offset dst), 
                                        SizeT (sizeof64<'T> * Tensor.nElems src))
            dstMemHnd.Dispose ()
            

        override this.CopyTo (dest: Tensor<'T>) =
            Tensor<'T>.CheckSameShape this dest
            match dest with
            | :? ArrayNDCudaT<'T> as dest ->
                if Tensor.hasContiguousMemory this && Tensor.hasContiguousMemory dest &&
                        Tensor.stride this = Tensor.stride dest then
                    // use fast CUDA memcpy
                    dest.Storage.Data.CopyToDevice (this.Storage.Data, 
                                                    SizeT (sizeof64<'T> * Tensor.offset this),
                                                    SizeT (sizeof64<'T> * Tensor.offset dest),
                                                    SizeT (sizeof64<'T> * Tensor.nElems this))
                else
                    // use slow element by element copy over host
                    base.CopyTo dest
            | :? ArrayNDHostT<'T> as dest when Tensor.isC dest ->
                ArrayNDCudaT<'T>.CopyIntoHost dest this
            | _ -> base.CopyTo dest

        /// Copies this ArrayNDCudaT to the host
        member this.ToHost () =
            let dst = ArrayNDHost.newC (Tensor.shape this) 
            ArrayNDCudaT<_>.CopyIntoHost dst this
            dst

        interface IArrayNDCudaT with
            member this.ToHost () = this.ToHost () :> IArrayNDHostT
            member this.Storage = this.Storage :> ICudaStorage

        interface IToArrayNDHostT<'T> with
            member this.ToHost () = this.ToHost ()

        override this.Invert () =
            failwith "not implemented"

        override this.SymmetricEigenDecomposition () =
            failwith "not implemented"

        /// device pointer to first element of this array
        member this.DevicePtr : nativeint =
            CudaSup.getIntPtr this.Storage.Data.DevicePointer + nativeint (this.Layout.Offset * sizeof64<'T>)


module ArrayNDCuda = 

    /// creates a new contiguous (row-major) ArrayNDCudaT in device memory of the given shape 
    let newC<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> shp : ArrayNDCudaT<'T> =
        ArrayNDCudaT<'T>.NewC shp

    /// creates a new Fortran (column-major) ArrayNDCudaT in device memory of the given shape
    let newF<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> shp : ArrayNDCudaT<'T> =
        ArrayNDCudaT<_>.NewF shp

    /// ArrayNDCudaT with zero dimensions (scalar) and given value
    let scalar value =
        let a = newC [] 
        Tensor.set [] value a
        a

    /// ArrayNDCudaT of given shape filled with zeros.
    let zeros<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> shape : ArrayNDCudaT<'T> =
        newC shape

    /// ArrayNDCudaT of given shape filled with ones.
    let ones<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> shape : ArrayNDCudaT<'T> =
        let a = newC shape
        Tensor.fillWithOnes a
        a

    /// ArrayNDCudaT identity matrix
    let identity<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> size : ArrayNDCudaT<'T> =
        let a = zeros [size; size]
        Tensor.fillDiagonalWithOnes a
        a

    /// Copies a ArrayNDHostT into the specified ArrayNDCudaT.
    /// Both must of same shape. dst must also be contiguous and with offset zero.
    let copyIntoDev dst src = ArrayNDCudaT.CopyIntoDev dst src

    /// Copies an ArrayNDHostT to the device
    let toDev src = ArrayNDCudaT.OfHost src

    /// Copies an IArrayNDHostT to the device 
    let toDevUntyped (src: IArrayNDHostT) =
        let devVarType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|src.DataType|]
        let ofHost = devVarType.GetMethod("OfHost", BindingFlags.Static ||| BindingFlags.Public)
        ofHost.Invoke (null, [|box src|]) :?> IArrayNDCudaT  

    /// Copies a ArrayNDCudaT into the specified ArrayNDHostT.
    /// Both must of same shape. dst must also be contiguous and with offset zero.
    let copyIntoHost dst src = ArrayNDCudaT.CopyIntoHost dst src

    /// Copies the specified ArrayNDCudaT to the host
    let toHost (src: ArrayNDCudaT<_>) = src.ToHost()

    /// Copies the specified IArrayNDCudaT to the host
    let toHostUntyped (src: IArrayNDCudaT) = src.ToHost()

    /// Creates a new IArrayNDT of given type and layout in device memory.
    let newOfType typ (layout: TensorLayout) = 
        let aryType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (aryType, [|box layout|]) :?> IArrayNDCudaT

    /// Creates a IArrayNDT for the given pointer, allocation size in bytes, type and layout.
    let fromPtrAndType (ptr: CUdeviceptr) (sizeInBytes: SizeT) typ (layout: TensorLayout) = 
        let devVarType = typedefof<CudaDeviceVariable<_>>.MakeGenericType [|typ|]
        let devVar = Activator.CreateInstance (devVarType, [|box ptr; box sizeInBytes|])

        let devStorType = typedefof<CudaStorageT<_>>.MakeGenericType [|typ|]
        let devStor = Activator.CreateInstance (devStorType, [|devVar|])

        let aryType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (aryType, [|box layout; devStor|]) :?> IArrayNDCudaT


