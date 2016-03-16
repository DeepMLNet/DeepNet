namespace ArrayNDNS

open System
open ManagedCuda
open ManagedCuda.BasicTypes
open Basics.Cuda

open Basics
open ArrayND
open ArrayNDHost


[<AutoOpen>]
module ArrayNDCudaTypes =

    /// variable stored on CUDA device
    let LocDev = ArrayLoc "CUDA"

    let (|LocDev|_|) arg =
        if arg = ArrayLoc "CUDA" then Some () else None

    type IDeviceStorage =
        abstract ByteData: CudaDeviceVariable<byte>

    type IDeviceStorage<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> =
        inherit IDeviceStorage
        abstract Item: int -> 'T with get, set
        abstract Data: CudaDeviceVariable<'T>

    /// Storage in a CudaDeviceVariable. 
    /// The underlying CudaDeviceVariable is disposed when this object is finalized.
    type CudaDeviceVariableStorageT<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                                                               (data: CudaDeviceVariable<'T>) =

        new (size: int) = 
            CudaSup.init()
            CudaDeviceVariableStorageT<'T> (new CudaDeviceVariable<'T> (SizeT(size)))

        interface IDeviceStorage<'T> with
            member this.Item 
                with get(index) = 
                    let hostBuf = ref (new 'T())
                    data.CopyToHost(hostBuf, SizeT(index * sizeof<'T>))
                    !hostBuf
                and set index (value: 'T) = 
                    data.CopyToDevice(value, SizeT(index * sizeof<'T>))

            member this.Data = data

        interface IDeviceStorage with
            member this.ByteData =
                new CudaDeviceVariable<byte> (data.DevicePointer, data.SizeInBytes)

        override this.Finalize() =
            data.Dispose()

        member this.CudaDeviceVariabe = data


    /// type-neutral interface to an ArrayNDCudaT
    type IArrayNDCudaT =
        inherit IArrayNDT
        abstract Storage: IDeviceStorage
        abstract ToHost: unit -> IArrayNDHostT

    /// an N-dimensional array with reshape and subview abilities stored in GPU device memory
    type ArrayNDCudaT<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                    (layout: ArrayNDLayoutT, 
                                     storage: IDeviceStorage<'T>) = 
        inherit ArrayNDT<'T>(layout)
        
        /// a new ArrayND stored on the GPU using newly allocated device memory
        new (layout: ArrayNDLayoutT) =
            let elems = ArrayNDLayout.nElems layout
            // CUDA cannot allocate memory of size zero
            let size = if elems > 0 then elems else 1
            ArrayNDCudaT<'T> (layout, CudaDeviceVariableStorageT<'T> size)

        /// storage
        member this.Storage = storage

        override this.Location = LocDev

        override this.Item
            with get pos = storage.[ArrayNDLayout.addr pos layout]
            and set pos value = 
                ArrayND.doCheckFinite value            
                storage.[ArrayNDLayout.addr pos layout] <- value 

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            ArrayNDCudaT<'T>(layout) :> ArrayNDT<'T>

        override this.NewView (layout: ArrayNDLayoutT) = 
            ArrayNDCudaT<'T>(layout, storage) :> ArrayNDT<'T>

        member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =
            ArrayND.view (this.ToRng allArgs) this
        member this.Item
            with get ([<System.ParamArray>] allArgs: obj []) = this.GetSlice (allArgs)
            and set (arg0: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj, arg6: obj) (value: ArrayNDT<'T>) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; arg6; value :> obj|])

        /// creates a new contiguous (row-major) ArrayNDCudaT in device memory of the given shape 
        static member NewContiguous shp =
            ArrayNDCudaT<_> (ArrayNDLayout.newContiguous shp)    

        /// creates a new Fortran (column-major) ArrayNDCudaT in device memory of the given shape
        static member NewColumnMajor shp =
            ArrayNDCudaT<_>(ArrayNDLayout.newColumnMajor shp)

        /// Copies a ArrayNDHostT into the specified ArrayNDCudaT.
        /// Both must of same shape. dst must also be contiguous and with offset zero.
        static member CopyIntoDev (dst: ArrayNDCudaT<'T>) (src: ArrayNDHostT<'T>) =
            if ArrayND.shape dst <> ArrayND.shape src then
                invalidArg "dst" "dst and src must be of same shape"
            if not (ArrayND.isContiguous dst && ArrayND.offset dst = 0) then
                invalidArg "dst" "dst must be contiguous without offset"

            let src = ArrayND.makeContiguousAndOffsetFree src 
            use srcMem = src.Storage.Pin()
        
            match dst.Storage with
            | :? CudaDeviceVariableStorageT<'T> as ds ->
                ds.CudaDeviceVariabe.CopyToDevice(srcMem.Ptr, SizeT(0), SizeT(0), 
                                                  SizeT(sizeof<'T> * ArrayND.nElems src))
            | _ -> failwith "cannot copy to that device storage"

        /// Copies the specified ArrayNDHostT to the device
        static member OfHost (src: ArrayNDHostT<'T>) =
            let dst = ArrayNDCudaT<_>.NewContiguous (ArrayND.shape src)
            ArrayNDCudaT<_>.CopyIntoDev dst src
            dst

        /// Copies a ArrayNDCudaT into the specified ArrayNDHostT.
        /// Both must of same shape. dst must also be contiguous and with offset zero.
        static member CopyIntoHost (dst: ArrayNDHostT<'T>) (src: ArrayNDCudaT<'T>) =
            if ArrayND.shape dst <> ArrayND.shape src then
                invalidArg "dst" "dst and src must be of same shape"
            if not (ArrayND.isContiguous dst && ArrayND.offset dst = 0) then
                invalidArg "dst" "dst must be contiguous without offset"

            let src = ArrayND.makeContiguousAndOffsetFree src 
            use dstMem = dst.Storage.Pin()
        
            match src.Storage with
            | :? CudaDeviceVariableStorageT<'T> as ds ->
                ds.CudaDeviceVariabe.CopyToHost(dstMem.Ptr, SizeT(0), SizeT(0), 
                                                SizeT(sizeof<'T> * ArrayND.nElems src))
            | _ -> failwith "cannot copy from unkown device storage"

        /// Copies this ArrayNDCudaT to the host
        member this.ToHost () =
            let dst = ArrayNDHost.newContiguous (ArrayND.shape this) 
            ArrayNDCudaT<_>.CopyIntoHost dst this
            dst

        interface IArrayNDCudaT with
            member this.Storage = this.Storage :> IDeviceStorage
            member this.ToHost () = this.ToHost () :> IArrayNDHostT



module ArrayNDCuda = 

    /// creates a new contiguous (row-major) ArrayNDCudaT in device memory of the given shape 
    let inline newContiguous shp =
        ArrayNDCudaT<_>.NewContiguous shp

    /// creates a new Fortran (column-major) ArrayNDCudaT in device memory of the given shape
    let inline newColumnMajor shp =
        ArrayNDCudaT<_>.NewColumnMajor shp

    /// ArrayNDCudaT with zero dimensions (scalar) and given value
    let inline scalar value =
        let a = newContiguous [] 
        ArrayND.set [] value a
        a

    /// ArrayNDCudaT of given shape filled with zeros.
    let inline zeros shape =
        newContiguous shape

    /// ArrayNDCudaT of given shape filled with ones.
    let inline ones shape =
        let a = newContiguous shape
        ArrayND.fillWithOnes a
        a

    /// ArrayNDCudaT identity matrix
    let inline identity size =
        let a = zeros [size; size]
        ArrayND.fillDiagonalWithOnes a
        a

    /// Copies a ArrayNDHostT into the specified ArrayNDCudaT.
    /// Both must of same shape. dst must also be contiguous and with offset zero.
    let copyIntoDev dst src = ArrayNDCudaT.CopyIntoDev dst src

    /// Copies a ArrayNDHostT to the device
    let toDev src = ArrayNDCudaT.OfHost src

    /// Copies a ArrayNDCudaT into the specified ArrayNDHostT.
    /// Both must of same shape. dst must also be contiguous and with offset zero.
    let copyIntoHost dst src = ArrayNDCudaT.CopyIntoHost dst src

    /// Copies the specified ArrayNDCudaT to the host
    let toHost (src: ArrayNDCudaT<_>) = src.ToHost()

    /// Creates a new IArrayNDT of given type and layout in device memory.
    let newOfType typ (layout: ArrayNDLayoutT) = 
        let aryType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (aryType, [|box layout|]) :?> IArrayNDCudaT

    /// Creates a IArrayNDT for the given pointer, type and layout.
    let fromPtrAndType (ptr: CUdeviceptr) typ (layout: ArrayNDLayoutT) = 
        let devVarType = typedefof<CudaDeviceVariable<_>>.MakeGenericType [|typ|]
        let devVar = Activator.CreateInstance (devVarType, [|box ptr|])

        let devStorType = typedefof<CudaDeviceVariableStorageT<_>>.MakeGenericType [|typ|]
        let devStor = Activator.CreateInstance (devStorType, [|devVar|])

        let aryType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (aryType, [|box layout; devStor|]) :?> IArrayNDCudaT

