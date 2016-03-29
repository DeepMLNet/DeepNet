namespace ArrayNDNS

open System
open System.Reflection
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

    /// type neutral interface to a CudaStorageT
    type ICudaStorage =
        abstract ByteData: CudaDeviceVariable<byte>

    /// CUDA memory allocation
    type CudaStorageT<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                    (data: CudaDeviceVariable<'T>) =

        new (elems: int) =
            // CUDA cannot allocate memory of size zero
            let elems = if elems > 0 then elems else 1
            CudaSup.init ()
            CudaStorageT<'T> (new CudaDeviceVariable<'T> (SizeT elems))

        member this.Data = data

        override this.Finalize() = data.Dispose()

        interface ICudaStorage with
            member this.ByteData =
                new CudaDeviceVariable<byte> (data.DevicePointer, data.SizeInBytes)
        

    /// type-neutral interface to an ArrayNDCudaT
    type IArrayNDCudaT =
        inherit IArrayNDT
        abstract Storage: ICudaStorage
        abstract ToHost: unit -> IArrayNDHostT

    /// an N-dimensional array with reshape and subview abilities stored in GPU device memory
    type ArrayNDCudaT<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                    (layout:    ArrayNDLayoutT, 
                                     storage:   CudaStorageT<'T>) = 
        inherit ArrayNDT<'T>(layout)
       
        let getElement index =
            if typeof<'T> = typeof<bool> then
                let hostBuf : byte ref = ref 0uy
                (storage :> ICudaStorage).ByteData.CopyToHost(hostBuf, SizeT (index * sizeof<byte>))
                !hostBuf <> 0uy |> box |> unbox
            else
                let hostBuf = ref (new 'T())
                storage.Data.CopyToHost(hostBuf, SizeT (index * sizeof<'T>))
                !hostBuf

        let setElement index (value: 'T) =
            if typeof<'T> = typeof<bool> then
                let byteVal = if (box value :?> bool) then 1uy else 0uy
                (storage :> ICudaStorage).ByteData.CopyToDevice(byteVal, SizeT (index * sizeof<bool>))
            else
                storage.Data.CopyToDevice(value, SizeT (index * sizeof<'T>))

        /// a new ArrayND stored on the GPU using newly allocated device memory
        new (layout: ArrayNDLayoutT) =
            let elems = ArrayNDLayout.nElems layout
            ArrayNDCudaT<'T> (layout, new CudaStorageT<'T> (elems))

        /// storage
        member this.Storage = storage

        override this.Location = LocDev

        override this.Item
            with get pos = getElement (ArrayNDLayout.addr pos layout)
            and set pos value = 
                ArrayND.doCheckFinite value            
                setElement (ArrayNDLayout.addr pos layout) value 

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            ArrayNDCudaT<'T>(layout) :> ArrayNDT<'T>

        override this.NewOfType<'N> (layout: ArrayNDLayoutT) = 
            // drop constraint on 'N
            let aryType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|typeof<'N>|]
            Activator.CreateInstance (aryType, [|box layout|]) :?> ArrayNDT<'N>

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

        static member (====) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) ==== b :?> ArrayNDCudaT<bool>
        static member (<<<<) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) <<<< b :?> ArrayNDCudaT<bool>
        static member (>>>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) >>>> b :?> ArrayNDCudaT<bool>
        static member (<<>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) <<>> b :?> ArrayNDCudaT<bool>

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
            use srcMem = src.Pin()       
            dst.Storage.Data.CopyToDevice(srcMem.Ptr, SizeT(0), SizeT(0), 
                                          SizeT(sizeof<'T> * ArrayND.nElems src))

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
            use dstMem = dst.Pin()
            src.Storage.Data.CopyToHost(dstMem.Ptr, SizeT 0, SizeT 0, 
                                        SizeT (sizeof<'T> * ArrayND.nElems src))

        /// Copies this ArrayNDCudaT to the host
        member this.ToHost () =
            let dst = ArrayNDHost.newContiguous (ArrayND.shape this) 
            ArrayNDCudaT<_>.CopyIntoHost dst this
            dst

        interface IArrayNDCudaT with
            member this.ToHost () = this.ToHost () :> IArrayNDHostT
            member this.Storage = this.Storage :> ICudaStorage



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
    let newOfType typ (layout: ArrayNDLayoutT) = 
        let aryType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (aryType, [|box layout|]) :?> IArrayNDCudaT

    /// Creates a IArrayNDT for the given pointer, allocation size in bytes, type and layout.
    let fromPtrAndType (ptr: CUdeviceptr) typ (layout: ArrayNDLayoutT) = 
        let devVarType = typedefof<CudaDeviceVariable<_>>.MakeGenericType [|typ|]
        let devVar = Activator.CreateInstance (devVarType, [|box ptr|])

        let devStorType = typedefof<CudaStorageT<_>>.MakeGenericType [|typ|]
        let devStor = Activator.CreateInstance (devStorType, [|devVar|])

        let aryType = typedefof<ArrayNDCudaT<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (aryType, [|box layout; devStor|]) :?> IArrayNDCudaT


