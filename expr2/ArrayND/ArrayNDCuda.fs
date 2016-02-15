namespace ArrayNDNS

open ManagedCuda
open ManagedCuda.BasicTypes

open Util
open ArrayND
open ArrayNDHost


module ArrayNDCuda =

    type IDeviceStorage<'T> =
        abstract Item: int -> 'T with get, set

    /// Storage in a CudaDeviceVariable. 
    /// The underlying CudaDeviceVariable is disposed when this object is finalized.
    type CudaDeviceVariableStorageT<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                                                               (data: CudaDeviceVariable<'T>) =

        new (size: int) = CudaDeviceVariableStorageT<'T> (new CudaDeviceVariable<'T> (SizeT(size)))
        interface IDeviceStorage<'T> with
            member this.Item 
                with get(index) = 
                    let hostBuf = ref (new 'T())
                    data.CopyToHost(hostBuf, SizeT(index * sizeof<'T>))
                    !hostBuf
                and set index (value: 'T) = 
                    data.CopyToDevice(value, SizeT(index * sizeof<'T>))

        override this.Finalize() =
            data.Dispose()

        member this.CudaDeviceVariabe = data


    /// an N-dimensional array with reshape and subview abilities stored in GPU device memory
    type ArrayNDCudaT<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> 
                                    (layout: ArrayNDLayoutT, storage: IDeviceStorage<'T>) = 
        inherit ArrayNDT<'T>(layout)
        
        /// a new ArrayND stored on the GPU using newly allocated device memory
        new (layout: ArrayNDLayoutT) =
            ArrayNDCudaT<'T> (layout, CudaDeviceVariableStorageT<'T>(ArrayNDLayout.nElems layout))

        /// storage
        member inline this.Storage = storage

        override this.Item
            with get pos = storage.[ArrayNDLayout.addr pos layout]
            and set pos value = 
                ArrayND.doCheckFinite value            
                storage.[ArrayNDLayout.addr pos layout] <- value 

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            ArrayNDCudaT<'T>(layout) :> ArrayNDT<'T>

        override this.NewView (layout: ArrayNDLayoutT) = 
            ArrayNDCudaT<'T>(layout, storage) :> ArrayNDT<'T>


    /// creates a new contiguous (row-major) ArrayNDCudaT in device memory of the given shape 
    let inline newContiguous<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> shp =
        ArrayNDCudaT<'T>(ArrayNDLayout.newContiguous shp)

    /// creates a new Fortran (column-major) ArrayNDCudaT in device memory of the given shape
    let inline newColumnMajor<'T when 'T: (new: unit -> 'T) and 'T: struct and 'T :> System.ValueType> shp =
        ArrayNDCudaT<'T>(ArrayNDLayout.newColumnMajor shp)

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

    /// makes a contiguous copy of ary if it is not contiguous and with zero offset
    let inline ensureContiguousAndOffsetFree ary = 
        if ArrayND.isContiguous ary && ArrayND.offset ary = 0 then ary
        else ArrayND.copy ary 

    /// Copies a ArrayNDHostT into the specified ArrayNDCudaT.
    /// Both must of same shape. dst must also be contiguous and with offset zero.
    let copyIntoDev (dst: ArrayNDCudaT<'T>) (src: ArrayNDHostT<'T>) =
        if ArrayND.shape dst <> ArrayND.shape src then
            invalidArg "dst" "dst and src must be of same shape"
        if not (ArrayND.isContiguous dst && ArrayND.offset dst = 0) then
            invalidArg "dst" "dst must be contiguous without offset"

        let src = ensureContiguousAndOffsetFree src :?> ArrayNDHostT<'T>
        use srcMem = src.Storage.Pin()
        
        match dst.Storage with
        | :? CudaDeviceVariableStorageT<'T> as ds ->
            ds.CudaDeviceVariabe.CopyToDevice(srcMem.Ptr, SizeT(0), SizeT(0), 
                                              SizeT(sizeof<'T> * ArrayND.nElems src))
        | _ -> failwith "cannot copy to that device storage"

    /// Copies a ArrayNDHostT to the device
    let toDev (src: ArrayNDHostT<'T>) =
        let dst = newContiguous<'T> (shape src)
        copyIntoDev dst src
        dst

    /// Copies a ArrayNDCudaT into the specified ArrayNDHostT.
    /// Both must of same shape. dst must also be contiguous and with offset zero.
    let copyIntoHost (dst: ArrayNDHostT<'T>) (src: ArrayNDCudaT<'T>) =
        if ArrayND.shape dst <> ArrayND.shape src then
            invalidArg "dst" "dst and src must be of same shape"
        if not (ArrayND.isContiguous src && ArrayND.offset src = 0) then
            invalidArg "src" "src must be contiguous without offset"

        let src = ensureContiguousAndOffsetFree src :?> ArrayNDCudaT<'T>
        use dstMem = dst.Storage.Pin()
        
        match src.Storage with
        | :? CudaDeviceVariableStorageT<'T> as ds ->
            ds.CudaDeviceVariabe.CopyToHost(dstMem.Ptr, SizeT(0), SizeT(0), 
                                            SizeT(sizeof<'T> * ArrayND.nElems src))
        | _ -> failwith "cannot copy from that device storage"

    /// Copies a ArrayNDCudaT to the host
    let toHost (src: ArrayNDCudaT<'T>) =
        let dst = ArrayNDHost.newContiguous<'T> (shape src)
        copyIntoHost dst src
        dst

