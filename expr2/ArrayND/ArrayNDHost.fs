namespace ArrayNDNS

open System
open System.Runtime.InteropServices

open Util
open ArrayND

module ArrayNDHost = 

    /// pinned memory (from .NET or other source)          
    type IPinnedMemory =
        inherit IDisposable
        abstract Ptr: IntPtr with get

    /// host storage for an NDArray
    type IHostStorage<'T> = 
        abstract Item: int -> 'T with get, set
        abstract Pin: unit -> IPinnedMemory

    /// pinned .NET managed memory
    type PinnedManagedMemoryT (gcHnd: GCHandle) =       
        interface IPinnedMemory with
            member this.Ptr = gcHnd.AddrOfPinnedObject()
            member this.Dispose() = gcHnd.Free()

    // NDArray storage in a managed .NET array
    type ManagedArrayStorageT<'T> (data: 'T[]) =
        new (size: int) = ManagedArrayStorageT<'T>(Array.zeroCreate size)
        interface IHostStorage<'T> with
            member this.Item 
                with get(index) = data.[index]
                and set index value = data.[index] <- value
            member this.Pin () =
                let gcHnd = GCHandle.Alloc (data, GCHandleType.Pinned)
                new PinnedManagedMemoryT (gcHnd) :> IPinnedMemory                

    /// an N-dimensional array with reshape and subview abilities stored in host memory
    type ArrayNDHostT<'T> (layout: ArrayNDLayoutT, storage: IHostStorage<'T>) = 
        inherit ArrayNDT<'T>(layout)
        
        /// a new ArrayND in host memory using a managed array as storage
        new (layout: ArrayNDLayoutT) =
            ArrayNDHostT<'T>(layout, ManagedArrayStorageT<'T>(ArrayNDLayout.nElems layout))

        /// storage
        member this.Storage = storage

        override this.Item
            with get pos = storage.[ArrayNDLayout.addr pos layout]
            and set pos value = 
                ArrayND.doCheckFinite value
                storage.[ArrayNDLayout.addr pos layout] <- value 

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            ArrayNDHostT<'T>(layout) :> ArrayNDT<'T>

        override this.NewView (layout: ArrayNDLayoutT) = 
            ArrayNDHostT<'T>(layout, storage) :> ArrayNDT<'T>

    /// creates a new contiguous (row-major) ArrayNDHostT in host memory of the given shape 
    let inline newContiguous<'T> shp =
        ArrayNDHostT<'T>(ArrayNDLayout.newContiguous shp) :> ArrayNDT<'T>

    /// creates a new Fortran (column-major) ArrayNDHostT in host memory of the given shape
    let inline newColumnMajor<'T> shp =
        ArrayNDHostT<'T>(ArrayNDLayout.newColumnMajor shp) :> ArrayNDT<'T>

    /// ArrayNDHostT with zero dimensions (scalar) and given value
    let inline scalar value =
        let a = newContiguous [] 
        ArrayND.set [] value a
        a

    /// ArrayNDHostT of given shape filled with zeros.
    let inline zeros shape =
        newContiguous shape

    /// ArrayNDHostT of given shape filled with ones.
    let inline ones shape =
        let a = newContiguous shape
        ArrayND.fillWithOnes a
        a

    /// ArrayNDHostT identity matrix
    let inline identity size =
        let a = zeros [size; size]
        ArrayND.fillDiagonalWithOnes a
        a

        
