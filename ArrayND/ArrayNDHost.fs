namespace ArrayNDNS

open System
open System.Runtime.InteropServices

open Basics
open ArrayND


[<AutoOpen>]
module ArrayNDHostTypes = 

    /// pinned .NET managed memory
    type PinnedMemoryT (gcHnd: GCHandle) =       
        /// pointer to storage array 
        member this.Ptr = gcHnd.AddrOfPinnedObject()

        interface IDisposable with
            member this.Dispose() = gcHnd.Free()


    type IArrayNDHostT =
        inherit IArrayNDT
        abstract Pin: unit -> PinnedMemoryT
        abstract DataObj: obj
        abstract DataSizeInBytes: int


    /// an N-dimensional array with reshape and subview abilities stored in host memory
    type ArrayNDHostT<'T> (layout:      ArrayNDLayoutT, 
                           data:        'T []) = 
        inherit ArrayNDT<'T>(layout)
        
        /// a new ArrayND in host memory using a managed array as storage
        new (layout: ArrayNDLayoutT) =
            ArrayNDHostT<'T>(layout, Array.zeroCreate (ArrayNDLayout.nElems layout))

        /// underlying data array
        member this.Data = data

        /// optimized layout operations
        member this.FastLayout = FastLayout.ofLayout layout

        /// pins the underlying data array and returns the corresponding GCHandle
        member this.Pin () =
            let gcHnd = GCHandle.Alloc (data, GCHandleType.Pinned)
            new PinnedMemoryT (gcHnd) 

        /// size of underlying data array in bytes
        member this.DataSizeInBytes = data.Length * sizeof<'T>

        interface IArrayNDHostT with
            member this.Pin () = this.Pin ()
            member this.DataObj = box data
            member this.DataSizeInBytes = this.DataSizeInBytes

        override this.Location = LocHost

        override this.Item
            with get pos = data.[ArrayNDLayout.addr pos layout]
            and set pos value = 
                ArrayND.doCheckFinite value
                data.[ArrayNDLayout.addr pos layout] <- value 

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            ArrayNDHostT<'T>(layout) :> ArrayNDT<'T>

        override this.NewOfType<'N> (layout: ArrayNDLayoutT) =            
            ArrayNDHostT<'N>(layout) :> ArrayNDT<'N>

        override this.NewView (layout: ArrayNDLayoutT) = 
            ArrayNDHostT<'T>(layout, data) :> ArrayNDT<'T>

        override this.CopyTo (dest: ArrayNDT<'T>) =
            ArrayNDT<'T>.CheckSameShape this dest
            match dest with
            | :? ArrayNDHostT<'T> as dest ->

                if ArrayND.hasContiguousMemory this && ArrayND.hasContiguousMemory dest &&
                        ArrayND.stride this = ArrayND.stride dest then
                    // use array block copy
                    let nElems = ArrayNDLayout.nElems this.Layout
                    Array.Copy (this.Data, this.Layout.Offset, dest.Data, dest.Layout.Offset, nElems)
                else
                    // copy element by element
                    let destData = dest.Data
                    let destAddrs = FastLayout.allAddr dest.FastLayout
                    let thisAddrs = FastLayout.allAddr this.FastLayout
                    for destAddr, thisAddr in Seq.zip destAddrs thisAddrs do
                        destData.[destAddr] <- data.[thisAddr]

            | _ -> base.CopyTo dest
                              
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

        static member (====) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) ==== b :?> ArrayNDHostT<bool>
        static member (<<<<) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) <<<< b :?> ArrayNDHostT<bool>
        static member (>>>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) >>>> b :?> ArrayNDHostT<bool>
        static member (<<>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) <<>> b :?> ArrayNDHostT<bool>

module ArrayNDHost = 

    /// Creates a ArrayNDT of given type and layout in host memory.
    let newOfType typ (layout: ArrayNDLayoutT) = 
        let gt = typedefof<ArrayNDHostT<_>>
        let t = gt.MakeGenericType [|typ|]
        Activator.CreateInstance (t, [|box layout|]) :?> IArrayNDHostT

    /// creates a new contiguous (row-major) ArrayNDHostT in host memory of the given shape 
    let inline newContiguous<'T> shp =
        ArrayNDHostT<'T>(ArrayNDLayout.newContiguous shp) 

    /// creates a new contiguous (row-major) ArrayNDHostT in host memory of the given type and shape 
    let inline newContiguousOfType typ shp =
        newOfType typ (ArrayNDLayout.newContiguous shp)

    /// creates a new Fortran (column-major) ArrayNDHostT in host memory of the given shape
    let inline newColumnMajor<'T> shp =
        ArrayNDHostT<'T>(ArrayNDLayout.newColumnMajor shp) 

    /// creates a new Fortran (column-major) ArrayNDHostT in host memory of the given type and shape
    let inline newColumnMajorOfType typ shp =
        newOfType typ (ArrayNDLayout.newColumnMajor shp)

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

    /// Creates an ArrayNDT using the specified data and shape with contiguous (row major) layout.
    /// The data is referenced, not copied.
    let ofArray (data: 'T []) shp =
        let layout = ArrayNDLayout.newContiguous shp
        if ArrayNDLayout.nElems layout <> Array.length data then
            failwithf "specified shape %A has %d elements, but passed data array has %d elements"
                shp (ArrayNDLayout.nElems layout) (Array.length data)
        ArrayNDHostT<'T> (layout, data) 
        


