namespace ArrayNDNS

open System
open System.Runtime.InteropServices
open System.Collections.Generic

open Basics
open ArrayND
open MKL


[<AutoOpen>]
module ArrayNDHostTypes = 

    /// pinned .NET managed memory
    type PinnedMemoryT (gcHnd: GCHandle, size: int64) =       
        /// pointer to storage array 
        member this.Ptr = gcHnd.AddrOfPinnedObject()

        /// size of storage array in bytes
        member this.Size = size

        interface IDisposable with
            member this.Dispose() = gcHnd.Free()

    /// Information for calling BLAS/LAPACK routines.
    type private BlasInfo (memory: PinnedMemoryT,
                           offset: nativeint,
                           rows:   int,
                           cols:   int,
                           ld:     int) =

        member this.Ptr  : nativeint  = memory.Ptr + offset
        member this.Rows : lapack_int = int64 rows
        member this.Cols : lapack_int = int64 cols
        member this.Ld   : lapack_int = int64 ld

        interface IDisposable with
            member this.Dispose() = (memory :> IDisposable).Dispose()

    /// Call BLAS/LAPACK function depending on data type.
    let private blasTypeChoose<'T, 'R> (singleFn: unit -> 'R) (doubleFn: unit -> 'R) : 'R =
        match typeof<'T> with
        | t when t = typeof<single> -> singleFn () 
        | t when t = typeof<double> -> doubleFn () 
        | t -> failwithf "unsupported data type for BLAS operation: %A" t

    /// type-neutral interface to ArrayNDHostT<'T>
    type IArrayNDHostT =
        inherit IArrayNDT
        abstract Pin: unit -> PinnedMemoryT
        abstract DataObj: obj
        abstract DataSizeInBytes: int

    /// an ArrayNDT that can be copied to an ArrayNDHostT
    type IToArrayNDHostT<'T> =
        abstract ToHost: unit -> ArrayNDHostT<'T>

    /// an N-dimensional array with reshape and subview abilities stored in host memory
    and ArrayNDHostT<'T> (layout:      ArrayNDLayoutT, 
                           data:        'T []) = 
        inherit ArrayNDT<'T>(layout)
        
        let fastLayout = FastLayout.ofLayout layout

        /// a new ArrayND in host memory using a managed array as storage
        new (layout: ArrayNDLayoutT) =
            ArrayNDHostT<'T>(layout, Array.zeroCreate (ArrayNDLayout.nElems layout))

        /// underlying data array
        member this.Data = data

        /// optimized layout operations
        member this.FastLayout = fastLayout

        /// pins the underlying data array and returns the corresponding GCHandle
        member this.Pin () =
            let gcHnd = GCHandle.Alloc (data, GCHandleType.Pinned)
            new PinnedMemoryT (gcHnd, data.LongLength * int64 sizeof<'T>) 

        /// size of underlying data array in bytes
        member this.DataSizeInBytes = data.Length * sizeof<'T>

        interface IArrayNDHostT with
            member this.Pin () = this.Pin ()
            member this.DataObj = box data
            member this.DataSizeInBytes = this.DataSizeInBytes

        interface IToArrayNDHostT<'T> with
            member this.ToHost () = this

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

        override this.MapImpl (f: 'T -> 'R) (dest: ArrayNDT<'R>) =
            let dest = dest :?> ArrayNDHostT<'R>
            let destData = dest.Data
            let destAddrs = FastLayout.allAddr dest.FastLayout
            let thisAddrs = FastLayout.allAddr this.FastLayout
            for destAddr, thisAddr in Seq.zip destAddrs thisAddrs do
                destData.[destAddr] <- f data.[thisAddr]

        override this.MapInplaceImpl (f: 'T -> 'T) = 
            let thisAddrs = FastLayout.allAddr this.FastLayout
            for thisAddr in thisAddrs do
                data.[thisAddr] <- f data.[thisAddr]

        override this.Map2Impl (f: 'T -> 'T -> 'R) (other: ArrayNDT<'T>) (dest: ArrayNDT<'R>) =
            let dest = dest :?> ArrayNDHostT<'R>
            let other = other :?> ArrayNDHostT<'T>
            let destData = dest.Data
            let otherData = other.Data
            let destAddrs = FastLayout.allAddr dest.FastLayout
            let thisAddrs = FastLayout.allAddr this.FastLayout
            let otherAddrs = FastLayout.allAddr other.FastLayout
            for destAddr, thisAddr, otherAddr in Seq.zip3 destAddrs thisAddrs otherAddrs do
                destData.[destAddr] <- f data.[thisAddr] otherData.[otherAddr]

        override this.IfThenElseImpl (cond: ArrayNDT<bool>) (elseVal: ArrayNDT<'T>) (dest: ArrayNDT<'T>) =
            let cond = cond :?> ArrayNDHostT<bool>
            let elseVal = elseVal :?> ArrayNDHostT<'T>
            let dest = dest :?> ArrayNDHostT<'T>                
            let condData = cond.Data
            let ifValData = this.Data
            let elseValData = elseVal.Data
            let destData = dest.Data
            let condAddrs = FastLayout.allAddr cond.FastLayout
            let ifValAddrs = FastLayout.allAddr this.FastLayout
            let elseValAddrs = FastLayout.allAddr elseVal.FastLayout
            let destAddrs = FastLayout.allAddr dest.FastLayout
            for destAddr, condAddr, (ifValAddr, elseValAddr) in 
                    Seq.zip3 destAddrs condAddrs (Seq.zip ifValAddrs elseValAddrs) do
                destData.[destAddr] <- 
                    if condData.[condAddr] then ifValData.[ifValAddr] else elseValData.[elseValAddr]

        interface IEnumerable<'T> with
            member this.GetEnumerator() =
                FastLayout.allAddr this.FastLayout
                |> Seq.map (fun addr -> this.Data.[addr])
                |> fun s -> s.GetEnumerator()
            member this.GetEnumerator() =
                (this :> IEnumerable<'T>).GetEnumerator() :> System.Collections.IEnumerator
                              
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
        static member (<<==) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) <<== b :?> ArrayNDHostT<bool>
        static member (>>>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) >>>> b :?> ArrayNDHostT<bool>
        static member (>>==) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) >>== b :?> ArrayNDHostT<bool>
        static member (<<>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> ArrayNDT<'T>) <<>> b :?> ArrayNDHostT<bool>

        /// Returns a BlasInfo that exposes the transpose of this matrix to BLAS
        /// (in column-major order).
        member private this.GetTransposedBlas copyAllowed =
            if this.NDims <> 2 then failwithf "require a matrix but got shape %A" this.Shape
            if not (this.Shape.[0] > 0 && this.Shape.[1] > 0) then 
                failwithf "require a non-empty matrix but got shape %A" this.Shape
            let str = stride this
            if str.[0] >= 1 && str.[0] >= this.Shape.[1] && str.[1] = 1 then
                new BlasInfo (this.Pin(), nativeint (this.Layout.Offset * sizeof<'T>),
                              this.Shape.[1], this.Shape.[0], str.[0])
            else
                if copyAllowed then (ArrayND.copy this).GetTransposedBlas copyAllowed
                else failwith "ArrayNDHost incompatible with BLAS but copying not allowed"
               
        /// Computes the matrix inverse.    
        override this.Invert () =
            let nd = this.NDims
            if nd < 2 then 
                failwithf "require at least a two-dimensional tensor \
                           for matrix inversion but got %A" this.Shape
            if this.Shape.[nd-2] <> this.Shape.[nd-1] then
                failwithf "cannot invert non-square matrix of shape %A" this.Shape
            let batchShp = this.Shape.[0 .. nd-3]

            let inv = ArrayND.copy this

            // iterate over all batch dimensions
            for batchIdx in ArrayNDLayout.allIdxOfShape batchShp do
                let batchRng = batchIdx |> List.map RngElem
                let rng = batchRng @ [RngAll; RngAll]                  
                let aAry = inv.[rng]

                // compute LU factorization
                use a = aAry.GetTransposedBlas false
                let ipiv : lapack_int[] = Array.zeroCreate aAry.Shape.[0]
                let info =
                    blasTypeChoose<'T, lapack_int> 
                        (fun () -> LAPACKE_sgetrf (LAPACK_COL_MAJOR, a.Rows, a.Cols, a.Ptr, a.Ld, ipiv))
                        (fun () -> LAPACKE_dgetrf (LAPACK_COL_MAJOR, a.Rows, a.Cols, a.Ptr, a.Ld, ipiv))
                if info < 0L then failwithf "LAPACK argument error %d" info
                if info > 0L then raise (SingularMatrixError "cannot invert singular matrix")

                // compute matrix inverse
                let info =
                    blasTypeChoose<'T, lapack_int>
                        (fun () -> LAPACKE_sgetri (LAPACK_COL_MAJOR, a.Rows, a.Ptr, a.Ld, ipiv))
                        (fun () -> LAPACKE_dgetri (LAPACK_COL_MAJOR, a.Rows, a.Ptr, a.Ld, ipiv))
                if info < 0L then failwithf "LAPACK argument error %d" info
                if info > 0L then raise (SingularMatrixError "cannot invert singular matrix")

            inv :> ArrayNDT<'T>



module ArrayNDHost = 

    /// Creates a ArrayNDT of given type and layout in host memory.
    let newOfType typ (layout: ArrayNDLayoutT) = 
        let gt = typedefof<ArrayNDHostT<_>>
        let t = gt.MakeGenericType [|typ|]
        Activator.CreateInstance (t, [|box layout|]) :?> IArrayNDHostT

    /// creates a new contiguous (row-major) ArrayNDHostT in host memory of the given shape 
    let newC<'T> shp =
        ArrayNDHostT<'T>(ArrayNDLayout.newC shp) 

    /// creates a new contiguous (row-major) ArrayNDHostT in host memory of the given type and shape 
    let newCOfType typ shp =
        newOfType typ (ArrayNDLayout.newC shp)

    /// creates a new Fortran (column-major) ArrayNDHostT in host memory of the given shape
    let newF<'T> shp =
        ArrayNDHostT<'T>(ArrayNDLayout.newF shp) 

    /// creates a new Fortran (column-major) ArrayNDHostT in host memory of the given type and shape
    let newFOfType typ shp =
        newOfType typ (ArrayNDLayout.newF shp)

    /// ArrayNDHostT with zero dimensions (scalar) and given value
    let scalar value =
        let a = newC [] 
        ArrayND.set [] value a
        a

    /// ArrayNDHostT of given shape filled with zeros.
    let zeros<'T> shape : ArrayNDHostT<'T> =
        newC shape

    /// ArrayNDHostT of given shape filled with ones.
    let ones<'T> shape : ArrayNDHostT<'T> =
        let a = newC shape
        ArrayND.fillWithOnes a
        a

    /// ArrayNDHostT of given shape filled with the given value.
    let filled shape (value: 'T) : ArrayNDHostT<'T> =
        let a = newC shape
        a |> ArrayND.fillConst value
        a       

    /// ArrayNDHostT identity matrix
    let identity<'T> size : ArrayNDHostT<'T> =
        let a = zeros [size; size]
        ArrayND.fillDiagonalWithOnes a
        a

    /// Creates a new ArrayNDHostT of the given shape and uses the given function to initialize it.
    let init<'T> shp (f: unit -> 'T) =
        let a = newC<'T> shp
        ArrayND.fill f a
        a

    /// Creates a new ArrayNDHostT of the given shape and uses the given function to initialize it.
    let initIndexed<'T> shp (f: int list -> 'T) =
        let a = newC<'T> shp
        ArrayND.fillIndexed f a
        a   

    /// Creates a new vector with linearly spaced values from start to (including) stop.
    let inline linSpaced (start: 'T) (stop: 'T) (nElems: int) =
        let a = newC<'T> [nElems]
        ArrayND.fillLinSpaced start stop a
        a          

    /// If the specified tensor is on a device, copies it to the host and returns the copy.
    /// If the tensor is already on the host, this does nothing.
    let fetch (a: #ArrayNDT<'T>) : ArrayNDHostT<'T> =
        match box a with
        | :? ArrayNDHostT<'T> as a -> a
        | :? IToArrayNDHostT<'T> as a -> a.ToHost ()
        | _ -> failwithf "the type %A is not copyable to the host" (a.GetType())

    /// Creates a one-dimensional ArrayNDT using the specified data.
    /// The data is referenced, not copied.
    let ofArray (data: 'T []) =
        let shp = [Array.length data]
        let layout = ArrayNDLayout.newC shp
        ArrayNDHostT<'T> (layout, data) 

    /// Creates a two-dimensional ArrayNDT using the specified data. 
    /// The data is copied.
    let ofArray2D (data: 'T [,]) =
        let shp = [Array2D.length1 data; Array2D.length2 data]
        initIndexed shp (fun idx -> data.[idx.[0], idx.[1]])

    /// Creates a three-dimensional ArrayNDT using the specified data. 
    /// The data is copied.
    let ofArray3D (data: 'T [,,]) =
        let shp = [Array3D.length1 data; Array3D.length2 data; Array3D.length3 data]
        initIndexed shp (fun idx -> data.[idx.[0], idx.[1], idx.[2]])

    /// Creates a four-dimensional ArrayNDT using the specified data. 
    /// The data is copied.
    let ofArray4D (data: 'T [,,,]) =
        let shp = [Array4D.length1 data; Array4D.length2 data; Array4D.length3 data; Array4D.length4 data]
        initIndexed shp (fun idx -> data.[idx.[0], idx.[1], idx.[2], idx.[3]])

    /// Creates a one-dimensional ArrayNDT using the specified sequence.       
    let ofSeq (data: 'T seq) =
        data |> Array.ofSeq |> ofArray

    /// Creates a one-dimensional ArrayNDT using the specified sequence and shape.       
    let ofSeqWithShape (shape: int list) (data: 'T seq) =
        let nElems = shape |> List.fold (*) 1
        data |> Seq.take nElems |> ofSeq |> ArrayND.reshape shape

    /// Creates a one-dimensional ArrayNDT using the specified list.       
    let ofList (data: 'T list) =
        data |> Array.ofList |> ofArray

    /// Creates a two-dimensional ArrayNDT using the specified list of lists.       
    let ofList2D (data: 'T list list) =
        data |> array2D |> ofArray2D

    /// Creates an Array from the data in this ArrayNDT. The data is copied.
    let toArray (ary: ArrayNDHostT<_>) =
        if ArrayND.nDims ary <> 1 then failwith "ArrayNDT must have 1 dimension"
        let shp = ArrayND.shape ary
        Array.init shp.[0] (fun i0 -> ary.[[i0]])

    /// Creates an Array2D from the data in this ArrayNDT. The data is copied.
    let toArray2D (ary: ArrayNDHostT<_>) =
        if ArrayND.nDims ary <> 2 then failwith "ArrayNDT must have 2 dimensions"
        let shp = ArrayND.shape ary
        Array2D.init shp.[0] shp.[1] (fun i0 i1 -> ary.[[i0; i1]])

    /// Creates an Array3D from the data in this ArrayNDT. The data is copied.
    let toArray3D (ary: ArrayNDHostT<_>) =
        if ArrayND.nDims ary <> 3 then failwith "ArrayNDT must have 3 dimensions"
        let shp = ArrayND.shape ary
        Array3D.init shp.[0] shp.[1] shp.[2] (fun i0 i1 i2 -> ary.[[i0; i1; i2]])
       
    /// Creates an Array4D from the data in this ArrayNDT. The data is copied.
    let toArray4D (ary: ArrayNDHostT<_>) =
        if ArrayND.nDims ary <> 4 then failwith "ArrayNDT must have 4 dimensions"
        let shp = ArrayND.shape ary
        Array4D.init shp.[0] shp.[1] shp.[2] shp.[3] (fun i0 i1 i2 i3 -> ary.[[i0; i1; i2; i3]])

    /// Creates a list from the data in this ArrayNDT. The data is copied.
    let toList (ary: ArrayNDHostT<_>) =
        ary |> toArray |> Array.toList
