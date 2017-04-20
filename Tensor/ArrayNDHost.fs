namespace ArrayNDNS

open System
open System.Runtime.InteropServices
open System.Collections.Generic

open Basics
open Tensor
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
                           rows:   int64,
                           cols:   int64,
                           ld:     int64) =

        member this.Ptr  : nativeint  = memory.Ptr + offset
        member this.Rows : lapack_int = rows
        member this.Cols : lapack_int = cols
        member this.Ld   : lapack_int = ld

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
        inherit ITensor
        abstract Pin: unit -> PinnedMemoryT
        abstract DataObj: obj
        abstract DataSizeInBytes: int64

    /// an ArrayNDT that can be copied to an ArrayNDHostT
    type IToArrayNDHostT<'T> =
        abstract ToHost: unit -> ArrayNDHostT<'T>

    /// an N-dimensional array with reshape and subview abilities stored in host memory
    and ArrayNDHostT<'T> (layout:      TensorLayout, 
                          data:        'T []) = 
        inherit Tensor<'T>(layout)
        
        let fastLayout = FastLayout.ofLayout layout

        /// a new ArrayND in host memory using a managed array as storage
        new (layout: TensorLayout) =
            let nElems = TensorLayout.nElems layout
            if nElems > int64 Microsoft.FSharp.Core.int32.MaxValue then
                failwithf "The current ArrayNDHostT implementation is limited to %d elements."
                          Microsoft.FSharp.Core.int32.MaxValue
            ArrayNDHostT<'T>(layout, Array.zeroCreate (int32 nElems))

        /// underlying data array
        member this.Data = data

        /// optimized layout operations
        member this.FastLayout = fastLayout

        /// pins the underlying data array and returns the corresponding GCHandle
        member this.Pin () =
            let gcHnd = GCHandle.Alloc (data, GCHandleType.Pinned)
            new PinnedMemoryT (gcHnd, data.LongLength * sizeof64<'T>) 

        /// size of underlying data array in bytes
        member this.DataSizeInBytes = int64 data.Length * sizeof64<'T>

        interface IArrayNDHostT with
            member this.Pin () = this.Pin ()
            member this.DataObj = box data
            member this.DataSizeInBytes = this.DataSizeInBytes

        interface IToArrayNDHostT<'T> with
            member this.ToHost () = this

        override this.Location = LocHost

        override this.Item
            with get pos = data.[int32 (TensorLayout.addr pos layout)]
            and set pos value = 
                Tensor.doCheckFinite value
                data.[int32 (TensorLayout.addr pos layout)] <- value 

        override this.NewOfSameType (layout: TensorLayout) = 
            ArrayNDHostT<'T>(layout) :> Tensor<'T>

        override this.NewOfType<'N> (layout: TensorLayout) =            
            ArrayNDHostT<'N>(layout) :> Tensor<'N>

        override this.NewView (layout: TensorLayout) = 
            ArrayNDHostT<'T>(layout, data) :> Tensor<'T>

        override this.CopyTo (dest: Tensor<'T>) =
            Tensor<'T>.CheckSameShape this dest
            match dest with
            | :? ArrayNDHostT<'T> as dest ->
                if Tensor.hasContiguousMemory this && Tensor.hasContiguousMemory dest &&
                        Tensor.stride this = Tensor.stride dest then
                    // use array block copy
                    let nElems = TensorLayout.nElems this.Layout
                    Array.Copy (this.Data, this.Layout.Offset, dest.Data, dest.Layout.Offset, nElems)
                else
                    // copy element by element
                    let destData = dest.Data
                    let destAddrs = FastLayout.allAddr dest.FastLayout
                    let thisAddrs = FastLayout.allAddr this.FastLayout
                    for destAddr, thisAddr in Seq.zip destAddrs thisAddrs do
                        destData.[int32 destAddr] <- data.[int32 thisAddr]

            | _ -> base.CopyTo dest

        override this.MapImpl (f: 'T -> 'R) (dest: Tensor<'R>) =
            let dest = dest :?> ArrayNDHostT<'R>
            let destData = dest.Data
            let destAddrs = FastLayout.allAddr dest.FastLayout
            let thisAddrs = FastLayout.allAddr this.FastLayout
            for destAddr, thisAddr in Seq.zip destAddrs thisAddrs do
                destData.[int32 destAddr] <- f data.[int32 thisAddr]

        override this.MapInplaceImpl (f: 'T -> 'T) = 
            let thisAddrs = FastLayout.allAddr this.FastLayout
            for thisAddr in thisAddrs do
                data.[int32 thisAddr] <- f data.[int32 thisAddr]

        override this.Map2Impl (f: 'T -> 'T -> 'R) (other: Tensor<'T>) (dest: Tensor<'R>) =
            let dest = dest :?> ArrayNDHostT<'R>
            let other = other :?> ArrayNDHostT<'T>
            let destData = dest.Data
            let otherData = other.Data
            let destAddrs = FastLayout.allAddr dest.FastLayout
            let thisAddrs = FastLayout.allAddr this.FastLayout
            let otherAddrs = FastLayout.allAddr other.FastLayout
            for destAddr, thisAddr, otherAddr in Seq.zip3 destAddrs thisAddrs otherAddrs do
                destData.[int32 destAddr] <- f data.[int32 thisAddr] otherData.[int32 otherAddr]

        override this.IfThenElseImpl (cond: Tensor<bool>) (elseVal: Tensor<'T>) (dest: Tensor<'T>) =
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
                destData.[int32 destAddr] <- 
                    if condData.[int32 condAddr] then ifValData.[int32 ifValAddr] else elseValData.[int32 elseValAddr]

        interface IEnumerable<'T> with
            member this.GetEnumerator() =
                FastLayout.allAddr this.FastLayout
                |> Seq.map (fun addr -> this.Data.[int32 addr])
                |> fun s -> s.GetEnumerator()
            member this.GetEnumerator() =
                (this :> IEnumerable<'T>).GetEnumerator() :> System.Collections.IEnumerator
                              
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

        static member (====) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) ==== b :?> ArrayNDHostT<bool>
        static member (<<<<) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) <<<< b :?> ArrayNDHostT<bool>
        static member (<<==) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) <<== b :?> ArrayNDHostT<bool>
        static member (>>>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) >>>> b :?> ArrayNDHostT<bool>
        static member (>>==) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) >>== b :?> ArrayNDHostT<bool>
        static member (<<>>) (a: ArrayNDHostT<'T>, b: ArrayNDHostT<'T>) = (a :> Tensor<'T>) <<>> b :?> ArrayNDHostT<bool>

        /// Returns a BlasInfo that exposes the transpose of this matrix to BLAS
        /// (in column-major order).
        member private this.GetTransposedBlas copyAllowed =
            if this.NDims <> 2 then failwithf "require a matrix but got shape %A" this.Shape
            if not (this.Shape.[0] > 0L && this.Shape.[1] > 0L) then 
                failwithf "require a non-empty matrix but got shape %A" this.Shape
            let str = stride this
            if str.[0] >= 1L && str.[0] >= this.Shape.[1] && str.[1] = 1L then
                new BlasInfo (this.Pin(), nativeint (this.Layout.Offset * sizeof64<'T>),
                              this.Shape.[1], this.Shape.[0], str.[0])
            else
                if copyAllowed then (Tensor.copy this).GetTransposedBlas copyAllowed
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

            let inv = Tensor.copy this

            // iterate over all batch dimensions
            for batchIdx in TensorLayout.allIdxOfShape batchShp do
                let batchRng = batchIdx |> List.map RngElem
                let rng = batchRng @ [RngAll; RngAll]                  
                let aAry = inv.[rng]

                // compute LU factorization
                use a = aAry.GetTransposedBlas false
                let ipiv : lapack_int[] = Array.zeroCreate (int32 aAry.Shape.[0])
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

            inv :> Tensor<'T>

        override this.SymmetricEigenDecomposition () =
            let nd = this.NDims
            if nd <> 2 || this.Shape.[0] <> this.Shape.[1] then 
                failwithf "require a square matrix for symmetric eigen decomposition but got %A" this.Shape
            let size = this.Shape.[0]

            let eigVecs = Tensor.copy this
            let eigVals = this.NewOfSameType (TensorLayout.newC [1L; size]) :?> ArrayNDHostT<'T>

            use a = eigVecs.GetTransposedBlas false
            use w = eigVals.GetTransposedBlas false
            let info = 
                blasTypeChoose<'T, lapack_int>
                    (fun () -> LAPACKE_ssyevd (LAPACK_COL_MAJOR, 'V', 'L', a.Rows, a.Ptr, a.Ld, w.Ptr))
                    (fun () -> LAPACKE_dsyevd (LAPACK_COL_MAJOR, 'V', 'L', a.Rows, a.Ptr, a.Ld, w.Ptr))
            if info < 0L then failwithf "LAPACK argument error %d" info
            if info > 0L then raise (SingularMatrixError "cannot compute eigen decomposition of singular matrix")

            eigVals.[0L, *] :> Tensor<'T>, eigVecs.T 


module ArrayNDHost = 

    /// Creates a ArrayNDT of given type and layout in host memory.
    let newOfType typ (layout: TensorLayout) = 
        let gt = typedefof<ArrayNDHostT<_>>
        let t = gt.MakeGenericType [|typ|]
        Activator.CreateInstance (t, [|box layout|]) :?> IArrayNDHostT

    /// creates a new contiguous (row-major) ArrayNDHostT in host memory of the given shape 
    let newC<'T> shp =
        ArrayNDHostT<'T>(TensorLayout.newC shp) 

    /// creates a new contiguous (row-major) ArrayNDHostT in host memory of the given type and shape 
    let newCOfType typ shp =
        newOfType typ (TensorLayout.newC shp)

    /// creates a new Fortran (column-major) ArrayNDHostT in host memory of the given shape
    let newF<'T> shp =
        ArrayNDHostT<'T>(TensorLayout.newF shp) 

    /// creates a new Fortran (column-major) ArrayNDHostT in host memory of the given type and shape
    let newFOfType typ shp =
        newOfType typ (TensorLayout.newF shp)

    /// ArrayNDHostT with zero dimensions (scalar) and given value
    let scalar value =
        let a = newC [] 
        Tensor.set [] value a
        a

    /// ArrayNDHostT of given shape filled with zeros.
    let zeros<'T> shape : ArrayNDHostT<'T> =
        newC shape

    /// ArrayNDHostT of given shape filled with ones.
    let ones<'T> shape : ArrayNDHostT<'T> =
        let a = newC shape
        Tensor.fillWithOnes a
        a

    /// ArrayNDHostT of given shape filled with the given value.
    let filled shape (value: 'T) : ArrayNDHostT<'T> =
        let a = newC shape
        a |> Tensor.fillConst value
        a       

    /// ArrayNDHostT identity matrix
    let identity<'T> size : ArrayNDHostT<'T> =
        let a = zeros [size; size]
        Tensor.fillDiagonalWithOnes a
        a

    /// Creates a new ArrayNDHostT of the given shape and uses the given function to initialize it.
    let init<'T> shp (f: unit -> 'T) =
        let a = newC<'T> shp
        Tensor.fill f a
        a

    /// Creates a new ArrayNDHostT of the given shape and uses the given function to initialize it.
    let initIndexed<'T> shp f =
        let a = newC<'T> shp
        Tensor.fillIndexed f a
        a   

    /// Creates a new vector with linearly spaced values from start to (including) stop.
    let inline linSpaced (start: 'T) (stop: 'T) nElems =
        let a = newC<'T> [nElems]
        Tensor.fillLinSpaced start stop a
        a          

    /// If the specified tensor is on a device, copies it to the host and returns the copy.
    /// If the tensor is already on the host, this does nothing.
    let fetch (a: #Tensor<'T>) : ArrayNDHostT<'T> =
        match box a with
        | :? ArrayNDHostT<'T> as a -> a
        | :? IToArrayNDHostT<'T> as a -> a.ToHost ()
        | _ -> failwithf "the type %A is not copyable to the host" (a.GetType())

    /// converts the from one data type to another
    let convert (a: ArrayNDHostT<'T>) : ArrayNDHostT<'C> =
        a |> Tensor.convert :> Tensor<'C> :?> ArrayNDHostT<'C>

    /// Creates a one-dimensional ArrayNDT using the specified data.
    /// The data is referenced, not copied.
    let ofArray (data: 'T []) =
        let shp = [Array.length data]
        let shp = shp |> List.map int64
        let layout = TensorLayout.newC shp
        ArrayNDHostT<'T> (layout, data) 

    /// Creates a two-dimensional ArrayNDT using the specified data. 
    /// The data is copied.
    let ofArray2D (data: 'T [,]) =
        let shp = [Array2D.length1 data; Array2D.length2 data]
        let shp = shp |> List.map int64
        initIndexed shp (fun idx -> data.[int32 idx.[0], int32 idx.[1]])

    /// Creates a three-dimensional ArrayNDT using the specified data. 
    /// The data is copied.
    let ofArray3D (data: 'T [,,]) =
        let shp = [Array3D.length1 data; Array3D.length2 data; Array3D.length3 data]
        let shp = shp |> List.map int64
        initIndexed shp (fun idx -> data.[int32 idx.[0], int32 idx.[1], int32 idx.[2]])

    /// Creates a four-dimensional ArrayNDT using the specified data. 
    /// The data is copied.
    let ofArray4D (data: 'T [,,,]) =
        let shp = [Array4D.length1 data; Array4D.length2 data; 
                   Array4D.length3 data; Array4D.length4 data]
        let shp = shp |> List.map int64
        initIndexed shp (fun idx -> data.[int32 idx.[0], int32 idx.[1], int32 idx.[2], int32 idx.[3]])

    /// Creates a one-dimensional ArrayNDT using the specified sequence.       
    let ofSeq (data: 'T seq) =
        data |> Array.ofSeq |> ofArray

    /// Creates a one-dimensional ArrayNDT using the specified sequence and shape.       
    let ofSeqWithShape shape (data: 'T seq) =
        let nElems = shape |> List.fold (*) 1L
        data |> Seq.take (int32 nElems) |> ofSeq |> Tensor.reshape shape

    /// Creates a one-dimensional ArrayNDT using the specified list.       
    let ofList (data: 'T list) =
        data |> Array.ofList |> ofArray

    /// Creates a two-dimensional ArrayNDT using the specified list of lists.       
    let ofList2D (data: 'T list list) =
        data |> array2D |> ofArray2D

    /// Creates an Array from the data in this ArrayNDT. The data is copied.
    let toArray (ary: ArrayNDHostT<_>) =
        if Tensor.nDims ary <> 1 then failwith "ArrayNDT must have 1 dimension"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array.init shp.[0] (fun i0 -> ary.[[int64 i0]])

    /// Creates an Array2D from the data in this ArrayNDT. The data is copied.
    let toArray2D (ary: ArrayNDHostT<_>) =
        if Tensor.nDims ary <> 2 then failwith "ArrayNDT must have 2 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array2D.init shp.[0] shp.[1] (fun i0 i1 -> ary.[[int64 i0; int64 i1]])

    /// Creates an Array3D from the data in this ArrayNDT. The data is copied.
    let toArray3D (ary: ArrayNDHostT<_>) =
        if Tensor.nDims ary <> 3 then failwith "ArrayNDT must have 3 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array3D.init shp.[0] shp.[1] shp.[2] (fun i0 i1 i2 -> ary.[[int64 i0; int64 i1; int64 i2]])
       
    /// Creates an Array4D from the data in this ArrayNDT. The data is copied.
    let toArray4D (ary: ArrayNDHostT<_>) =
        if Tensor.nDims ary <> 4 then failwith "ArrayNDT must have 4 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array4D.init shp.[0] shp.[1] shp.[2] shp.[3] (fun i0 i1 i2 i3 -> ary.[[int64 i0; int64 i1; int64 i2; int64 i3]])

    /// Creates a list from the data in this ArrayNDT. The data is copied.
    let toList (ary: ArrayNDHostT<_>) =
        ary |> toArray |> Array.toList

    /// One-dimensional int tensor containing the numbers [0L; 1L; ...; size-1L].
    let arange size =
        {0L .. size-1L} |> ofSeq
