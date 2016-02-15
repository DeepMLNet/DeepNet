namespace ArrayNDNS

open Util
open ArrayND

module ArrayNDHost = 

    /// if true, then setting NaN or Inf causes and exception to be thrown.
    let CheckFinite = false

    // host storage for an NDArray
    type IHostStorage<'T> = 
        abstract Item: int -> 'T with get, set

    // NDArray storage in a managed .NET array
    type ManagedArrayStorageT<'T> (data: 'T[]) =
        new (size: int) = ManagedArrayStorageT<'T>(Array.zeroCreate size)
        interface IHostStorage<'T> with
            member this.Item 
                with get(index) = data.[index]
                and set index value = data.[index] <- value

    /// an N-dimensional array with reshape and subview abilities stored in host memory
    type ArrayNDHostT<'T> (layout: ArrayNDLayoutT, storage: IHostStorage<'T>) = 
        inherit ArrayNDT<'T>(layout)
        
        /// a new ArrayND in host memory using a managed array as storage
        new (layout: ArrayNDLayoutT) =
            ArrayNDHostT<'T>(layout, ManagedArrayStorageT<'T>(ArrayNDLayout.nElems layout))

        /// storage
        member inline this.Storage = storage

        override this.Item
            with get pos = storage.[ArrayNDLayout.addr pos layout]
            and set pos value = 
                if CheckFinite then
                    let isNonFinite =
                        match box value with
                        | :? double as dv -> System.Double.IsInfinity(dv) || System.Double.IsNaN(dv) 
                        | :? single as sv -> System.Single.IsInfinity(sv) || System.Single.IsNaN(sv) 
                        | _ -> false
                    if isNonFinite then raise (System.ArithmeticException("non-finite value encountered"))
                storage.[ArrayNDLayout.addr pos layout] <- value 

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            ArrayNDHostT<'T>(layout) :> ArrayNDT<'T>

        override this.NewView (layout: ArrayNDLayoutT) = 
            ArrayNDHostT<'T>(layout, storage) :> ArrayNDT<'T>

    /// creates a new contiguous (row-major) NDArray in host memory of the given shape 
    let inline newContinguous<'T> shp =
        ArrayNDHostT<'T>(ArrayNDLayout.newContiguous shp)

    /// creates a new Fortran (column-major) NDArray in host memory of the given shape
    let inline newColumnMajor<'T> shp =
        ArrayNDHostT<'T>(ArrayNDLayout.newColumnMajor shp)

    /// NDArray with zero dimensions (scalar) and given value
    let inline scalar value =
        let a = newContinguous [] 
        ArrayND.set [] value a
        a

    /// NDArray of given shape filled with zeros.
    let inline zeros shape =
        newContinguous shape

    /// NDArray of given shape filled with ones.
    let inline ones shape =
        let a = newContinguous shape
        ArrayND.fillWithOnes a
        a

    /// NDArray with ones on the diagonal of given shape.
    let inline identity shape =
        let a = zeros shape
        let ndim = List.length shape
        for i = 0 to (List.min shape) - 1 do
            set (List.replicate ndim i) a.One a
        
