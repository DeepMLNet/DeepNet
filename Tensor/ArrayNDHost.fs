namespace ArrayNDNS

open System
open System.Runtime.InteropServices
open System.Collections.Generic

open Basics
open Tensor
open MKL


[<AutoOpen>]
module ArrayNDHostTypes = 





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
        


        interface IArrayNDHostT with
            member this.Pin () = this.Pin ()
            member this.DataObj = box data
            member this.DataSizeInBytes = this.DataSizeInBytes

        interface IToArrayNDHostT<'T> with
            member this.ToHost () = this


             

        override this.SymmetricEigenDecomposition () =



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
