namespace Tensor

open System
open System.Reflection
open System.Numerics
open System.Threading.Tasks
open System.Linq.Expressions
open System.Collections.Generic
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

open Tensor.Utils
open Tensor.Backend
open Tensor.Host


module internal HostTensorHelpers = 

    let ensureCAndOffsetFree (x: Tensor<'T>) =
        if x.Dev <> (TensorHostDevice.Instance :> ITensorDevice) then
            invalidOp "Require a tensor stored on host device, but got a %s tensor." x.Dev.Id
        if TensorLayout.isC x.Layout && x.Layout.Offset = 0L then x
        else Tensor.copy (x, order=RowMajor)


type private HDFFuncs =

    static member Write<'T> (hdf5: HDF5, path: string, x: Tensor<'T>) =
        let x = HostTensorHelpers.ensureCAndOffsetFree x
        let storage = x.Storage :?> TensorHostStorage<'T>
        hdf5.Write (path, storage.Data, Tensor.shape x)

    static member Read<'T> (hdf5: HDF5, name: string) =
        let (data: 'T []), shape = hdf5.Read (name)
        Tensor<'T> (TensorLayout.newC shape, TensorHostStorage<'T> data)         



/// Host tensor functions.
module HostTensor =

    /// Tensor located on host using a .NET array as storage.
    let Dev = TensorHostDevice.Instance :> ITensorDevice

    /// Gets backend of a host tensor.
    let internal backend (trgt: Tensor<'T>) =
        if trgt.Dev <> Dev then
            invalidOp "This operation requires a tensor stored on the host, but got storage %A." trgt.Dev
        trgt.Backend :?> TensorHostBackend<'T>

    /// Fills the tensor with the values returned by the function.
    let FillIndexed (trgt: Tensor<'T>) (fn: int64[] -> 'T) =
        (backend trgt).FillIndexed (fn=fn, trgt=trgt, useThreads=false)

    /// Creates a tensor with the values returned by the function.
    let init (shape: int64 list) (fn: int64[] -> 'T) : Tensor<'T> =
        let x = Tensor<'T> (shape, Dev)
        FillIndexed x fn
        x           

    let transfer x = Tensor.transfer Dev x

    let empty<'T> = Tensor<'T>.empty Dev

    let zeros<'T> = Tensor<'T>.zeros Dev 

    let ones<'T> = Tensor<'T>.ones Dev

    let falses = Tensor.falses Dev

    let trues = Tensor.trues Dev

    let scalar<'T> = Tensor<'T>.scalar Dev

    let filled<'T> = Tensor<'T>.filled Dev

    let identity<'T> = Tensor<'T>.identity Dev

    let counting = Tensor.counting Dev

    let inline arange start incr stop = 
        Tensor.arange Dev start incr stop

    let inline linspace start stop nElems = 
        Tensor.linspace Dev start stop nElems
  
    /// Creates a one-dimensional Tensor using the specified data.
    /// The data is referenced, not copied.
    let usingArray (data: 'T []) =
        let shp = [data.LongLength]
        let layout = TensorLayout.newC shp
        let storage = TensorHostStorage<'T> (data)
        Tensor<'T> (layout, storage) 

    /// Creates a one-dimensional Tensor using the specified data.
    /// The data is copied.
    let ofArray (data: 'T []) =
        let shp = [Array.length data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0]])

    /// Creates a two-dimensional Tensor using the specified data. 
    /// The data is copied.
    let ofArray2D (data: 'T [,]) =
        let shp = [Array2D.length1 data; Array2D.length2 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1]])

    /// Creates a three-dimensional Tensor using the specified data. 
    /// The data is copied.
    let ofArray3D (data: 'T [,,]) =
        let shp = [Array3D.length1 data; Array3D.length2 data; Array3D.length3 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1], int32 idx.[2]])

    /// Creates a four-dimensional Tensor using the specified data. 
    /// The data is copied.
    let ofArray4D (data: 'T [,,,]) =
        let shp = [Array4D.length1 data; Array4D.length2 data; 
                   Array4D.length3 data; Array4D.length4 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1], int32 idx.[2], int32 idx.[3]])

    /// Creates a one-dimensional Tensor using the specified sequence.       
    let ofSeq (data: 'T seq) =
        data |> Array.ofSeq |> usingArray

    /// Creates a one-dimensional Tensor using the specified sequence and shape.       
    let ofSeqWithShape shape (data: 'T seq) =
        let nElems = shape |> List.fold (*) 1L
        data |> Seq.take (int32 nElems) |> ofSeq |> Tensor.reshape shape

    /// Creates a one-dimensional Tensor using the specified list.       
    let ofList (data: 'T list) =
        data |> Array.ofList |> usingArray

    /// Creates a two-dimensional Tensor using the specified list of lists.       
    let ofList2D (data: 'T list list) =
        data |> array2D |> ofArray2D

    /// Creates an Array from the data in this Tensor. The data is copied.
    let toArray (ary: Tensor<_>) =
        if Tensor.nDims ary <> 1 then failwith "Tensor must have 1 dimension"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array.init shp.[0] (fun i0 -> ary.[[int64 i0]])

    /// Creates an Array2D from the data in this Tensor. The data is copied.
    let toArray2D (ary: Tensor<_>) =
        if Tensor.nDims ary <> 2 then failwith "Tensor must have 2 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array2D.init shp.[0] shp.[1] (fun i0 i1 -> ary.[[int64 i0; int64 i1]])

    /// Creates an Array3D from the data in this Tensor. The data is copied.
    let toArray3D (ary: Tensor<_>) =
        if Tensor.nDims ary <> 3 then failwith "Tensor must have 3 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array3D.init shp.[0] shp.[1] shp.[2] (fun i0 i1 i2 -> ary.[[int64 i0; int64 i1; int64 i2]])
       
    /// Creates an Array4D from the data in this Tensor. The data is copied.
    let toArray4D (ary: Tensor<_>) =
        if Tensor.nDims ary <> 4 then failwith "Tensor must have 4 dimensions"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array4D.init shp.[0] shp.[1] shp.[2] shp.[3] (fun i0 i1 i2 i3 -> ary.[[int64 i0; int64 i1; int64 i2; int64 i3]])

    /// Creates a list from the data in this Tensor. The data is copied.
    let toList (ary: Tensor<_>) =
        ary |> toArray |> Array.toList

    /// Creates a list of lists from the data in the two-dimensional Tensor. The data is copied.
    let toList2D (ary: Tensor<_>) =
        if Tensor.nDims ary <> 2 then failwith "Tensor must have 2 dimensions"
        [0L .. ary.Shape.[0]-1L] |> List.map (fun i0 -> toList ary.[i0, *])

    /// Writes the given host tensor into the HDF5 file under the given path.
    let write (hdf5: HDF5) (path: string) (x: ITensor) =
        callGeneric<HDFFuncs, unit> "Write" [x.DataType] (hdf5, path, x)

    /// Reads the tensor of data type 'T with the given path from an HDF5 file.
    let read<'T> (hdf5: HDF5) (path: string) : Tensor<'T> =
        HDFFuncs.Read (hdf5, path)

    /// Reads the tensor with the given path from an HDF5 file and returns it
    /// as an ITensor with the data type as stored in the HDF5 file.
    let readUntyped (hdf5: HDF5) (path: string) = 
        let dataType = hdf5.GetDataType path
        callGeneric<HDFFuncs, ITensor> "Read" [dataType] (hdf5, path)

    /// Creates a tensor of given shape filled with random integer numbers between
    /// minValue and maxValue.
    let randomInt (rnd: Random) (minValue, maxValue) shp =
        rnd.Seq (minValue, maxValue) 
        |> ofSeqWithShape shp  

    /// Creates a tensor of given shape filled with random floating-point numbers 
    /// uniformly placed between minValue and maxValue.
    let randomUniform (rnd: Random) (minValue: 'T, maxValue: 'T) shp =
        rnd.SeqDouble (conv<float> minValue, conv<float> maxValue) 
        |> Seq.map conv<'T>
        |> ofSeqWithShape shp    

    /// Creates a tensor of given shape filled with random samples from a normal
    /// distribution with the specified mean and variance.
    let randomNormal (rnd: Random) (mean: 'T, variance: 'T) shp =
        rnd.SeqNormal (conv<float> mean, conv<float> variance) 
        |> Seq.map conv<'T>
        |> ofSeqWithShape shp       
    
    /// Fills the tensor with the values returned by the function.
    let Fill (trgt: Tensor<'T>) (fn: unit -> 'T)  =
        (backend trgt).Fill (fn=fn, trgt=trgt, useThreads=false)

    /// Fills the tensor with the values returned by the given sequence.
    let FillSeq (trgt: Tensor<'T>) (data: 'T seq) =
        use enumerator = data.GetEnumerator()
        Fill trgt (fun () -> 
            if enumerator.MoveNext() then enumerator.Current
            else invalidArg "data" "Sequence ended before tensor of shape %A was filled." trgt.Shape)

    /// maps all elements using the specified function into this tensor
    let FillMap (trgt: Tensor<'T>) (fn: 'TA -> 'T) (a: Tensor<'TA>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        (backend trgt).Map (fn=fn, trgt=trgt, a=a, useThreads=false)

    /// maps all elements using the specified function into a new tensor
    let map (fn: 'T -> 'R) (a: Tensor<'T>) =
        let trgt, a = Tensor.PrepareElemwise (a)
        FillMap trgt fn a
        trgt       

    /// maps all elements using the specified indexed function into this tensor
    let FillMapIndexed (trgt: Tensor<'T>) (fn: int64[] -> 'TA -> 'T) (a: Tensor<'TA>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        (backend trgt).MapIndexed (fn=fn, trgt=trgt, a=a, useThreads=false)

    /// maps all elements using the specified indexed function into a new tensor
    let mapi (fn: int64[] -> 'T -> 'R) (a: Tensor<'T>) =
        let trgt, a = Tensor.PrepareElemwise (a)
        FillMapIndexed trgt fn a
        trgt     

    /// maps all elements using the specified function into this tensor
    let FillMap2 (trgt: Tensor<'T>) (fn: 'TA -> 'TB -> 'T) (a: Tensor<'TA>) (b: Tensor<'TB>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        (backend trgt).Map2 (fn=fn, trgt=trgt, a=a, b=b, useThreads=false)

    /// maps all elements using the specified function into a new tensor
    let map2 (fn: 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        FillMap2 trgt fn a b
        trgt       

    /// maps all elements using the specified indexed function into this tensor
    let FillMapIndexed2 (trgt: Tensor<'T>) (fn: int64[] -> 'TA -> 'TB -> 'T) (a: Tensor<'TA>) (b: Tensor<'TB>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        (backend trgt).MapIndexed2 (fn=fn, trgt=trgt, a=a, b=b, useThreads=false)

    /// maps all elements using the specified indexed function into a new tensor
    let mapi2 (fn: int64[] -> 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        FillMapIndexed2 trgt fn a b
        trgt       

    // TODO: change to Tensor folder function      
    /// folds the function over the given axis, using this tensor as target 
    let FillFoldAxis (trgt: Tensor<'T>) (fn: 'T -> 'TA -> 'T) (initial: Tensor<'T>) (axis: int) (a: Tensor<'TA>) =
        let a, initial = Tensor.PrepareAxisReduceSources (trgt, axis, a, Some initial)
        (backend trgt).FoldLastAxis (fn=fn, initial=initial.Value, trgt=trgt, a=a, useThreads=false)        

    // TODO: change to Tensor folder function
    /// folds the function over the given axis
    let foldAxis (fn: 'T -> 'TA -> 'T) (initial: Tensor<'T>) (axis: int) (a: Tensor<'TA>) =
        let trgt, a = Tensor.PrepareAxisReduceTarget (axis, a)
        FillFoldAxis trgt fn initial axis a
        trgt


    /// Multi-threaded operations of Tensor<'T>.
    module Parallel = 

        /// Fills the tensor with the values returned by the function using multiple threads.
        let FillIndexed (trgt: Tensor<'T>) (fn: int64[] -> 'T) =
            (backend trgt).FillIndexed (fn=fn, trgt=trgt, useThreads=true)

        /// Fills the tensor with the values returned by the function using multiple threads.
        let Fill (trgt: Tensor<'T>) (fn: unit -> 'T)  =
            (backend trgt).Fill (fn=fn, trgt=trgt, useThreads=true)

        /// maps all elements using the specified function into this tensor using multiple threads
        let FillMap (trgt: Tensor<'T>) (fn: 'TA -> 'T) (a: Tensor<'TA>) = 
            let a = Tensor.PrepareElemwiseSources (trgt, a)
            (backend trgt).Map (fn=fn, trgt=trgt, a=a, useThreads=false)

        /// maps all elements using the specified indexed function into this tensor using multiple threads
        let FillMapIndexed (trgt: Tensor<'T>) (fn: int64[] -> 'TA -> 'T) (a: Tensor<'TA>) = 
            let a = Tensor.PrepareElemwiseSources (trgt, a)
            (backend trgt).MapIndexed (fn=fn, trgt=trgt, a=a, useThreads=true)

        /// maps all elements using the specified function into this tensor using multiple threads
        let FillMap2 (trgt: Tensor<'T>) (fn: 'TA -> 'TB -> 'T) (a: Tensor<'TA>) (b: Tensor<'TB>) = 
            let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
            (backend trgt).Map2 (fn=fn, trgt=trgt, a=a, b=b, useThreads=true)

        /// maps all elements using the specified indexed function into this tensor using multiple threads
        let FillMapIndexed2 (trgt: Tensor<'T>) (fn: int64[] -> 'TA -> 'TB -> 'T) (a: Tensor<'TA>) (b: Tensor<'TB>) = 
            let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
            (backend trgt).MapIndexed2 (fn=fn, trgt=trgt, a=a, b=b, useThreads=true)

        // TODO: change to Tensor folder function
        /// folds the function over the given axis, using this tensor as target and multiple threads
        let FillFoldAxis (trgt: Tensor<'T>) (fn: 'T -> 'TA -> 'T) (initial: Tensor<'T>) (axis: int) (a: Tensor<'TA>) =
            let a, initial = Tensor.PrepareAxisReduceSources (trgt, axis, a, Some initial)
            (backend trgt).FoldLastAxis (fn=fn, initial=initial.Value, trgt=trgt, a=a, useThreads=true) 

        /// Creates a new tensor with the values returned by the function.
        let init<'T> (shape: int64 list) (fn: int64[] -> 'T) : Tensor<'T> =
            let x = Tensor<'T> (shape, Dev)
            FillIndexed x fn
            x          

        /// Maps all elements using the specified function into a new tensor.
        let map (fn: 'T -> 'R) (a: Tensor<'T>) =
            let trgt, a = Tensor.PrepareElemwise (a)
            FillMap trgt fn a
            trgt       

        /// Maps all elements using the specified indexed function into a new tensor.
        let mapi (fn: int64[] -> 'T -> 'R) (a: Tensor<'T>) =
            let trgt, a = Tensor.PrepareElemwise (a)
            FillMapIndexed trgt fn a
            trgt      

        /// Maps all elements using the specified function into a new tensor.
        let map2 (fn: 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
            let trgt, a, b = Tensor.PrepareElemwise (a, b)
            FillMap2 trgt fn a b
            trgt           

        /// Maps all elements using the specified indexed function into a new tensor.
        let mapi2 (fn: int64[] -> 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
            let trgt, a, b = Tensor.PrepareElemwise (a, b)
            FillMapIndexed2 trgt fn a b
            trgt            

        /// Folds the function over the given axis.
        let foldAxis (fn: 'T -> 'TA -> 'T) (initial: 'T) (axis: int) (a: Tensor<'TA>) =
            let trgt, a = Tensor.PrepareAxisReduceTarget (axis, a)
            FillFoldAxis trgt fn initial axis a
            trgt


