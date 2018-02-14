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
            let msg = sprintf "require a Host tensor but got a %s tensor" x.Dev.Id
            raise (StorageMismatch msg)
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

    let transfer x = Tensor.transfer Dev x

    let empty<'T> = Tensor.empty<'T> Dev

    let zeros<'T> = Tensor.zeros<'T> Dev 

    let ones<'T> = Tensor.ones<'T> Dev

    let falses = Tensor.falses Dev

    let trues = Tensor.trues Dev

    let scalar<'T> = Tensor.scalar<'T> Dev

    let init<'T> = Tensor.init<'T> Dev

    let filled<'T> = Tensor.filled<'T> Dev

    let identity<'T> = Tensor.identity<'T> Dev

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
    
