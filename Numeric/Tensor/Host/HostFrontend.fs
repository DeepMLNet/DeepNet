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



/// <summary>Functions for creating and operating on tensors stored in host memory.</summary>
/// <remarks>This module contains functions for creating tensors stored in host memory.
/// It further contains functions that only work with tensors stored in host memory.
/// Calling these functions with tensors stored on other devices will result in an
/// <see cref="System.InvalidOperationException"/>.</remarks>
/// <example><code language="fsharp">
/// let x = HostTensor.zeros [3L; 3L]  // x.Dev = HostTensor.Dev
/// </code></example>
/// <seealso cref="Tensor`1"/><seealso cref="HostTensor.Parallel"/>
module HostTensor =

    /// <summary>Tensor device using a .NET array in host memory as data storage.</summary>
    /// <seealso cref="Tensor`1.Dev"/>
    let Dev = TensorHostDevice.Instance :> ITensorDevice

    /// Gets the backend of a host tensor.
    let internal backend (trgt: Tensor<'T>) =
        if trgt.Dev <> Dev then
            invalidOp "This operation requires a tensor stored on the host, but got device %A." trgt.Dev
        trgt.Backend :?> TensorHostBackend<'T>

    /// <summary>Fills the tensor with values returned by the specifed function.</summary>
    /// <param name="trgt">The target tensor to fill.</param>
    /// <param name="fn">A function that takes the index of the element to fill and returns
    /// the corresponding value.</param>
    /// <seealso cref="init``1"/><seealso cref="HostTensor.Parallel.FillIndexed``1"/>      
    let FillIndexed (trgt: Tensor<'T>) (fn: int64[] -> 'T) =
        (backend trgt).FillIndexed (fn=fn, trgt=trgt, useThreads=false)

    /// <summary>Creates a new tensor with values returned by the specified function.</summary>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <param name="fn">A function that takes the index of the element to fill and returns
    /// the corresponding value.</param>
    /// <seealso cref="FillIndexed``1"/><seealso cref="HostTensor.Parallel.init``1"/>  
    let init (shape: int64 list) (fn: int64[] -> 'T) : Tensor<'T> =
        let x = Tensor<'T> (shape, Dev)
        FillIndexed x fn
        x           

    /// <summary>Transfers a tensor to the host device.</summary>
    /// <typeparam name="'T">The data type of the tensor.</typeparam>    
    /// <param name="a">The tensor to transfer.</param>
    /// <returns>A tensor on the host device.</returns>
    /// <seealso cref="Tensor`1.transfer"/>
    let transfer (a: Tensor<'T>) = Tensor.transfer Dev a

    /// <summary>Creates a new, empty tensor with the given number of dimensions.</summary>
    /// <typeparam name="'T">The data type of the new tensor.</typeparam>    
    /// <param name="nDims">The number of dimensions of the new, empty tensor.</param>
    /// <returns>The new tensor.</returns>    
    /// <seealso cref="Tensor`1.empty"/>
    let empty<'T> nDims = Tensor<'T>.empty Dev nDims

    /// <summary>Creates a new tensor filled with zeros (0).</summary>
    /// <typeparam name="'T">The data type of the new tensor.</typeparam>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.zeros"/>
    let zeros<'T> shape = Tensor<'T>.zeros Dev shape

    /// <summary>Creates a new tensor filled with ones (1).</summary>
    /// <typeparam name="'T">The data type of the new tensor.</typeparam>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.ones"/>
    let ones<'T> shape = Tensor<'T>.ones Dev shape

    /// <summary>Creates a new boolean tensor filled with falses.</summary>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.falses"/>
    let falses shape = Tensor.falses Dev shape

    /// <summary>Creates a new boolean tensor filled with trues.</summary>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.trues"/>
    let trues shape = Tensor.trues Dev shape

    /// <summary>Creates a new zero-dimensional (scalar) tensor with the specified value.</summary>
    /// <param name="value">The value of the new, scalar tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.scalar"/>
    let scalar (value: 'T) = Tensor<'T>.scalar Dev value

    /// <summary>Creates a new tensor filled with the specified value.</summary>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <param name="value">The value to fill the new tensor with.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.filled"/>
    let filled shape (value: 'T) = Tensor<'T>.filled Dev shape value

    /// <summary>Creates a new identity matrix.</summary>
    /// <typeparam name="'T">The data type of the new tensor.</typeparam>
    /// <param name="size">The size of the square identity matrix.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.identity"/>
    let identity<'T> size = Tensor<'T>.identity Dev size

    /// <summary>Creates a new vector filled with the integers from zero to the specified maximum.</summary>
    /// <param name="nElems">The number of elements of the new vector.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.counting"/>
    let counting nElems = Tensor.counting Dev nElems

    /// <summary>Creates a new vector filled with equaly spaced values using a specifed increment.</summary>
    /// <typeparam name="^V">The data type of the new tensor.</typeparam>
    /// <param name="start">The starting value.</param>
    /// <param name="incr">The increment between successive element.</param>   
    /// <param name="stop">The end value, which is not included.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.arange``3"/>
    let inline arange (start: 'V) (incr: 'V) (stop: 'V) = 
        Tensor.arange Dev start incr stop

    /// <summary>Creates a new vector of given size filled with equaly spaced values.</summary>
    /// <typeparam name="'V">The data type of the new tensor.</typeparam>
    /// <param name="start">The starting value.</param>
    /// <param name="stop">The end value, which is not included.</param>
    /// <param name="nElems">The size of the vector.</param>   
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.linspace``2"/>
    let inline linspace (start: 'V) (stop: 'V) nElems = 
        Tensor.linspace Dev start stop nElems
  
    /// <summary>Creates a one-dimensional tensor referencing the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor using the array <c>data</c> as its storage.</returns>
    /// <remarks>The data array is referenced, not copied.
    /// Thus changing the tensor modifies the specified data array and vice versa.</remarks>
    /// <seealso cref="ofArray``1"/>
    let usingArray (data: 'T []) =
        let shp = [data.LongLength]
        let layout = TensorLayout.newC shp
        let storage = TensorHostStorage<'T> (data)
        Tensor<'T> (layout, storage) 

    /// <summary>Creates a one-dimensional tensor copying the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor filled with the values from <c>data</c>.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="usingArray``1"/>
    /// <seealso cref="ofArray2D``1"/><seealso cref="ofArray3D``1"/><seealso cref="ofArray4D``1"/>
    let ofArray (data: 'T []) =
        let shp = [Array.length data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0]])

    /// <summary>Creates a two-dimensional tensor copying the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor using filled with the values from <c>data</c>.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="ofArray``1"/><seealso cref="ofArray3D``1"/><seealso cref="ofArray4D``1"/>
    let ofArray2D (data: 'T [,]) =
        let shp = [Array2D.length1 data; Array2D.length2 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1]])

    /// <summary>Creates a three-dimensional tensor copying the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor using filled with the values from <c>data</c>.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="ofArray``1"/><seealso cref="ofArray2D``1"/><seealso cref="ofArray4D``1"/>
    let ofArray3D (data: 'T [,,]) =
        let shp = [Array3D.length1 data; Array3D.length2 data; Array3D.length3 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1], int32 idx.[2]])

    /// <summary>Creates a four-dimensional tensor copying the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor using filled with the values from <c>data</c>.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="ofArray``1"/><seealso cref="ofArray2D``1"/><seealso cref="ofArray3D``1"/>
    let ofArray4D (data: 'T [,,,]) =
        let shp = [Array4D.length1 data; Array4D.length2 data; 
                   Array4D.length3 data; Array4D.length4 data]
        let shp = shp |> List.map int64
        init shp (fun idx -> data.[int32 idx.[0], int32 idx.[1], int32 idx.[2], int32 idx.[3]])

    /// <summary>Creates a one-dimensional tensor from the specified sequence.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data to fill the tensor with.</param>
    /// <returns>A tensor containing values from the specifed sequence.</returns>
    /// <remarks>The sequence must be finite.</remarks>
    /// <seealso cref="ofSeqWithShape``1"/><seealso cref="ofList``1"/> 
    let ofSeq (data: 'T seq) =
        data |> Array.ofSeq |> usingArray

    /// <summary>Creates a one-dimensional Tensor using the specified sequence and shape.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <param name="data">The data to fill the tensor with.</param>
    /// <returns>A tensor containing values from the specifed sequence.</returns>
    /// <remarks>Only the number of elements required to fill the tensor of the specified
    /// shape are consumed from the sequence. Thus it may be infinite.</remarks>
    /// <seealso cref="ofSeq``1"/>
    let ofSeqWithShape shape (data: 'T seq) =
        let nElems = shape |> List.fold (*) 1L
        data |> Seq.take (int32 nElems) |> ofSeq |> Tensor.reshape shape

    /// <summary>A sequence of all elements contained in the tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A sequence of all elements of the tensor.</returns>
    /// <remarks>The enumeration is done so that the last index is the fastest changing index.</remarks>
    /// <seealso cref="ofSeq``1"/>
    let toSeq (a: Tensor<'T>) : seq<'T> =
        backend a :> IEnumerable<'T> 

    /// <summary>Creates a one-dimensional tensor from the specified list.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data to fill the tensor with.</param>
    /// <returns>A tensor containing values from the specifed list.</returns>
    /// <seealso cref="ofSeq``1"/><seealso cref="ofList2D``1"/>    
    /// <seealso cref="toList``1"/>
    let ofList (data: 'T list) =
        data |> Array.ofList |> usingArray

    /// <summary>Creates a two-dimensional tensor from the specified list of lists.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data to fill the tensor with.</param>
    /// <returns>A tensor containing values from the specifed lists.</returns>
    /// <seealso cref="ofSeq``1"/><seealso cref="ofList``1"/>    
    /// <seealso cref="toList2D``1"/>
    let ofList2D (data: 'T list list) =
        data |> array2D |> ofArray2D

    /// <summary>Creates an array from a one-dimensional tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>An array containing the values from the tensor.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="toArray2D``1"/><seealso cref="toArray3D``1"/><seealso cref="toArray4D``1"/>
    /// <seealso cref="ofArray``1"/>
    let toArray (ary: Tensor<'T>) =
        if Tensor.nDims ary <> 1 then invalidOp "Tensor must have 1 dimension"
        let shp = Tensor.shape ary
        let shp = shp |> List.map int32
        Array.init shp.[0] (fun i0 -> ary.[[int64 i0]])

    /// <summary>Creates an array from a two-dimensional tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>An array containing the values from the tensor.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="toArray``1"/><seealso cref="toArray2D``1"/><seealso cref="toArray3D``1"/>
    /// <seealso cref="ofArray2D``1"/>
    let toArray2D (a: Tensor<'T>) =
        if Tensor.nDims a <> 2 then invalidOp "Tensor must have 2 dimensions"
        let shp = Tensor.shape a
        let shp = shp |> List.map int32
        Array2D.init shp.[0] shp.[1] (fun i0 i1 -> a.[[int64 i0; int64 i1]])

    /// <summary>Creates an array from a three-dimensional tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>An array containing the values from the tensor.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="toArray``1"/><seealso cref="toArray2D``1"/><seealso cref="toArray4D``1"/>
    /// <seealso cref="ofArray3D``1"/>
    let toArray3D (a: Tensor<'T>) =
        if Tensor.nDims a <> 3 then invalidOp "Tensor must have 3 dimensions"
        let shp = Tensor.shape a
        let shp = shp |> List.map int32
        Array3D.init shp.[0] shp.[1] shp.[2] (fun i0 i1 i2 -> a.[[int64 i0; int64 i1; int64 i2]])
       
    /// <summary>Creates an array from a four-dimensional tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>An array containing the values from the tensor.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="toArray``1"/><seealso cref="toArray2D``1"/><seealso cref="toArray3D``1"/>
    /// <seealso cref="ofArray4D``1"/>
    let toArray4D (a: Tensor<'T>) =
        if Tensor.nDims a <> 4 then invalidOp "Tensor must have 4 dimensions"
        let shp = Tensor.shape a
        let shp = shp |> List.map int32
        Array4D.init shp.[0] shp.[1] shp.[2] shp.[3] (fun i0 i1 i2 i3 -> a.[[int64 i0; int64 i1; int64 i2; int64 i3]])

    /// <summary>Creates a list from a one-dimensional tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A list containing the values from the tensor.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="toList2D``1"/><seealso cref="ofList``1"/>
    let toList (a: Tensor<'T>) =
        a |> toArray |> Array.toList

    /// <summary>Creates a list of lists from a two-dimensional tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A list of lists containing the values from the tensor.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="toList``1"/><seealso cref="ofList2D``1"/>
    let toList2D (a: Tensor<'T>) =
        if Tensor.nDims a <> 2 then invalidOp "Tensor must have 2 dimensions"
        [0L .. a.Shape.[0]-1L] |> List.map (fun i0 -> toList a.[i0, *])

    /// <summary>Writes the tensor into the HDF5 file under the specfied HDF5 object path.</summary>
    /// <param name="hdf5">The HDF5 file.</param>
    /// <param name="path">The HDF5 object path.</param>
    /// <param name="x">The tensor to write.</param>
    let write (hdf5: HDF5) (path: string) (x: ITensor) =
        callGeneric<HDFFuncs, unit> "Write" [x.DataType] (hdf5, path, x)

    /// <summary>Reads a tensor from the specified HDF5 object path in an HDF5 file.</summary>
    /// <typeparam name="'T">The type of the data. This must match the type of the data stored in the
    /// HDF5 file.</typeparam>
    /// <param name="hdf5">The HDF5 file.</param>
    /// <param name="path">The HDF5 object path.</param>
    /// <returns>A tensor filled with data read from the HDF5 file.</returns>
    /// <exception cref="System.InvalidOperationException">The data type stored in the HDF5 does 
    /// not match type <c>'T</c>.</exception>
    /// <seealso cref="write"/><seealso cref="readUntyped"/>
    let read<'T> (hdf5: HDF5) (path: string) : Tensor<'T> =
        HDFFuncs.Read (hdf5, path)

    /// <summary>Reads a tensor with unspecified data type from the specified HDF5 object path in an HDF5 file.</summary>
    /// <param name="hdf5">The HDF5 file.</param>
    /// <param name="path">The HDF5 object path.</param>
    /// <returns>A tensor filled with data read from the HDF5 file.</returns>
    /// <seealso cref="read``1"/>
    let readUntyped (hdf5: HDF5) (path: string) = 
        let dataType = hdf5.GetDataType path
        callGeneric<HDFFuncs, ITensor> "Read" [dataType] (hdf5, path)

    /// <summary>Creates a tensor filled with random integer numbers from a uniform distribution.</summary>
    /// <param name="rnd">The random generator to use.</param>
    /// <param name="minValue">The minimum value.</param>
    /// <param name="maxValue">The maximum value.</param>
    /// <param name="shp">The shape of the new tensor.</param>
    /// <returns>A tensor of specified shape filled with random numbers.</returns>
    let randomInt (rnd: Random) (minValue, maxValue) shp =
        rnd.Seq (minValue, maxValue) 
        |> ofSeqWithShape shp  

    /// <summary>Creates a tensor filled with random floating-point numbers from a uniform distribution.</summary>
    /// <param name="rnd">The random generator to use.</param>
    /// <param name="minValue">The minimum value.</param>
    /// <param name="maxValue">The maximum value.</param>
    /// <param name="shp">The shape of the new tensor.</param>
    /// <returns>A tensor of specified shape filled with random numbers.</returns>
    let randomUniform (rnd: Random) (minValue: 'T, maxValue: 'T) shp =
        rnd.SeqDouble (conv<float> minValue, conv<float> maxValue) 
        |> Seq.map conv<'T>
        |> ofSeqWithShape shp    

    /// <summary>Creates a tensor filled with random numbers from a normale distribution.</summary>
    /// <param name="rnd">The random generator to use.</param>
    /// <param name="mean">The mean of the normal distribution.</param>
    /// <param name="variance">The variance of the normal distribution.</param>
    /// <param name="shp">The shape of the new tensor.</param>
    /// <returns>A tensor of specified shape filled with random numbers.</returns>
    let randomNormal (rnd: Random) (mean: 'T, variance: 'T) shp =
        rnd.SeqNormal (conv<float> mean, conv<float> variance) 
        |> Seq.map conv<'T>
        |> ofSeqWithShape shp       
    
    /// <summary>Fills the tensor with the values returned by the function.</summary>
    /// <param name="trgt">The target tensor to fill.</param>
    /// <param name="fn">A function that returns the values to fill the tensor with.</param>
    /// <seealso cref="HostTensor.Parallel.Fill``1"/>
    let Fill (trgt: Tensor<'T>) (fn: unit -> 'T)  =
        (backend trgt).Fill (fn=fn, trgt=trgt, useThreads=false)

    /// <summary>Fills the tensor with the values returned by the given sequence.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="trgt">The target tensor to fill.</param>
    /// <param name="data">The sequence of data to fill the tensor with.</param>    
    let FillSeq (trgt: Tensor<'T>) (data: 'T seq) =
        use enumerator = data.GetEnumerator()
        Fill trgt (fun () -> 
            if enumerator.MoveNext() then enumerator.Current
            else invalidArg "data" "Sequence ended before tensor of shape %A was filled." trgt.Shape)

    /// <summary>Applies to specified function to all elements of the tensor using the specified tensor as 
    /// target.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="trgt">The output tensor to fill.</param>
    /// <param name="fn">A function that takes a value from the input tensor and returns the corresponding output 
    /// value.</param>        
    /// <param name="a">The input tensor.</param>
    /// <seealso cref="map``2"/><seealso cref="HostTensor.Parallel.FillMap``2"/>
    let FillMap (trgt: Tensor<'T>) (fn: 'TA -> 'T) (a: Tensor<'TA>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        (backend trgt).Map (fn=fn, trgt=trgt, a=a, useThreads=false)

    /// <summary>Applies to specified function to all elements of the tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="fn">A function that takes a value from the input tensor and returns the corresponding output 
    /// value.</param>        
    /// <param name="a">The source tensor.</param>
    /// <returns>The output tensor.</returns>
    /// <seealso cref="FillMap``2"/><seealso cref="mapi``2"/><seealso cref="map2``3"/>
    /// <seealso cref="HostTensor.Parallel.map``2"/>
    let map (fn: 'T -> 'R) (a: Tensor<'T>) =
        let trgt, a = Tensor.PrepareElemwise (a)
        FillMap trgt fn a
        trgt       

    /// <summary>Applies to specified indexed function to all elements of the tensor using the specified tensor as 
    /// target.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="trgt">The output tensor to fill.</param>
    /// <param name="fn">A function that takes an index and the corresponding value from the input tensor and returns 
    /// the corresponding output value.</param>        
    /// <param name="a">The input tensor.</param>
    /// <seealso cref="mapi``2"/><seealso cref="HostTensor.Parallel.FillMapIndexed``2"/>
    let FillMapIndexed (trgt: Tensor<'T>) (fn: int64[] -> 'TA -> 'T) (a: Tensor<'TA>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        (backend trgt).MapIndexed (fn=fn, trgt=trgt, a=a, useThreads=false)

    /// <summary>Applies to specified indexed function to all elements of the tensor.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="fn">A function that takes an index and the corresponding value from the input tensor and returns 
    /// the corresponding output value.</param>        
    /// <param name="a">The source tensor.</param>
    /// <returns>The output tensor.</returns>
    /// <seealso cref="FillMapIndexed``2"/><seealso cref="map``2"/><seealso cref="HostTensor.Parallel.map``2"/>
    let mapi (fn: int64[] -> 'T -> 'R) (a: Tensor<'T>) =
        let trgt, a = Tensor.PrepareElemwise (a)
        FillMapIndexed trgt fn a
        trgt     

    /// <summary>Applies to specified function to all elements of the two tensors using the specified tensor as 
    /// target.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="trgt">The output tensor to fill.</param>
    /// <param name="fn">A function that takes a value from the first input tensor and a value from the second input 
    /// tensor and returns the corresponding output value.</param>        
    /// <param name="a">The first input tensor.</param>
    /// <param name="b">The second input tensor.</param>
    /// <seealso cref="map2``3"/><seealso cref="HostTensor.Parallel.FillMap2``3"/>
    let FillMap2 (trgt: Tensor<'T>) (fn: 'TA -> 'TB -> 'T) (a: Tensor<'TA>) (b: Tensor<'TB>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        (backend trgt).Map2 (fn=fn, trgt=trgt, a=a, b=b, useThreads=false)

    /// <summary>Applies to specified function to all elements of the two tensors.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="fn">A function that takes a value from the first input tensor and a value from the second input 
    /// tensor and returns the corresponding output value.</param>        
    /// <param name="a">The first input tensor.</param>
    /// <param name="b">The second input tensor.</param>
    /// <returns>The output tensor.</returns>
    /// <seealso cref="FillMap2``3"/><seealso cref="map``2"/><seealso cref="mapi2``3"/>
    /// <seealso cref="HostTensor.Parallel.map2``3"/>
    let map2 (fn: 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        FillMap2 trgt fn a b
        trgt       

    /// <summary>Applies to specified indexed function to all elements of the two tensors using the specified tensor as 
    /// target.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="trgt">The output tensor to fill.</param>
    /// <param name="fn">A function that takes an index, the corresponding value from the first input and second input 
    /// tensor and returns the corresponding output value.</param>        
    /// <param name="a">The first input tensor.</param>
    /// <param name="b">The second input tensor.</param>
    /// <seealso cref="mapi2``3"/><seealso cref="HostTensor.Parallel.FillMapIndexed2``3"/>
    let FillMapIndexed2 (trgt: Tensor<'T>) (fn: int64[] -> 'TA -> 'TB -> 'T) (a: Tensor<'TA>) (b: Tensor<'TB>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        (backend trgt).MapIndexed2 (fn=fn, trgt=trgt, a=a, b=b, useThreads=false)

    /// <summary>Applies to specified indexed function to all elements of the two tensors.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="fn">A function that takes an index, the corresponding value from the first input and second input 
    /// tensor and returns the corresponding output value.</param>        
    /// <param name="a">The first input tensor.</param>
    /// <param name="b">The second input tensor.</param>
    /// <returns>The output tensor.</returns>
    /// <seealso cref="FillMapIndexed2``3"/><seealso cref="map2``3"/><seealso cref="HostTensor.Parallel.mapi2``3"/>
    let mapi2 (fn: int64[] -> 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        FillMapIndexed2 trgt fn a b
        trgt       

    /// <summary>Applies to specified function to all elements of the tensor, threading an accumulator through the 
    /// computation.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="trgt">The output tensor that will contain the final state values.</param>
    /// <param name="fn">A function that takes a state value and a value from the input tensor and returns a new state 
    /// value.</param>        
    /// <param name="initial">The initial state value.</param>
    /// <param name="axis">The axis to fold over.</param>
    /// <param name="a">The source tensor.</param>
    /// <seealso cref="foldAxis``2"/><seealso cref="HostTensor.Parallel.FillFoldAxis``3"/>
    let FillFoldAxis (trgt: Tensor<'T>) (fn: 'T -> 'TA -> 'T) (initial: Tensor<'T>) (axis: int) (a: Tensor<'TA>) =
        let a, initial = Tensor.PrepareAxisReduceSources (trgt, axis, a, Some initial)
        (backend trgt).FoldLastAxis (fn=fn, initial=initial.Value, trgt=trgt, a=a, useThreads=false)        

    /// <summary>Applies to specified function to all elements of the tensor, threading an accumulator through the 
    /// computation.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="fn">A function that takes a state value and a value from the input tensor and returns a new state 
    /// value.</param>        
    /// <param name="initial">The initial state value.</param>
    /// <param name="axis">The axis to fold over.</param>
    /// <param name="a">The source tensor.</param>
    /// <returns>The output tensor containg the final states.</returns>
    /// <seealso cref="FillFoldAxis``2"/><seealso cref="HostTensor.Parallel.foldAxis``2"/>
    let foldAxis (fn: 'T -> 'TA -> 'T) (initial: Tensor<'T>) (axis: int) (a: Tensor<'TA>) =
        let trgt, a = Tensor.PrepareAxisReduceTarget (axis, a)
        FillFoldAxis trgt fn initial axis a
        trgt


    /// <summary>Multi-threaded operations for tensors stored on the host device.</summary>
    /// <seealso cref="HostTensor"/>
    module Parallel = 

        /// <summary>Fills the tensor with values returned by the specifed function using multiple threads.</summary>
        /// <param name="trgt">The target tensor to fill.</param>
        /// <param name="fn">A function that takes the index of the element to fill and returns
        /// the corresponding value.</param>
        /// <seealso cref="HostTensor.FillIndexed``1"/> 
        let FillIndexed (trgt: Tensor<'T>) (fn: int64[] -> 'T) =
            (backend trgt).FillIndexed (fn=fn, trgt=trgt, useThreads=true)

        /// <summary>Fills the tensor with the values returned by the function using multiple threads.</summary>
        /// <param name="trgt">The target tensor to fill.</param>
        /// <param name="fn">A function that returns the values to fill the tensor with.</param>
        /// <seealso cref="HostTensor.Fill``1"/>
        let Fill (trgt: Tensor<'T>) (fn: unit -> 'T)  =
            (backend trgt).Fill (fn=fn, trgt=trgt, useThreads=true)

        /// <summary>Applies to specified function to all elements of the tensor using the specified tensor as target 
        /// using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="trgt">The output tensor to fill.</param>
        /// <param name="fn">A function that takes a value from the input tensor and returns the corresponding output 
        /// value.</param>        
        /// <param name="a">The input tensor.</param>
        /// <seealso cref="HostTensor.FillMap``2"/>
        let FillMap (trgt: Tensor<'T>) (fn: 'TA -> 'T) (a: Tensor<'TA>) = 
            let a = Tensor.PrepareElemwiseSources (trgt, a)
            (backend trgt).Map (fn=fn, trgt=trgt, a=a, useThreads=false)

        /// <summary>Applies to specified indexed function to all elements of the tensor using the specified tensor as 
        /// target using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="trgt">The output tensor to fill.</param>
        /// <param name="fn">A function that takes an index and the corresponding value from the input tensor and returns 
        /// the corresponding output value.</param>        
        /// <param name="a">The input tensor.</param>
        /// <seealso cref="HostTensor.FillMapIndexed``2"/>
        let FillMapIndexed (trgt: Tensor<'T>) (fn: int64[] -> 'TA -> 'T) (a: Tensor<'TA>) = 
            let a = Tensor.PrepareElemwiseSources (trgt, a)
            (backend trgt).MapIndexed (fn=fn, trgt=trgt, a=a, useThreads=true)

        /// <summary>Applies to specified function to all elements of the two tensors using the specified tensor as 
        /// target using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="trgt">The output tensor to fill.</param>
        /// <param name="fn">A function that takes a value from the first input tensor and a value from the second input 
        /// tensor and returns the corresponding output value.</param>        
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <seealso cref="HostTensor.FillMap2``3"/>
        let FillMap2 (trgt: Tensor<'T>) (fn: 'TA -> 'TB -> 'T) (a: Tensor<'TA>) (b: Tensor<'TB>) = 
            let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
            (backend trgt).Map2 (fn=fn, trgt=trgt, a=a, b=b, useThreads=true)

        /// <summary>Applies to specified indexed function to all elements of the two tensors using the specified tensor 
        /// as target using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="trgt">The output tensor to fill.</param>
        /// <param name="fn">A function that takes an index, the corresponding value from the first input and second input 
        /// tensor and returns the corresponding output value.</param>        
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <seealso cref="HostTensor.FillMapIndexed2``3"/>
        let FillMapIndexed2 (trgt: Tensor<'T>) (fn: int64[] -> 'TA -> 'TB -> 'T) (a: Tensor<'TA>) (b: Tensor<'TB>) = 
            let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
            (backend trgt).MapIndexed2 (fn=fn, trgt=trgt, a=a, b=b, useThreads=true)

        /// <summary>Applies to specified function to all elements of the tensor, threading an accumulator through the 
        /// computation using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="trgt">The output tensor that will contain the final state values.</param>
        /// <param name="fn">A function that takes a state value and a value from the input tensor and returns a new 
        /// state value.</param>        
        /// <param name="initial">The initial state value.</param>
        /// <param name="axis">The axis to fold over.</param>
        /// <param name="a">The source tensor.</param>
        /// <seealso cref="HostTensor.FillFoldAxis``3"/>
        let FillFoldAxis (trgt: Tensor<'T>) (fn: 'T -> 'TA -> 'T) (initial: Tensor<'T>) (axis: int) (a: Tensor<'TA>) =
            let a, initial = Tensor.PrepareAxisReduceSources (trgt, axis, a, Some initial)
            (backend trgt).FoldLastAxis (fn=fn, initial=initial.Value, trgt=trgt, a=a, useThreads=true) 

        /// <summary>Creates a new tensor with values returned by the specified function using multiple threads.</summary>
        /// <param name="shape">The shape of the new tensor.</param>
        /// <param name="fn">A function that takes the index of the element to fill and returns
        /// the corresponding value.</param>
        /// <seealso cref="HostTensor.FillIndexed``1"/>      
        let init (shape: int64 list) (fn: int64[] -> 'T) : Tensor<'T> =
            let x = Tensor<'T> (shape, Dev)
            FillIndexed x fn
            x          

        /// <summary>Applies to specified function to all elements of the tensor using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="fn">A function that takes a value from the input tensor and returns the corresponding output 
        /// value.</param>        
        /// <param name="a">The source tensor.</param>
        /// <returns>The output tensor.</returns>
        /// <seealso cref="HostTensor.map``2"/>
        let map (fn: 'T -> 'R) (a: Tensor<'T>) =
            let trgt, a = Tensor.PrepareElemwise (a)
            FillMap trgt fn a
            trgt       

        /// <summary>Applies to specified indexed function to all elements of the tensor using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="fn">A function that takes an index and the corresponding value from the input tensor and returns 
        /// the corresponding output value.</param>        
        /// <param name="a">The source tensor.</param>
        /// <returns>The output tensor.</returns>
        /// <seealso cref="HostTensor.map``2"/>
        let mapi (fn: int64[] -> 'T -> 'R) (a: Tensor<'T>) =
            let trgt, a = Tensor.PrepareElemwise (a)
            FillMapIndexed trgt fn a
            trgt      

        /// <summary>Applies to specified function to all elements of the two tensors using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="fn">A function that takes a value from the first input tensor and a value from the second input 
        /// tensor and returns the corresponding output value.</param>        
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        /// <seealso cref="HostTensor.map2``3"/>
        let map2 (fn: 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
            let trgt, a, b = Tensor.PrepareElemwise (a, b)
            FillMap2 trgt fn a b
            trgt           

        /// <summary>Applies to specified indexed function to all elements of the two tensors using multiple 
        /// threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="fn">A function that takes an index, the corresponding value from the first input and second 
        /// input tensor and returns the corresponding output value.</param>        
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        /// <seealso cref="HostTensor.mapi2``3"/>
        let mapi2 (fn: int64[] -> 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
            let trgt, a, b = Tensor.PrepareElemwise (a, b)
            FillMapIndexed2 trgt fn a b
            trgt            

        /// <summary>Applies to specified function to all elements of the tensor, threading an accumulator through 
        /// the computation using multiple threads.</summary>
        /// <typeparam name="'T">The type of the data.</typeparam>
        /// <param name="fn">A function that takes a state value and a value from the input tensor and returns a new 
        /// state value.</param>        
        /// <param name="initial">The initial state value.</param>
        /// <param name="axis">The axis to fold over.</param>
        /// <param name="a">The source tensor.</param>
        /// <returns>The output tensor containg the final states.</returns>
        /// <seealso cref="HostTensor.foldAxis``2"/>
        let foldAxis (fn: 'T -> 'TA -> 'T) (initial: 'T) (axis: int) (a: Tensor<'TA>) =
            let trgt, a = Tensor.PrepareAxisReduceTarget (axis, a)
            FillFoldAxis trgt fn initial axis a
            trgt


