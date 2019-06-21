namespace Tensor

open System
open System.Collections.Generic

open DeepNet.Utils
open Tensor.Backend
open Tensor.Host



/// <summary>Functions for creating and operating on tensors stored in native host memory.</summary>
/// <remarks>
/// <p>This module contains functions for creating tensors stored in native host memory.
/// Contrary to a tensor in native memory, this allows using reference types and value types that
/// contain references.</p>
/// <p>Only tensor creation functions are provided in this module. For other purposes use functions from the
/// <see cref="HostTensor"/> module.</p>
/// </remarks>
/// <seealso cref="Tensor`1"/><seealso cref="HostTensor"/>
module ManagedHostTensor =
    
    /// <summary>Tensor device using a .NET array in host memory as data storage.</summary>
    /// <seealso cref="HostTensor.Dev"/><seealso cref="Tensor`1.Dev"/>    
    let Dev = TensorHostManagedDevice.Instance :> ITensorDevice
    
    /// <summary>Creates a new tensor with values returned by the specified function.</summary>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <param name="fn">A function that takes the index of the element to fill and returns
    /// the corresponding value.</param>
    /// <seealso cref="FillIndexed``1"/><seealso cref="HostTensor.Parallel.init``1"/>  
    let init (shape: int64 list) (fn: int64[] -> 'T) : Tensor<'T> =
        let x = Tensor<'T> (shape, Dev)
        HostTensor.FillIndexed x fn
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
    /// <typeparam name="'T">The data type of the new tensor.</typeparam>
    /// <param name="start">The starting value.</param>
    /// <param name="incr">The increment between successive element.</param>   
    /// <param name="stop">The end value, which is not included.</param>
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.arange``3"/>
    let arange (start: 'T) (incr: 'T) (stop: 'T) = Tensor.arange Dev start incr stop

    /// <summary>Creates a new vector of given size filled with equaly spaced values.</summary>
    /// <typeparam name="'T">The data type of the new tensor.</typeparam>
    /// <param name="start">The starting value.</param>
    /// <param name="stop">The end value, which is not included.</param>
    /// <param name="nElems">The size of the vector.</param>   
    /// <returns>The new tensor.</returns>
    /// <seealso cref="Tensor`1.linspace``2"/>
    let linspace (start: 'T) (stop: 'T) nElems = Tensor.linspace Dev start stop nElems
  
    /// <summary>Creates a one-dimensional tensor referencing the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor using the array <c>data</c> as its storage.</returns>
    /// <remarks>The data array is referenced, not copied.
    /// Thus changing the tensor modifies the specified data array and vice versa.</remarks>
    /// <seealso cref="ofArray``1"/>
    let usingArray (data: 'T []) =
        let shp = [data.LongLength]
        let layout = TensorLayout.newRowMajor shp
        Tensor<'T> (layout, TensorHostStorage<'T>.make data) 

    /// <summary>Creates a one-dimensional tensor copying the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor filled with the values from <c>data</c>.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="usingArray``1"/>
    /// <seealso cref="ofArray2D``1"/><seealso cref="ofArray3D``1"/><seealso cref="ofArray4D``1"/>
    let ofArray data = HostTensor.ofArrayDev Dev data

    /// <summary>Creates a two-dimensional tensor copying the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor using filled with the values from <c>data</c>.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="ofArray``1"/><seealso cref="ofArray3D``1"/><seealso cref="ofArray4D``1"/>
    let ofArray2D data = HostTensor.ofArray2DDev Dev data

    /// <summary>Creates a three-dimensional tensor copying the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor using filled with the values from <c>data</c>.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="ofArray``1"/><seealso cref="ofArray2D``1"/><seealso cref="ofArray4D``1"/>
    let ofArray3D data = HostTensor.ofArray3DDev Dev data

    /// <summary>Creates a four-dimensional tensor copying the specified data.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data array to use.</param>
    /// <returns>A tensor using filled with the values from <c>data</c>.</returns>
    /// <remarks>The data is copied.</remarks>
    /// <seealso cref="ofArray``1"/><seealso cref="ofArray2D``1"/><seealso cref="ofArray3D``1"/>
    let ofArray4D data = HostTensor.ofArray4DDev Dev data

    /// <summary>Creates a one-dimensional tensor from the specified sequence.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data to fill the tensor with.</param>
    /// <returns>A tensor containing values from the specifed sequence.</returns>
    /// <remarks>The sequence must be finite.</remarks>
    let ofSeq (data: 'T seq) =
        data |> Array.ofSeq |> usingArray

    /// <summary>Creates a one-dimensional Tensor using the specified sequence and shape.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <param name="data">The data to fill the tensor with.</param>
    /// <returns>A tensor containing values from the specifed sequence.</returns>
    /// <remarks>Only the number of elements required to fill the tensor of the specified
    /// shape are consumed from the sequence. Thus it may be infinite.</remarks>
    let ofSeqWithShape shape (data: 'T seq) =
        let nElems = shape |> List.fold (*) 1L
        data |> Seq.take (int32 nElems) |> ofSeq |> Tensor.reshape shape
        
    /// <summary>Creates a one-dimensional tensor from the specified list.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data to fill the tensor with.</param>
    /// <returns>A tensor containing values from the specifed list.</returns>
    let ofList (data: 'T list) =
        data |> Array.ofList |> usingArray

    /// <summary>Creates a two-dimensional tensor from the specified list of lists.</summary>
    /// <typeparam name="'T">The type of the data.</typeparam>
    /// <param name="data">The data to fill the tensor with.</param>
    /// <returns>A tensor containing values from the specifed lists.</returns>
    let ofList2D (data: 'T list list) =
        data |> array2D |> ofArray2D
                                
    /// <summary>Reads a tensor from the specified HDF5 object path in an HDF5 file.</summary>
    /// <typeparam name="'T">The type of the data. This must match the type of the data stored in the
    /// HDF5 file.</typeparam>
    /// <param name="hdf5">The HDF5 file.</param>
    /// <param name="path">The HDF5 object path.</param>
    /// <returns>A tensor filled with data read from the HDF5 file.</returns>
    /// <exception cref="System.InvalidOperationException">The data type stored in the HDF5 does 
    /// not match type <c>'T</c>.</exception>
    /// <example><code language="fsharp">
    /// use hdfFile = HDF5.OpenRead "tensors.h5"
    /// let k = HostTensor.read&lt;float&gt; hdfFile "k"
    /// </code></example>    
    let read<'T> hdf5 path = HostTensor.readDev<'T> Dev hdf5 path
    
    /// <summary>Reads a tensor with unspecified data type from the specified HDF5 object path in an HDF5 file.</summary>
    /// <param name="hdf5">The HDF5 file.</param>
    /// <param name="path">The HDF5 object path.</param>
    /// <returns>A tensor filled with data read from the HDF5 file.</returns>
    /// <seealso cref="read``1"/>
    let readUntyped hdf5 path = HostTensor.readUntypedDev Dev hdf5 path    
       