namespace Tensor

open System
open System.Runtime
open System.Threading
open System.Runtime.InteropServices

open ManagedCuda
open ManagedCuda.BasicTypes

open Tensor
open Tensor.Utils
open Tensor.Cuda
open Tensor.Backend
open Tensor.Utils
open DeepNet.Utils



/// <summary>Provides access to nVidia CUDA GPUs.</summary>
module CudaDev =

    /// <summary>Number of CUDA-capable devices.</summary>
    let count = 
        try CudaContext.GetDeviceCount()
        with _ -> 0

    /// <summary>Device properties for all available CUDA-capable devices.</summary>
    let info = [
        for i in 0..count-1 do
            yield CudaContext.GetDeviceInfo i
    ]
    
    /// <summary>TensorCudaDevices for each CUDA-capable device.</summary>
    let private devices: WeakReference<TensorCudaDevice> option [] = Array.create count None

    /// <summary>Returns a tensor device for the specified CUDA-capable device.</summary>
    /// <param name="id">The index of the CUDA device.</param>
    /// <remarks>
    /// <p>This method creates a private CudaContext for library's use.</p>
    /// </remarks>
    /// <returns>A tensor device for the specified CUDA device.</returns>
    let get id =
        if id < 0 || id >= count then
            failwithf "Cannot use CUDA device %d because only %d devices are available."
                id count
        lock devices (fun () ->
            let dev = 
                match devices.[id] with
                | Some weakDev ->
                    match weakDev.TryGetTarget () with
                    | true, dev -> Some dev
                    | _ -> None
                | None -> None
            match dev with
            | Some dev -> dev
            | None ->
                let ctx = new CudaContext (id)
                ctx.PopContext()
                let dev = TensorCudaDevice (ctx, true)
                devices.[id] <- Some (WeakReference<_> dev)
                dev)
        :> ITensorDevice

    /// <summary>Returns a tensor device for the specified CudaContext.</summary>
    /// <returns>A tensor device associated with the specified CUDA context.</returns>
    let forContext (ctx: CudaContext) =
        TensorCudaDevice (ctx, false) :> ITensorDevice



/// <summary>Functions for creating and operating on tensors stored on a nVidia CUDA GPU.</summary>
/// <remarks>
/// <p>This module contains functions for creating tensors stored on a nVidia CUDA GPU.
/// It further contains functions that only work with tensors stored on a nVidia CUDA GPU.
/// Calling these functions with tensors stored on other devices will result in an
/// <see cref="System.InvalidOperationException"/>.</p>
/// <p>The CUDA backend provides options that can be configured via <see cref="Tensor.Cuda.Cfg"/>.</p>
/// </remarks>
/// <example><code language="fsharp">
/// let x = CudaTensor.zeros [3L; 3L]  // x.Dev = CudaTensor.Dev
/// </code></example>
/// <seealso cref="Tensor`1"/><seealso cref="Tensor.Cuda.Cfg"/>
module CudaTensor =

    /// <summary>Tensor device using the default CUDA GPU as data storage.</summary>
    /// <seealso cref="Tensor`1.Dev"/>
    /// <seealso cref="CudaDev"/>
    let Dev = 
        if CudaDev.count > 0 then CudaDev.get 0 
        else failwith "No CUDA-capable device is available."

    /// <summary>Transfers a tensor to the CUDA device.</summary>
    /// <typeparam name="'T">The data type of the tensor.</typeparam>    
    /// <param name="a">The tensor to transfer.</param>
    /// <returns>A tensor on the CUDA device.</returns>
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
  
    /// <summary>Creates a tensor from the given CUDA pointer, allocation size in bytes, type and layout.</summary>
    /// <param name="ptr">A CUDA device pointer.</param>
    /// <param name="sizeInBytes">Size of the allocation referenced by <paramref name="ptr"/> in bytes.</param>
    /// <param name="typ">Type of contained data.</param>
    /// <param name="layout">Layout of the tensor.</param>
    /// <remarks>
    /// <p>This function creates a tensor using existing data in GPU memory.</p>
    /// <p>The data is referenced. Thus changing values within the tensor affects the original data.</p>
    /// </remarks>
    let usingPtrAndType (ptr: CUdeviceptr) (sizeInBytes: SizeT) (typ: Type) (layout: TensorLayout) = 
        let devVarType = typedefof<CudaDeviceVariable<_>>.MakeGenericType [|typ|]
        let devVar = Activator.CreateInstance (devVarType, [|box ptr; box sizeInBytes|])

        let devStorType = typedefof<TensorCudaStorage<_>>.MakeGenericType [|typ|]
        let devStor = Activator.CreateInstance (devStorType, [|devVar|])

        let tensorType = typedefof<Tensor<_>>.MakeGenericType [|typ|]
        Activator.CreateInstance (tensorType, [|box layout; devStor|]) :?> ITensor

    /// <summary>Creates a tensor from the given CUDA pointer, allocation size in bytes and layout.</summary>
    /// <typeparam name="'T">Type of contained data.</typeparam>
    /// <param name="ptr">A CUDA device pointer.</param>
    /// <param name="sizeInBytes">Size of the allocation referenced by <paramref name="ptr"/> in bytes.</param>
    /// <param name="layout">Layout of the tensor.</param>
    /// <remarks>
    /// <p>This function creates a tensor using existing data in GPU memory.</p>
    /// <p>The data is referenced. Thus changing values within the tensor affects the original data.</p>
    /// </remarks>
    let usingPtr<'T> (ptr: CUdeviceptr) (sizeInBytes: SizeT) (layout: TensorLayout) =
        usingPtrAndType ptr sizeInBytes typeof<'T> layout :?> Tensor<'T>

        