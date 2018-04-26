namespace rec Tensor

open System
open System.Collections
open System.Collections.Generic
open System.Diagnostics

open Tensor.Utils
open Tensor.Backend



/// <summary>A singular matrix was encountered during an operation that does not allow singular matrices.</summary>
/// <param name="msg">Detailed error message.</param>
/// <remarks>
/// See the documentation of the method that raised this exception for a detailed description of the error conditions.
/// </remarks>
exception SingularMatrixException of msg:string with 
    /// <summary>Detailed error message.</summary>    
    override __.Message = __.msg



/// <summary>Block tensor specification.</summary>
/// <typeparam name="'T">The type of the data stored within the tensor.</typeparam>
/// <remarks>See <see cref="Tensor`1.ofBlocks"/> for usage information.</remarks>
/// <seealso cref="Tensor`1.ofBlocks"/>
type BlockTensor<'T> =
    /// A block consisting of multiple sub-blocks.
    | SubBlocks of BlockTensor<'T> list
    /// A block consisting of a single tensor.
    | Block of Tensor<'T> 



/// <summary>An N-dimensional array with elements of type 'T.</summary>
/// <typeparam name="'T">The type of the data stored within the tensor.</typeparam>
/// <param name="layout">The memory layout to use.</param>
/// <param name="storage">The storage to use.</param>
/// <returns>A tensor using the specified memory layout and storage.</returns>
/// <remarks>
/// <para>The data of a tensor can be stored on different devices. Currently supported devices are host memory
/// and CUDA GPU memory.</para>
/// <para>Different tensors can share the whole or parts of the underlying data.</para>
/// <para>The recommended way to create a new tensor is to use <see cref="zeros"/>.
/// The implicit constructor creates a view into the specified storage using the specified memory layout.
/// In most cases, it is not necessary to use the implicit constructor.</para>
/// </remarks> 
/// <seealso cref="ITensor"/> 
type [<StructuredFormatDisplay("{Pretty}"); DebuggerDisplay("{Shape}-Tensor: {Pretty}")>] 
        Tensor<'T> (layout: TensorLayout, storage: ITensorStorage<'T>) =

    do TensorLayout.check layout
    let backend = storage.Backend layout

    /// <summary>Memory layout of this tensor.</summary>
    /// <value>Memory layout.</value>
    /// <remarks>Provides information of how the data is stored within this tensor.</remarks>
    /// <seealso cref="Storage"/><seealso cref="Shape"/>
    member val Layout = layout

    /// <summary>Memory layout of the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Memory layout.</returns>
    /// <seealso cref="Layout"/>
    static member inline layout (a: Tensor<'T>) = a.Layout

    /// <summary>The storage object that holds the data of this tensor.</summary>
    /// <value>Storage object.</value>
    /// <remarks>
    /// <para>The storage object holds the actual data of the tensor.
    /// A storage object can be associated with one or more tensors, i.e. it can be shared between multiple tensors.
    /// Sharing occurs, for example, when a view into an existing tensor is created or the tensor is reshapred.</para>
    /// <para>The actual type of the storage object depends on the device the data of the tensor is stored on.</para>
    /// <para>For tensors stored in host memory the storage object type is <see cref="Tensor.Host.TensorHostStorage`1"/>.</para>
    /// <para>For tensors stored on a CUDA GPU the storage object type is <see cref="Tensor.Cuda.TensorCudaStorage`1"/>.</para>
    /// </remarks>
    /// <seealso cref="Dev"/><seealso cref="Layout"/>
    member val Storage = storage

    /// <summary>Device the data of tensor is stored on.</summary>
    /// <value>Data storage device.</value>
    /// <remarks>
    /// <para>For tensors stored in host memory the value of this property is <see cref="HostTensor.Dev"/>.</para>
    /// <para>For tensors stored on a CUDA GPU the value of this property is <see cref="CudaTensor.Dev"/>.</para>
    /// </remarks>
    /// <seealso cref="Storage"/>
    member inline this.Dev = this.Storage.Dev

    /// <summary>Device the data of tensor is stored on.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Data storage device.</returns>
    /// <seealso cref="Dev"/><seealso cref="transfer``1"/>
    static member inline dev (a: Tensor<'T>) = a.Dev

    /// backend 
    member internal this.Backend = backend

    /// <summary>Shape of this tensor.</summary>
    /// <value>Shape.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0; 5.0]
    ///                            [3.0; 4.0; 6.0]]
    /// let c = a.Shape // [2L; 3L]
    /// </code></example>
    /// <remarks>
    /// <para>Provides the shape of this tensor.</para>
    /// <para>A tensor is empty of any dimension has size zero.</para>
    /// <para>A zero-dimensional tensor has an empty shape and contains one element.</para>
    /// </remarks>
    /// <seealso cref="reshape"/><seealso cref="NDims"/><seealso cref="NElems"/>
    member inline this.Shape = this.Layout.Shape

    /// <summary>Shape of the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Shape.</returns>
    /// <seealso cref="Shape"/>
    static member inline shape (a: Tensor<'T>) = a.Shape

    /// <summary>Dimensionality of this tensor.</summary>
    /// <value>Number of dimensions.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0; 5.0]
    ///                            [3.0; 4.0; 6.0]]
    /// let c = a.NDims // 2
    /// </code></example>
    /// <remarks>
    /// <para>Provides the number of dimensions of this tensor.</para>
    /// <para>A zero-dimensional tensor contains one element, i.e. it is a scalar.</para>
    /// </remarks>
    /// <seealso cref="Shape"/>
    member inline this.NDims = this.Layout.NDims

    /// <summary>Dimensionality of the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Number of dimensions.</returns>
    /// <seealso cref="NDims"/>
    static member inline nDims (a: Tensor<'T>) = a.NDims

    /// <summary>Total number of elements within this tensor.</summary>
    /// <value>Number of elements.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0; 5.0]
    ///                            [3.0; 4.0; 6.0]]
    /// let c = a.NElems // 6L
    /// </code></example>
    /// <remarks>
    /// <para>Counts the total number of elements of this tensor.</para>
    /// <para>A zero-dimensional tensor contains one element, i.e. it is a scalar.</para>
    /// </remarks>
    /// <seealso cref="Shape"/>
    member inline this.NElems = this.Layout.NElems

    /// <summary>Total number of elements within the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Number of elements.</returns>
    /// <seealso cref="NElems"/>
    static member inline nElems (a: Tensor<'T>) = a.NElems

    /// <summary>Type of data stored within this tensor.</summary>
    /// <value>Data type.</value>
    /// <remarks>
    /// <para>The data type is <c>typeof&lt;'T&gt;</c>.</para>
    /// </remarks>
    /// <seealso cref="convert``1"/>
    member inline this.DataType = typeof<'T>

    /// <summary>Type of data stored within the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Data type.</returns>
    /// <seealso cref="DataType"/>
    static member inline dataType (a: Tensor<'T>) = a.DataType

    /// a tensor with the same storage but new layout
    member internal this.Relayout (newLayout: TensorLayout) =
        Tensor<'T> (newLayout, storage)

    /// <summary>Creates a tensor with the specified layout sharing its storage with the original tensor.</summary>
    /// <param name="newLayout">The new tensor memory layout.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    static member relayout newLayout (a: Tensor<'T>) =
        a.Relayout newLayout 

    /// a view of this tensor over the given range 
    member internal this.Range (rng: Rng list) =
        this.Relayout (this.Layout |> TensorLayout.view rng)

    /// <summary>Get a slice (part) of the tensor.</summary>
    /// <param name="rng">The range of the tensor to select.</param>    
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Item(Microsoft.FSharp.Collections.FSharpList{Tensor.Rng})"/>
    static member range (rng: Rng list) (a: Tensor<'T>) = ITensor.range rng a :?> Tensor<'T>
   
    /// <summary>Checks the the specified axis is valid for this tensor.</summary>   
    /// <param name="ax">The axis number to check.</param>
    /// <remarks>If the axis is valid, this function does nothing.</remarks>
    /// <exception cref="System.IndexOutOfRangeException">Raised when the axis is invalid.</exception>
    member inline this.CheckAxis ax = this.Layout |> TensorLayout.checkAxis ax

    /// <summary>Gets a sequence of all indices to enumerate all elements within the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Sequence of indicies.</returns>
    /// <remarks>The sequence sequentially enumerates the indices of all elements of the tensor.</remarks>
    /// <seealso cref="allIdxOfDim"/><seealso cref="allElems"/>
    static member allIdx (a: Tensor<'T>) = ITensor.allIdx a

    /// <summary>Gets a sequence of all all elements within the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Sequence of elements.</returns>
    /// <remarks>The sequence sequentially enumerates all elements of the tensor.</remarks>
    /// <seealso cref="allIdx"/>
    static member allElems (a: Tensor<'T>) = a |> Tensor<_>.allIdx |> Seq.map (fun idx -> a.[idx])
    
    /// <summary>Insert a dimension of size one as the first dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 4L; 5L]
    /// let b = Tensor.padLeft a // b.Shape = [1L; 3L; 4L; 5L]
    /// </code></example>    
    /// <remarks>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="padRight"/><seealso cref="insertAxis"/>
    static member padLeft (a: Tensor<'T>) = ITensor.padLeft a :?> Tensor<'T>

    /// <summary>Append a dimension of size one after the last dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 4L; 5L]
    /// let b = Tensor.padRight a // b.Shape = [3L; 4L; 5L; 1L]
    /// </code></example>    
    /// <remarks>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="padLeft"/><seealso cref="insertAxis"/>
    static member padRight (a: Tensor<'T>) = ITensor.padRight a :?> Tensor<'T>

    /// <summary>Insert a dimension of size one before the specifed dimension.</summary>
    /// <param name="ax">The dimension to insert before.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 4L; 5L]
    /// let b = Tensor.insertAxis 1 a // b.Shape = [3L; 1L 4L; 5L]
    /// </code></example>    
    /// <remarks>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="padLeft"/><seealso cref="padRight"/>
    static member insertAxis ax (a: Tensor<'T>) = ITensor.insertAxis ax a :?> Tensor<'T>

    /// <summary>Removes the first dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 4L; 5L]
    /// let b = Tensor.cutLeft a // b.Shape = [4L; 5L]
    /// </code></example>    
    /// <remarks>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="cutRight"/>
    static member cutLeft (a: Tensor<'T>) = ITensor.cutLeft a :?> Tensor<'T>
      
    /// <summary>Removes the last dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 4L; 5L]
    /// let b = Tensor.cutRight a // b.Shape = [3L; 4L]
    /// </code></example>    
    /// <remarks>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="cutLeft"/>
    static member cutRight (a: Tensor<'T>) = ITensor.cutRight a :?> Tensor<'T>

    /// <summary>Broadcast a dimension to a specified size.</summary>
    /// <param name="dim">The size-one dimension to broadcast.</param>
    /// <param name="size">The size to broadcast to.</param>    
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 1L; 5L]
    /// let b = Tensor.broadCastDim 1 9L a // b.Shape = [3L; 9L; 5L]
    /// </code></example>    
    /// <remarks>
    /// <para>The broadcasted dimension must be of size one. The tensor is repeated <paramref name="size"/> times along
    /// the axis <paramref name="dim"/>.</para>
    /// <para>Broadcasting is usually performed automatically when the shapes allow for it. See broadcasting rules
    /// for details.</para>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="insertAxis"/>
    static member broadcastDim dim size (a: Tensor<'T>) = ITensor.broadcastDim dim size a :?> Tensor<'T>

    /// <summary>Creates a new, uninitialized tensor with a new storage.</summary>
    /// <param name="shape">The shape of the tensor to create.</param>
    /// <param name="dev">The device to store the data of the tensor on.</param>
    /// <param name="order">The memory layout to use for the new tensor. (default: row-major)</param>
    /// <returns>The new, uninitialized tensor.</returns>
    /// <remarks>
    /// <para>The contents of the new tensor are undefined. The default memory layout is row-major.</para>
    /// <para>The recommended way to create a new tensor is to use <see cref="zeros"/>.</para>
    /// </remarks>
    /// <seealso cref="NewOfType"/><seealso cref="zeros"/>
    new (shape: int64 list, dev: ITensorDevice, ?order: TensorOrder) =
        let order = defaultArg order RowMajor
        let layout = 
            match order with
            | RowMajor -> TensorLayout.newC shape
            | ColumnMajor -> TensorLayout.newF shape
            | CustomOrder perm -> TensorLayout.newOrdered shape perm
        let storage = dev.Create layout.NElems
        Tensor<'T> (layout, storage)

    /// Applies the given function to the tensors' layouts.
    static member inline internal ApplyLayoutFn (fn, a: Tensor<'TA>, b: Tensor<'TB>) =
        let layouts = [Tensor<_>.layout a; Tensor<_>.layout b]
        let newLayouts = fn layouts
        match newLayouts with
        | [al; bl] -> 
            Tensor<_>.relayout al a, Tensor<_>.relayout bl b
        | _ -> failwith "unexpected layout function result"

    /// Applies the given function to the tensors' layouts.
    static member inline internal ApplyLayoutFn (fn, a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>) =
        let layouts = [Tensor<_>.layout a; Tensor<_>.layout b; Tensor<_>.layout c]
        let newLayouts = fn layouts
        match newLayouts with
        | [al; bl; cl] -> 
            Tensor<_>.relayout al a, Tensor<_>.relayout bl b, Tensor<_>.relayout cl c
        | _ -> failwith "unexpected layout function result"

    /// Applies the given function to the tensors' layouts.
    static member inline internal ApplyLayoutFn (fn, xs: Tensor<'T> list) =
        let layouts = fn (xs |> List.map Tensor<_>.layout)
        (layouts, xs) ||> List.map2 Tensor<_>.relayout

    /// <summary>Pads all specified tensors from the left with dimensions of size one until they have the 
    /// same dimensionality.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <param name="b">The tensor to operate on.</param>    
    /// <returns>A tuple of the resulting tensors, all having the same dimensionality.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [4L; 5L]
    /// let b = HostTensor.zeros [3L; 4L; 5L]
    /// let pa, pb = Tensor.padToSame (a, b) // pa.Shape = [1L; 4L; 5L]; pb.Shape = [3L; 4L; 5L]
    /// </code></example>    
    /// <remarks>
    /// <para>Size one dimensions are added from the left to each tensor until all of them have the same 
    /// dimensionality.</para>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="padLeft"/><seealso cref="broadcastToSame"/>
    static member padToSame (a: Tensor<'TA>, b: Tensor<'TB>) = 
        Tensor<_>.ApplyLayoutFn (TensorLayout.padToSameMany, a, b)

    /// <summary>Pads all specified tensors from the left with dimensions of size one until they have the 
    /// same dimensionality.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <param name="b">The tensor to operate on.</param>    
    /// <param name="c">The tensor to operate on.</param>    
    /// <returns>A tuple of the resulting tensors, all having the same dimensionality.</returns>
    /// <seealso cref="padToSame``2"/>
    static member padToSame (a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>) = 
        Tensor<_>.ApplyLayoutFn (TensorLayout.padToSameMany, a, b, c)

    /// <summary>Pads all specified tensors from the left with dimensions of size one until they have the 
    /// same dimensionality.</summary>
    /// <param name="xs">A list of tensors to operate on.</param>
    /// <returns>A list of the resulting tensors, all having the same dimensionality.</returns>
    /// <seealso cref="padToSame``2"/>
    static member padToSame (xs: Tensor<'T> list) = 
        Tensor<_>.ApplyLayoutFn (TensorLayout.padToSameMany, xs)

    /// <summary>Broadcasts all specified tensors to have the same shape.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <param name="b">The tensor to operate on.</param>    
    /// <returns>A tuple of the resulting tensors, all having the same shape.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [4L; 5L]
    /// let b = HostTensor.zeros [3L; 4L; 5L]
    /// let pa, pb = Tensor.broadcastToSame (a, b) // pa.Shape = [3L; 4L; 5L]; pb.Shape = [3L; 4L; 5L]
    /// </code></example>    
    /// <remarks>
    /// <para>First, size one dimensions are added from the left to each tensor until all of them have the same 
    /// dimensionality. Then, size one dimensions are broadcasted to match the size of non-size-one dimensions.</para>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <exception cref="System.InvalidOperationException">Raised when broadcasting to a common shape is impossible.</exception>
    /// <seealso cref="padToSame``2"/><seealso cref="broadcastToSameInDims``2"/><seealso cref="broadcastTo"/>
    static member broadcastToSame (a: Tensor<'TA>, b: Tensor<'TB>) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameMany, a, b)

    /// <summary>Broadcasts all specified tensors to have the same shape.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <param name="b">The tensor to operate on.</param>    
    /// <param name="c">The tensor to operate on.</param>    
    /// <returns>A tuple of the resulting tensors, all having the same shape.</returns>
    /// <seealso cref="broadcastToSame``2"/>
    static member broadcastToSame (a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameMany, a, b, c)

    /// <summary>Broadcasts all specified tensors to have the same shape.</summary>
    /// <param name="xs">A list of tensors to operate on.</param>    
    /// <returns>A list of the resulting tensors, all having the same shape.</returns>
    /// <seealso cref="broadcastToSame``2"/>
    static member broadcastToSame (xs: Tensor<'T> list) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameMany, xs)

    /// <summary>Broadcasts all specified tensors to have the same size in the specified dimensions.</summary>
    /// <param name="dims">A list of dimensions that should be broadcasted to have the same size.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <param name="b">The tensor to operate on.</param>    
    /// <returns>A tuple of the resulting tensors, all having the same size in the specified dimensions.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [1L; 7L; 1L]
    /// let b = HostTensor.zeros [3L; 4L; 5L]
    /// let pa, pb = Tensor.broadcastToSameInDims ([0; 2], a, b) // pa.Shape = [3L; 7L; 5L]; pb.Shape = [3L; 4L; 5L]
    /// </code></example>    
    /// <remarks>
    /// <para>The specified dimensions are broadcasted to match the size of non-size-one dimensions.</para>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <exception cref="System.InvalidOperationException">Raised when broadcasting to a common shape is impossible.</exception>
    /// <seealso cref="broadcastToSame``2"/><seealso cref="broadcastTo"/>
    static member broadcastToSameInDims (dims, a: Tensor<'TA>, b: Tensor<'TB>) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameInDimsMany dims, a, b)

    /// <summary>Broadcasts all specified tensors to have the same size in the specified dimensions.</summary>
    /// <param name="dims">A list of dimensions that should be broadcasted to have the same size.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <param name="b">The tensor to operate on.</param>    
    /// <param name="c">The tensor to operate on.</param>    
    /// <returns>A tuple of the resulting tensors, all having the same size in the specified dimensions.</returns>
    /// <seealso cref="broadcastToSameInDims``2"/>
    static member broadcastToSameInDims (dims, a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameInDimsMany dims, a, b, c)

    /// <summary>Broadcasts all specified tensors to have the same size in the specified dimensions.</summary>
    /// <param name="dims">A list of dimensions that should be broadcasted to have the same size.</param>
    /// <param name="xs">A list of tensors to operate on.</param>
    /// <returns>A list of the resulting tensors, all having the same size in the specified dimensions.</returns>
    /// <seealso cref="broadcastToSameInDims``2"/>
    static member broadcastToSameInDims (dims, xs: Tensor<'T> list) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameInDimsMany dims, xs)

    /// <summary>Broadcasts the specified tensor to the specified shape.</summary>
    /// <param name="shp">The target shape.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Tensor of shape <paramref name="shp"/>.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [1L; 7L; 1L]
    /// let pa = Tensor.broadcastTo [2L; 7L; 3L] a // pa.Shape = [2L; 7L; 3L]
    /// </code></example>    
    /// <remarks>
    /// <para>Size one dimensions are broadcasted to match the corresponding dimension of the target shape 
    /// <paramref name="shp"/>. Non-size-one dimensions must match the target shape.</para>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <exception cref="System.InvalidOperationException">Raised when broadcasting to the specified shape is impossible.</exception>
    /// <seealso cref="broadcastToSame``2"/>
    static member broadcastTo shp (a: Tensor<'T>) = ITensor.broadcastTo shp a :?> Tensor<'T>

    /// <summary>Checks if the specified tensor is broadcasted in at least one dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>true if at least one dimension is broadcasted, otherwise false.</returns>
    /// <remarks>
    /// <para>If any stride is zero, it is assumed that the tensor is broadcasted.
    /// If this is the case, changing an element of the tensor may change other elements as well.</para>    
    /// </remarks>
    /// <seealso cref="broadcastToSame``2"/><seealso cref="broadcastTo"/>
    static member isBroadcasted (a: Tensor<'T>) = ITensor.isBroadcasted a

    /// <summary>Tries to create a reshaped view of the tensor (without copying).</summary>
    /// <param name="shp">The target shape.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The reshaped tensor, if reshaping without copying is possible. Otherwise <c>None</c>.</returns>
    /// <remarks>
    /// <para>Changes the shape of the tensor to the specified shape.
    /// The total number of elements must not change.
    /// One dimension of the <paramref name="shp"/> can be specified as <see cref="Tensor.Remainder"/>, 
    /// in which case the size of that dimension is inferred automatically.</para>
    /// <para>If a reshape is not possible without copying the data of the tensor, <c>None</c> is returned.</para>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="reshapeView"/><seealso cref="reshape"/>
    static member tryReshapeView shp (a: Tensor<'T>) =
        ITensor.tryReshapeView shp a |> Option.map (fun r -> r :?> Tensor<'T>)

    /// <summary>Creates a reshaped view of the tensor (without copying).</summary>
    /// <param name="shp">The target shape.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A reshaped view of the original tensor.</returns>
    /// <remarks>
    /// <para>Changes the shape of the tensor to the specified shape.
    /// The total number of elements must not change.
    /// One dimension of the <paramref name="shp"/> can be specified as <see cref="Tensor.Remainder"/>, 
    /// in which case the size of that dimension is inferred automatically.</para>
    /// <para>If a reshape is not possible without copying the data of the tensor, an exception is raised.
    /// To avoid this, use <see cref="tryReshapeView"/> instead.</para>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="tryReshapeView"/><seealso cref="reshape"/>
    static member reshapeView shp (a: Tensor<'T>) = ITensor.reshapeView shp a :?> Tensor<'T>

    /// <summary>Changes the shape of a tensor.</summary>
    /// <param name="shp">The target shape.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor of the specified shape.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [2L; 3L; 4L]
    /// let b = Tensor.reshape [6L; 4L] a // b.Shape = [6L; 4L]
    /// let c = Tensor.reshape [2L; Remainder; 1L] a // c.Shape = [2L; 12L; 1L]
    /// </code></example>    
    /// <remarks>
    /// <para>Changes the shape of the tensor to the specified shape.
    /// The total number of elements must not change.
    /// One dimension of the <paramref name="shp"/> can be specified as <see cref="Tensor.Remainder"/>, 
    /// in which case the size of that dimension is inferred automatically.</para>
    /// <para>If a reshape is possible without copying the data of the tensor, a view of the original tensor is returned
    /// and the storage is shared. In this case, modifications done to the returned tensor will affect the original 
    /// tensor.</para>
    /// <para>If a reshape is not possible without copying the data of the tensor, a new tensor of the specified shape
    /// and a new storage is allocated and the data is copied into the new tensor.</para>
    /// </remarks>
    /// <seealso cref="tryReshapeView"/><seealso cref="reshapeView"/><seealso cref="flatten"/><seealso cref="Shape"/>
    static member reshape shp (a: Tensor<'T>) = ITensor.reshape shp a :?> Tensor<'T>

    /// <summary>Flattens the tensor into a (one-dimensional) vector.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A vector.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [2L; 3L; 4L]
    /// let b = Tensor.flatten a // b.Shape = [24L]
    /// </code></example>    
    /// <remarks>    
    /// <para>If a reshape is possible without copying the data of the tensor, a view of the original tensor is returned
    /// and the storage is shared. In this case, modifications done to the returned tensor will affect the original 
    /// tensor.</para>
    /// </remarks>    
    /// <seealso cref="reshape"/>
    static member flatten (a: Tensor<'T>) = ITensor.flatten a :?> Tensor<'T>

    /// <summary>Swaps the specified dimensions of the tensor.</summary>
    /// <param name="ax1">The dimension to swap.</param>
    /// <param name="ax2">The dimension to swap with.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The tensor with the dimensions swapped.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [2L; 3L; 4L]
    /// let b = Tensor.swapDim 0 2 a // b.Shape = [4L; 3L; 2L]
    /// </code></example>    
    /// <remarks>    
    /// <para>A view of the original tensor is returned and the storage is shared. Modifications done to the returned 
    /// tensor will affect the original tensor.</para>
    /// </remarks>    
    /// <seealso cref="permuteAxes"/><seealso cref="T"/>
    static member swapDim ax1 ax2 (a: Tensor<'T>) = ITensor.swapDim ax1 ax2 a :?> Tensor<'T>

    /// <summary>(Batched) transpose of a matrix.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The result of this operation.</returns>
    /// <seealso cref="T"/>
    static member transpose (a: Tensor<'T>) = ITensor.transpose a :?> Tensor<'T>

    /// <summary>Permutes the axes as specified.</summary>
    /// <param name="permut">The permutation to apply to the dimensions of tensor.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The tensor with the dimensions permuted.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [0L; 11L; 22L; 33L; 44L]
    /// let b = Tensor.permuteAxes [3; 2; 4; 1; 0] a // b.Shape = [44L; 33L; 11L; 0L; 22L]
    /// </code></example>    
    /// <remarks>    
    /// <para>Each entry in the specified permutation specifies the new position of the corresponding axis, i.e. to 
    /// which position the axis moves.</para>
    /// <para>A view of the original tensor is returned and the storage is shared. Modifications done to the returned 
    /// tensor will affect the original tensor.</para>
    /// </remarks>    
    /// <seealso cref="swapDim"/><seealso cref="T"/>
    static member permuteAxes (permut: int list) (a: Tensor<'T>) = ITensor.permuteAxes permut a :?> Tensor<'T>

    /// <summary>Reverses the elements in the specified dimension.</summary>
    /// <param name="ax">The axis to reverse.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The tensor with the dimensions permuted.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [0; 1; 2; 3]
    /// let b = Tensor.reverseAxis 0 a // b = [3; 2; 1; 0]
    /// </code></example>    
    /// <remarks>    
    /// <para>The elements along the specified axis are reversed.</para>
    /// <para>A view of the original tensor is returned and the storage is shared. Modifications done to the returned 
    /// tensor will affect the original tensor.</para>
    /// </remarks>    
    static member reverseAxis ax (a: Tensor<'T>) = ITensor.reverseAxis ax a :?> Tensor<'T>

    /// <summary>Pads the tensor from the left with size-one dimensions until it has at least the specified number of
    /// dimensions.</summary>
    /// <param name="minDims">The minimum number of dimensions.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor with at least <paramref name="minDims"/> dimensions.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [2L; 3L]
    /// let b = Tensor.atLeastND 5 a // b.Shape = [1L; 1L; 1L; 2L; 3L]
    /// </code></example>    
    /// <remarks>    
    /// <para>Size-one dimensions are inserted at the front until the tensor has at least the specified number of 
    /// dimensions. If it already has the specified number of dimensions or more, it is returned unchanged.</para>
    /// <para>A view of the original tensor is returned and the storage is shared. Modifications done to the returned 
    /// tensor will affect the original tensor.</para>
    /// </remarks>    
    /// <seealso cref="padLeft"/><seealso cref="reshape"/>    
    static member atLeastND minDims (a: Tensor<'T>) = ITensor.atLeastND minDims a :?> Tensor<'T>

    /// <summary>Pads the tensor from the left with size-one dimensions until it has at least one dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor with at least one dimensions.</returns>
    /// <seealso cref="atLeastND"/>
    static member atLeast1D (a: Tensor<'T>) = a |> Tensor<_>.atLeastND 1

    /// <summary>Pads the tensor from the left with size-one dimensions until it has at least two dimensions.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor with at least two dimensions.</returns>
    /// <seealso cref="atLeastND"/>
    static member atLeast2D (a: Tensor<'T>) = a |> Tensor<_>.atLeastND 2

    /// <summary>Pads the tensor from the left with size-one dimensions until it has at least three dimensions.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor with at least three dimensions.</returns>
    /// <seealso cref="atLeastND"/>
    static member atLeast3D (a: Tensor<'T>) = a |> Tensor<_>.atLeastND 3

    /// <summary>Transpose of a matrix.</summary>
    /// <value>The transposed matrx.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 5L]
    /// let b = a.T // b.Shape = [5L; 3L]
    /// </code></example>    
    /// <remarks>
    /// <para>If the given tensor has more then two dimensions, the last two axes are swapped.</para>
    /// <para>The operation returns a view of the original tensor and shares its storage. Modifications done to the
    /// returned tensor will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// </remarks>
    /// <seealso cref="permuteAxes"/><seealso cref="swapDim"/>    
    member inline this.T = 
        Tensor<_>.transpose this

    /// Returns a copy of the tensor.
    member internal this.Copy (?order) =
        let trgt, src = Tensor.PrepareElemwise (this, ?order=order)
        trgt.Backend.Copy (trgt=trgt, src=src)
        trgt      
        
    /// <summary>Returns a copy of the tensor.</summary>
    /// <param name="a">The tensor to copy.</param>
    /// <param name="order">The memory layout of the copy. (default: row-major)</param>
    /// <returns>A copy of the tensor.</returns>
    /// <remarks>    
    /// <para>A new tensor is created with the specified memory layout on the same device as the orignal tensor.</para>
    /// <para>The elements of the original tensor are copied into the new tensor.</para>
    /// </remarks>    
    /// <seealso cref="CopyFrom"/><seealso cref="transfer``1"/>
    static member copy (a: Tensor<'T>, ?order) : Tensor<'T> =
        a.Copy (?order=order) 

    /// <summary>Fills this tensor with a copy of the specified tensor.</summary>
    /// <param name="src">The tensor to copy from.</param>
    /// <remarks>
    /// <para>The source tensor must have the same shape and be stored on the same device as this tensor.</para>
    /// </remarks>
    /// <seealso cref="copy"/><seealso cref="FillFrom"/>
    member trgt.CopyFrom (src: Tensor<'T>) =
        Tensor.CheckSameShape trgt src
        Tensor.CheckSameStorage [trgt; src]
        trgt.Backend.Copy (trgt=trgt, src=src)

    /// <summary>Transfers the specified tensor located on another device into this tensor.</summary>
    /// <param name="src">The tensor to transfer from.</param>
    /// <remarks>    
    /// <para>The elements of the original tensor are copied into the new tensor.</para>
    /// <para>Both tensors must have same shape and type.</para>
    /// <para>If both tensors are located on the same device, a copy is performed.</para>
    /// </remarks>    
    /// <see cref="transfer``1"/>
    member trgt.TransferFrom (src: Tensor<'T>) =
        Tensor.CheckSameShape trgt src
        if trgt.Dev = src.Dev then
            trgt.CopyFrom (src)
        else
            if not (trgt.Backend.Transfer (trgt=trgt, src=src) ||
                    src.Backend.Transfer (trgt=trgt, src=src)) then
                invalidOp "Cannot transfer from storage %s to storage %s." src.Dev.Id trgt.Dev.Id

    /// Transfers this tensor to the specifed device.
    member internal src.Transfer (dev: ITensorDevice) =
        if src.Dev <> dev then
            let trgt = Tensor<'T> (src.Shape, dev)
            trgt.TransferFrom src
            trgt
        else src

    /// <summary>Transfers a tensor to the specifed device.</summary>
    /// <param name="dev">The target device.</param>
    /// <param name="a">The tensor to transfer.</param>
    /// <returns>A tensor on the target device.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 5L] // a.Dev = HostTensor.Dev
    /// let b = Tensor.transfer CudaTensor.Dev a // b.Dev = CudaTensor.Dev
    /// </code></example>       
    /// <remarks>    
    /// <para>A new tensor is created on the specified device.</para>
    /// <para>The elements of the original tensor are copied into the new tensor.</para>
    /// <para>If the target device matches the current device of the tensor, the original tensor is returned.</para>
    /// </remarks>    
    /// <seealso cref="TransferFrom"/><seealso cref="Dev"/><seealso cref="copy"/>
    static member transfer (dev: ITensorDevice) (src: Tensor<'T>) =
        src.Transfer (dev) 

    /// this tensor as Tensor<bool>
    member internal this.AsBool : Tensor<bool> =
        if this.DataType = typeof<bool> then
            this |> box :?> Tensor<bool>
        else
            invalidOp "The operation requires a Tensor<bool> but the data type of the specified tensor is %s." 
                      this.DataType.Name

    /// this tensor as Tensor<int64>
    member internal this.AsInt64 : Tensor<int64> =
        if this.DataType = typeof<int64> then
            this |> box :?> Tensor<int64>
        else
            invalidOp "The operation requires a Tensor<int64> but the data type of the specified tensor is %s." 
                      this.DataType.Name

    /// <summary>Fills this tensor with a copy of the specified tensor.</summary>
    /// <param name="src">The tensor to copy from.</param>
    /// <remarks>
    /// <para>The source tensor is broadcasted to the size of this tensor.</para>
    /// <para>The source tensor must be stored on the same device as this tensor.</para>
    /// </remarks>
    /// <seealso cref="CopyFrom"/>
    member trgt.FillFrom (src: Tensor<'T>) = 
        let src = Tensor.PrepareElemwiseSources (trgt, src)
        trgt.CopyFrom src

    /// <summary>Fills this tensor with the specified constant value.</summary>
    /// <param name="value">The value to use.</param>
    /// <seealso cref="filled"/>
    member trgt.FillConst (value: 'T) =
        trgt.Backend.FillConst (value=value, trgt=trgt)

    /// <summary>Fills this vector with an equispaced sequence of elements.</summary>
    /// <param name="start">The starting value.</param>
    /// <param name="incr">The increment between successive elements.</param>    
    /// <remarks>
    /// <para>This tensor must be one dimensional.</para>
    /// </remarks>
    /// <seealso cref="arange``3"/>
    member trgt.FillIncrementing (start: 'T, incr: 'T) =
        if trgt.NDims <> 1 then invalidOp "FillIncrementing requires a vector."
        trgt.Backend.FillIncrementing (start=start, incr=incr, trgt=trgt)

    /// <summary>Copies elements from a tensor of different data type into this tensor and converts their type.</summary>
    /// <typeparam name="'C">The data type to convert from.</typeparam>
    /// <param name="a">The tensor to copy from.</param>    
    /// <seealso cref="convert``1"/>
    member trgt.FillConvert (a: Tensor<'C>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Convert (trgt=trgt, src=a)

    /// <summary>Convert the elements of a tensor to the specifed type.</summary>
    /// <typeparam name="'C">The data type to convert from.</typeparam>
    /// <param name="a">The tensor to convert.</param>
    /// <returns>A tensor of the new data type.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [1; 2; 3] 
    /// let b = Tensor&lt;float>.convert a // b = [1.0; 2.0; 3.0]
    /// </code></example>       
    /// <remarks>    
    /// <para>The elements of the original tensor are copied into the new tensor and their type is converted
    /// during the copy.</para>
    /// <para>For tensors that contain data of non-primitive types and are stored on the host, 
    /// the <c>op_Explicit</c> or <c>op_Implicit</c> methods of the source or destination type are used to perform
    /// the conversion.</para>
    /// </remarks>    
    /// <seealso cref="FillConvert``1"/>
    static member convert (a: Tensor<'C>) : Tensor<'T> =
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillConvert (a)
        trgt   

    /// <summary>Fills this tensor with the element-wise prefix plus of the argument.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <seealso cref="op_UnaryPlus"/>
    member trgt.FillUnaryPlus (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.UnaryPlus (trgt=trgt, src1=a)

    /// <summary>Element-wise prefix plus.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let c = +a  // c = [5.0; 6.0; 7.0]
    /// </code></example>
    /// <remarks>
    /// <para>Applies the unary plus operator to each element of tensor <paramref name="a"/> and returns the result 
    /// as a new tensor.</para>
    /// <para>For most data types, this operation does not change the value.</para>
    /// </remarks>
    /// <seealso cref="FillUnaryPlus"/>
    static member (~+) (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillUnaryPlus (a)
        trgt

    /// <summary>Fills this tensor with the element-wise negation of the argument.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <seealso cref="op_UnaryNegation"/>
    member trgt.FillUnaryMinus (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.UnaryMinus (trgt=trgt, src1=a)

    /// <summary>Element-wise negation.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let c = -a  // c = [-5.0; -6.0; -7.0]
    /// </code></example>
    /// <remarks>
    /// <para>Negates each element of tensor <paramref name="a"/> and returns the result as a new tensor.</para>
    /// </remarks>
    /// <seealso cref="FillNegate"/>
    static member (~-) (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillUnaryMinus (a)
        trgt

    /// <summary>Fills this tensor with the element-wise absolute value of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Abs"/>
    member trgt.FillAbs (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Abs (trgt=trgt, src1=a)

    /// <summary>Element-wise absolute value.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-2; -1; 1]
    /// let b = abs a // b = [2; 1; 1]
    /// </code></example>
    /// <remarks>Computes the absolute value of each element of the specified tensor and returns them as a new tensor.
    /// Do not call this function directly; instead use the F# <c>abs</c> function.</remarks>
    /// <seealso cref="FillAbs"/>
    static member Abs (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillAbs (a)
        trgt

    /// <summary>Fills this tensor with the element-wise sign of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Sgn"/>
    member trgt.FillSgn (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Sgn (trgt=trgt, src1=a)

    /// <summary>Element-wise sign.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-2; -1; 0; 2]
    /// let b = sgn a // b = [-1; -1; 0; 1]
    /// </code></example>
    /// <remarks>Computes the sign of each element of the specified tensor and returns them as a new tensor.
    /// The type of the returned tensor matches the type of the argument tensor.
    /// Do not call this function directly; instead use the F# <c>sgn</c> function.</remarks>
    /// <seealso cref="FillSgn"/>
    static member Sgn (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillSgn (a)
        trgt

    /// <summary>Fills this tensor with the element-wise natural logarithm of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Log"/>
    member trgt.FillLog (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Log (trgt=trgt, src1=a)

    /// <summary>Element-wise natural logarithm.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [1.0; 2.71828; 4.0]
    /// let b = log a // b = [0.0; 1.0; 1.38529]
    /// </code></example>
    /// <remarks>Computes the natural logarithm of each element of the specified tensor and returns them as a new tensor.
    /// Do not call this function directly; instead use the F# <c>log</c> function.</remarks>
    /// <seealso cref="FillLog"/><seealso cref="Log10"/><seealso cref="Exp"/>
    static member Log (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillLog (a)
        trgt

    /// <summary>Fills this tensor with the element-wise common logarithm of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Log10"/>
    member trgt.FillLog10 (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Log10 (trgt=trgt, src1=a)

    /// <summary>Element-wise common logarithm.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [1.0; 10.0; 100.0]
    /// let b = log10 a // b = [0.0; 1.0; 2.0]
    /// </code></example>
    /// <remarks>Computes the common logarithm (to base 10) of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>log10</c> function.</remarks>
    /// <seealso cref="FillLog10"/><seealso cref="Log"/>
    static member Log10 (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillLog10 (a)
        trgt

    /// <summary>Fills this tensor with the element-wise exponential function of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Exp"/>
    member trgt.FillExp (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Exp (trgt=trgt, src1=a)

    /// <summary>Element-wise exponential function.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.0; 0.0; 1.0; 10.0]
    /// let b = exp a // b = [0.36787; 1.0; 2.71828; 22026.4657]
    /// </code></example>
    /// <remarks>Computes the exponential function of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>exp</c> function.</remarks>
    /// <seealso cref="FillExp"/><seealso cref="Log"/>
    static member Exp (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillExp (a)
        trgt

    /// <summary>Fills this tensor with the element-wise sine of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Sin"/>
    member trgt.FillSin (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Sin (trgt=trgt, src1=a)

    /// <summary>Element-wise sine.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.57079; 0.0; 1.57079]
    /// let b = sin a // b = [-1.0; 0.0; 1.0]
    /// </code></example>
    /// <remarks>Computes the sine of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>sin</c> function.</remarks>
    /// <seealso cref="FillSin"/><seealso cref="Asin"/>
    static member Sin (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillSin (a)
        trgt

    /// <summary>Fills this tensor with the element-wise cosine of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Cos"/>
    member trgt.FillCos (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Cos (trgt=trgt, src1=a)

    /// <summary>Element-wise cosine.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.57079; 0.0; 1.57079]
    /// let b = cos a // b = [0.0; 1.0; 0.0]
    /// </code></example>
    /// <remarks>Computes the cosine of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>cos</c> function.</remarks>
    /// <seealso cref="FillCos"/><seealso cref="Acos"/>
    static member Cos (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillCos (a)
        trgt

    /// <summary>Fills this tensor with the element-wise tangent of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Tan"/>
    member trgt.FillTan (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Tan (trgt=trgt, src1=a)

    /// <summary>Element-wise tangent.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.57079; 0.0; 1.57079]
    /// let b = tan a // b = [-158057.9134; 0.0; 158057.9134]
    /// </code></example>
    /// <remarks>Computes the tangent of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>tan</c> function.</remarks>
    /// <seealso cref="FillTan"/><seealso cref="Atan"/>
    static member Tan (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillTan (a)
        trgt

    /// <summary>Fills this tensor with the element-wise arcsine (inverse sine) of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Asin"/>
    member trgt.FillAsin (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Asin (trgt=trgt, src1=a)

    /// <summary>Element-wise arcsine (inverse sine).</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.0; 0.0; 1.0]
    /// let b = asin a // b = [-1.57079; 0.0; 1.57079]
    /// </code></example>
    /// <remarks>Computes the arcsine (inverse sine) of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>asin</c> function.</remarks>
    /// <seealso cref="FillAsin"/><seealso cref="Sin"/>
    static member Asin (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillAsin (a)
        trgt

    /// <summary>Fills this tensor with the element-wise arccosine (inverse cosine) of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Acos"/>
    member trgt.FillAcos (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Acos (trgt=trgt, src1=a)

    /// <summary>Element-wise arccosine (inverse cosine).</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.0; 0.0; 1.0]
    /// let b = acos a // b = [3.15159; 1.57079; 0.0]
    /// </code></example>
    /// <remarks>Computes the arccosine (inverse cosine) of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>acos</c> function.</remarks>
    /// <seealso cref="FillAcos"/><seealso cref="Cos"/>
    static member Acos (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillAcos (a)
        trgt

    /// <summary>Fills this tensor with the element-wise arctanget (inverse tangent) of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Atan"/>
    member trgt.FillAtan (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Atan (trgt=trgt, src1=a)

    /// <summary>Element-wise arctanget (inverse tangent).</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.0; 0.0; 1.0]
    /// let b = atan a // b = [-0.78539; 0.0; 0.78539]
    /// </code></example>
    /// <remarks>Computes the arctanget (inverse tangent) of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>atan</c> function.</remarks>
    /// <seealso cref="FillAcos"/><seealso cref="Tan"/>
    static member Atan (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillAtan (a)
        trgt

    /// <summary>Fills this tensor with the element-wise hyperbolic sine of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Sinh"/>
    member trgt.FillSinh (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Sinh (trgt=trgt, src1=a)

    /// <summary>Element-wise hyperbolic sine.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.57079; 0.0; 1.57079]
    /// let b = sinh a // b = [-2.30128; 0.0; 2.30128]
    /// </code></example>
    /// <remarks>Computes the hyperbolic sine of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>sinh</c> function.</remarks>
    /// <seealso cref="FillSinh"/>
    static member Sinh (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillSinh (a)
        trgt

    /// <summary>Fills this tensor with the element-wise hyperbolic cosine of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Cosh"/>
    member trgt.FillCosh (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Cosh (trgt=trgt, src1=a)

    /// <summary>Element-wise hyperbolic cosine.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.57079; 0.0; 1.57079]
    /// let b = cosh a // b = [2.50916; 1.0; 2.50916]
    /// </code></example>
    /// <remarks>Computes the hyperbolic cosine of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>cosh</c> function.</remarks>
    /// <seealso cref="FillCosh"/>
    static member Cosh (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillCosh (a)
        trgt

    /// <summary>Fills this tensor with the element-wise hyperbolic tangent of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Tanh"/>
    member trgt.FillTanh (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Tanh (trgt=trgt, src1=a)

    /// <summary>Element-wise hyperbolic tangent.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-1.57079; 0.0; 1.57079]
    /// let b = tanh a // b = [-0.91715; 0.0; 0.91715]
    /// </code></example>
    /// <remarks>Computes the hyperbolic tangent of each element of the specified tensor and returns them 
    /// as a new tensor.
    /// Do not call this function directly; instead use the F# <c>tanh</c> function.</remarks>
    /// <seealso cref="FillTanh"/>
    static member Tanh (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillTanh (a)
        trgt

    /// <summary>Fills this tensor with the element-wise square root of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Sqrt"/>
    member trgt.FillSqrt (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Sqrt (trgt=trgt, src1=a)

    /// <summary>Element-wise square root.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [1.0; 4.0; 16.0]
    /// let b = sqrt a // b = [1.0; 2.0; 4.0]
    /// </code></example>
    /// <remarks>Computes the square root of each element of the specified tensor and returns them as a new tensor.
    /// Do not call this function directly; instead use the F# <c>sqrt</c> function.</remarks>
    /// <seealso cref="FillSqrt"/>
    static member Sqrt (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillSqrt (a)
        trgt

    /// <summary>Fills this tensor with the element-wise ceiling (round towards positive infinity) of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Ceiling"/>
    member trgt.FillCeiling (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Ceiling (trgt=trgt, src1=a)

    /// <summary>Element-wise ceiling (round towards positive infinity).</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-3.0; -2.7; 2.7; 3.0]
    /// let b = ceil a // b = [-3.0; -2.0; 3.0; 3.0]
    /// </code></example>
    /// <remarks>Computes the ceiling (round towards positive infinity) of each element of the specified tensor and 
    /// returns them as a new tensor.
    /// Do not call this function directly; instead use the F# <c>ceil</c> function.</remarks>
    /// <seealso cref="FillCeiling"/>
    static member Ceiling (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillCeiling (a)
        trgt

    /// <summary>Fills this tensor with the element-wise floor (round towards negative infinity) of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Floor"/>
    member trgt.FillFloor (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Floor (trgt=trgt, src1=a)

    /// <summary>Element-wise floor (round towards negative infinity).</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-3.0; -2.7; 2.7; 3.0]
    /// let b = floor a // b = [-3.0; -3.0; 2.0; 3.0]
    /// </code></example>
    /// <remarks>Computes the floor (round towards negative infinity) of each element of the specified tensor and 
    /// returns them as a new tensor.
    /// Do not call this function directly; instead use the F# <c>floor</c> function.</remarks>
    /// <seealso cref="FillFloor"/>
    static member Floor (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillFloor (a)
        trgt

    /// <summary>Fills this tensor with the element-wise rounding of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Round"/>
    member trgt.FillRound (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Round (trgt=trgt, src1=a)

    /// <summary>Element-wise rounding.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-3.0; -2.7; 2.7; 3.0]
    /// let b = round a // b = [-3.0; -3.0; 3.0; 3.0]
    /// </code></example>
    /// <remarks>Computes the rounding of each element of the specified tensor and returns them as a new tensor.
    /// Do not call this function directly; instead use the F# <c>round</c> function.</remarks>
    /// <seealso cref="FillRound"/>
    static member Round (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillRound (a)
        trgt

    /// <summary>Fills this tensor with the element-wise truncation (rounding towards zero) of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="Truncate"/>
    member trgt.FillTruncate (a: Tensor<'T>) = 
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Truncate (trgt=trgt, src1=a)

    /// <summary>Element-wise truncation (rounding towards zero).</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-3.0; -2.7; 2.7; 3.0]
    /// let b = truncate a // b = [-3.0; -2.0; 2.0; 3.0]
    /// </code></example>
    /// <remarks>Computes the truncation (rounding towards zero) of each element of the specified tensor and returns
    /// them as a new tensor.
    /// Do not call this function directly; instead use the F# <c>truncate</c> function.</remarks>
    /// <seealso cref="FillTruncate"/>
    static member Truncate (a: Tensor<'T>) = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillTruncate (a)
        trgt

    /// <summary>Fills this tensor with the element-wise finity check (not -Inf, Inf or NaN) of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="isFinite"/>
    member trgt.FillIsFinite (a: Tensor<'R>) = 
        let trgt = trgt.AsBool
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        a.Backend.IsFinite (trgt=trgt, src1=a)

    /// <summary>Element-wise finity check (not -Inf, Inf or NaN).</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [-infinity; -3.0; nan; 3.0; infinity]
    /// let b = isFinite a // b = [false; true; false; true; false]
    /// </code></example>
    /// <remarks>Checks each element of the specified tensor for finity (not -Inf, Inf or NaN) and returns
    /// the results as a new tensor of type <c>bool</c>.</remarks>
    /// <seealso cref="FillIsFinite``1"/><seealso crf="allFinite``1"/>
    static member isFinite (a: Tensor<'T>) : Tensor<bool> = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillIsFinite (a)
        trgt

    /// <summary>Fills this tensor with the element-wise logical negation of the argument.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <seealso cref="op_TwiddleTwiddleTwiddleTwiddle"/>
    member trgt.FillNegate (a: Tensor<bool>) = 
        let trgt = trgt.AsBool
        let a = Tensor.PrepareElemwiseSources (trgt, a)
        trgt.Backend.Negate (trgt=trgt, src1=a)

    /// <summary>Element-wise logical negation.</summary>
    /// <param name="a">The tensor to apply this operation to.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [true; false]
    /// let b = ~~~~a // b = [false; true]
    /// </code></example>
    /// <remarks>Logically negates each element of the specified tensor and returns the results as a new tensor.
    /// </remarks>
    /// <seealso cref="FillNegate"/>
    static member (~~~~) (a: Tensor<bool>) : Tensor<bool> = 
        let trgt, a = Tensor.PrepareElemwise (a)
        trgt.FillNegate (a)
        trgt

    /// <summary>Fills this tensor with the element-wise addition of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_Addition"/>
    member trgt.FillAdd (a: Tensor<'T>) (b: Tensor<'T>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        trgt.Backend.Add (trgt=trgt, src1=a, src2=b)
   
    /// <summary>Element-wise addition.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = a + b // c = [7.0; 9.0; 11.0]
    /// </code></example>
    /// <remarks>
    /// <para>Adds each element of tensor <paramref name="a"/> to the corresponding element of tensor <paramref name="b"/>
    /// and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same type and storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillAdd"/>
    static member (+) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillAdd a b
        trgt

    /// <summary>Element-wise addition with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Addition"/>
    static member (+) (a: Tensor<'T>, b: 'T) = a + Tensor.scalarLike a b

    /// <summary>Element-wise addition with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Addition"/>
    static member (+) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a + b

    /// <summary>Fills this tensor with the element-wise substraction of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_Subtraction"/>
    member trgt.FillSubtract (a: Tensor<'T>) (b: Tensor<'T>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        trgt.Backend.Subtract (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise substraction.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = a - b // c = [3.0; 3.0; 3.0]
    /// </code></example>
    /// <remarks>
    /// <para>Substracts each element of tensor <paramref name="b"/> from the corresponding element of tensor <paramref name="a"/>
    /// and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same type and storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillSubtract"/>
    static member (-) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillSubtract a b
        trgt

    /// <summary>Element-wise substraction with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Subtraction"/>
    static member (-) (a: Tensor<'T>, b: 'T) = a - Tensor.scalarLike a b

    /// <summary>Element-wise substraction with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Subtraction"/>
    static member (-) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a - b

    /// <summary>Fills this tensor with the element-wise multiplication of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_Multiply"/>
    member trgt.FillMultiply (a: Tensor<'T>) (b: Tensor<'T>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        trgt.Backend.Multiply (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise multiplication.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = a * b // c = [10.0; 18.0; 28.0]
    /// </code></example>
    /// <remarks>
    /// <para>Multiplies each element of tensor <paramref name="a"/> with the corresponding element of tensor <paramref name="b"/>
    /// and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same type and storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillMultiply"/>
    static member (*) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillMultiply a b
        trgt

    /// <summary>Element-wise multiplication with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Multiply"/>
    static member (*) (a: Tensor<'T>, b: 'T) = a * Tensor.scalarLike a b

    /// <summary>Element-wise multiplication with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Multiply"/>
    static member (*) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a * b

    /// <summary>Fills this tensor with the element-wise division of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_Division"/>
    member trgt.FillDivide (a: Tensor<'T>) (b: Tensor<'T>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        trgt.Backend.Divide (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise division.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = a / b // c = [2.5; 2.0; 1.75]
    /// </code></example>
    /// <remarks>
    /// <para>Divides each element of tensor <paramref name="a"/> by the corresponding element of tensor <paramref name="b"/>
    /// and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same type and storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillDivide"/>
    static member (/) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillDivide a b
        trgt

    /// <summary>Element-wise division with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Division"/>
    static member (/) (a: Tensor<'T>, b: 'T) = a / Tensor.scalarLike a b

    /// <summary>Element-wise division with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Division"/>
    static member (/) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a / b

    /// <summary>Fills this tensor with the element-wise remainder of the division of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_Modulus"/>
    member trgt.FillModulo (a: Tensor<'T>) (b: Tensor<'T>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        trgt.Backend.Modulo (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise remainder of division.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = a % b // c = [1.0; 0.0; 3.0]
    /// </code></example>
    /// <remarks>
    /// <para>Computes the remainder of dividing each element of tensor <paramref name="a"/> by the corresponding 
    /// element of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same type and storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillModulo"/>
    static member (%) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillModulo a b
        trgt

    /// <summary>Element-wise remainder of division with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Modulus"/>
    static member (%) (a: Tensor<'T>, b: 'T) = a % Tensor.scalarLike a b

    /// <summary>Element-wise division with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_Modulus"/>
    static member (%) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a % b

    /// <summary>Fills this tensor with the element-wise exponentiation.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="Pow"/>
    member trgt.FillPower (a: Tensor<'T>) (b: Tensor<'T>) = 
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        trgt.Backend.Power (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise exponentiation.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = a ** b // c = [25.0; 216.0; 2401.0]
    /// </code></example>
    /// <remarks>
    /// <para>Computes the exponentiation of each element of tensor <paramref name="a"/> to the power given by the
    /// corresponding element of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>Do not call this function directly; instead use the F# <c>**</c> operator.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same type and storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillPower"/>
    static member Pow (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillPower a b
        trgt

    /// <summary>Element-wise exponentiation with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="Pow"/>
    static member Pow (a: Tensor<'T>, b: 'T) = a ** Tensor.scalarLike a b

    /// <summary>Element-wise exponentiation with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="Pow"/>
    static member Pow (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a ** b

    /// <summary>Fills this tensor with the element-wise logical and of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_AmpAmpAmpAmp"/>
    member trgt.FillAnd (a: Tensor<bool>) (b: Tensor<bool>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        trgt.Backend.And (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise loigcal and.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [true; true; false; false]
    /// let b = HostTensor.ofList [true; false; true; false]
    /// let c = a &amp;&amp;&amp;&amp; b // c = [true; false; false; false]
    /// </code></example>
    /// <remarks>
    /// <para>Computes the logical and of each element of tensor <paramref name="a"/> with the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillAnd"/>
    static member (&&&&) (a: Tensor<bool>, b: Tensor<bool>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillAnd a b
        trgt

    /// <summary>Element-wise loigcal and with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_AmpAmpAmpAmp"/>
    static member (&&&&) (a: Tensor<bool>, b: bool) = a &&&& Tensor.scalarLike a b

    /// <summary>Element-wise loigcal and with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_AmpAmpAmpAmp"/>
    static member (&&&&) (a: bool, b: Tensor<bool>) = Tensor.scalarLike b a &&&& b
    
    /// <summary>Fills this tensor with the element-wise logical or of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_BarBarBarBar"/>
    member trgt.FillOr (a: Tensor<bool>) (b: Tensor<bool>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        trgt.Backend.Or (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise loigcal or.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [true; true; false; false]
    /// let b = HostTensor.ofList [true; false; true; false]
    /// let c = a |||| b // c = [true; true; true; false]
    /// </code></example>
    /// <remarks>
    /// <para>Computes the logical or of each element of tensor <paramref name="a"/> with the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillOr"/>
    static member (||||) (a: Tensor<bool>, b: Tensor<bool>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillOr a b
        trgt

    /// <summary>Element-wise loigcal or with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_BarBarBarBar"/>
    static member (||||) (a: Tensor<bool>, b: bool) = a |||| Tensor.scalarLike a b

    /// <summary>Element-wise loigcal or with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_BarBarBarBar"/>
    static member (||||) (a: bool, b: Tensor<bool>) = Tensor.scalarLike b a |||| b

    /// <summary>Fills this tensor with the element-wise logical xor of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_HatHatHatHat"/>
    member trgt.FillXor (a: Tensor<bool>) (b: Tensor<bool>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        trgt.Backend.Xor (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise loigcal xor.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [true; true; false; false]
    /// let b = HostTensor.ofList [true; false; true; false]
    /// let c = a ^^^^ b // c = [false; true; true; false]
    /// </code></example>
    /// <remarks>
    /// <para>Computes the logical xor of each element of tensor <paramref name="a"/> with the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillXor"/>
    static member (^^^^) (a: Tensor<bool>, b: Tensor<bool>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillXor a b
        trgt

    /// <summary>Element-wise loigcal xor with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_HatHatHatHat"/>
    static member (^^^^) (a: Tensor<bool>, b: bool) = a ^^^^ Tensor.scalarLike a b

    /// <summary>Element-wise loigcal xor with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_HatHatHatHat"/>
    static member (^^^^) (a: bool, b: Tensor<bool>) = Tensor.scalarLike b a ^^^^ b

    /// <summary>Fills this tensor with the element-wise equality test of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_EqualsEqualsEqualsEquals"/>
    member trgt.FillEqual (a: Tensor<'R>) (b: Tensor<'R>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        a.Backend.Equal (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise equality test.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [2.0; 4.0; 6.0]
    /// let b = HostTensor.ofList [1.0; 4.0; 7.0]
    /// let c = a ==== b // c = [false; true; false]
    /// </code></example>
    /// <remarks>
    /// <para>Test each element of tensor <paramref name="a"/> for being equal to the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage and type.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillEqual``1"/>
    static member (====) (a: Tensor<'T>, b: Tensor<'T>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillEqual a b
        trgt

    /// <summary>Element-wise equality test with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_EqualsEqualsEqualsEquals"/>
    static member (====) (a: Tensor<'T>, b: 'T) = a ==== Tensor.scalarLike a b

    /// <summary>Element-wise equality test with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_EqualsEqualsEqualsEquals"/>
    static member (====) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a ==== b

    /// <summary>Fills this tensor with the element-wise not-equality test or of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_LessLessGreaterGreater"/>
    member trgt.FillNotEqual (a: Tensor<'R>) (b: Tensor<'R>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        a.Backend.NotEqual (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise not-equality test.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [2.0; 4.0; 6.0]
    /// let b = HostTensor.ofList [1.0; 4.0; 7.0]
    /// let c = a &lt;&lt;>> b // c = [true; false; true]
    /// </code></example>
    /// <remarks>
    /// <para>Test each element of tensor <paramref name="a"/> for being not equal to the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage and type.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillNotEqual``1"/>
    static member (<<>>) (a: Tensor<'T>, b: Tensor<'T>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillNotEqual a b
        trgt

    /// <summary>Element-wise not-equality test with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_LessLessGreaterGreater"/>
    static member (<<>>) (a: Tensor<'T>, b: 'T) = a <<>> Tensor.scalarLike a b

    /// <summary>Element-wise not-equality test with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_LessLessGreaterGreater"/>
    static member (<<>>) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a <<>> b

    /// <summary>Fills this tensor with the element-wise less-than test of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_LessLessLessLess"/>
    member trgt.FillLess (a: Tensor<'R>) (b: Tensor<'R>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        a.Backend.Less (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise less-than test.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [2.0; 4.0; 6.0]
    /// let b = HostTensor.ofList [1.0; 4.0; 7.0]
    /// let c = a &lt;&lt;&lt;&lt; b // c = [false; false; true]
    /// </code></example>
    /// <remarks>
    /// <para>Test each element of tensor <paramref name="a"/> for being less than the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage and type.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillLess``1"/>
    static member (<<<<) (a: Tensor<'T>, b: Tensor<'T>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillLess a b
        trgt

    /// <summary>Element-wise less-than test with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_LessLessLessLess"/>
    static member (<<<<) (a: Tensor<'T>, b: 'T) = a <<<< Tensor.scalarLike a b

    /// <summary>Element-wise less-than test with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_LessLessLessLess"/>
    static member (<<<<) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a <<<< b

    /// <summary>Fills this tensor with the element-wise less-than-or-equal test of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_LessLessEqualsEquals"/>
    member trgt.FillLessOrEqual (a: Tensor<'R>) (b: Tensor<'R>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        a.Backend.LessOrEqual (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise less-than-or-equal test.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [2.0; 4.0; 6.0]
    /// let b = HostTensor.ofList [1.0; 4.0; 7.0]
    /// let c = a &lt;&lt;== b // c = [false; true; true]
    /// </code></example>
    /// <remarks>
    /// <para>Test each element of tensor <paramref name="a"/> for being less than or equal to the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage and type.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillLessOrEqual``1"/>
    static member (<<==) (a: Tensor<'T>, b: Tensor<'T>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillLessOrEqual a b
        trgt

    /// <summary>Element-wise less-than-or-equal test with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_LessLessEqualsEquals"/>
    static member (<<==) (a: Tensor<'T>, b: 'T) = a <<== Tensor.scalarLike a b

    /// <summary>Element-wise less-than-or-equal test with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_LessLessEqualsEquals"/>
    static member (<<==) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a <<== b

    /// <summary>Fills this tensor with the element-wise greater-than test of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_GreaterGreaterGreaterGreater"/>
    member trgt.FillGreater (a: Tensor<'R>) (b: Tensor<'R>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        a.Backend.Greater (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise greater-than test.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [2.0; 4.0; 6.0]
    /// let b = HostTensor.ofList [1.0; 4.0; 7.0]
    /// let c = a >>>> b // c = [true; false; false]
    /// </code></example>
    /// <remarks>
    /// <para>Test each element of tensor <paramref name="a"/> for being greater than the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage and type.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillGreater``1"/>
    static member (>>>>) (a: Tensor<'T>, b: Tensor<'T>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillGreater a b
        trgt

    /// <summary>Element-wise greater-than test with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_GreaterGreaterGreaterGreater"/>
    static member (>>>>) (a: Tensor<'T>, b: 'T) = a >>>> Tensor.scalarLike a b

    /// <summary>Element-wise greater-than test with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_GreaterGreaterGreaterGreater"/>
    static member (>>>>) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a >>>> b

    /// <summary>Fills this tensor with the element-wise greater-than-or-equal test of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_GreaterGreaterEqualsEquals"/>
    member trgt.FillGreaterOrEqual (a: Tensor<'R>) (b: Tensor<'R>) = 
        let trgt = trgt.AsBool
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)           
        a.Backend.GreaterOrEqual (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise greater-than-or-equal test.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [2.0; 4.0; 6.0]
    /// let b = HostTensor.ofList [1.0; 4.0; 7.0]
    /// let c = a >>== b // c = [true; true; false]
    /// </code></example>
    /// <remarks>
    /// <para>Test each element of tensor <paramref name="a"/> for being greater than or equal to the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage and type.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para> 
    /// </remarks>
    /// <seealso cref="FillGreaterOrEqual``1"/>
    static member (>>==) (a: Tensor<'T>, b: Tensor<'T>) : Tensor<bool> = 
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillGreaterOrEqual a b
        trgt

    /// <summary>Element-wise greater-than-or-equal test with scalar.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The scalar on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_GreaterGreaterEqualsEquals"/>
    static member (>>==) (a: Tensor<'T>, b: 'T) = a >>== Tensor.scalarLike a b

    /// <summary>Element-wise greater-than-or-equal test with scalar.</summary>
    /// <param name="a">The scalar on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="op_GreaterGreaterEqualsEquals"/>
    static member (>>==) (a: 'T, b: Tensor<'T>) = Tensor.scalarLike b a >>== b

    /// <summary>Fills this tensor with the element-wise maximum of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="maxElemwise"/>
    member trgt.FillMaxElemwise (a: Tensor<'T>) (b: Tensor<'T>) =
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        trgt.Backend.MaxElemwise (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise maximum.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = Tensor.maxElemwise a b // c = [5.0; 6.0; 7.0]
    /// </code></example>
    /// <remarks>
    /// <para>Finds the maximum of each element of tensor <paramref name="a"/> and the corresponding element of 
    /// tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same type and storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillMaxElemwise"/>
    static member maxElemwise (a: Tensor<'T>) (b: Tensor<'T>) =
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillMaxElemwise a b
        trgt

    /// <summary>Fills this tensor with the element-wise minimum of the arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="minElemwise"/>
    member trgt.FillMinElemwise (a: Tensor<'T>) (b: Tensor<'T>) =
        let a, b = Tensor.PrepareElemwiseSources (trgt, a, b)
        trgt.Backend.MinElemwise (trgt=trgt, src1=a, src2=b)

    /// <summary>Element-wise minimum.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = Tensor.minElemwise a b // c = [2.0; 3.0; 4.0]
    /// </code></example>
    /// <remarks>
    /// <para>Finds the minimum of each element of tensor <paramref name="a"/> and the corresponding element of 
    /// tensor <paramref name="b"/> and returns the results as a new tensor.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same type and storage.
    /// Broadcasting rules apply if <paramref name="a"/> and <paramref name="b"/> have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillMinElemwise"/>
    static member minElemwise (a: Tensor<'T>) (b: Tensor<'T>) =
        let trgt, a, b = Tensor.PrepareElemwise (a, b)
        trgt.FillMinElemwise a b
        trgt

    /// <summary>Fills this tensor with an element-wise choice between two sources depending on a condition.</summary>
    /// <param name="cond">The condition tensor.</param>
    /// <param name="ifTrue">The tensor containing the values to use for when an element of the condition is true.</param>
    /// <param name="ifFalse">The tensor containing the values to use for when an element of the condition is false.</param>    
    /// <seealso cref="ifThenElse"/>
    member trgt.FillIfThenElse (cond: Tensor<bool>) (ifTrue: Tensor<'T>) (ifFalse: Tensor<'T>) = 
        let cond, ifTrue, ifFalse = Tensor.PrepareElemwiseSources (trgt, cond, ifTrue, ifFalse)
        trgt.Backend.IfThenElse (trgt=trgt, cond=cond, ifTrue=ifTrue, ifFalse=ifFalse)

    /// <summary>Element-wise choice between two sources depending on a condition.</summary>
    /// <param name="cond">The condition tensor.</param>
    /// <param name="ifTrue">The tensor containing the values to use for when an element of the condition is true.</param>
    /// <param name="ifFalse">The tensor containing the values to use for when an element of the condition is false.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let cond = HostTensor.ofList [true; false; false]
    /// let ifTrue = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let ifFalse = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let target = Tensor.ifThenElse cond ifTrue ifFalse // target = [2.0; 6.0; 7.0]
    /// </code></example>
    /// <remarks>
    /// <para>Evaluates each element of tensor <paramref name="cond"/>. If it evaluates to true, the corresponding
    /// element of tensor <paramref name="ifTrue"/> is written into the result. Otherwise, the corresponding element
    /// of tensor <paramref name="ifFalse"/> is written into the result.</para>
    /// <para>The tensors <paramref name="cond"/>, <paramref name="ifTrue"/> and <paramref name="ifFalse"/> must have 
    /// the same storage.
    /// Broadcasting rules apply if <paramref name="cond"/>, <paramref name="ifTrue"/> and <paramref name="ifFalse"/> 
    /// have different shapes.</para>
    /// </remarks>
    /// <seealso cref="FillIfThenElse"/>
    static member ifThenElse (cond: Tensor<bool>) (ifTrue: Tensor<'T>) (ifFalse: Tensor<'T>) =
        let trgt, cond, ifTrue, ifFalse = Tensor.PrepareElemwise(cond, ifTrue, ifFalse)
        trgt.FillIfThenElse cond ifTrue ifFalse
        trgt

    /// <summary>Selects elements from a tensor according to specified indices.</summary>
    /// <param name="indices">A list of tensors, one per dimension of <paramref name="src"/>, containing the indicies
    /// to pick from <paramref name="src"/> for each output element.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="gather"/>
    member trgt.FillGather (indices: Tensor<int64> option list) (src: Tensor<'T>) =
        Tensor.CheckSameStorage ([src :> ITensor] @ 
            List.choose (Option.map (fun t -> t :> ITensor)) indices)
        if src.NDims <> indices.Length then
            invalidArg "indices" "For each dimension of src an index tensor must be specified."        
        if indices |> List.skip trgt.NDims |> List.exists Option.isNone then
            invalidArg "indices" "Index dimensions beyond the number of target dimensions must not be None."
        let indices = indices |> List.map (Option.map (fun t -> t |> Tensor<_>.broadcastTo trgt.Shape :> ITensorFrontend<_>))
        trgt.Backend.Gather (trgt=trgt, srcIdxs=indices, src=src)

    /// <summary>Selects elements from a tensor according to specified indices.</summary>
    /// <param name="indices">A list of tensors, one per dimension of <paramref name="src"/>, containing the indicies
    /// to pick from <paramref name="src"/> for each output element.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>Result with the shape of the (broadcasted) tensors specified in <paramref name="indices"/>.</returns>
    /// <example><code language="fsharp">
    /// let src = HostTensor.ofList2D [[0.0; 0.1; 0.2; 0.3]
    ///                                [1.0; 1.1; 1.2; 1.3]
    ///                                [2.0; 2.1; 2.2; 2.3]]
    /// let i0 = HostTensor.ofList [1L; 2L; 0L; 0L]
    /// let i1 = HostTensor.ofList [3L; 1L; 0L; 3L]
    /// let g = Tensor.gather [Some i0; Some i1] src // g = [1.3000    2.1000    0.0000    0.3000]
    ///
    /// // Using None instead of an index tensor.    
    /// let j1 = HostTensor.ofList [3L; 1L; 0L]
    /// let g2 = Tensor.gather [None; Some j1] src // g2 = [0.3000    1.1000    2.0000]
    /// </code></example>
    /// <remarks>
    /// <para>The output element with indices <c>[i_0; i_1; i_2; ...]</c> is given by the source element with indices 
    /// <c>[j_0; j_1; j_2; ...]</c>, where each index <c>j_k</c> is given by <c>j_k = indices.[k].[i_0; i_1; i_2; ...]</c>.
    /// If <c>indices.[k]</c> is <c>None</c>, then <c>j_k = i_k</c> is assumed instead.
    /// Index dimensions beyond the number of target dimensions must not be <c>None</c>.
    /// </para>
    /// <para>The tensors <paramref name="indices"/> and <paramref name="src"/> must have the same storage.
    /// All index tensors are broadcasted to the same size.</para>
    /// </remarks>
    /// <seealso cref="FillGather"/><seealso cref="scatter"/>
    static member gather (indices: Tensor<int64> option list) (src: Tensor<'T>) =
        // broadcast specified indices to same shape
        let specIndices = indices |> List.choose id
        if List.isEmpty specIndices then
            invalidArg "indicies" "At least one index tensor must not be None."
        let bcSpecIndices = Tensor<_>.broadcastToSame specIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs -> Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSpecIndices

        // apply gather
        let trgt = Tensor<'T> (bcSpecIndices.Head.Shape, src.Dev)
        trgt.FillGather bcIndices src
        trgt        

    /// <summary>Disperses elements from a source tensor to this tensor according to the specified indices.</summary>
    /// <param name="indices">A list of tensors, one per dimension of this tensor, containing the target indicies
    /// for each element of <paramref name="src"/>.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="scatter"/>
    member trgt.FillScatter (indices: Tensor<int64> option list) (src: Tensor<'T>) =
        Tensor.CheckSameStorage ([src :> ITensor] @ 
            List.choose (Option.map (fun t -> t :> ITensor)) indices)
        if trgt.NDims <> indices.Length then
            invalidArg "indices" "For each dimension of the target an index tensor must be specified."        
        if indices |> List.skip src.NDims |> List.exists Option.isNone then
            invalidArg "indices" "Index dimensions beyond the number of source dimensions must not be None."
        let indices = indices |> List.map (Option.map (fun t -> t |> Tensor<_>.broadcastTo src.Shape :> ITensorFrontend<_>))
        trgt.Backend.FillConst (trgt=trgt, value=zero<'T>)
        trgt.Backend.Scatter (trgt=trgt, trgtIdxs=indices, src=src)

    /// <summary>Disperses elements from a source tensor to a new tensor according to the specified indices.</summary>
    /// <param name="indices">A list of tensors, one per dimension of this tensor, containing the target indicies
    /// for each element of <paramref name="src"/>.</param>
    /// <param name="trgtShp">The shape of the resulting tensor.</param>    
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>Result with the shape specified in <paramref name="trgtShp"/>.</returns>
    /// <example><code language="fsharp">
    /// // Sum first row of src into last element and swap rows 1 and 2.
    /// let src = HostTensor.ofList2D [[0.0; 0.1; 0.2; 0.3]
    ///                                [1.0; 1.1; 1.2; 1.3]
    ///                                [2.0; 2.1; 2.2; 2.3]]
    /// let i0 = HostTensor.ofList2D [[0L; 0L; 0L; 0L]
    ///                               [2L; 2L; 2L; 2L]
    ///                               [1L; 1L; 1L; 1L]]
    /// let i1 = HostTensor.ofList2D [[3L; 3L; 3L; 3L]
    ///                               [0L; 1L; 2L; 3L]
    ///                               [0L; 1L; 2L; 3L]]
    /// let s = Tensor.scatter [Some i0; Some i1] [4L; 4L] src
    /// // s =
    /// //     [[   0.0000    0.0000    0.0000    0.6000]
    /// //      [   2.0000    2.1000    2.2000    2.3000]
    /// //      [   1.0000    1.1000    1.2000    1.3000]
    /// //      [   0.0000    0.0000    0.0000    0.0000]]    
    /// </code></example>
    /// <remarks>
    /// <para>The source element with indices <c>[i_0; i_1; i_2; ...]</c> is written to the target element with indices 
    /// <c>[j_0; j_1; j_2; ...]</c>, where each index <c>j_k</c> is given by <c>j_k = indices.[k].[i_0; i_1; i_2; ...]</c>.
    /// If <c>indices.[k]</c> is <c>None</c>, then <c>j_k = i_k</c> is assumed instead.</para>
    /// <para>If a target index occurs multiple times, the corresponding source values are summed.
    /// If a target element is not referenced by any index, it is set to zero.</para>
    /// <para>The tensors <paramref name="indices"/> and <paramref name="src"/> must have the same storage.</para>
    /// </remarks>
    /// <seealso cref="FillScatter"/><seealso cref="gather"/>
    static member scatter (indices: Tensor<int64> option list) (trgtShp: int64 list) (src: Tensor<'T>) =
        let trgt = Tensor<'T> (trgtShp, src.Dev)
        trgt.FillScatter indices src
        trgt
        
    /// <summary>Counts the elements being true along the specified axis and writes the result into this tensor.</summary>
    /// <param name="ax">The axis the count along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="countTrueAxis"/>        
    member trgt.FillCountTrueAxis (ax: int) (src: Tensor<bool>) =
        let trgt = trgt.AsInt64
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        trgt.Backend.CountTrueLastAxis (trgt=trgt, src1=src)

    /// <summary>Counts the elements being true along the specified axis.</summary>
    /// <param name="ax">The axis the count along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[true; false; true; false]
    ///                              [false; true; true; false]]
    /// let b = Tensor.countTrueAxis 1 a // b = [2L; 2L]
    /// </code></example>
    /// <remarks>The number of elements that are true is accumulated along the specified axis.</remarks>
    /// <seealso cref="FillCountTrueAxis"/><seealso cref="countTrue"/>
    static member countTrueAxis (ax: int) (src: Tensor<bool>) : Tensor<int64> =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillCountTrueAxis ax src
        trgt

    /// <summary>Counts the elements being true returning the result as a Tensor.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new scalar tensor containing the result of this operation.</returns>
    /// <seealso cref="countTrue"/>
    static member countTrueTensor (src: Tensor<bool>) =
        src |> Tensor<_>.flatten |> Tensor<_>.countTrueAxis 0

    /// <summary>Counts the elements being true.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A scalar containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[true; false; true; false]
    ///                              [false; true; true; false]]
    /// let b = Tensor.countTrue a // b = 4L
    /// </code></example>
    /// <remarks>The number of elements that are true is counted and returned as a scalar.</remarks>
    /// <seealso cref="countTrueTensor"/><seealso cref="countTrueAxis"/>    
    static member countTrue (src: Tensor<bool>) =
        src |> Tensor.countTrueTensor |> Tensor.value
    
    /// <summary>Finds the indices of all element that are true.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A matrix that has one row per true entry in <paramref name="src"/>.
    /// The columns correspond to the dimensions of <paramref name="src"/>.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[true; false; true; false]
    ///                              [false; true; true; false]]
    /// let b = Tensor.trueIdx a // b = [[0L; 0L]
    ///                          //      [0L; 2L]
    ///                          //      [1L; 1L]
    ///                          //      [1L; 2L]]
    /// </code></example>
    /// <remarks>The function searches for elements that are true and for each one it creates a row
    /// in the result matrix consisting of the indices of the element.</remarks>
    static member trueIdx (src: Tensor<bool>) : Tensor<int64> =
        let nTrue = Tensor<_>.countTrue src
        let trgt = Tensor<int64> ([nTrue; int64 src.NDims], src.Storage.Dev)
        trgt.Backend.TrueIndices (trgt=trgt, src1=src)                
        trgt

    /// <summary>Sums the elements over the specified axis and writes the result into this tensor.</summary>
    /// <param name="ax">The axis to sum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="sumAxis"/>        
    member trgt.FillSumAxis (ax: int) (src: Tensor<'T>) =
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        trgt.Backend.SumLastAxis (trgt=trgt, src1=src)

    /// <summary>Sums the elements along the specified axis.</summary>
    /// <param name="ax">The axis to sum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.sumAxis 1 a // b = [10.0; 26.0]
    /// </code></example>
    /// <remarks>Elements are summed along the specified axis. An empty sum equals zero.</remarks>
    /// <seealso cref="FillSumAxis"/><seealso cref="sum"/>
    static member sumAxis (ax: int) (src: Tensor<'T>) =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillSumAxis ax src
        trgt

    /// <summary>Sums all elements returning a Tensor.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new scalar tensor containing the result of this operation.</returns>
    /// <seealso cref="sum"/>
    static member sumTensor (src: Tensor<'T>) =
        src |> Tensor<_>.flatten |> Tensor<_>.sumAxis 0

    /// <summary>Sums all elements.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A scalar containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.sum a // b = 36.0
    /// </code></example>
    /// <remarks>All elements are summed. An empty sum equals zero.</remarks>
    /// <seealso cref="sumAxis"/><seealso cref="sumTensor"/>
    static member sum (src: Tensor<'T>) =
        src |> Tensor.sumTensor |> Tensor.value

    /// <summary>Calculates the product of the elements over the specified axis and writes the result into this tensor.</summary>
    /// <param name="ax">The axis to calculate the product along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="productAxis"/>     
    member trgt.FillProductAxis (ax: int) (src: Tensor<'T>) =
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        trgt.Backend.ProductLastAxis (trgt=trgt, src1=src)

    /// <summary>Calculates the product of the elements along the specified axis.</summary>
    /// <param name="ax">The axis to calculate the product along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.productAxis 1 a // b = [24.0; 1680.0]
    /// </code></example>
    /// <remarks>The product is calculated along the specified axis. An empty product equals one.</remarks>
    /// <seealso cref="FillProductAxis"/><seealso cref="product"/>
    static member productAxis (ax: int) (src: Tensor<'T>) =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillProductAxis ax src
        trgt

    /// <summary>Calculates the product all elements returning a Tensor.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new scalar tensor containing the result of this operation.</returns>
    /// <seealso cref="product"/>
    static member productTensor (src: Tensor<'T>) =
        src |> Tensor<_>.flatten |> Tensor<_>.productAxis 0

    /// <summary>Calculates the product of all elements.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A scalar containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.product a // b = 40320.0
    /// </code></example>
    /// <remarks>The product of all elements is calculated. An empty product equals one.</remarks>
    /// <seealso cref="productAxis"/><seealso cref="productTensor"/>
    static member product (src: Tensor<'T>) =
        src |> Tensor.productTensor |> Tensor.value

    /// <summary>Calculates the minimum value of the elements over the specified axis and writes the result into this tensor.</summary>
    /// <param name="ax">The axis to calculate the minimum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="minAxis"/>     
    member trgt.FillMinAxis (ax: int) (src: Tensor<'T>) =
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        trgt.Backend.MinLastAxis (trgt=trgt, src1=src)

    /// <summary>Calculates the minimum value of the elements along the specified axis.</summary>
    /// <param name="ax">The axis to calculate the minimum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.minAxis 1 a // b = [1.0; 5.0]
    /// </code></example>
    /// <remarks>The minimum is calculated along the specified axis. An empty minimum gives the largest possible value 
    /// of the used data type.</remarks>
    /// <seealso cref="FillMinAxis"/><seealso cref="min"/>
    static member minAxis (ax: int) (src: Tensor<'T>) =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillMinAxis ax src
        trgt

    /// <summary>Calculates the minimum all elements returning a Tensor.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new scalar tensor containing the result of this operation.</returns>
    /// <seealso cref="min"/>
    static member minTensor (src: Tensor<'T>) =
        src |> Tensor<_>.flatten |> Tensor<_>.minAxis 0

    /// <summary>Calculates the minimum of all elements.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A scalar containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.min a // b = 1.0
    /// </code></example>
    /// <remarks>The minimum of all elements is calculated. An empty minimum gives the largest possible value 
    /// of the used data type.</remarks>
    /// <seealso cref="minAxis"/><seealso cref="minTensor"/>
    static member min (src: Tensor<'T>) =
        src |> Tensor.minTensor |> Tensor.value

    /// <summary>Calculates the maximum value of the elements over the specified axis and writes the result into this tensor.</summary>
    /// <param name="ax">The axis to calculate the maximum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="maxAxis"/>   
    member trgt.FillMaxAxis (ax: int) (src: Tensor<'T>) =
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        trgt.Backend.MaxLastAxis (trgt=trgt, src1=src)

    /// <summary>Calculates the maximum value of the elements along the specified axis.</summary>
    /// <param name="ax">The axis to calculate the maximum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.maxAxis 1 a // b = [4.0; 8.0]
    /// </code></example>
    /// <remarks>The maximum is calculated along the specified axis. An empty maximum gives the smallest possible value 
    /// of the used data type.</remarks>
    /// <seealso cref="FillMaxAxis"/><seealso cref="max"/>
    static member maxAxis (ax: int) (src: Tensor<'T>) =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillMaxAxis ax src
        trgt

    /// <summary>Calculates the maximum all elements returning a Tensor.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new scalar tensor containing the result of this operation.</returns>
    /// <seealso cref="max"/> 
    static member maxTensor (src: Tensor<'T>) =
        src |> Tensor<_>.flatten |> Tensor<_>.maxAxis 0

    /// <summary>Calculates the maximum of all elements.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A scalar containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.min a // b = 8.0
    /// </code></example>
    /// <remarks>The maximum of all elements is calculated. An empty maximum gives the smallest possible value 
    /// of the used data type.</remarks>
    /// <seealso cref="maxAxis"/><seealso cref="maxTensor"/>
    static member max (src: Tensor<'T>) =
        src |> Tensor.maxTensor |> Tensor.value

    /// <summary>Finds the index of the minimum value along the specified axis and writes it into this tensor.</summary>
    /// <param name="ax">The axis to calculate the minimum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="argMinAxis"/>
    member trgt.FillArgMinAxis (ax: int) (src: Tensor<'R>) =
        let trgt = trgt.AsInt64
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        src.Backend.ArgMinLastAxis (trgt=trgt, src1=src)

    /// <summary>Finds the index of the minimum value along the specified axis.</summary>
    /// <param name="ax">The axis to calculate the minimum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.argMinAxis 1 a // b = [0L; 0L]
    /// </code></example>
    /// <remarks>The index of the minimum is calculated along the specified axis. 
    /// An empty tensor gives <see cref="Tensor.NotFound"/>.</remarks>
    /// <seealso cref="FillArgMinAxis``1"/><seealso cref="argMin"/>
    static member argMinAxis (ax: int) (src: Tensor<'T>) : Tensor<int64> =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillArgMinAxis ax src
        trgt

    /// <summary>Finds the index of the maximum value along the specified axis and writes it into this tensor.</summary>
    /// <param name="ax">The axis to calculate the maximum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="argMaxAxis"/>
    member trgt.FillArgMaxAxis (ax: int) (src: Tensor<'R>) =
        let trgt = trgt.AsInt64
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        src.Backend.ArgMaxLastAxis (trgt=trgt, src1=src)

    /// <summary>Finds the index of the maximum value along the specified axis.</summary>
    /// <param name="ax">The axis to calculate the maximum along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.argMaxAxis 1 a // b = [3L; 3L]
    /// </code></example>
    /// <remarks>The index of the maximum is calculated along the specified axis. 
    /// An empty tensor gives <see cref="Tensor.NotFound"/>.</remarks>
    /// <seealso cref="FillArgMaxAxis``1"/><seealso cref="argMax"/>
    static member argMaxAxis (ax: int) (src: Tensor<'T>) : Tensor<int64> =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillArgMaxAxis ax src
        trgt

    /// <summary>Finds the indicies of the minimum value of the tensor.</summary>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>The indices of the position of the minimum value.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.argMin 1 a // b = [0L; 0L]
    /// </code></example>
    /// <remarks>The minimum value within the specified tensor is found and its indicies are returned. 
    /// The function fails for an empty tensor.</remarks>
    /// <seealso cref="argMinAxis"/>
    static member argMin (a: Tensor<'T>) =
        a 
        |> Tensor<_>.flatten 
        |> Tensor<_>.argMinAxis 0 
        |> Tensor.value 
        |> TensorLayout.linearToIdx a.Layout

    /// <summary>Finds the indicies of the maximum value of the tensor.</summary>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>The indices of the position of the maximum value.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.argMax a // b = [1L; 3L]
    /// </code></example>
    /// <remarks>The maximum value within the specified tensor is found and its indicies are returned. 
    /// The function fails for an empty tensor.</remarks>
    /// <seealso cref="argMaxAxis"/>
    static member argMax (a: Tensor<'T>) =
        a 
        |> Tensor<_>.flatten 
        |> Tensor<_>.argMaxAxis 0 
        |> Tensor.value 
        |> TensorLayout.linearToIdx a.Layout
        
    /// <summary>Finds the first occurence of the specfied value along the specified axis and write its index into this tensor.</summary>
    /// <param name="value">The value to find.</param>
    /// <param name="ax">The axis to find the value along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="findAxis"/>
    member trgt.FillFindAxis (value: 'R) (ax: int) (src: Tensor<'R>) =
        let trgt = trgt.AsInt64
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        src.Backend.FindLastAxis (value=value, trgt=trgt, src1=src)

    /// <summary>Finds the first occurence of the specfied value along the specified axis and returns its index.</summary>
    /// <param name="value">The value to find.</param>
    /// <param name="ax">The axis to find the value along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the indices of the first occurence of <paramref name="value"/>.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 3.0]]
    /// let b = Tensor.findAxis 3.0 1 a // b = [2L; 3L]
    /// </code></example>
    /// <remarks>The values is searched for an the index of the first occurence is returned.
    /// If the value is not found <see cref="Tensor.NotFound"/> is returned instead.</remarks>
    /// <seealso cref="FillFindAxis``1"/>
    static member findAxis (value: 'T) (ax: int) (src: Tensor<'T>) : Tensor<int64> =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillFindAxis value ax src
        trgt

    /// <summary>Finds the first occurence of the specfied value and returns its indices.</summary>
    /// <param name="value">The value to find.</param>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>The indices if the value was found, otherwise <c>None</c>.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 3.0]]
    /// let b = Tensor.tryFind 3.0 a // b = Some [0L; 2L]
    /// </code></example>
    /// <remarks>The values is searched for an the index of the first occurence is returned.
    /// If the value is not found <c>None</c> is returned instead.</remarks>
    /// <seealso cref="find"/><seealso cref="findAxis"/>
    static member tryFind (value: 'T) (a: Tensor<'T>) =
        let pos = 
            a 
            |> Tensor<_>.flatten 
            |> Tensor<_>.findAxis value 0 
            |> Tensor.value
        if pos <> NotFound then
            pos |> TensorLayout.linearToIdx a.Layout |> Some
        else None
        
    /// <summary>Finds the first occurence of the specfied value and returns its indices.</summary>
    /// <param name="value">The value to find.</param>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>The indices of the value.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 3.0]]
    /// let b = Tensor.find 3.0 a // b = [0L; 2L]
    /// </code></example>
    /// <remarks>The values is searched for an the index of the first occurence is returned.
    /// If the value is not found, an <see cref="System.InvalidOperationException"/> is raised.
    /// Use <see cref="tryFind"/> instead, if the value might not be present.
    /// </remarks>
    /// <exception cref="System.InvalidOperationException">Raised if value is not found.</exception>
    /// <seealso cref="tryFind"/><seealso cref="findAxis"/>
    static member find (value: 'T) (a: Tensor<'T>) =
        match Tensor<_>.tryFind value a with
        | Some pos -> pos
        | None -> invalidOp "Value %A was not found in specifed tensor." value

    /// <summary>Checks if all elements along the specified axis are true using this tensor as target.</summary>
    /// <param name="ax">The axis to check along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="allAxis"/>
    member trgt.FillAllAxis (ax: int) (src: Tensor<bool>) =
        let trgt = trgt.AsBool
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        trgt.Backend.AllLastAxis (trgt=trgt, src1=src)

    /// <summary>Checks if all elements along the specified axis are true.</summary>
    /// <param name="ax">The axis to check along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[true; true; true]
    ///                              [false; true; true]]
    /// let b = Tensor.allAxis 1 a // b = [true; false]
    /// </code></example>
    /// <remarks>
    /// <para>It is checked whether all elements along the specified axis are true.
    /// If so, true is returned; otherwise false is returned.</para>
    /// <para>If the tensor is empty true is returned.</para>
    /// </remarks>
    /// <seealso cref="FillAllAxis"/><seealso cref="all"/>
    static member allAxis (ax: int) (src: Tensor<bool>) : Tensor<bool> =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillAllAxis ax src
        trgt

    /// <summary>Checks if all elements of the tensor are true returning the result as a tensor.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="all"/>
    static member allTensor (src: Tensor<bool>) =
        src |> Tensor<_>.flatten |> Tensor<_>.allAxis 0 

    /// <summary>Checks if all elements of the tensor are true.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A scalar containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[true; true; true]
    ///                              [false; true; true]]
    /// let b = Tensor.all a // b = false
    /// </code></example>
    /// <remarks>
    /// <para>It is checked whether all elements of the tensor are true.
    /// If so, true is returned; otherwise false is returned.</para>
    /// <para>If the tensor is empty true is returned.</para>
    /// </remarks>
    /// <seealso cref="allTensor"/><seealso cref="allAxis"/>
    static member all (src: Tensor<bool>) =
        src |> Tensor.allTensor |> Tensor.value

    /// <summary>Checks if any element along the specified axis is true using this tensor as target.</summary>
    /// <param name="ax">The axis to check along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <seealso cref="anyAxis"/>
    member trgt.FillAnyAxis (ax: int) (src: Tensor<bool>) =
        let trgt = trgt.AsBool
        let src, _ = Tensor.PrepareAxisReduceSources (trgt, ax, src, None)
        trgt.Backend.AnyLastAxis (trgt=trgt, src1=src)

    /// <summary>Checks if any element along the specified axis is true.</summary>
    /// <param name="ax">The axis to check along.</param>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[true; true; true]
    ///                              [false; true; true]]
    /// let b = Tensor.anyAxis 1 a // b = [true; true]
    /// </code></example>
    /// <remarks>
    /// <para>It is checked whether any element along the specified axis is true.
    /// If so, true is returned; otherwise false is returned.</para>
    /// <para>If the tensor is empty false is returned.</para>
    /// </remarks>
    /// <seealso cref="FillAnyAxis"/><seealso cref="any"/>
    static member anyAxis (ax: int) (src: Tensor<bool>) : Tensor<bool> =
        let trgt, src = Tensor.PrepareAxisReduceTarget (ax, src)
        trgt.FillAnyAxis ax src
        trgt

    /// <summary>Checks if any element of the tensor is true returning the result as a tensor.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <seealso cref="any"/>
    static member anyTensor (src: Tensor<bool>) =
        src |> Tensor<_>.flatten |> Tensor<_>.anyAxis 0 

    /// <summary>Checks if any elements of the tensor are true.</summary>
    /// <param name="src">The tensor containing the source values.</param>    
    /// <returns>A scalar containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[true; true; true]
    ///                              [false; true; true]]
    /// let b = Tensor.any a // b = true
    /// </code></example>
    /// <remarks>
    /// <para>It is checked whether any element of the tensor is true.
    /// If so, true is returned; otherwise false is returned.</para>
    /// <para>If the tensor is empty false is returned.</para>
    /// </remarks>
    /// <seealso cref="anyTensor"/><seealso cref="anyAxis"/>
    static member any (src: Tensor<bool>) =
        src |> Tensor.anyTensor |> Tensor.value

    /// <summary>Fill this tensor with the (batched) matrix product, matrix-vector product or scalar product of the 
    /// arguments.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <seealso cref="op_DotMultiply"/>    
    member trgt.FillDot (a: Tensor<'T>) (b: Tensor<'T>) = 
        Tensor.CheckSameStorage [trgt; a; b]
        match trgt.NDims, a.NDims, b.NDims with
        | 0, 1, 1 when a.Shape = b.Shape -> 
            trgt.Backend.VecVecDot (trgt, a, b)
        | 1, 2, 1 when trgt.Shape.[0] = a.Shape.[0] && a.Shape.[1] = b.Shape.[0] -> 
            trgt.Backend.MatVecDot (trgt, a, b)
        | 2, 2, 2 when trgt.Shape.[0] = a.Shape.[0] && trgt.Shape.[1] = b.Shape.[1] &&
                       a.Shape.[1] = b.Shape.[0] ->
            trgt.Backend.MatMatDot (trgt, a, b)
        | nt, na, nb when na > 2 && nt = na && na = nb && a.Shape.[na-1] = b.Shape.[na-2] ->
            let a = a |> Tensor.broadcastTo (trgt.Shape.[0 .. na-3] @ a.Shape.[na-2 ..])
            let b = b |> Tensor.broadcastTo (trgt.Shape.[0 .. na-3] @ b.Shape.[na-2 ..])
            trgt.Backend.BatchedMatMatDot (trgt, a, b)
        | _ -> 
            invalidOp "Cannot compute dot product between tensors of shapes %A and %A 
                       into tensor of shape %A." a.Shape b.Shape trgt.Shape

    /// <summary>Computes the (batched) matrix product, (batched) matrix-vector product or scalar product.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// // Scalar product
    /// let a = HostTensor.ofList [5.0; 6.0; 7.0]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = a .* b // c = [56.0]
    ///
    /// // Matrix-vector product
    /// let a = HostTensor.ofList2D [[5.0; 6.0; 7.0]
    ///                              [8.0; 9.0; 0.0]]
    /// let b = HostTensor.ofList [2.0; 3.0; 4.0]
    /// let c = a .* b // c = [56.0; 43.0]
    ///
    /// // Matrix product
    /// let a = HostTensor.ofList2D [[5.0; 6.0; 7.0]
    ///                              [8.0; 9.0; 0.0]]
    /// let b = HostTensor.ofList2D [[2.0; 1.0] 
    ///                              [3.0; 1.0]
    ///                              [4.0; 1.0]]
    /// let c = a .* b // c = [[56.0; 18.0] 
    ///                //      [43.0; 17.0]]
    /// </code></example>
    /// <remarks>
    /// <para>If <paramref name="a"/> and <paramref name="b"/> are vectors of the same length, then the scalar
    /// product is computed. The result is a scalar.</para>
    /// <para>If <paramref name="a"/> is a matrix and <paramref name="b"/> is a vector of compatible shape, then 
    /// the matrix-vector product is computed. The result is a vector.</para>
    /// <para>If both <paramref name="a"/> and <paramref name="b"/> are matrices of compatibles shapes, then
    /// the matrix product is computed. The result is a matrix.</para>
    /// <para>If <paramref name="a"/> is a tensor of shape <c>[b_1; ...; b_n; i; j]</c> and <paramref name="b"/> 
    /// is a tensor of shape <c>[b_1; ...; b_n; j]</c>, then the batched matrix-vector product is computed resulting in
    /// a tensor of shape <c>[b_1; ...; b_n; i]</c>. Broadcasting rules apply for the batch dimensions.</para>
    /// <para>If <paramref name="a"/> is a tensor of shape <c>[b_1; ...; b_n; i; j]</c> and <paramref name="b"/> 
    /// is a tensor of shape <c>[b_1; ...; b_n; j; k]</c>, then the batched matrix product is computed resulting in
    /// a tensor of shape <c>[b_1; ...; b_n; i; k]</c>. Broadcasting rules apply for the batch dimensions.</para>
    /// </remarks>
    /// <seealso cref="FillDot"/>
    static member (.*) (a: Tensor<'T>, b: Tensor<'T>) : Tensor<'T> = 
        Tensor.CheckSameStorage [a; b]
        match a.NDims, b.NDims with
        | 1, 1 when a.Shape = b.Shape -> 
            let trgt = Tensor<'T> ([], a.Dev)
            trgt.FillDot a b
            trgt
        | 2, 1 when a.Shape.[1] = b.Shape.[0] -> 
            let trgt = Tensor<'T> ([a.Shape.[0]], a.Dev)
            trgt.FillDot a b
            trgt
        | 2, 2 when a.Shape.[1] = b.Shape.[0] -> 
            let trgt = Tensor<'T> ([a.Shape.[0]; b.Shape.[1]], a.Dev)
            trgt.FillDot a b
            trgt
        | na, nb when na > 2 && na = nb && a.Shape.[na-1] = b.Shape.[na-2] ->
            let a, b = Tensor.broadcastToSameInDims ([0 .. na-3], a, b)
            let trgt = Tensor<'T> (a.Shape.[0 .. na-3] @ [a.Shape.[na-2]; b.Shape.[na-1]], a.Dev)
            trgt.FillDot a b
            trgt
        | na, nb when na > 2 && na = nb+1 && a.Shape.[na-1] = b.Shape.[nb-1] ->
            let bPad = Tensor.padRight b
            let resPad = a .* bPad
            resPad.[Fill, 0L]
        | _ -> 
            invalidOp "Cannot compute dot product between tensors of shapes %A and %A." a.Shape b.Shape 

    /// <summary>Computes the (batched) matrix product, (batched) matrix-vector product or scalar product.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>    
    /// <seealso cref="op_DotMultiply"/>        
    static member dot (a: Tensor<'T>) (b: Tensor<'T>) =
        a .* b        

    /// <summary>Fills this tensor with the (batch) inverse of a matrix.</summary>
    /// <param name="a">The input to this operation.</param>
    /// <seealso cref="invert"/>        
    member trgt.FillInvert (a: Tensor<'T>)  = 
        Tensor.CheckSameStorage [trgt; a]
        if a.NDims < 2 then
            invalidArg "a" "Need at least a matrix to invert but got shape %A." a.Shape
        let a = a |> Tensor.broadcastTo trgt.Shape
        trgt.Backend.BatchedInvert (trgt, a)

    /// <summary>(Batch) inverts a matrix.</summary>
    /// <param name="a">The input matrix or tensor to this operation.</param>
    /// <returns>A new matrix or tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0]
    ///                            [3.0; 4.0]]
    /// let c = Tensor.invert a // c = [[-2.0; 1.0]    
    ///                         //      [1.5; -0.5]]  
    /// </code></example>
    /// <remarks>
    /// <para>If <paramref name="a"/> is a square matrix, its inverse is computed. The result is a matrix.</para>
    /// <para>If <paramref name="a"/> is a tensor of shape <c>[b_1; ...; b_n; i; i]</c>, the inverse
    /// of all square matrices consisting of the last two dimensions of the tensor are computed. 
    /// The result is a tensor of same shape.</para>
    /// <para>If the matrix is not invertible a <see cref="Tensor.SingularMatrixException"/> is raised.
    /// Use <see cref="pseudoInvert"/> for such matrices instead.</para>
    /// </remarks>
    /// <exception cref="Tensor.SingularMatrixException">Raised when the matrix is not invertible.</exception>
    /// <seealso cref="FillInvert"/><seealso cref="pseudoInvert"/>         
    static member invert (a: Tensor<'T>) = 
        let trgt = Tensor<'T> (a.Shape, a.Dev)
        trgt.FillInvert a
        trgt

    /// Computes the sizes of an SVD decomposition.
    static member internal SVDSizes (a: ITensorFrontend<'T>) =
        if a.NDims < 2 then
            invalidArg "a" "Need at least a matrix to SVD but got shape %A." a.Shape
        let batchShp = a.Shape.[0 .. a.NDims-3]
        let M, N = a.Shape.[a.NDims-2], a.Shape.[a.NDims-1]
        let K = min M N
        batchShp, M, N, K

    /// <summary>Fills this tensor with the (batched) singular values of the specified matrix.</summary>
    /// <param name="a">The input matrix or tensor to this operation.</param>
    /// <param name="trgtUV">The optional target tensors for the transformation matrices.</param>
    /// <remarks>
    /// <para>The singular values are stored in this vector.</para>
    /// </remarks>
    /// <seealso cref="SVD"/><seealso cref="SVDWithoutUV"/>
    member trgtS.FillSVD (a: Tensor<'T>, ?trgtUV: Tensor<'T> * Tensor<'T>) =
        let batchShp, M, N, K = Tensor.SVDSizes a
        Tensor.CheckSameStorage [trgtS; a]
        if trgtS.Shape <> batchShp @ [K] then
            invalidArg "trgtS" "Need a tensor of shape %A for SVD singular values but got shape %A."
                               (batchShp @ [K]) trgtS.Shape
        match trgtUV with
        | Some (trgtU, trgtV) -> 
            Tensor.CheckSameStorage [trgtS; a; trgtU; trgtV]
            if trgtU.Shape <> batchShp @ [M; M] then
                invalidArg "trgtUV" "Need a tensor of shape %A for SVD left unitary matrices but got shape %A."
                                    (batchShp @ [M; M]) trgtU.Shape
            if trgtV.Shape <> batchShp @ [N; N] then
                invalidArg "trgtUV" "Need a tensor of shape %A for SVD right unitary matrices but got shape %A."
                                    (batchShp @ [N; N]) trgtV.Shape
        | None -> ()
        let trgtUV = trgtUV |> Option.map (fun (trgtU, trgtV) -> trgtU :> ITensorFrontend<_>, 
                                                                 trgtV :> ITensorFrontend<_>)
        trgtS.Backend.BatchedSVD (trgtS, trgtUV, a)                

    /// <summary>Computes the (batched) singular value decomposition (SVD) of the specified matrix.</summary>
    /// <param name="a">The input matrix or tensor to this operation.</param>
    /// <returns>A tuple consisting of <c>(U, S, V)</c> where <c>S</c> is a vector consisting of the singular values and
    /// <c>U</c>, <c>V</c> are the transformation matrices.</returns>
    /// <remarks>
    /// <para>The singular value decomposition of matrix <paramref name="a"/> is computed.
    /// It is defined by the property that <c>a = U .* Tensor.diagMat(S) .* V.T</c>.</para>
    /// <para>If <paramref name="a"/> is a tensor, the operation is batched over the matrices consisting
    /// of the last two dimensions.</para>
    /// </remarks>    
    /// <seealso cref="FillSVD"/><seealso cref="SVDWithoutUV"/>
    static member SVD (a: Tensor<'T>) =
        let batchShp, M, N, K = Tensor.SVDSizes a
        let U = Tensor<'T> (batchShp @ [M;M], a.Dev, order=ColumnMajor)
        let S = Tensor<'T> (batchShp @ [K], a.Dev, order=ColumnMajor)
        let V = Tensor<'T> (batchShp @ [N;N], a.Dev, order=RowMajor)
        S.FillSVD(a, trgtUV=(U, V))
        U, S, V

    /// <summary>Computes the (batched) singular values of the specified matrix.</summary>
    /// <param name="a">The input matrix or tensor to this operation.</param>
    /// <returns>A vector consisting of the singular values.</returns>
    /// <seealso cref="SVD"/>
    static member SVDWithoutUV (a: Tensor<'T>) =
        let batchShp, M, N, K = Tensor.SVDSizes a
        let S = Tensor<'T> (batchShp @ [K], a.Dev, order=ColumnMajor)
        S.FillSVD(a)
        S

    /// <summary>Fills this tensor with the (batched) Moore-Penrose pseudo-inverse of the specified matrix.</summary>
    /// <param name="a">The input matrix or tensor to this operation.</param>
    /// <param name="rCond">The cut-off value for the singular values. (default: 1e-15)</param>
    /// <seealso cref="pseudoInvert"/>
    member trgt.FillPseudoInvert (a: Tensor<'T>, ?rCond: 'T)  = 
        let rCond = defaultArg rCond (conv<'T> 1e-15)
        Tensor.CheckSameStorage [trgt; a]
        if a.NDims < 2 then
            invalidArg "a" "Need at least a matrix to pseudo invert but got shape %A." a.Shape
        let a = a |> Tensor.broadcastTo trgt.Shape

        let u, s, v = Tensor.SVD a
        let rCond = Tensor.scalarLike s rCond
        let zero = Tensor.scalarLike s (conv<'T> 0)
        let one = Tensor.scalarLike s (conv<'T> 1)
        s.FillIfThenElse (s >>== rCond) (one / s) (zero)
        trgt.FillDot (v) (Tensor.padRight s * u.T)

    /// <summary>Computes the (batched) Moore-Penrose pseudo-inverse of the specified matrix.</summary>
    /// <param name="a">The input matrix or tensor to this operation.</param>
    /// <param name="rCond">The cut-off value for the singular values. (default: 1e-15)</param>
    /// <returns>A new matrix or tensor containing the result of this operation.</returns>
    /// <remarks>    
    /// <para>If <paramref name="a"/> is a matrix, its pseudo-inverse is computed. The result is a matrix.</para>
    /// <para>If <paramref name="a"/> is a tensor of shape <c>[b_1; ...; b_n; i; j]</c>, the pseudo-inverse
    /// of all matrices consisting of the last two dimensions of the tensor are computed. 
    /// The result is a tensor of shape <c>[b_1; ...; b_n; j; i]</c>.</para>
    /// </remarks>
    /// <seealso cref="FillPseudoInvert"/><seealso cref="invert"/>   
    static member pseudoInvert (a: Tensor<'T>, ?rCond: 'T) = 
        let trgt = Tensor<'T> (a.Shape, a.Dev)
        trgt.FillPseudoInvert (a, ?rCond=rCond)
        trgt

    /// <summary>Computes the (real) eigendecomposition of a symmetric matrix and writes it into the specified 
    /// target tensors.</summary>
    /// <param name="part">Specifies which part of the matrix should be used.</param>
    /// <param name="trgtEigVals">The target vector that will receive the eigenvalues.</param>
    /// <param name="trgtEigVecs">The target matrix that will receive the eigenvectors.</param>    
    /// <param name="a">The input matrix to this operation.</param>
    /// <seealso cref="symmetricEigenDecomposition"/>
    static member FillSymmetricEigenDecomposition (part: MatrixPart)
            (trgtEigVals: Tensor<'T>) (trgtEigVecs: Tensor<'T>) (a: Tensor<'T>) =
        Tensor.CheckSameStorage [trgtEigVals; trgtEigVecs; a]
        if a.NDims <> 2 || a.Shape.[0] <> a.Shape.[1] then 
            invalidArg "a" "Require a square matrix for symmetric eigen-decomposition but got %A." a.Shape
        if trgtEigVecs.Shape <> a.Shape then
            invalidArg "trgtEigVecs" "trgtEigVecs and src must have the same shapes but got %A and %A." 
                                     trgtEigVecs.Shape a.Shape
        if trgtEigVals.NDims <> 1 || trgtEigVals.Shape.[0] <> a.Shape.[0] then
            invalidArg "trgtEigVals" "trgtEigVals must be a vector of length %d but it has shape %A."
                                     a.Shape.[0] trgtEigVals.Shape
        trgtEigVals.Backend.SymmetricEigenDecomposition (part, trgtEigVals, trgtEigVecs, a)

    /// <summary>Computes the (real) eigendecomposition of a symmetric matrix.</summary>
    /// <param name="part">Specifies which part of the matrix should be used.</param>
    /// <param name="a">The input matrix to this operation.</param>
    /// <returns>A tuple consisting of <c>(vals, vecs)</c> where each column of <c>vecs</c> is the eigenvector for the
    /// corresponding eigenvalue in <c>vals</c>.</returns>
    /// <remarks>
    /// <para>The eigendecomposition of a symmetric matrix is real.
    /// Only the part of the matrix specified by <paramref name="part"/> is used. The other part is ignored and can
    /// contain arbitrary values.</para>
    /// </remarks>
    /// <seealso cref="FillSymmetricEigenDecomposition"/>
    static member symmetricEigenDecomposition (part: MatrixPart) (a: Tensor<'T>) =
        if a.NDims <> 2 then
            invalidArg "a" "require a square matrix for symmetric eigen-decomposition"
        let trgtEigVals = Tensor<'T> ([a.Shape.[0]], a.Dev)
        let trgtEigVecs = Tensor<'T> (a.Shape, a.Dev, order=ColumnMajor)
        Tensor.FillSymmetricEigenDecomposition part trgtEigVals trgtEigVecs a
        trgtEigVals, trgtEigVecs
        
    // Helper functions for getting slices.
    member inline internal this.GetRng (rngArgs: obj[]) =
        this.Range (Rng.ofItemOrSliceArgs rngArgs) 
    member inline internal this.IGetRng (rngArgs: obj[]) =
        this.GetRng rngArgs :> ITensor
    member inline internal this.GetRngWithRest (rngArgs: obj[]) (restArgs: obj[]) =
        Array.concat [rngArgs; restArgs] |> this.GetRng
    member inline internal this.IGetRngWithRest (rngArgs: obj[]) (restArgs: obj[]) =
        Array.concat [rngArgs; restArgs] |> this.IGetRng

    // Helper functions for setting slices.
    member inline internal this.SetRng (rngArgs: obj[]) (value: Tensor<'T>) =
        Tensor.CheckSameStorage [this; value]
        let trgt = this.Range (Rng.ofItemOrSliceArgs rngArgs) 
        value |> Tensor<_>.broadcastTo trgt.Shape |> trgt.CopyFrom
    member inline internal this.ISetRng (rngArgs: obj[]) (value: ITensor) =
        match value with
        | :? Tensor<'T> as value -> this.SetRng rngArgs value
        | _ ->
            invalidOp "Cannot assign data type %s to tensor of data type %s." value.DataType.Name this.DataType.Name
    member inline internal this.SetRngWithRest (rngArgs: obj[]) (restArgs: obj[]) =
        let allArgs = Array.concat [rngArgs; restArgs]
        let value = Array.last allArgs :?> Tensor<'T>
        let args = allArgs.[0 .. allArgs.Length-2]
        this.SetRng args value
    member inline internal this.ISetRngWithRest (rngArgs: obj[]) (restArgs: obj[]) =
        let allArgs = Array.concat [rngArgs; restArgs]
        let value = Array.last allArgs :?> ITensor
        let args = allArgs.[0 .. allArgs.Length-2]
        this.ISetRng args value

    /// Computes the shape of the targets and sources of a mask operation. 
    static member internal MaskShapes (masks: Tensor<bool> option list) (shape: int64 list) =
        match masks with
        | None :: rMasks ->
            // non-specified mask consumes one dimension of tensor
            match List.tryHead shape with
            | Some s -> (s, s) :: Tensor<_>.MaskShapes rMasks (List.tail shape)
            | None -> invalidArg "masks" "Dimension mismatch between masks and tensor shape."
        | Some mask :: rMasks ->
            // specified mask consumes as many dimensions as it has
            if mask.NDims > List.length shape then 
                invalidArg "masks" "Dimension mismatch between masks and tensor shape."
            let s, rShape = shape.[..mask.NDims-1], shape.[mask.NDims..]
            if mask.Shape <> s then
                invalidArg "masks" "Shape of mask %A does not match part %A of tensor shape it applies to." mask.Shape s
            (Tensor<_>.countTrue mask, mask.NElems) :: Tensor<_>.MaskShapes rMasks rShape
        | [] ->
            if not (List.isEmpty shape) then
                invalidArg "masks" "Dimension mismatch between masks and tensor shape."
            []

    /// Converts a list of masks (which may be null) to a list of mask options.
    static member internal MaskOptions (masks: Tensor<bool> list) =
        masks |> List.map (fun m -> match box m with
                                    | null -> None
                                    | _ -> Some m)

    /// Cast ITensor to Tensor<bool> for use as mask.
    static member internal MaskAsBoolTensor (m: ITensor) =
        match m with
        | :? Tensor<bool> as m -> m
        | _ -> invalidArg "mask" "Masks must be of type Tensor<bool>."         

    /// Collect all elements of this tensor where mask is true.
    member internal this.MaskedGet (masks: Tensor<bool> list) =
        let masks = Tensor<_>.MaskOptions masks
        masks |> List.iter (Option.iter (fun m -> Tensor.CheckSameStorage [this; m]))
        let trgtShp, srcShp = Tensor<_>.MaskShapes masks this.Shape |> List.unzip
        let trgt = Tensor<'T> (trgtShp, this.Dev)
        let src = this |> Tensor.reshape srcShp
        let masks = masks |> List.map (Option.map (fun t -> t |> Tensor<_>.flatten :> ITensorFrontend<_>)) |> List.toArray
        backend.MaskedGet (trgt=trgt, src=src, masks=masks) 
        trgt    
    member inline internal this.IMaskedGet (masks: ITensor list) = 
        this.MaskedGet (masks |> List.map Tensor<_>.MaskAsBoolTensor) :> ITensor    

    /// Set all elements of this tensor where mask is true to the specfied values.
    member internal this.MaskedSet (masks: Tensor<bool> list) (value: Tensor<'T>) =
        let masks = Tensor<_>.MaskOptions masks
        Tensor.CheckSameStorage [this; value]
        masks |> List.iter (Option.iter (fun m -> Tensor.CheckSameStorage [this; m]))       
        let valueShp, trgtShp = Tensor<_>.MaskShapes masks this.Shape |> List.unzip        
        let masks = masks |> List.map (Option.map (fun t -> t |> Tensor<_>.flatten :> ITensorFrontend<_>)) |> List.toArray
        let value = value |> Tensor<_>.broadcastTo valueShp        
        match this |> Tensor.tryReshapeView trgtShp with
        | Some trgtView -> backend.MaskedSet (trgt=trgtView, masks=masks, src=value)
        | None ->
            let trgt = this |> Tensor.reshape trgtShp
            backend.MaskedSet (trgt=trgt, masks=masks, src=value)
            this.CopyFrom (trgt |> Tensor.reshape this.Shape)       
    member inline internal this.IMaskedSet (masks: ITensor list) (value: ITensor) =         
        match value with
        | :? Tensor<'T> as value -> this.MaskedSet (masks |> List.map Tensor<_>.MaskAsBoolTensor) value
        | _ ->
            invalidOp "Cannot assign data type %s to tensor of data type %s." value.DataType.Name this.DataType.Name

    /// <summary>Accesses a single element within the tensor.</summary>
    /// <param name="idx">An array consisting of the indicies of the element to access. The arry must have one entry
    /// per dimension of this tensor.</param>
    /// <value>The value of the selected element.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0]
    ///                            [3.0; 4.0]]
    /// let b = a.[[|1L; 1L|]] // b = 4.0
    /// a.[[|1L; 0L|]] &lt;- 6.0 // a = [[1.0; 2.0]
    ///                       //      [6.0; 4.0]]                        
    /// </code></example>    
    /// <remarks>
    /// <para>Indexing is zero-based.</para>
    /// </remarks>
    /// <exception cref="System.IndexOutOfRangeException">Raised when the specified indicies are out of range.</exception>
    /// <seealso cref="Item(Microsoft.FSharp.Collections.FSharpList{System.Int64})"/>
    member this.Item
        with get (idx: int64[]) : 'T = backend.[idx]
        and set (idx: int64[]) (value: 'T) = backend.[idx] <- value
          
    /// <summary>Accesses a single element within the tensor.</summary>
    /// <param name="idx">A list consisting of the indicies of the element to access. The list must have one entry
    /// per dimension of this tensor.</param>
    /// <value>The value of the selected element.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0]
    ///                            [3.0; 4.0]]
    /// let b = a.[[1L; 1L]] // b = 4.0
    /// a.[[1L; 0L]] &lt;- 6.0 // a = [[1.0; 2.0]
    ///                     //      [6.0; 4.0]]                        
    /// </code></example>    
    /// <remarks>
    /// <para>Indexing is zero-based.</para>
    /// <para>Use <see cref="Item(System.Int64[])"/> for faster element access.</para>
    /// </remarks>
    /// <exception cref="System.IndexOutOfRangeException">Raised when the specified indicies are out of range.</exception>
    /// <seealso cref="Item(System.Int64[])"/><seealso cref="Value"/>
    member this.Item
        with get (idx: int64 list) : 'T = backend.[Array.ofList idx]
        and set (idx: int64 list) (value: 'T) = backend.[Array.ofList idx] <- value

    /// <summary>Picks elements from a tensor using one or more boolean mask tensors.</summary>
    /// <param name="m0">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <value>All elements from the tensor for which the mask is true.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0; 3.0]
    ///                            [4.0; 5.0; 6.0]]
    ///
    /// // masked get with one mask for the whole tensor
    /// let m = HostTensor.ofList [[true;  true;  false]
    ///                            [false; false; true ]]
    /// let b = a.M(m) // b = [1.0; 2.0; 6.0]
    ///
    /// // an element-wise comparison operator can be used to create the mask
    /// let c = a.M(a >>>> 3.5) // c = [4.0; 5.0; 6.0]
    ///
    /// // masked get with one mask per dimension
    /// let m0 = HostTensor.ofList [true; false]
    /// let m1 = HostTensor.ofList [false; false; true]
    /// let d = a.M(m0, m1) // d = [3.0]
    ///
    /// // masked get using only one dimension
    /// let m0 = HostTensor.ofList [true; false]
    /// let e = a.M(m0, NoMask) // e = [[1.0; 2.0; 3.0]]
    ///
    /// // masked set with one mask for the whole tensor
    /// let m = HostTensor.ofList [[true;  true;  false]
    ///                            [false; false; true ]]
    /// a.M(m) &lt;- [8.0; 9.0; 0.0]  // a = [[8.0; 9.0; 3.0]
    ///                            //      [4.0; 5.0; 0.0]]
    /// </code></example>    
    /// <remarks>
    /// <para>Masking picks elements from the tensor for which the corresponding element in the mask tensor is true.
    /// The mask can, for example, be generated by one or more element-wise comparison operation.</para>
    /// <para>The get operation returns a copy of the selected elements of the tensor.</para>
    /// <para>The set operation replaces the selected elements with a copy of the specified tensor.</para>
    /// <para>If a dimension should not be masked, specify <see cref="Tensor.NoMask"/> instead of a mask tensor.</para>
    /// <para>For clarity the documentation does not list all overloads of <c>M</c>.
    /// However, this masking method can be used for up to 5 dimensions, as shown in the example.
    /// For programmatically generated ranges or for more than 5 dimensions, the mask specification variant 
    /// <see cref="M(Microsoft.FSharp.Collections.FSharpList{Tensor.Tensor{System.Boolean}})"/> is available.</para>
    /// <para>Currently this operation is only supported for tensors stored on the host. Support for CUDA tensors is
    /// planned in the future.</para>
    /// </remarks>
    /// <exception cref="System.InvalidArgumentException">Raised when the mask is incompatible with the tensor.</exception>
    /// <seealso cref="M(Microsoft.FSharp.Collections.FSharpList{Tensor.Tensor{System.Boolean}})"/>
    member this.M
        with get (m0: Tensor<bool>) = this.MaskedGet [m0]
        and set (m0: Tensor<bool>) (value: Tensor<'T>) = this.MaskedSet [m0] value                          

    /// <summary>Picks elements from a tensor using one or more boolean mask tensors.</summary>
    /// <param name="m0">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m1">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    member this.M
        with get (m0: Tensor<bool>, m1: Tensor<bool>) = this.MaskedGet [m0; m1]
        and set (m0: Tensor<bool>, m1: Tensor<bool>) (value: Tensor<'T>) = this.MaskedSet [m0; m1] value                          

    /// <summary>Picks elements from a tensor using one or more boolean mask tensors.</summary>
    /// <param name="m0">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m1">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m2">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>    
    member this.M
        with get (m0: Tensor<bool>, m1: Tensor<bool>, m2: Tensor<bool>) = this.MaskedGet [m0; m1; m2]
        and set (m0: Tensor<bool>, m1: Tensor<bool>, m2: Tensor<bool>) (value: Tensor<'T>) = this.MaskedSet [m0; m1; m2] value                          

    /// <summary>Picks elements from a tensor using one or more boolean mask tensors.</summary>
    /// <param name="m0">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m1">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m2">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m3">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>    
    member this.M
        with get (m0: Tensor<bool>, m1: Tensor<bool>, m2: Tensor<bool>, m3: Tensor<bool>) = this.MaskedGet [m0; m1; m2; m3]
        and set (m0: Tensor<bool>, m1: Tensor<bool>, m2: Tensor<bool>, m3: Tensor<bool>) (value: Tensor<'T>) = this.MaskedSet [m0; m1; m2; m3] value                          

    /// <summary>Picks elements from a tensor using one or more boolean mask tensors.</summary>
    /// <param name="m0">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m1">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m2">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>
    /// <param name="m3">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>    
    /// <param name="m4">A boolean mask tensor or <see cref="Tensor.NoMask"/>.</param>    
    member this.M
        with get (m0: Tensor<bool>, m1: Tensor<bool>, m2: Tensor<bool>, m3: Tensor<bool>, m4: Tensor<bool>) = this.MaskedGet [m0; m1; m2; m3; m4]
        and set (m0: Tensor<bool>, m1: Tensor<bool>, m2: Tensor<bool>, m3: Tensor<bool>, m4: Tensor<bool>) (value: Tensor<'T>) = this.MaskedSet [m0; m1; m2; m3; m4] value                          

    /// <summary>Picks elements from a tensor using one or more boolean mask tensors.</summary>
    /// <param name="masks">A list of boolean mask tensors or <see cref="Tensor.NoMask"/>.</param>
    /// <value>All elements from the tensor for which the mask is true.</value>
    /// <remarks>
    /// <para>Masking picks elements from the tensor for which the corresponding element in the mask tensor is true.
    /// The mask can, for example, be generated by one or more element-wise comparison operation.</para>
    /// <para>The get operation returns a copy of the selected elements of the tensor.</para>
    /// <para>The set operation replaces the selected elements with a copy of the specified tensor.</para>
    /// <para>If a dimension should not be masked, specify <see cref="Tensor.NoMask"/> instead of a mask tensor.</para>
    /// <para>This mask specification variant is intended for programmatically generated ranges. For most use cases
    /// the variant <seealso cref="M(Tensor.Tensor{System.Boolean})"/> is more succinct and thus the 
    /// recommended method.</para>    
    /// </remarks>
    /// <exception cref="System.InvalidArgumentException">Raised when the mask is incompatible with the tensor.</exception>
    /// <seealso cref="M(Tensor.Tensor{System.Boolean})"/>
    member this.M
        with get (masks: Tensor<bool> list) = this.MaskedGet masks
        and set (masks: Tensor<bool> list) (value: Tensor<'T>) = this.MaskedSet masks value                          

    /// <summary>Accesses a slice (part) of the tensor.</summary>
    /// <param name="rng">The range of the tensor to select.</param>
    /// <value>A view of the selected part of the tensor.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0; 3.0]
    ///                            [4.0; 5.0; 6.0]]
    ///
    /// // get view
    /// let b = a.[[Rng.Elem 0L; Rng.Elem 1L]] // b = 2.0
    /// let c = a.[[Rng.Elem 0L; Rng.All]] // b = [1.0; 2.0; 3.0]
    /// let d = a.[[Rng.Elem 1L; Rng.Rng (Some 0L, Some 1L)]] // b = [4.0; 5.0]
    /// let e = a.[[Rng.Rng (Some 1L, Some 1L); Rng (Some 0L, Some 1L)]] // b = [[4.0; 5.0]]
    ///
    /// // set view
    /// a.[[Rng.Elem 0L; Rng.All]] &lt;- HostTensor.ofList [7.0; 8.0; 9.0] // a = [[7.0; 8.0; 9.0]
    ///                                                                 //      [4.0; 5.0; 6.0]]
    ///
    /// // modifiying view affects original tensor
    /// d.[[1L]] &lt;- 0.0 // a = [[7.0; 8.0; 9.0]
    ///                 //      [4.0; 0.0; 6.0]]
    /// </code></example>    
    /// <remarks>
    /// <para>This range specification variant is intended for programmatically generated ranges. For most use cases
    /// the variant <seealso cref="Item(System.Int64)"/> allows vastly simpler range specifications and is the 
    /// recommended method.</para>
    /// <para>Indexing is zero-based.</para>
    /// <para>This indexing options allows to select a part (called slice) of the tensor.</para>
    /// <para>The get operation returns a view of the specified part of the tensor. Modifications done to that
    /// view will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// <para>See <see cref="Tensor.Rng"/> for available range specifications.</para>
    /// </remarks>
    /// <exception cref="System.IndexOutOfRangeException">Raised when the specified range is out of range.</exception>
    /// <seealso cref="Item(System.Int64)"/>
    member this.Item
        with get (rng: Rng list) = this.GetRng [|rng|]
        and set (rng: Rng list) (value: Tensor<'T>) = this.SetRng [|rng|] value

    /// <summary>Accesses a slice (part) of the tensor.</summary>
    /// <param name="i0">The range of the tensor to select.</param>
    /// <value>A view of the selected part of the tensor.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList [[1.0; 2.0; 3.0]
    ///                            [4.0; 5.0; 6.0]]
    ///
    /// // get view
    /// let b = a.[0L, 1L] // b = 2.0
    /// let c = a.[0L, *] // b = [1.0; 2.0; 3.0]
    /// let d = a.[1L, 0L..1L] // b = [4.0; 5.0]
    /// let e = a.[1L..1L, 0L..1L] // b = [[4.0; 5.0]]
    ///
    /// // set view
    /// a.[0L, *] &lt;- HostTensor.ofList [7.0; 8.0; 9.0] // a = [[7.0; 8.0; 9.0]
    ///                                                //      [4.0; 5.0; 6.0]]
    ///
    /// // modifiying view affects original tensor
    /// d.[[1L]] &lt;- 0.0 // a = [[7.0; 8.0; 9.0]
    ///                 //      [4.0; 0.0; 6.0]]
    /// </code></example>    
    /// <remarks>
    /// <para>Indexing is zero-based.</para>
    /// <para>This indexing options allows to select a part (called slice) of the tensor.</para>
    /// <para>The get operation returns a view of the specified part of the tensor. Modifications done to that
    /// view will affect the original tensor. Also, modifying the orignal tensor will affect the view.</para>
    /// <para>The slicing specifications follows standard F# practice. 
    /// Specifying an integer for the index of a dimension, selects that index for the dimension.
    /// Specifying <c>*</c> for a dimension, selects all indices of the dimension.
    /// Specifying <c>f..l</c> for a dimension, select all indices from <c>f</c> to (including) <c>l</c> for the dimension.
    /// </para>
    /// <para>For clarity the documentation does not list all overloads of the Item property and GetSlice, 
    /// SetSlice methods. However, this slicing method can be used for up to 5 dimensions, as shown in the example.
    /// For programmatically generated ranges or for more than 5 dimensions, the range specification variant 
    /// <seealso cref="Item(Microsoft.FSharp.Collections.FSharpList{Tensor.Rng})"/> is available.</para>
    /// </remarks>
    /// <exception cref="System.IndexOutOfRangeException">Raised when the specified range is out of range.</exception>
    /// <seealso cref="Item(Microsoft.FSharp.Collections.FSharpList{Tensor.Rng})"/>
    member this.Item
        with get (i0: int64) = this.GetRng [|i0|]
        and set (i0: int64) (value: Tensor<'T>) = this.SetRng [|i0|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option) = this.GetRng [|i0s; i0f|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, value: Tensor<'T>) = this.SetRng [|i0s; i0f|] value

    // two-dimensional slicing using indices and special axes
    member this.Item
        with get (i0: int64, i1: int64) = this.GetRng [|i0; i1|]
        and set (i0: int64, i1: int64) (value: Tensor<'T>) = this.SetRng [|i0; i1|] value
    member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option) = this.GetRng [|i0; i1s; i1f|]
    member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, value: Tensor<'T>) = this.SetRng [|i0; i1s; i1f|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64) = this.GetRng [|i0s; i0f; i1|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, value: Tensor<'T>) = this.SetRng [|i0s; i0f; i1|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option) = this.GetRng [|i0s; i0f; i1s; i1f|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, value: Tensor<'T>) = this.SetRng [|i0s; i0f; i1s; i1f|] value

    // three-dimensional slicing using indices and special axes
    member this.Item
        with get (i0: int64, i1: int64, i2: int64) = this.GetRng [|i0; i1; i2|]
        and set (i0: int64, i1: int64, i2: int64) (value: Tensor<'T>) = this.SetRng [|i0; i1; i2|] value
    member this.GetSlice (i0: int64, i1: int64, i2: int64) = this.GetRng [|i0; i1; i2|]
    member this.SetSlice (i0: int64, i1: int64, i2: int64, value: Tensor<'T>) = this.SetRng [|i0; i1; i2|] value
    member this.GetSlice (i0: int64, i1: int64, i2s: int64 option, i2f: int64 option) = this.GetRng [|i0; i1; i2s; i2f|]
    member this.SetSlice (i0: int64, i1: int64, i2s: int64 option, i2f: int64 option, value: Tensor<'T>) = this.SetRng [|i0; i1; i2s; i2f|] value
    member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2: int64) = this.GetRng [|i0; i1s; i1f; i2|]
    member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2: int64, value: Tensor<'T>) = this.SetRng [|i0; i1s; i1f; i2|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2: int64) = this.GetRng [|i0s; i0f; i1; i2|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2: int64, value: Tensor<'T>) = this.SetRng [|i0s; i0f; i1; i2|] value
    member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option) = this.GetRng [|i0; i1s; i1f; i2s; i2f|]
    member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, value: Tensor<'T>) = this.SetRng [|i0; i1s; i1f; i2s; i2f|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2s: int64 option, i2f: int64 option) = this.GetRng [|i0s; i0f; i1; i2s; i2f|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2s: int64 option, i2f: int64 option, value: Tensor<'T>) = this.SetRng [|i0s; i0f; i1; i2s; i2f|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2: int64) = this.GetRng [|i0s; i0f; i1s; i1f; i2|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2: int64, value: Tensor<'T>) = this.SetRng [|i0s; i0f; i1s; i1f; i2|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option) = this.GetRng [|i0s; i0f; i1s; i1f; i2s; i2f|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, value: Tensor<'T>) = this.SetRng [|i0s; i0f; i1s; i1f; i2s; i2f|] value

    // four- and more-dimensional slicing using indices and special axes
    member this.Item
        with get (o0: obj, o1: obj, o2: obj, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|o0; o1; o2; o3|] r
        and set (o0: obj, o1: obj, o2: obj, o3: obj) (value: Tensor<'T>) = this.SetRng [|o0; o1; o2; o3|] value
    member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj) (value: Tensor<'T>) = this.SetRng [|o0; o1; o2; o3; o4|] value
    member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj) (value: Tensor<'T>) = this.SetRng [|o0; o1; o2; o3; o4; o5|] value
    member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj, o6: obj) (value: Tensor<'T>) = this.SetRng [|o0; o1; o2; o3; o4; o5; o6|] value
    member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj, o6: obj, o7: obj) (value: Tensor<'T>) = this.SetRng [|o0; o1; o2; o3; o4; o5; o6; o7|] value
    member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj, o6: obj, o7: obj, o8: obj) (value: Tensor<'T>) = this.SetRng [|o0; o1; o2; o3; o4; o5; o6; o7; o8|] value
    member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj, o6: obj, o7: obj, o8: obj, o9: obj) (value: Tensor<'T>) = this.SetRng [|o0; o1; o2; o3; o4; o5; o6; o7; o8; o9|] value
    member this.GetSlice (i0: int64, i1: int64, i2: int64, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|i0; i1; i2; o3|] r
    member this.SetSlice (i0: int64, i1: int64, i2: int64, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.SetRngWithRest [|i0; i1; i2; o3; o4|] r
    member this.GetSlice (i0: int64, i1: int64, i2s: int64 option, i2f: int64 option, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|i0; i1; i2s; i2f; o3|] r
    member this.SetSlice (i0: int64, i1: int64, i2s: int64 option, i2f: int64 option, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.SetRngWithRest [|i0; i1; i2s; i2f; o3; o4|] r
    member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2: int64, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|i0; i1s; i1f; i2; o3|] r
    member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2: int64, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.SetRngWithRest [|i0; i1s; i1f; i2; o3; o4|] r
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2: int64, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|i0s; i0f; i1; i2; o3|] r
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2: int64, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.SetRngWithRest [|i0s; i0f; i1; i2; o3; o4|] r
    member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|i0; i1s; i1f; i2s; i2f; o3|] r
    member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.SetRngWithRest [|i0; i1s; i1f; i2s; i2f; o3; o4|] r
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2s: int64 option, i2f: int64 option, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|i0s; i0f; i1; i2s; i2f; o3|] r
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2s: int64 option, i2f: int64 option, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.SetRngWithRest [|i0s; i0f; i1; i2s; i2f; o3; o4|] r
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2: int64, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|i0s; i0f; i1s; i1f; i2; o3|] r
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2: int64, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.SetRngWithRest [|i0s; i0f; i1s; i1f; i2; o3; o4|] r
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, o3: obj, [<System.ParamArray>] r: obj[]) = this.GetRngWithRest [|i0s; i0f; i1s; i1f; i2s; i2f; o3|] r
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.SetRngWithRest [|i0s; i0f; i1s; i1f; i2s; i2f; o3; o4|] r

    /// <summary>Gets the value of a single element of the tensor.</summary>
    /// <param name="a">The tensor to read from.</param>
    /// <param name="pos">A list consisting of the indicies of the element to access. The list must have one entry
    /// per dimension of this tensor.</param>
    /// <returns>The value of the selected element.</returns>
    /// <seealso cref="Item(Microsoft.FSharp.Collections.FSharpList{System.Int64})"/>
    static member inline get (a: Tensor<'T>) (pos: int64 list) = 
        a.[pos]
    
    /// <summary>Sets the value of a single element of the tensor.</summary>
    /// <param name="a">The tensor to write to.</param>
    /// <param name="pos">A list consisting of the indicies of the element to access. The list must have one entry
    /// per dimension of this tensor.</param>
    /// <param name="value">The new value of the element.</param>
    /// <seealso cref="Item(Microsoft.FSharp.Collections.FSharpList{System.Int64})"/>
    static member inline set (a: Tensor<'T>) (pos: int64 list) value = 
        a.[pos] <- value

    /// Checks that this Tensor is a scalar tensor.
    member inline internal this.CheckScalar () =
        if this.NDims <> 0 then 
            indexOutOfRange "This operation requires a scalar (0-dimensional) tensor, but its shape is %A." this.Shape

    /// <summary>Accesses the value of a zero-dimensional (scalar) tensor.</summary>
    /// <value>The scalar value of the tensor.</value>
    /// <example><code language="fsharp">
    /// let a = HostTensor.sclar 2.0
    /// let b = a.Value // 2.0
    /// a.Value &lt;- 3.0 // a = 3.0
    /// </code></example>
    /// <remarks>
    /// <para>Gets or sets the value of a scalar tensor.</para>
    /// <para>The tensor must have zero dimensions.</para>
    /// </remarks>
    /// <exception cref="System.IndexOutOfRangeException">Raised when the tensor is not zero-dimensional.</exception>
    /// <seealso cref="Item(Microsoft.FSharp.Collections.FSharpList{System.Int64})"/>
    member this.Value 
        with get () = 
            this.CheckScalar()
            this.[[||]]
        and set value = 
            this.CheckScalar()
            this.[[||]] <- value

    /// <summary>Gets the value of a zero-dimensional (scalar) tensor.</summary>
    /// <param name="a">The zero-dimensional tensor to read from.</param>    
    /// <returns>The scalar value of the tensor.</returns>
    /// <seealso cref="Value"/>
    static member value (a: Tensor<'T>) : 'T =
        a.Value

    /// <summary>String representation of the tensor limited to a specific number of elements per dimension.</summary>
    /// <param name="maxElems">Maximum number of element per dimension to include in string representation.</param>
    /// <returns>A (shortened) string representation of this tensor</returns>
    /// <seealso cref="Full"/><seealso cref="Pretty"/>
    member this.ToString (maxElems) =
        let rec prettyDim lineSpace (a: Tensor<'T>) =
            let ls () = a.Shape.[0]
            let subPrint idxes = 
                idxes
                |> Seq.map (fun i -> 
                    prettyDim (lineSpace + " ") (a.[i, Fill])) 
                |> Seq.toList                   
            let subStrs () = 
                if ls() <= maxElems then
                    subPrint (seq {0L .. ls() - 1L})
                else
                    let leftTo = maxElems / 2L - 1L
                    let remaining = maxElems - 1L - leftTo - 1L
                    let rightFrom = ls() - remaining
                    let leftIdx = seq {0L .. leftTo}
                    let rightIdx = seq {rightFrom .. (ls()-1L)}
                    let elipsis =
                        match typeof<'T> with
                        | t when t=typeof<single> -> "      ..."
                        | t when t=typeof<double> -> "      ..."
                        | t when t=typeof<int>    -> " ..."
                        | t when t=typeof<byte>   -> "..."
                        | t when t=typeof<bool>   -> " ... "
                        | _ -> "..."
                    (subPrint leftIdx) @ [elipsis] @ (subPrint rightIdx)
            match a.NDims with
            | 0 -> 
                let v = box a.Value
                match typeof<'T> with
                | t when t = typeof<single> && unbox v >= 0.0f -> sprintf "%9.4f" (v :?> single)
                | t when t = typeof<single> && unbox v <  0.0f -> sprintf "%9.3f" (v :?> single)
                | t when t = typeof<double> && unbox v >= 0.0  -> sprintf "%9.4f" (v :?> double)
                | t when t = typeof<double> && unbox v <  0.0  -> sprintf "%9.3f" (v :?> double)
                | t when t = typeof<int>                       -> sprintf "%4d"   (v :?> int)
                | t when t = typeof<int64>                     -> sprintf "%4d"   (v :?> int64)
                | t when t = typeof<byte>                      -> sprintf "%3d"   (v :?> byte)
                | t when t = typeof<bool>   && unbox v = true  -> "true "
                | t when t = typeof<bool>   && unbox v = false -> "false"
                | _                                            -> sprintf "%A;" v
            | 1 -> "[" + (String.concat " " (subStrs ())) + "]"
            | _ -> "[" + (String.concat ("\n" + lineSpace) (subStrs ())) + "]"
        prettyDim " " this                       

    /// <summary>String representation of the tensor limited to 10 elements per dimension.</summary>
    /// <value>A (shortened) string representation of this tensor</value>
    /// <seealso cref="ToString(System.Int64)"/><seealso cref="Full"/>
    member this.Pretty = this.ToString (maxElems=10L)

    /// <summary>String representation of the tensor limited to 10 elements per dimension.</summary>
    /// <returns>A (shortened) string representation of this tensor</returns>
    /// <seealso cref="ToString(System.Int64)"/><seealso cref="Pretty"/><seealso cref="Full"/>   
    override this.ToString() = this.Pretty

    /// <summary>Unabreviated string representation of the tensor.</summary>
    /// <value>An unabreviated string representation of this tensor</value>
    /// <seealso cref="ToString(System.Int64)"/><seealso cref="Pretty"/>
    member this.Full = this.ToString (maxElems=Int64.MaxValue)
                               
    // interface for access from backend                             
    interface ITensorFrontend<'T> with
        member this.Storage = this.Storage
        member this.Dev = this.Dev
        member this.Backend = this.Backend
        member this.Layout = this.Layout
        member this.Shape = this.Shape
        member this.Stride = this.Layout.Stride
        member this.Offset = this.Layout.Offset
        member this.NDims = this.NDims
        member this.NElems = this.NElems
        member this.Relayout layout = this.Relayout layout :> ITensorFrontend<'T>
        member this.Copy (?order) = this.Copy (?order=order) :> ITensorFrontend<'T>
        member this.CopyFrom (src) = this.CopyFrom (src :?> Tensor<'T>)
        member this.Transfer (dev) = this.Transfer (dev) :> ITensorFrontend<'T>
        member this.T = this.T :> ITensorFrontend<'T>

    // type-neural interface
    interface ITensor with
        member this.Layout = this.Layout
        member this.Relayout layout = this.Relayout layout :> ITensor
        member this.Shape = this.Shape
        member this.NDims = this.NDims
        member this.NElems = this.NElems
        member this.DataType = this.DataType
        member this.Storage = this.Storage :> ITensorStorage
        member this.Dev = this.Dev
        member this.Copy (?order) = this.Copy (?order=order) :> ITensor
        member this.Transfer (dev) = this.Transfer (dev) :> ITensor
        member this.FillZero () = this.FillConst zero<'T>
        member this.Pretty = this.Pretty
        member this.Full = this.Full

        member this.Item
            with get (rng: Rng list) = this.IGetRng [|rng|]
            and set (rng: Rng list) (value: ITensor) = this.ISetRng [|rng|] value

        member this.M
            with get (m0: ITensor) = this.IMaskedGet [m0]
            and set (m0: ITensor) (value: ITensor) = this.IMaskedSet [m0] value                          
    
        member this.M
            with get (m0: ITensor, m1: ITensor) = this.IMaskedGet [m0; m1]
            and set (m0: ITensor, m1: ITensor) (value: ITensor) = this.IMaskedSet [m0; m1] value                          
    
        member this.M
            with get (m0: ITensor, m1: ITensor, m2: ITensor) = this.IMaskedGet [m0; m1; m2]
            and set (m0: ITensor, m1: ITensor, m2: ITensor) (value: ITensor) = this.IMaskedSet [m0; m1; m2] value                          
    
        member this.M
            with get (m0: ITensor, m1: ITensor, m2: ITensor, m3: ITensor) = this.IMaskedGet [m0; m1; m2; m3]
            and set (m0: ITensor, m1: ITensor, m2: ITensor, m3: ITensor) (value: ITensor) = this.IMaskedSet [m0; m1; m2; m3] value                          
    
        member this.M
            with get (m0: ITensor, m1: ITensor, m2: ITensor, m3: ITensor, m4: ITensor) = this.IMaskedGet [m0; m1; m2; m3; m4]
            and set (m0: ITensor, m1: ITensor, m2: ITensor, m3: ITensor, m4: ITensor) (value: ITensor) = this.IMaskedSet [m0; m1; m2; m3; m4] value                          
    
        member this.M
            with get (masks: ITensor list) = this.IMaskedGet masks
            and set (masks: ITensor list) (value: ITensor) = this.IMaskedSet masks value                          

        member this.Item
            with get (i0: int64) = this.IGetRng [|i0|]
            and set (i0: int64) (value: ITensor) = this.ISetRng [|i0|] value
        member this.GetSlice (i0s: int64 option, i0f: int64 option) = this.IGetRng [|i0s; i0f|] 
        member this.SetSlice (i0s: int64 option, i0f: int64 option, value: ITensor) = this.ISetRng [|i0s; i0f|] value

        member this.Item
            with get (i0: int64, i1: int64) = this.IGetRng [|i0; i1|]
            and set (i0: int64, i1: int64) (value: ITensor) = this.ISetRng [|i0; i1|] value
        member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option) = this.IGetRng [|i0; i1s; i1f|]
        member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, value: ITensor) = this.ISetRng [|i0; i1s; i1f|] value
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64) = this.IGetRng [|i0s; i0f; i1|]
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, value: ITensor) = this.ISetRng [|i0s; i0f; i1|] value
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option) = this.IGetRng [|i0s; i0f; i1s; i1f|]
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, value: ITensor) = this.ISetRng [|i0s; i0f; i1s; i1f|] value

        member this.Item
            with get (i0: int64, i1: int64, i2: int64) = this.IGetRng [|i0; i1; i2|]
            and set (i0: int64, i1: int64, i2: int64) (value: ITensor) = this.ISetRng [|i0; i1; i2|] value
        member this.GetSlice (i0: int64, i1: int64, i2: int64) = this.IGetRng [|i0; i1; i2|]
        member this.SetSlice (i0: int64, i1: int64, i2: int64, value: ITensor) = this.ISetRng [|i0; i1; i2|] value
        member this.GetSlice (i0: int64, i1: int64, i2s: int64 option, i2f: int64 option) = this.IGetRng [|i0; i1; i2s; i2f|]
        member this.SetSlice (i0: int64, i1: int64, i2s: int64 option, i2f: int64 option, value: ITensor) = this.ISetRng [|i0; i1; i2s; i2f|] value
        member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2: int64) = this.IGetRng [|i0; i1s; i1f; i2|]
        member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2: int64, value: ITensor) = this.ISetRng [|i0; i1s; i1f; i2|] value
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2: int64) = this.IGetRng [|i0s; i0f; i1; i2|]
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2: int64, value: ITensor) = this.ISetRng [|i0s; i0f; i1; i2|] value
        member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option) = this.IGetRng [|i0; i1s; i1f; i2s; i2f|]
        member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, value: ITensor) = this.ISetRng [|i0; i1s; i1f; i2s; i2f|] value
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2s: int64 option, i2f: int64 option) = this.IGetRng [|i0s; i0f; i1; i2s; i2f|]
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2s: int64 option, i2f: int64 option, value: ITensor) = this.ISetRng [|i0s; i0f; i1; i2s; i2f|] value
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2: int64) = this.IGetRng [|i0s; i0f; i1s; i1f; i2|]
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2: int64, value: ITensor) = this.ISetRng [|i0s; i0f; i1s; i1f; i2|] value
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option) = this.IGetRng [|i0s; i0f; i1s; i1f; i2s; i2f|]
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, value: ITensor) = this.ISetRng [|i0s; i0f; i1s; i1f; i2s; i2f|] value

        member this.Item
            with get (o0: obj, o1: obj, o2: obj, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|o0; o1; o2; o3|] r
            and set (o0: obj, o1: obj, o2: obj, o3: obj) (value: ITensor) = this.ISetRng [|o0; o1; o2; o3|] value
        member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj) (value: ITensor) = this.ISetRng [|o0; o1; o2; o3; o4|] value
        member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj) (value: ITensor) = this.ISetRng [|o0; o1; o2; o3; o4; o5|] value
        member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj, o6: obj) (value: ITensor) = this.ISetRng [|o0; o1; o2; o3; o4; o5; o6|] value
        member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj, o6: obj, o7: obj) (value: ITensor) = this.ISetRng [|o0; o1; o2; o3; o4; o5; o6; o7|] value
        member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj, o6: obj, o7: obj, o8: obj) (value: ITensor) = this.ISetRng [|o0; o1; o2; o3; o4; o5; o6; o7; o8|] value
        member this.Item with set (o0: obj, o1: obj, o2: obj, o3: obj, o4: obj, o5: obj, o6: obj, o7: obj, o8: obj, o9: obj) (value: ITensor) = this.ISetRng [|o0; o1; o2; o3; o4; o5; o6; o7; o8; o9|] value
        member this.GetSlice (i0: int64, i1: int64, i2: int64, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|i0; i1; i2; o3|] r
        member this.SetSlice (i0: int64, i1: int64, i2: int64, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.ISetRngWithRest [|i0; i1; i2; o3; o4|] r
        member this.GetSlice (i0: int64, i1: int64, i2s: int64 option, i2f: int64 option, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|i0; i1; i2s; i2f; o3|] r
        member this.SetSlice (i0: int64, i1: int64, i2s: int64 option, i2f: int64 option, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.ISetRngWithRest [|i0; i1; i2s; i2f; o3; o4|] r
        member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2: int64, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|i0; i1s; i1f; i2; o3|] r
        member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2: int64, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.ISetRngWithRest [|i0; i1s; i1f; i2; o3; o4|] r
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2: int64, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|i0s; i0f; i1; i2; o3|] r
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2: int64, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.ISetRngWithRest [|i0s; i0f; i1; i2; o3; o4|] r
        member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|i0; i1s; i1f; i2s; i2f; o3|] r
        member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.ISetRngWithRest [|i0; i1s; i1f; i2s; i2f; o3; o4|] r
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2s: int64 option, i2f: int64 option, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|i0s; i0f; i1; i2s; i2f; o3|] r
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, i2s: int64 option, i2f: int64 option, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.ISetRngWithRest [|i0s; i0f; i1; i2s; i2f; o3; o4|] r
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2: int64, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|i0s; i0f; i1s; i1f; i2; o3|] r
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2: int64, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.ISetRngWithRest [|i0s; i0f; i1s; i1f; i2; o3; o4|] r
        member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, o3: obj, [<System.ParamArray>] r: obj[]) = this.IGetRngWithRest [|i0s; i0f; i1s; i1f; i2s; i2f; o3|] r
        member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, i2s: int64 option, i2f: int64 option, o3: obj, o4: obj, [<System.ParamArray>] r: obj[]) = this.ISetRngWithRest [|i0s; i0f; i1s; i1f; i2s; i2f; o3; o4|] r

    /// <summary>Tests for equality to another object.</summary>
    /// <param name="other">The other object.</param>
    /// <returns>true if the objects are equal. Otherwise false.</returns>
    /// <remarks>
    /// <para>Two tensors are equal if they have the same storage and same layout.
    /// In this case, changing one tensor will have the exact same effect on the other tensor.</para>
    /// <para>Two tensors can overlap, i.e. one can partially or fully affect the other, without being equal.</para>
    /// <para>The elements of a tensor do not affect equality, i.e. two tensors can contain exactly the same values 
    /// without being equal.</para>
    /// </remarks>
    /// <seealso cref="op_EqualsEqualsEqualsEquals"/><seealso cref="almostEqual"/>
    override this.Equals other =
        match other with
        | :? Tensor<'T> as ot ->
            this.Storage = ot.Storage && this.Layout = ot.Layout 
        | _ -> false

    /// <summary>Calculates the hash code of the tensor.</summary>
    /// <returns>The hash code.</returns>
    /// <remarks>
    /// <para>The hash code is calculated from the storage and layout of the tensor.
    /// If two tensors are equal, they will have the same hash code.</para>
    /// </remarks>
    /// <seealso cref="Equals"/>
    override this.GetHashCode () =
        hash (this.Storage, this.Layout)

    /// <summary>Type-neutral function for creating a new, uninitialized tensor with a new storage.</summary>
    /// <param name="shape">The shape of the tensor to create.</param>
    /// <param name="dataType">The data type of the tensor to create.</param>
    /// <param name="dev">The device to store the data of the tensor on.</param>
    /// <param name="order">The memory layout to use for the new tensor.</param>
    /// <returns>The new, uninitialized tensor.</returns>
    /// <remarks>
    /// <para>The contents of the new tensor are undefined.</para>
    /// <para>Use this function only if you require a type-neutral function.
    /// The recommended way is to use <see cref="zeros"/> to create a typed tensor.</para>
    /// </remarks>
    /// <seealso cref="#ctor"/>
    static member NewOfType (shape: int64 list, dataType: Type, dev: ITensorDevice, ?order: TensorOrder) =
        let gt = typedefof<Tensor<_>>.MakeGenericType (dataType)
        Activator.CreateInstance (gt, [|box shape; box dev; box order|]) :?> ITensor

    /// <summary>Creates a new, empty tensor with the given number of dimensions.</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="nDims">The number of dimensions of the new, empty tensor.</param>
    /// <returns>The new, empty tensor.</returns>
    /// <remarks>
    /// <para>The shape of the tensor is <c>[0L; ...; 0L]</c>. It contains no elements.</para>
    /// </remarks>
    static member empty (dev: ITensorDevice) (nDims: int) : Tensor<'T> =
        Tensor<'T> (List.init nDims (fun _ -> 0L), dev)

    /// <summary>Creates a new tensor filled with zeros (0).</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor&lt;float&gt;.zeros HostTensor.Dev [2L; 3L]
    /// // a = [[0.0; 0.0; 0.0]
    /// //      [0.0; 0.0; 0.0]]
    /// </code></example>    
    /// <remarks>
    /// <para>A new tensor of the specified shape is created on the specified device.</para>
    /// <para>The tensor is filled with zeros.</para>
    /// </remarks>
    /// <seealso cref="zerosLike"/><seealso cref="ones"/>
    static member zeros (dev: ITensorDevice) (shape: int64 list) : Tensor<'T> =
        let x = Tensor<'T> (shape, dev)
        if not dev.Zeroed then 
            x.FillConst zero<'T>
        x
   
    /// <summary>Creates a new tensor filled with zeros using the specified tensor as template.</summary>
    /// <param name="tmpl">The template tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <remarks>
    /// <para>A new tensor is created with the same shape and on the same device as <paramref name="tmpl"/>.</para>
    /// <para>The tensor is filled with zeros.</para>
    /// </remarks>
    /// <seealso cref="zeros"/>
    static member zerosLike (tmpl: Tensor<'T>) : Tensor<'T> =
        Tensor<'T>.zeros tmpl.Storage.Dev tmpl.Shape

    /// <summary>Creates a new tensor filled with ones (1).</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor&lt;float&gt;.ones HostTensor.Dev [2L; 3L]
    /// // a = [[1.0; 1.0; 1.0]
    /// //      [1.0; 1.0; 1.0]]
    /// </code></example>      
    /// <remarks>
    /// <para>A new tensor of the specified shape is created on the specified device.</para>
    /// <para>The tensor is filled with ones.</para>
    /// </remarks>
    /// <seealso cref="onesLike"/><seealso cref="zeros"/>    
    static member ones (dev: ITensorDevice) (shape: int64 list) : Tensor<'T> =
        let x = Tensor<'T> (shape, dev)
        x.FillConst one<'T>
        x
        
    /// <summary>Creates a new tensor filled with ones using the specified tensor as template.</summary>
    /// <param name="tmpl">The template tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <remarks>
    /// <para>A new tensor is created with the same shape and on the same device as <paramref name="tmpl"/>.</para>
    /// <para>The tensor is filled with ones.</para>
    /// </remarks>
    /// <seealso cref="ones"/>
    static member onesLike (tmpl: Tensor<'T>) : Tensor<'T> =
        Tensor<'T>.ones tmpl.Storage.Dev tmpl.Shape 

    /// <summary>Creates a new boolean tensor filled with falses.</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor.falses HostTensor.Dev [2L; 3L]
    /// // a = [[false; false; false]
    /// //      [false; false; false]]
    /// </code></example>        
    /// <remarks>
    /// <para>A new tensor of the specified shape is created on the specified device.</para>
    /// <para>The tensor is filled with falses.</para>
    /// </remarks>
    /// <seealso cref="trues"/>
    static member falses (dev: ITensorDevice) (shape: int64 list) : Tensor<bool> =
        let x = Tensor<bool> (shape, dev)
        x.FillConst false
        x

    /// <summary>Creates a new boolean tensor filled with trues.</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor.trues HostTensor.Dev [2L; 3L]
    /// // a = [[true; true; true]
    /// //      [true; true; true]]
    /// </code></example>       
    /// <remarks>
    /// <para>A new tensor of the specified shape is created on the specified device.</para>
    /// <para>The tensor is filled with trues.</para>
    /// </remarks>
    /// <seealso cref="falses"/>
    static member trues (dev: ITensorDevice) (shape: int64 list) : Tensor<bool> =
        let x = Tensor<bool> (shape, dev)
        x.FillConst true
        x   

    /// <summary>Creates a new zero-dimensional (scalar) tensor with the specified value.</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="value">The value of the new, scalar tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor.scalar HostTensor.Dev 2.5f
    /// // a = 2.5f
    /// </code></example>       
    /// <remarks>
    /// <para>A new tensor of zero-dimensional shape is created on the specified device.</para>
    /// <para>The values of the tensor is set to the specified value.</para>
    /// </remarks>
    /// <seealso cref="scalarLike"/>
    static member scalar (dev: ITensorDevice) (value: 'T) : Tensor<'T> =
        let x = Tensor<'T> ([], dev)
        x.Value <- value
        x

    /// <summary>Creates a new zero-dimensional (scalar) tensor using the specified tensor as template and with 
    /// the specified value.</summary>
    /// <param name="tmpl">The template tensor.</param>
    /// <param name="value">The value of the new, scalar tensor.</param>
    /// <returns>The new tensor.</returns>
    /// <remarks>
    /// <para>A new tensor of zero-dimensional shape is created on the same device as <paramref name="tmpl"/>.</para>
    /// <para>The values of the tensor is set to the specified value.</para>
    /// </remarks>
    /// <seealso cref="scalar"/>
    static member scalarLike (tmpl: ITensor) (value: 'T) : Tensor<'T> =
        Tensor<'T>.scalar tmpl.Storage.Dev value 

    /// <summary>Creates a new tensor filled with the specified value.</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <param name="value">The value to fill the new tensor with.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor.filled HostTensor.Dev [2L; 3L] 1.5
    /// // a = [[1.5; 1.5; 1.5]
    /// //      [1.5; 1.5; 1.5]]
    /// </code></example>         
    /// <remarks>
    /// <para>A new tensor of the specified shape is created on the specified device.</para>
    /// <para>The tensor is filled with the specified value.</para>
    /// </remarks>
    /// <seealso cref="FillConst"/>
    static member filled (dev: ITensorDevice) (shape: int64 list) (value: 'T) : Tensor<'T> =
        let x = Tensor<'T> (shape, dev)
        x.FillConst value
        x           

    /// <summary>Creates a new identity matrix.</summary>
    /// <param name="dev">The device to create the matrix on.</param>
    /// <param name="size">The size of the square identity matrix.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor&lt;float&gt;.identity HostTensor.Dev 3L
    /// // a = [[1.0; 0.0; 0.0]
    /// //      [0.0; 1.0; 0.0]
    /// //      [0.0; 0.0; 1.0]]    
    /// </code></example>         
    /// <remarks>
    /// <para>A new square matrix of the specified size is created on the specified device.</para>
    /// <para>The tensor is filled with ones on the diagonal and zeros elsewhere.</para>
    /// </remarks>
    static member identity (dev: ITensorDevice) (size: int64) : Tensor<'T> =
        let x = Tensor<'T>.zeros dev [size; size]
        let d : Tensor<'T> = Tensor.diag x
        d.FillConst one<'T>
        x           

    /// <summary>Creates a new vector filled with the integers from zero to the specified maximum.</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="nElems">The number of elements of the new vector.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor.counting HostTensor.Dev 5L
    /// // a = [0L; 1L; 2L; 3L; 4L]
    /// </code></example>          
    /// <remarks>
    /// <para>A new vector with the specified number of elements is created on the specified device.</para>
    /// <para>The tensor is filled with <c>[0L; 1L; 2L; ...; nElems-1L]</c>. </para>
    /// </remarks>
    /// <seealso cref="arange``1"/>
    static member counting (dev: ITensorDevice) (nElems: int64) =
        let x = Tensor<int64> ([nElems], dev)
        x.FillIncrementing (0L, 1L)
        x

    /// <summary>Creates a new vector filled with equaly spaced values using a specifed increment.</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="start">The starting value.</param>
    /// <param name="incr">The increment between successive element.</param>   
    /// <param name="stop">The end value, which is not included.</param>
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor.arange HostTensor.Dev 1.0 0.1 2.0
    /// // a = [1.0; 1.1; 1.2; 1.3; 1.4; 1.5; 1.6; 1.7; 1.8; 1.9]
    /// </code></example>         
    /// <remarks>
    /// <para>A new vector with <c>floor ((stop - start) / incr)</c> elements is created on the specified device.</para>
    /// <para>The vector is filled with <c>[start; start+1*incr; start+2*incr; ...]</c>.</para>
    /// <para>If stop is smaller or equal to start, an empty vector is returned.</para>
    /// </remarks>
    /// <seealso cref="counting"/><seealso cref="linspace``1"/>
    static member arange (dev: ITensorDevice) (start: 'T) (incr: 'T) (stop: 'T) = 
        let op = ScalarPrimitives.For<'T, 'T> ()
        let opc = ScalarPrimitives.For<int64, 'T> ()
        let nElemsT = op.Divide (op.Subtract stop start) incr
        let nElemsInt = opc.Convert nElemsT
        let nElems = max 0L nElemsInt
        let x = Tensor<'T> ([nElems], dev)
        x.FillIncrementing (start, incr)
        x

    /// <summary>Creates a new vector of given size filled with equaly spaced values.</summary>
    /// <param name="dev">The device to create the tensor on.</param>
    /// <param name="start">The starting value.</param>
    /// <param name="stop">The end value, which is not included.</param>
    /// <param name="nElems">The size of the vector.</param>   
    /// <returns>The new tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = Tensor.linspace HostTensor.Dev 1.0 2.0 5L
    /// // a = [1.0; 1.2; 1.4; 1.6; 1.8]
    /// </code></example>        
    /// <remarks>
    /// <para>A new vector with <paramref name="nElems"/> elements is created on the specified device.</para>
    /// <para>The vector is filled with <c>[start; start+1*incr; start+2*incr; ...; stop]</c> where
    /// <c>incr = (stop - start) / (nElems - 1)</c>.</para>
    /// </remarks>
    /// <seealso cref="arange``1"/>
    static member linspace (dev: ITensorDevice) (start: 'T) (stop: 'T) (nElems: int64) =
        if nElems < 2L then invalidArg "nElems" "linspace requires at least two elements."
        let op = ScalarPrimitives.For<'T, int64> ()
        let nElemsT = op.Convert (nElems - 1L)
        let incr = op.Divide (op.Subtract stop start) nElemsT
        let x = Tensor<'T> ([nElems], dev)
        x.FillIncrementing (start, incr)
        x

    /// <summary>Element-wise check if two tensors have same (within machine precision) values.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <param name="absTol">The absolute tolerance. (default 1e-8)</param>
    /// <param name="relTol">The relative tolerance. (default 1e-5)</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <remarks>
    /// <para>Test each element of tensor <paramref name="a"/> for being almost equal to the corresponding element 
    /// of tensor <paramref name="b"/> and returns the results as a new tensor. For integer data types the check
    /// is exact.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage and type.</para>
    /// </remarks>
    /// <seealso crf="almostEqual``1"/>
    static member isClose (a: Tensor<'T>, b: Tensor<'T>, ?absTol: 'T, ?relTol: 'T) =
        match typeof<'T> with
        | t when t=typeof<single> || t=typeof<double> ->
            let absTol = defaultArg absTol (conv<'T> 1e-8) |> Tensor.scalarLike a
            let relTol = defaultArg relTol (conv<'T> 1e-5) |> Tensor.scalarLike a
            abs (a - b) <<== absTol + relTol * abs b
        | _ -> a ==== b

    /// <summary>Checks if two tensors have the same (within machine precision) values in all elements.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <param name="absTol">The absolute tolerance. (default 1e-8)</param>
    /// <param name="relTol">The relative tolerance. (default 1e-5)</param>
    /// <returns>true if two tensors have same (within specified precision) values in all elements, otherwise false.</returns>
    /// <remarks>
    /// <para>Test each element of tensor <paramref name="a"/> for being almost equal to the corresponding element 
    /// of tensor <paramref name="b"/>. For integer data types the check is exact.</para>
    /// <para>If tensors have different shape, then false is returned.</para>
    /// <para>The tensors <paramref name="a"/> and <paramref name="b"/> must have the same storage and type.</para>
    /// </remarks>
    /// <seealso crf="isClose``1"/>
    static member almostEqual (a: Tensor<'T>, b: Tensor<'T>, ?absTol: 'T, ?relTol: 'T) =
        if a.Shape = b.Shape then
            Tensor.isClose (a, b, ?absTol=absTol, ?relTol=relTol) |> Tensor.all
        else false

    /// <summary>Checks that all elements of the tensor are finite.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>true if all elements are finite, otherwise false.</returns>
    /// <remarks>
    /// <para>Checks each element of the specified tensor for finity (not -Inf, Inf or NaN).</para>
    /// </remarks>
    /// <seealso crf="isFinite``1"/>
    static member allFinite (a: Tensor<'T>) =
        a |> Tensor.isFinite |> Tensor.all

    /// <summary>Calculates the mean of the elements along the specified axis.</summary>
    /// <param name="ax">The axis to operate along.</param>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.meanAxis 1 a // b = [2.5; 6.5]
    /// </code></example>
    /// <remarks>The mean is calculated along the specified axis.</remarks>
    /// <seealso cref="mean"/><seealso cref="varAxis"/><seealso cref="stdAxis"/>
    static member meanAxis axis (a: Tensor<'T>) = 
        Tensor.sumAxis axis a / Tensor.scalarLike a (conv<'T> a.Shape.[axis])

    /// <summary>Calculates the mean of the tensor.</summary>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>The mean estimate.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.mean a // b = 4.5
    /// </code></example>
    /// <remarks>The mean is calculated over all elements of the tensor.</remarks>
    /// <seealso cref="meanAxis"/><seealso cref="var"/><seealso cref="std"/>
    static member mean (a: Tensor<'T>) =
        a |> Tensor.flatten |> Tensor.meanAxis 0 |> Tensor.value

    /// <summary>Calculates the variance of the elements along the specified axis.</summary>
    /// <param name="ax">The axis to operate along.</param>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <param name="ddof">The delta degrees of freedom. (default: 0L)</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.varAxis (1, a, ddof=1L) 
    /// </code></example>
    /// <remarks>
    /// <para>The variance is calculated along the specified axis.</para>
    /// <para>The parameter <paramref name="ddof"/> specifies the difference between the number of elements and the
    /// degrees of freedom for the computation of the variance. Use <c>ddof=1</c> to obtain an unbiased estimate and
    /// <c>ddof=0</c> for a maximum-likelihood estimate.</para>
    /// </remarks>
    /// <seealso cref="var"/><seealso cref="meanAxis"/><seealso cref="stdAxis"/>
    static member varAxis (axis, a: Tensor<'T>, ?ddof) =
        let ddof = defaultArg ddof 0L
        let m = Tensor.meanAxis axis a |> Tensor.insertAxis axis
        let v = a - m
        let n = a.Shape.[axis] - ddof
        Tensor.sumAxis axis (v * v) / Tensor.scalarLike a (conv<'T> n)

    /// <summary>Calculates the variance of the tensor.</summary>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>The variance estimate.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.var a 
    /// </code></example>
    /// <remarks>
    /// <para>The variance is calculated over all elements of the tensor.</para>
    /// <para>The parameter <paramref name="ddof"/> specifies the difference between the number of elements and the
    /// degrees of freedom for the computation of the variance. Use <c>ddof=1</c> to obtain an unbiased estimate and
    /// <c>ddof=0</c> for a maximum-likelihood estimate.</para>
    /// </remarks>
    /// <seealso cref="varAxis"/><seealso cref="mean"/><seealso cref="std"/>
    static member var (a: Tensor<'T>, ?ddof) =
        Tensor.varAxis (0, Tensor.flatten a, ?ddof=ddof) |> Tensor.value

    /// <summary>Calculates the standard deviation of the elements along the specified axis.</summary>
    /// <param name="ax">The axis to operate along.</param>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <param name="ddof">The delta degrees of freedom. (default: 0L)</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.stdAxis (1, a, ddof=1L) 
    /// </code></example>
    /// <remarks>
    /// <para>The standard deviation is calculated along the specified axis.</para>
    /// <para>The parameter <paramref name="ddof"/> specifies the difference between the number of elements and the
    /// degrees of freedom for the computation of the variance. Use <c>ddof=1</c> to obtain an unbiased estimate and
    /// <c>ddof=0</c> for a maximum-likelihood estimate.</para>
    /// </remarks>
    /// <seealso cref="std"/><seealso cref="meanAxis"/><seealso cref="varAxis"/>
    static member stdAxis (ax, a: Tensor<'T>, ?ddof) =
        Tensor.varAxis (ax, a, ?ddof=ddof) |> sqrt

    /// <summary>Calculates the standard deviation of the tensor.</summary>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <param name="ddof">The delta degrees of freedom. (default: 0L)</param>
    /// <returns>The standard deviation estimate.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.std a 
    /// </code></example>
    /// <remarks>
    /// <para>The standard deviation is calculated over all elements of the tensor.</para>
    /// <para>The parameter <paramref name="ddof"/> specifies the difference between the number of elements and the
    /// degrees of freedom for the computation of the variance. Use <c>ddof=1</c> to obtain an unbiased estimate and
    /// <c>ddof=0</c> for a maximum-likelihood estimate.</para>
    /// </remarks>
    /// <seealso cref="stdAxis"/><seealso cref="mean"/><seealso cref="var"/>
    static member std (a: Tensor<'T>, ?ddof) =
        Tensor.varAxis (0, Tensor.flatten a, ?ddof=ddof) |> sqrt |> Tensor.value

    /// <summary>Calculates the norm along the specified axis.</summary>
    /// <param name="axis">The axis to operate along.</param>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <param name="ord">The order (power) of the norm. (default: 2)</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.normAxis (1, a) // [5.477; 13.191]
    /// </code></example>
    /// <remarks>
    /// <para>The norm is calculated along the specified axis.</para>
    /// <para>It is defined by <c>sqrt (sum_i (x_i**ord))</c>.</para>
    /// </remarks>
    /// <seealso cref="norm"/>
    static member normAxis (axis, a: Tensor<'T>, ?ord: 'T) =
        let ord = defaultArg ord (conv<'T> 2)
        let tOrd = Tensor.scalarLike a ord
        let tOrdRep = Tensor.scalarLike a (conv<'T> 1) / tOrd
        let s = a ** tOrd |> Tensor.sumAxis axis
        s ** tOrdRep 

    /// <summary>Calculates the norm of the (flattened) tensor.</summary>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <param name="ord">The order (power) of the norm. (default: 2)</param>
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.ofList2D [[1.0; 2.0; 3.0; 4.0]
    ///                              [5.0; 6.0; 7.0; 8.0]]
    /// let b = Tensor.norm a // 14.283
    /// </code></example>
    /// <remarks>
    /// <para>The norm is calculated over all elements of the tensor.</para>
    /// <para>It is defined by <c>sqrt (sum_i (x_i**ord))</c>.</para>
    /// </remarks>
    /// <seealso cref="normAxis"/>
    static member norm (a: Tensor<'T>, ?ord: 'T) =
        Tensor.normAxis (0, Tensor.flatten a, ?ord=ord) |> Tensor.value

    /// <summary>Returns a view of the diagonal along the given axes.</summary>
    /// <param name="ax1">The first dimension of the diagonal.</param>
    /// <param name="ax2">The seconds dimension of the diagonal.</param>
    /// <param name="a">The tensor to operate on.</param>    
    /// <returns>A tensor where dimension <paramref name="ax1"/> is the diagonal and dimension
    /// <paramref name="ax2"/> is removed.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [4L; 3L; 3L; 5L]
    /// let b = Tensor.diagAxis 1 2 a // b.Shape = [4L; 3L; 5L]
    /// </code></example>    
    /// <remarks>    
    /// <para>The dimensions specified by <paramref name="ax1"/> and <paramref name="ax2"/> must have the same size.</para>
    /// <para>A view of the original tensor is returned and the storage is shared. Modifications done to the returned 
    /// tensor will affect the original tensor.</para>
    /// </remarks>    
    /// <seealso cref="diag"/><seealso cref="diagMatAxis"/>
    static member diagAxis ax1 ax2 (a: Tensor<'T>) =
        a |> Tensor.relayout (a.Layout |> TensorLayout.diagAxis ax1 ax2)

    /// <summary>Returns a view of the diagonal of the matrix.</summary>
    /// <param name="a">A square matrix.</param>    
    /// <returns>The diagonal vector.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 3L]
    /// let b = Tensor.diag a // b.Shape = [3L]
    /// </code></example>    
    /// <remarks>    
    /// <para>The matrix must be square.</para>
    /// <para>If the specified tensor has more than two dimensions, the diagonals along the last two dimensions 
    /// are returned as a tensor.</para>
    /// <para>A view of the original tensor is returned and the storage is shared. Modifications done to the returned 
    /// tensor will affect the original tensor.</para>
    /// </remarks>    
    /// <seealso cref="diagAxis"/><seealso cref="diagMat"/>
    static member diag (a: Tensor<'T>) =
        if a.NDims < 2 then
            invalidArg "a" "Need at least a two dimensional array for diagonal but got shape %A." a.Shape
        Tensor.diagAxis (a.NDims-2) (a.NDims-1) a

    /// <summary>Creates a tensor with the specified diagonal along the given axes.</summary>
    /// <param name="ax1">The first dimension of the diagonal.</param>
    /// <param name="ax2">The seconds dimension of the diagonal.</param>
    /// <param name="a">The values for the diagonal.</param>    
    /// <returns>A tensor having the values <paramref name="a"/> on the diagonal specified by the axes 
    /// <paramref name="ax1"/> and <paramref name="ax2"/>.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [4L; 3L]
    /// let b = Tensor.diagMatAxis 0 1 a // b.Shape = [4L; 4L; 3L]
    /// </code></example>    
    /// <remarks>    
    /// <para>A new tensor with the same shape as <paramref name="a"/> but with axis <paramref name="ax2"/> inserted
    /// is created. The size of axis <paramref name="ax2"/> is set to the size of axis <paramref name="ax1"/>.</para>
    /// <para>The diagonal over axes <paramref name="ax1"/> and <paramref name="ax2"/> is filled with the elements of 
    /// tensor <paramref name="a"/>. The other elements are set to zero.</para>
    /// </remarks>    
    /// <seealso cref="diagMat"/><seealso cref="diagAxis"/>
    static member diagMatAxis ax1 ax2 (a: Tensor<'T>) =
        if ax1 = ax2 then 
            invalidArg "ax1" "axes to use for diagonal must be different"
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        a.CheckAxis ax1
        if not (0 <= ax2 && ax2 <= a.NDims) then
            invalidArg "ax2" "Cannot insert axis at position %d into array of shape %A." ax2 a.Shape
        let dShp = a.Shape |> List.insert ax2 a.Shape.[ax1]
        let d = Tensor.zeros a.Dev dShp
        let dDiag = Tensor.diagAxis ax1 ax2 d
        dDiag.FillFrom a
        d

    /// <summary>Creates a matrix with the specified diagonal.</summary>
    /// <param name="a">The vector containing the values for the diagonal.</param>    
    /// <returns>A matrix having the values <paramref name="a"/> on its diagonal.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [5L]
    /// let b = Tensor.diag a // b.Shape = [5L; 5L]
    /// </code></example>    
    /// <remarks>    
    /// <para>A square matrix with the same size as <paramref name="a"/> is created.</para>
    /// <para>The diagonal is filled with the elements of vector <paramref name="a"/>. 
    /// The other elements are set to zero.</para>
    /// <para>If the specified tensor has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.</para>
    /// </remarks>    
    /// <seealso cref="diagMatAxis"/><seealso cref="diag"/>
    static member diagMat (a: Tensor<'T>) =
        if a.NDims < 1 then
            invalidArg "a" "need at leat a one-dimensional array to create a diagonal matrix"
        Tensor.diagMatAxis (a.NDims-1) a.NDims a

    /// <summary>Calculates the trace along the specified axes.</summary>
    /// <param name="ax1">The first axis of the diagonal to compute the trace along.</param>
    /// <param name="ax2">The second axis of the diagonal to compute the trace along.</param>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>A new tensor containing the result of this operation.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [2L; 3L; 4L; 3L]
    /// let b = Tensor.traceAxis 1 3 a // b.Shape = [2L; 4L]
    /// </code></example>
    /// <remarks>
    /// <para>The trace is calculated along the specified axes. It is defined by the sum of the elements on the
    /// diagonal.</para>
    /// <para>The tensor must have the same size in dimensions <paramref name="ax1"/> and <paramref name="ax2"/>.</para>
    /// </remarks>
    /// <seealso cref="trace"/>
    static member traceAxis ax1 ax2 (a: Tensor<'T>) =
        let tax = if ax1 < ax2 then ax1 else ax1 - 1
        a |> Tensor.diagAxis ax1 ax2 |> Tensor.sumAxis tax

    /// <summary>Calculates the trace of the matrix.</summary>
    /// <param name="a">A square matrix.</param>    
    /// <returns>The trace of the matrix.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [3L; 3L]
    /// let b = Tensor.trace a 
    /// </code></example>    
    /// <remarks>    
    /// <para>The trace is is defined by the sum of the elements on the diagonal.</para>
    /// <para>The matrix must be square.</para>
    /// <para>If the specified tensor has more than two dimensions, the traces along the last two dimensions 
    /// are returned as a tensor.</para>
    /// </remarks>    
    /// <seealso cref="traceAxis"/><seealso cref="diag"/>
    static member trace (a: Tensor<'T>) =
        if a.NDims < 2 then
            invalidArg "a" "Need at least a two dimensional array for trace but got shape %A." a.Shape
        Tensor.traceAxis (a.NDims-2) (a.NDims-1) a 

    /// <summary>Builds a tensor out of tensor blocks.</summary>
    /// <param name="bs">The block tensor specification.</param>    
    /// <returns>The resulting tensor.</returns>
    /// <example><code language="fsharp">
    /// // Consider a block matrix of the follow structure.
    /// // +-----------------------------+---------------+
    /// // |                             |               |
    /// // |                             |               |
    /// // |            b1               |      b2       |        
    /// // |         (5 x 28)            |   (5 x 15)    |
    /// // |                             |               |
    /// // +---------------------+-------+-------+-------+        
    /// // |                     |               |       |
    /// // |       b3            |      b4       |  b5   |
    /// // |    (3 x 22)         |    (3 x 14)   |(3 x 7)|
    /// // +---------------------+---------------+-------+        
    /// //
    /// // It can be specified as follows.
    /// let b1 = HostTensor.zeros [5L; 28L]
    /// let b2 = HostTensor.zeros [5L; 15L]
    /// let b3 = HostTensor.zeros [3L; 22L]
    /// let b4 = HostTensor.zeros [3L; 14L]
    /// let b5 = HostTensor.zeros [3L; 7L]
    /// let r1 = SubBlocks [Block b1; Block b2]
    /// let r2 = SubBlocks [Block b3; Block b4; Block b5]
    /// let a = Tensor.ofBlocks (SubBlocks [r1; r2])
    /// // a.Shape = [8L; 43L]
    /// </code></example>    
    /// <remarks>    
    /// <para>The contents of a sub-block are jointed along the dimension determined by its nesting level, i.e. 
    /// a sub-block nested <c>n</c> levels within other sub-blocks will be concatenated along dimension <c>n</c>.</para>
    /// <para>The contents of a sub-block must have equal sizes in all dimensions except for the 
    /// concatenation dimensions.</para>
    /// </remarks>    
    /// <seealso cref="concat"/>
    static member ofBlocks (bs: BlockTensor<'T>) =
        let rec commonShape joinDim shps =               
            match shps with
            | [shp] -> List.set joinDim -1L shp
            | shp::rShps ->
                let commonShp = commonShape joinDim [shp]
                if commonShp <> commonShape joinDim rShps then
                    invalidArg "bs" "Block tensor blocks must have same number of dimensions and be 
                                     identical in all but the join dimension."
                commonShp
            | [] -> []

        let joinSize joinDim (shps: int64 list list) =
            shps |> List.map (fun shp -> shp.[joinDim]) |> List.sum

        let joinShape joinDim shps =
            commonShape joinDim shps 
            |> List.set joinDim (joinSize joinDim shps)

        let rec joinedBlocksShape joinDim bs =
            match bs with
            | SubBlocks blcks ->
                blcks |> List.map (joinedBlocksShape (joinDim + 1)) |> joinShape joinDim
            | Block ary -> ary.Shape

        let rec blockPosAndContents (joinDim: int) startPos bs = seq {
            match bs with
            | SubBlocks blcks ->
                let mutable pos = startPos
                for blck in blcks do
                    yield! blockPosAndContents (joinDim + 1) pos blck 
                    let blckShape = joinedBlocksShape (joinDim + 1) blck
                    pos <- List.set joinDim (pos.[joinDim] + blckShape.[joinDim]) pos
            | Block ary -> yield startPos, ary
        }

        let rec anyArray bs =
            match bs with
            | SubBlocks b -> List.tryPick anyArray b
            | Block a -> Some a
                  
        let tmplArray = Option.get (anyArray bs)
        let joinedShape = joinedBlocksShape 0 bs
        let joined = Tensor<_> (joinedShape, tmplArray.Dev)
        let startPos = List.replicate (List.length joinedShape) 0L

        for pos, ary in blockPosAndContents 0 startPos bs do
            let slice = (pos, ary.Shape) ||> List.map2 (fun p s -> Rng.Rng (Some p, Some (p + s - 1L))) 
            joined.[slice] <- ary
        joined

    /// <summary>Builds a vector out of vectors blocks.</summary>
    /// <param name="bs">The block vector specification.</param>    
    /// <returns>The resulting vector.</returns>
    /// <example><code language="fsharp">
    /// // Consider a block vector of the follow structure.
    /// // +-----------------------------+---------------+
    /// // |          b1 (28)            |    b2 (15)    |
    /// // +-----------------------------+---------------+        
    /// //
    /// // It can be specified as follows.
    /// let b1 = HostTensor.zeros [28L]
    /// let b2 = HostTensor.zeros [15L]
    /// let a = Tensor.ofBlocks [b1; b2]
    /// // a.Shape = [43L]
    /// </code></example>    
    /// <remarks>    
    /// <para>The contents of a the vectors are concatenated.</para>
    /// </remarks>    
    static member ofBlocks (bs: Tensor<'T> list) =
        bs |> List.map Block |> SubBlocks |> Tensor.ofBlocks

    /// <summary>Builds a matrix out of matrix blocks.</summary>
    /// <param name="bs">The matrix blocks.</param>    
    /// <returns>The resulting matrix.</returns>
    /// <example><code language="fsharp">
    /// // Consider a block matrix of the follow structure.
    /// // +-----------------------------+---------------+
    /// // |                             |               |
    /// // |                             |               |
    /// // |            b1               |      b2       |        
    /// // |         (5 x 28)            |   (5 x 15)    |
    /// // |                             |               |
    /// // +---------------------+-------+-------+-------+        
    /// // |                     |               |       |
    /// // |       b3            |      b4       |  b5   |
    /// // |    (3 x 22)         |    (3 x 14)   |(3 x 7)|
    /// // +---------------------+---------------+-------+        
    /// //
    /// // It can be specified as follows.
    /// let b1 = HostTensor.zeros [5L; 28L]
    /// let b2 = HostTensor.zeros [5L; 15L]
    /// let b3 = HostTensor.zeros [3L; 22L]
    /// let b4 = HostTensor.zeros [3L; 14L]
    /// let b5 = HostTensor.zeros [3L; 7L]
    /// let bs = [[b1;   b2  ]
    ///           [b3; b4; b5]]
    /// let a = Tensor.ofBlocks bs
    /// // a.Shape = [8L; 43L]
    /// </code></example>    
    /// <remarks>    
    /// <para>The contents of each list are jointed along the dimension determined by its nesting level, i.e. 
    /// the elements of the outer lists are concatenated along dimension zero (rows) and the elements of the inner lists
    /// are concatenated along dimension one (columns).</para>
    /// <para>The contents of a list must have equal sizes in all dimensions except for the 
    /// concatenation dimensions.</para>
    /// </remarks>    
    static member ofBlocks (bs: Tensor<'T> list list) =
        bs |> List.map (List.map Block >> SubBlocks) |> SubBlocks |> Tensor.ofBlocks

    /// <summary>Builds a three dimensional tensor out of tensor blocks.</summary>
    /// <param name="bs">The tensor blocks.</param>    
    /// <returns>The resulting tensor.</returns>
    /// <remarks>    
    /// <para>The contents of each list are jointed along the dimension determined by its nesting level, i.e. 
    /// the elements of the outer-most lists are concatenated along dimension zero and the elements of the middle lists
    /// are concatenated along dimension one and the elements of the inner-most lists are concatenated along dimension
    /// two.</para>
    /// <para>The contents of a list must have equal sizes in all dimensions except for the 
    /// concatenation dimensions.</para>
    /// </remarks>    
    static member ofBlocks (bs: Tensor<'T> list list list) =
        bs |> List.map (List.map (List.map Block >> SubBlocks) >> SubBlocks) |> SubBlocks |> Tensor.ofBlocks

    /// <summary>Computes the tensor product between two tensors.</summary>
    /// <param name="a">The tensor on the left side of this binary operation.</param>
    /// <param name="b">The tensor on the right side of this binary operation.</param>
    /// <returns>A new tensor containing the result of this operation.</returns>    
    static member tensorProduct (a: Tensor<'T>) (b: Tensor<'T>) =
        let a, b = Tensor.padToSame (a, b)
        let rec generate (pos: int64 list) = 
            match List.length pos with
            | dim when dim = a.NDims ->
                let slice = pos |> List.map Rng.Elem
                Block (a.[slice] * b)
            | dim ->
                seq {for p in 0L .. a.Shape.[dim] - 1L -> generate (pos @ [p])}
                |> Seq.toList |> SubBlocks
        generate [] |> Tensor.ofBlocks

    /// <summary>Concatenates tensors along an axis.</summary>
    /// <param name="ax">The concatenation axis.</param>        
    /// <param name="ts">Sequence of tensors to concatenate.</param>    
    /// <returns>The concatenated tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [4L; 28L]
    /// let b = HostTensor.zeros [4L; 15L]
    /// let c = HostTensor.zeros [4L; 10L]
    /// let d = Tensor.concat 1 [a; b; c] // d.Shape = [4L; 53L]
    /// </code></example>    
    /// <remarks>    
    /// <para>The contents of a the tensors are concatenated in the specifed dimension.</para>
    /// <para>The sizes of the tensors in all other dimensions must be equal.</para>
    /// </remarks>    
    /// <seealso cref="ofBlocks"/>
    static member concat (ax: int) (ts: Tensor<'T> seq) =
        let ts = List.ofSeq ts
        if List.isEmpty ts then
            invalidArg "ts" "Cannot concatenate empty sequence of tensors."

        // check for compatibility
        let shp = ts.Head.Shape
        if not (0 <= ax && ax < shp.Length) then
            invalidArg "ax" "Concatenation axis %d is out of range for shape %A." ax shp
        for aryIdx, ary in List.indexed ts do
            if List.without ax ary.Shape <> List.without ax shp then
                invalidArg "ts" "Concatentation element with index %d with shape %A must 
                                 be equal to shape %A of the first element, except in the concatenation axis %d" 
                                 aryIdx ary.Shape shp ax

        // calculate shape of concatenated tensors
        let totalSize = ts |> List.sumBy (fun ary -> ary.Shape.[ax])
        let concatShape = shp |> List.set ax totalSize

        // copy tensors into concatenated tensor
        let cc = Tensor(concatShape, ts.Head.Dev)
        let mutable pos = 0L
        for ary in ts do
            let aryLen = ary.Shape.[ax]
            if aryLen > 0L then
                let ccRng = 
                    List.init shp.Length (fun idx ->
                        if idx = ax then Rng.Rng (Some pos, Some (pos + aryLen - 1L))
                        else Rng.All)
                cc.[ccRng] <- ary
                pos <- pos + aryLen
        cc

    /// <summary>Repeats the tensor along an axis.</summary>
    /// <param name="ax">The axis to repeat along.</param>        
    /// <param name="reps">The number of repetitions.</param>
    /// <param name="a">The tensor to repeat.</param>    
    /// <returns>The repeated tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [4L; 28L]
    /// let b = Tensor.replicate 0 10L a // b.Shape = [40L; 28L]
    /// </code></example>    
    /// <remarks>    
    /// <para>The contents of a the tensors are replicated <paramref name="reps"/> times in the specifed dimension.</para>
    /// </remarks>    
    static member replicate (ax: int) (reps: int64) (a: Tensor<'T>) =
        a.CheckAxis ax
        if reps < 0L then invalidArg "reps" "Number of repetitions cannot be negative."

        // 1. insert axis of size one left to repetition axis
        // 2. broadcast along the new axis to number of repetitions
        // 3. reshape to result shape
        a 
        |> Tensor.reshape (a.Shape |> List.insert ax 1L)
        |> Tensor.broadcastDim ax reps
        |> Tensor.reshape (a.Shape |> List.set ax (reps * a.Shape.[ax]))

    /// <summary>Calculates the difference between adjoining elements along the specified axes.</summary>
    /// <param name="ax">The axis to operate along.</param>
    /// <param name="a">The tensor containing the source values.</param>    
    /// <returns>The differences tensor. It has one element less in dimension <paramref name="ax"/> 
    /// as the input tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [10L; 5L]
    /// let b = Tensor.diffAxis 0 a // b.Shape = [9L; 5L]
    /// </code></example>
    /// <remarks>
    /// <para>The resulting tensor has one element less in the differentiation dimension than the original tensor.</para>
    /// </remarks>
    /// <seealso cref="diff"/>
    static member diffAxis (ax: int) (a: Tensor<'T>) =
        a.CheckAxis ax 
        let shftRng = 
            [for d=0 to a.NDims-1 do
                if d = ax then yield Rng.Rng (Some 1L, None)
                else yield Rng.All]
        let cutRng = 
            [for d=0 to a.NDims-1 do
                if d = ax then yield Rng.Rng (None, Some (a.Shape.[d] - 2L))
                else yield Rng.All]
        a.[shftRng] - a.[cutRng]

    /// <summary>Calculates the difference between adjoining elements of the vector.</summary>
    /// <param name="a">The vector containing the source values.</param>    
    /// <returns>The differences vector. It has one element less than the input tensor.</returns>
    /// <example><code language="fsharp">
    /// let a = HostTensor.zeros [5L]
    /// let b = Tensor.diff a // b.Shape = [4L]
    /// </code></example>
    /// <remarks>
    /// <para>The value of output element <c>i</c> is given by <c>d_i = a_(i+1) - a_i</c>.</para>
    /// <para>The resulting vector has one element less in the last dimension than the original vector.</para>
    /// <para>If the input tensor has more than one dimension, this operation is applied batch-wise on the last
    /// dimension.</para>
    /// </remarks>
    /// <seealso cref="diffAxis"/>
    static member diff (a: Tensor<'T>) =
        if a.NDims < 1 then invalidArg "a" "Need at least a vector to calculate diff."
        Tensor.diffAxis (a.NDims-1) a
        


/// See Tensor<'T>.
type Tensor = 

    /// checks that all tensors have the same storage
    static member internal CheckSameStorage (xs: ITensor list) =
        match xs with
        | x::rs when rs |> List.exists (fun r -> x.Dev <> r.Dev) ->
            let storages = xs |> List.map (fun x -> x.Dev.Id)
            invalidOp "Storage devices must be equal for this operation, but they are %A." storages
        | _ -> ()            

    /// checks that two tensors have the same shape
    static member internal CheckSameShape (a: ITensor) (b: ITensor) =
        if a.Shape <> b.Shape then
            invalidArg "b" "Tensors of shapes %A and %A were expected to have same shape" a.Shape b.Shape

    /// prepares the sources of an elementwise operation by broadcasting them to the target shape
    static member internal PrepareElemwiseSources<'TR, 'TA> (trgt: Tensor<'TR>, a: Tensor<'TA>) : Tensor<'TA> =
        Tensor.CheckSameStorage [trgt; a]
        let a = a |> Tensor<_>.broadcastTo trgt.Shape
        a

    /// prepares the sources of an elementwise operation by broadcasting them to the target shape
    static member internal PrepareElemwiseSources<'TR, 'TA, 'TB> (trgt: Tensor<'TR>, a: Tensor<'TA>, b: Tensor<'TB>) 
            : (Tensor<'TA> * Tensor<'TB>) =
        Tensor.CheckSameStorage [trgt; a; b]
        let a = a |> Tensor<_>.broadcastTo trgt.Shape
        let b = b |> Tensor<_>.broadcastTo trgt.Shape
        a, b

    /// prepares the sources of an elementwise operation by broadcasting them to the target shape
    static member internal PrepareElemwiseSources<'TR, 'TA, 'TB, 'TC> (trgt: Tensor<'TR>, a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>) 
            : (Tensor<'TA> * Tensor<'TB> * Tensor<'TC>) =
        Tensor.CheckSameStorage [trgt; a; b; c]
        let a = a |> Tensor<_>.broadcastTo trgt.Shape
        let b = b |> Tensor<_>.broadcastTo trgt.Shape
        let c = c |> Tensor<_>.broadcastTo trgt.Shape
        a, b, c

    /// Prepares the sources of an axis reduce operation (e.g. sum over axis),
    /// by moving the reduction axis to be the last axis in the source.
    static member internal PrepareAxisReduceSources<'TR, 'TA> 
            (trgt: Tensor<'TR>, axis: int, a: Tensor<'TA>,
             initial: Tensor<'TR> option) : (Tensor<'TA> * Tensor<'TR> option) =
        Tensor.CheckSameStorage [trgt; a]
        a.CheckAxis axis
        let redShp = a.Shape |> List.without axis
        if trgt.Shape <> redShp then
            invalidOp "Reduction of tensor %A along axis %d gives shape %A but target has shape %A." 
                      a.Shape axis redShp trgt.Shape
        let initial =
            match initial with
            | Some initial -> 
                Tensor.CheckSameStorage [trgt; initial]
                initial |> Tensor.broadcastTo redShp |> Some  
            | None -> None                                       
        let axisToLast = [
            for d in 0 .. axis-1 do yield d
            yield a.NDims-1
            for d in axis+1 .. a.NDims-1 do yield d-1
        ]
        let a = a |> Tensor<_>.permuteAxes axisToLast
        if not (trgt.Shape = a.Shape.[0 .. a.NDims-2]) then
            failwith "Internal axis reduce shape computation error."
        a, initial

    /// prepares an axis reduce operation by allocating a target of appropriate size and storage
    static member internal PrepareAxisReduceTarget<'TR, 'TA> (axis: int, a: Tensor<'TA>, ?order: TensorOrder) : (Tensor<'TR> * Tensor<'TA>) =
        a.CheckAxis axis
        let redShp = a.Shape |> List.without axis        
        let trgt = Tensor<'TR> (redShp, a.Storage.Dev, ?order=order)
        trgt, a

    /// prepares an elementwise operation by allocating a target of same size and storage
    static member internal PrepareElemwise<'TR, 'TA> (a: Tensor<'TA>, ?order: TensorOrder) : (Tensor<'TR> * Tensor<'TA>) =
        let trgt = Tensor<'TR> (a.Shape, a.Storage.Dev, ?order=order)
        trgt, a

    /// prepares an elementwise operation by broadcasting both tensors to the same size
    /// and allocating a target of same size and storage
    static member internal PrepareElemwise<'TR, 'TA, 'TB> (a: Tensor<'TA>, b: Tensor<'TB>, ?order: TensorOrder) 
            : (Tensor<'TR> * Tensor<'TA> * Tensor<'TB>) =
        Tensor.CheckSameStorage [a; b]
        let a, b = Tensor<_>.broadcastToSame (a, b)
        let trgt = Tensor<'TR> (a.Shape, a.Storage.Dev, ?order=order)
        trgt, a, b

    /// prepares an elementwise operation by broadcasting all three tensors to the same size
    /// and allocating a target of same size and storage
    static member internal PrepareElemwise<'TR, 'TA, 'TB, 'TC> (a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>, ?order: TensorOrder) 
            : (Tensor<'TR> * Tensor<'TA> * Tensor<'TB> * Tensor<'TC>) =
        Tensor.CheckSameStorage [a; b; c]
        let a, b, c = Tensor<_>.broadcastToSame (a, b, c)
        let trgt = Tensor<'TR> (a.Shape, a.Storage.Dev, ?order=order)
        trgt, a, b, c



/// Special values that can be passed instead of masks.
[<AutoOpen>]
module internal SpecialMask =

    /// Indicates that the dimension is unmasked, i.e. equals specifying a tensor filled with trues. 
    let NoMask : Tensor<bool> = Unchecked.defaultof<_> // = null
        
        