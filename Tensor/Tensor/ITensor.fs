namespace Tensor

open System
open System.Collections
open System.Collections.Generic
open System.Diagnostics

open Tensor.Utils
open Tensor.Backend
open DeepNet.Utils



/// <summary>Type-neutral interface to Tensor&lt;'T&gt; of any type 'T.</summary>
/// <remarks>These functions are useful for working with tensors of unknown types.
/// For most use cases the functions provided by <see cref="Tensor`1"/> are better suited.</remarks>
/// <seealso cref="Tensor`1"/>
type ITensor =

    /// layout of this tensor (shape, offset and strides)
    abstract Layout:            TensorLayout
    /// storage of this tensor
    abstract Storage:           ITensorStorage
    /// storage factory
    abstract Dev:               ITensorDevice
    /// shape
    abstract Shape:             int64 list
    /// number of dimensions
    abstract NDims:             int
    /// number of elements
    abstract NElems:            int64    
    /// type of data stored in this tensor
    abstract DataType:          System.Type
    /// pretty contents string
    abstract Pretty:            string
    /// full contents string
    abstract Full:              string

    /// a tensor with the same storage but new layout
    abstract Relayout:          TensorLayout -> ITensor
    /// returns a copy of the tensor
    abstract Copy:              ?order:TensorOrder -> ITensor
    /// Transfers this tensor to the specifed device.
    abstract Transfer:          dev:ITensorDevice -> ITensor
    /// fills the tensors with zeros
    abstract FillZero:          unit -> unit

    /// Element selection using boolean mask. Specify NoMask for a dimension if no masking is desired.   
    abstract M : m0:ITensor -> ITensor with get, set
    /// Element selection using boolean mask. Specify NoMask for a dimension if no masking is desired.   
    abstract M : m0:ITensor * m1:ITensor -> ITensor with get, set
    /// Element selection using boolean mask. Specify NoMask for a dimension if no masking is desired.   
    abstract M : m0:ITensor * m1:ITensor * m2:ITensor -> ITensor with get, set
    /// Element selection using boolean mask. Specify NoMask for a dimension if no masking is desired.   
    abstract M : m0:ITensor * m1:ITensor * m2:ITensor * m3:ITensor -> ITensor with get, set
    /// Element selection using boolean mask. Specify NoMask for a dimension if no masking is desired.   
    abstract M : m0:ITensor * m1:ITensor * m2:ITensor * m3:ITensor * m4:ITensor -> ITensor with get, set
    /// Element selection using boolean mask. Specify NoMask for a dimension if no masking is desired.   
    abstract M : masks:ITensor list -> ITensor with get, set

    /// n-dimensional slicing using a list of Rngs
    abstract Item : rng:Rng list -> ITensor with get
    /// n-dimensional slicing using a list of Rngs
    abstract Item : rng:Rng list -> ITensor with set

    // type-neutral slicing 
    abstract Item : i0:int64 -> ITensor with get
    abstract Item : i0:int64 * i1:int64 -> ITensor with get
    abstract Item : i0:int64 * i1:int64 * i2:int64 -> ITensor with get
    abstract Item : o0:obj * o1:obj * o2:obj * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor with get

    abstract Item : i0:int64 -> ITensor with set
    abstract Item : i0:int64 * i1:int64 -> ITensor with set
    abstract Item : i0:int64 * i1:int64 * i2:int64 -> ITensor with set
    abstract Item : o0:obj * o1:obj * o2:obj * o3:obj -> ITensor with set
    abstract Item : o0:obj * o1:obj * o2:obj * o3:obj * o4:obj -> ITensor with set
    abstract Item : o0:obj * o1:obj * o2:obj * o3:obj * o4:obj * o5:obj -> ITensor with set
    abstract Item : o0:obj * o1:obj * o2:obj * o3:obj * o4:obj * o5:obj * o6:obj -> ITensor with set
    abstract Item : o0:obj * o1:obj * o2:obj * o3:obj * o4:obj * o5:obj * o6:obj * o7:obj -> ITensor with set
    abstract Item : o0:obj * o1:obj * o2:obj * o3:obj * o4:obj * o5:obj * o6:obj * o7:obj * o8:obj -> ITensor with set
    abstract Item : o0:obj * o1:obj * o2:obj * o3:obj * o4:obj * o5:obj * o6:obj * o7:obj * o8:obj * o9:obj -> ITensor with set

    abstract GetSlice : i0s:int64 option * i0f:int64 option -> ITensor
    abstract GetSlice : i0:int64 * i1s:int64 option * i1f:int64 option -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1:int64 -> ITensor
    abstract GetSlice : i0:int64 * i1:int64 * i2:int64 -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option -> ITensor
    abstract GetSlice : i0:int64 * i1:int64 * i2s:int64 option * i2f:int64 option -> ITensor
    abstract GetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * i2:int64 -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * i2:int64 -> ITensor
    abstract GetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * i2s:int64 option * i2f:int64 option -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * i2s:int64 option * i2f:int64 option -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * i2:int64 -> ITensor
    abstract GetSlice : i0:int64 * i1:int64 * i2:int64 * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * i2s:int64 option * i2f:int64 option -> ITensor
    abstract GetSlice : i0:int64 * i1:int64 * i2s:int64 option * i2f:int64 option * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor
    abstract GetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * i2:int64 * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * i2:int64 * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor
    abstract GetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * i2s:int64 option * i2f:int64 option * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * i2s:int64 option * i2f:int64 option * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * i2:int64 * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor
    abstract GetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * i2s:int64 option * i2f:int64 option * o3:obj * [<System.ParamArray>] r:obj [] -> ITensor

    abstract SetSlice : i0s:int64 option * i0f:int64 option * value:ITensor -> unit
    abstract SetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * value:ITensor -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * value:ITensor -> unit
    abstract SetSlice : i0:int64 * i1:int64 * i2:int64 * value:ITensor -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * value:ITensor -> unit
    abstract SetSlice : i0:int64 * i1:int64 * i2s:int64 option * i2f:int64 option * value:ITensor -> unit
    abstract SetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * i2:int64 * value:ITensor -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * i2:int64 * value:ITensor -> unit
    abstract SetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * i2s:int64 option * i2f:int64 option * value:ITensor -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * i2s:int64 option * i2f:int64 option * value:ITensor -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * i2:int64 * value:ITensor -> unit
    abstract SetSlice : i0:int64 * i1:int64 * i2:int64 * o3:obj * o4:obj * [<System.ParamArray>] r:obj [] -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * i2s:int64 option * i2f:int64 option * value:ITensor -> unit
    abstract SetSlice : i0:int64 * i1:int64 * i2s:int64 option * i2f:int64 option * o3:obj * o4:obj * [<System.ParamArray>] r:obj [] -> unit
    abstract SetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * i2:int64 * o3:obj * o4:obj * [<System.ParamArray>] r:obj [] -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * i2:int64 * o3:obj * o4:obj * [<System.ParamArray>] r:obj [] -> unit
    abstract SetSlice : i0:int64 * i1s:int64 option * i1f:int64 option * i2s:int64 option * i2f:int64 option * o3:obj * o4:obj * [<System.ParamArray>] r:obj [] -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1:int64 * i2s:int64 option * i2f:int64 option * o3:obj * o4:obj * [<System.ParamArray>] r:obj [] -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * i2:int64 * o3:obj * o4:obj * [<System.ParamArray>] r:obj [] -> unit
    abstract SetSlice : i0s:int64 option * i0f:int64 option * i1s:int64 option * i1f:int64 option * i2s:int64 option * i2f:int64 option * o3:obj * o4:obj * [<System.ParamArray>] r:obj [] -> unit

    // type-neutral unary operations
    abstract UnaryPlus: unit -> ITensor
    abstract UnaryMinus: unit -> ITensor
    abstract Abs: unit -> ITensor
    abstract Sgn: unit -> ITensor
    abstract Log: unit -> ITensor
    abstract Log10: unit -> ITensor
    abstract Exp: unit -> ITensor
    abstract Sin: unit -> ITensor
    abstract Cos: unit -> ITensor
    abstract Tan: unit -> ITensor
    abstract Asin: unit -> ITensor
    abstract Acos: unit -> ITensor
    abstract Atan: unit -> ITensor
    abstract Sinh: unit -> ITensor
    abstract Cosh: unit -> ITensor
    abstract Tanh: unit -> ITensor
    abstract Sqrt: unit -> ITensor
    abstract Ceiling: unit -> ITensor
    abstract Floor: unit -> ITensor
    abstract Round: unit -> ITensor
    abstract Truncate: unit -> ITensor
    abstract IsFinite: unit -> ITensor

    // type-neutral binary operations
    abstract Add: ITensor -> ITensor
    abstract Subtract: ITensor -> ITensor
    abstract Multiply: ITensor -> ITensor
    abstract Divide: ITensor -> ITensor
    abstract Modulo: ITensor -> ITensor
    abstract Pow: ITensor -> ITensor
    
    abstract Equal: ITensor -> ITensor
    abstract NotEqual: ITensor -> ITensor
    abstract Less: ITensor -> ITensor
    abstract LessOrEqual: ITensor -> ITensor
    abstract Greater: ITensor -> ITensor
    abstract GreaterOrEqual: ITensor -> ITensor
    abstract MaxElemwise: ITensor -> ITensor
    abstract MinElemwise: ITensor -> ITensor

    abstract IfThenElse: ifFalse:ITensor -> cond:ITensor -> ITensor
    abstract Gather: indices:ITensor option list -> ITensor
    abstract Scatter: indices:ITensor option list -> trgtShp:int64 list -> ITensor

    // type-neutral reduction operations
    abstract SumAxis: ax:int -> ITensor
    abstract SumTensor: unit -> ITensor
    abstract ProductAxis: ax:int -> ITensor
    abstract ProductTensor: unit -> ITensor
    abstract MinAxis: ax:int -> ITensor
    abstract MinTensor: unit -> ITensor
    abstract MaxAxis: ax:int -> ITensor
    abstract MaxTensor: unit -> ITensor
    abstract ArgMinAxis: ax:int -> ITensor
    abstract ArgMaxAxis: ax:int -> ITensor
    abstract ArgMin: unit -> int64 list
    abstract ArgMax: unit -> int64 list

    abstract Dot: ITensor -> ITensor
    abstract TensorProduct: ITensor -> ITensor
    abstract Invert: unit -> ITensor
    abstract SVD: unit -> ITensor * ITensor * ITensor
    abstract SVDWithoutUV: unit -> ITensor
    abstract PseudoInvert: unit -> ITensor
    abstract SymmetricEigenDecomposition: part:MatrixPart -> ITensor * ITensor

/// <summary>Type-neutral interface to Tensor&lt;'T&gt; of any type 'T.</summary>
/// <remarks>These functions are useful for working with tensors of unknown types.
/// For most use cases the functions provided by <see cref="Tensor`1"/> are better suited.</remarks>
/// <seealso cref="Tensor`1"/>
module ITensor =

    /// <summary>Memory layout of the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Memory layout.</returns>
    /// <seealso cref="Tensor`1.Layout"/>
    let layout (a: ITensor) = a.Layout

    /// <summary>Device the data of tensor is stored on.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Data storage device.</returns>
    /// <seealso cref="Tensor`1.Dev"/>
    let dev (a: ITensor) = a.Dev

    /// <summary>Shape of the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Shape.</returns>
    /// <seealso cref="Tensor`1.Shape"/>
    let shape (a: ITensor) = a.Shape

    /// <summary>Dimensionality of the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Number of dimensions.</returns>
    /// <seealso cref="Tensor`1.NDims"/>
    let nDims (a: ITensor) = a.Layout.NDims

    /// <summary>Total number of elements within the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Number of elements.</returns>
    /// <seealso cref="Tensor`1.NElems"/>
    let nElems (a: ITensor) = a.Layout.NElems

    /// <summary>Type of data stored within the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Data type.</returns>
    /// <seealso cref="Tensor`1.DataType"/>
    let dataType (a: ITensor) = a.DataType

    /// <summary>Creates a tensor with the specified layout sharing its storage with the original tensor.</summary>
    /// <param name="newLayout">The new tensor memory layout.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Tensor`1.relayout"/>
    let relayout newLayout (a: ITensor) =
        a.Relayout newLayout 

    let private ApplyLayoutFn (fn, xs: ITensor list) = 
        let layouts = fn (xs |> List.map layout)
        (layouts, xs) ||> List.map2 relayout              

    /// <summary>Get a slice (part) of the tensor.</summary>
    /// <param name="rng">The range of the tensor to select.</param>    
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Tensor`1.range"/>
    let range (rng: Rng list) (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.view rng)

    /// <summary>Gets a sequence of all indices to enumerate all elements within the tensor.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Sequence of indicies.</returns>
    /// <remarks>The sequence sequentially enumerates the indices of all elements of the tensor.</remarks>
    /// <seealso cref="Tensor`1.allIdxOfDim"/>
    let allIdx (a: ITensor) = a.Layout |> TensorLayout.allIdx

    /// <summary>Insert a dimension of size one as the first dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Tensor`1.padLeft"/>
    let padLeft (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.padLeft)

    /// <summary>Append a dimension of size one after the last dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Tensor`1.padRight"/>
    let padRight (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.padRight)

    /// <summary>Insert a dimension of size one before the specifed dimension.</summary>
    /// <param name="ax">The dimension to insert before.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Tensor`1.insertAxis"/>
    let insertAxis ax (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.insertAxis ax)

    /// <summary>Removes the first dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Tensor`1.cutLeft"/>
    let cutLeft (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.cutLeft)

    /// <summary>Removes the last dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Tensor`1.cutRight"/>
    let cutRight (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.cutRight)

    /// <summary>Broadcast a dimension to a specified size.</summary>
    /// <param name="dim">The size-one dimension to broadcast.</param>
    /// <param name="size">The size to broadcast to.</param>    
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The resulting tensor.</returns>
    /// <seealso cref="Tensor`1.broadcastDim"/>
    let broadcastDim dim size (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.broadcastDim dim size)       

    /// <summary>Pads all specified tensors from the left with dimensions of size one until they have the 
    /// same dimensionality.</summary>
    /// <param name="xs">A list of tensors to operate on.</param>
    /// <returns>A list of the resulting tensors, all having the same dimensionality.</returns>
    /// <seealso cref="Tensor`1.padToSame"/>
    let padToSame (xs: ITensor list) = 
        ApplyLayoutFn (TensorLayout.padToSameMany, xs)

    /// <summary>Broadcasts all specified tensors to have the same shape.</summary>
    /// <param name="xs">A list of tensors to operate on.</param>    
    /// <returns>A list of the resulting tensors, all having the same shape.</returns>
    /// <seealso cref="Tensor`1.broadcastToSame"/>
    let broadcastToSame (xs: ITensor list) =
        ApplyLayoutFn (TensorLayout.broadcastToSameMany, xs)

    /// <summary>Broadcasts all specified tensors to have the same size in the specified dimensions.</summary>
    /// <param name="dims">A list of dimensions that should be broadcasted to have the same size.</param>
    /// <param name="xs">A list of tensors to operate on.</param>
    /// <returns>A list of the resulting tensors, all having the same size in the specified dimensions.</returns>
    /// <seealso cref="Tensor`1.broadcastToSameInDims"/>
    let broadcastToSameInDims (dims, xs: ITensor list) =
        ApplyLayoutFn (TensorLayout.broadcastToSameInDimsMany dims, xs)

    /// <summary>Broadcasts the specified tensor to the specified shape.</summary>
    /// <param name="shp">The target shape.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>Tensor of shape <paramref name="shp"/>.</returns>
    /// <seealso cref="Tensor`1.broadcastTo"/>
    let broadcastTo shp (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.broadcastToShape shp)

    /// <summary>Checks if the specified tensor is broadcasted in at least one dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>true if at least one dimension is broadcasted, otherwise false.</returns>
    /// <seealso cref="Tensor`1.isBroadcasted"/>
    let isBroadcasted (a: ITensor) =
        a.Layout |> TensorLayout.isBroadcasted 

    /// <summary>Tries to create a reshaped view of the tensor (without copying).</summary>
    /// <param name="shp">The target shape.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The reshaped tensor, if reshaping without copying is possible. Otherwise <c>None</c>.</returns>
    /// <seealso cref="Tensor`1.tryReshapeView"/>
    let tryReshapeView shp (a: ITensor) =
        match a.Layout |> TensorLayout.tryReshape shp with
        | Some newLayout -> a.Relayout newLayout |> Some
        | None -> None

    /// <summary>Creates a reshaped view of the tensor (without copying).</summary>
    /// <param name="shp">The target shape.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A reshaped view of the original tensor.</returns>
    /// <seealso cref="Tensor`1.reshapeView"/>
    let reshapeView shp (a: ITensor) =
        match tryReshapeView shp a with
        | Some res -> res
        | None -> 
            let msg =
                sprintf "Cannot reshape tensor of shape %A and strides %A without copying."
                    a.Shape a.Layout.Stride
            raise (InvalidOperationException msg)

    /// <summary>Changes the shape of a tensor.</summary>
    /// <param name="shp">The target shape.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor of the specified shape.</returns>
    /// <seealso cref="Tensor`1.reshape"/>
    let reshape shp (a: ITensor) =
        match a |> tryReshapeView shp with
        | Some res -> res
        | None -> a.Copy() |> reshapeView shp

    /// <summary>Flattens the tensor into a (one-dimensional) vector.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A vector.</returns>
    /// <seealso cref="Tensor`1.flatten"/>
    let flatten (a: ITensor) =
        reshape [Remainder] a    

    /// <summary>Swaps the specified dimensions of the tensor.</summary>
    /// <param name="ax1">The dimension to swap.</param>
    /// <param name="ax2">The dimension to swap with.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The tensor with the dimensions swapped.</returns>
    /// <seealso cref="Tensor`1.swapDim"/>
    let swapDim ax1 ax2 (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.swapDim ax1 ax2)

    /// <summary>Transpose of a matrix.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The result of this operation.</returns>
    /// <seealso cref="Tensor`1.transpose"/>
    let transpose (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.transpose)

    /// <summary>Permutes the axes as specified.</summary>
    /// <param name="permut">The permutation to apply to the dimensions of tensor.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The tensor with the dimensions permuted.</returns>
    /// <seealso cref="Tensor`1.permuteAxes"/>
    let permuteAxes (permut: int list) (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.permuteAxes permut)

    /// <summary>Reverses the elements in the specified dimension.</summary>
    /// <param name="ax">The axis to reverse.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>The tensor with the dimensions permuted.</returns>
    /// <seealso cref="Tensor`1.reverseAxis"/>
    let reverseAxis ax (a: ITensor) =
        a.Relayout (a.Layout |> TensorLayout.reverseAxis ax)        

    /// <summary>Pads the tensor from the left with size-one dimensions until it has at least the specified number of
    /// dimensions.</summary>
    /// <param name="minDims">The minimum number of dimensions.</param>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor with at least <paramref name="minDims"/> dimensions.</returns>
    /// <seealso cref="Tensor`1.atLeastND"/>
    let atLeastND minDims (a: ITensor) =
        if a.NDims >= minDims then a
        else
            let newShp = List.init (minDims - a.NDims) (fun _ -> 1L)
            a |> reshape newShp

    /// <summary>Pads the tensor from the left with size-one dimensions until it has at least one dimension.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor with at least one dimensions.</returns>
    /// <seealso cref="Tensor`1.atLeast1D"/>
    let atLeast1D (a: ITensor) = a |> atLeastND 1

    /// <summary>Pads the tensor from the left with size-one dimensions until it has at least two dimensions.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor with at least two dimensions.</returns>
    /// <seealso cref="Tensor`1.atLeast2D"/>
    let atLeast2D (a: ITensor) = a |> atLeastND 2

    /// <summary>Pads the tensor from the left with size-one dimensions until it has at least three dimensions.</summary>
    /// <param name="a">The tensor to operate on.</param>
    /// <returns>A tensor with at least three dimensions.</returns>
    /// <seealso cref="Tensor`1.atLeast3D"/>
    let atLeast3D (a: ITensor) = a |> atLeastND 3

    /// <summary>Returns a copy of the tensor.</summary>
    /// <param name="a">The tensor to copy.</param>
    /// <param name="order">The memory layout of the copy. (default: row-major)</param>
    /// <returns>A copy of the tensor.</returns>
    /// <seealso cref="Tensor`1.copy"/>
    let copy (a: ITensor) = a.Copy()

    /// <summary>Transfers a tensor to the specifed device.</summary>
    /// <param name="dev">The target device.</param>
    /// <param name="a">The tensor to transfer.</param>
    /// <returns>A tensor on the target device.</returns>
    /// <seealso cref="Tensor`1.transfer"/>
    let transfer (dev: ITensorDevice) (src: ITensor) =
        src.Transfer (dev)
        

