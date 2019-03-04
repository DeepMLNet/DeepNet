namespace Tensor

open System

open DeepNet.Utils
open Tensor.Backend



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
    /// String representation of the tensor limited to a specific number of elements per dimension.
    abstract ToString:          int64 -> string

    /// a tensor with the same storage but new layout
    abstract Relayout:          TensorLayout -> ITensor
    /// Returns a copy of the tensor.
    abstract Copy:              ?order:TensorOrder -> ITensor
    /// Fills this tensor with a copy of the specified tensor.
    abstract CopyFrom:          src:ITensor -> unit
    /// Transfers this tensor to the specifed device.
    abstract Transfer:          dev:ITensorDevice -> ITensor
    /// fills the tensors with zeros
    abstract FillZero:          unit -> unit
    /// fills the tensors with ones
    abstract FillOnes:          unit -> unit
    /// Tensor of same type filled with zeros.
    abstract ZerosOfSameType:   dev:ITensorDevice -> shape:int64 list -> ITensor
    /// Convert this tensor to tensor of specified type.
    abstract Convert:           Type -> ITensor

    /// Value of a scalar tensor.
    abstract Value: obj with get, set

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
    abstract AllFinite: unit -> bool

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

    abstract IsClose: other:ITensor * ?absTol:obj * ?relTol:obj -> ITensor
    abstract AlmostEqual: other:ITensor * ?absTol:obj * ?relTol:obj -> bool

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

    // other type-neutral operations
    abstract DiagAxis: ax1:int -> ax2:int -> ITensor
    abstract Diag: unit -> ITensor
    abstract DiagMatAxis: ax1:int -> ax2:int -> ITensor
    abstract DiagMat: unit -> ITensor

