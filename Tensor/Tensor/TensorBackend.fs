namespace Tensor.Backend

open System
open System.Collections
open System.Collections.Generic
open System.Diagnostics

open Tensor
open Tensor.Utils



/// Tensor storage (type neutral).
type ITensorStorage =
    abstract Dev:               ITensorDevice

/// Tensor storage.
and ITensorStorage<'T> =
    inherit ITensorStorage
    abstract Backend:           TensorLayout -> ITensorBackend<'T>

/// Tensor device.
and ITensorDevice =
    inherit IComparable
    inherit IComparable<ITensorDevice>
    inherit IEquatable<ITensorDevice>

    abstract Id:                string
    abstract Create:            nElems:int64 -> ITensorStorage<'T>
    abstract Zeroed:            bool

/// Tensor frontend access (for use from backend).
and ITensorFrontend<'T> =
    /// storage of this tensor
    abstract Storage:           ITensorStorage<'T>    
    /// storage factory
    abstract Dev:               ITensorDevice
    /// the backend    
    abstract Backend:           ITensorBackend<'T>
    /// layout of this tensor (shape, offset and strides)
    abstract Layout:            TensorLayout
    /// shape
    abstract Shape:             int64 list
    /// stride
    abstract Stride:            int64 list
    /// stride
    abstract Offset:            int64 
    /// number of dimensions
    abstract NDims:             int
    /// number of elements
    abstract NElems:            int64    
    /// a tensor with the same storage but new layout
    abstract Relayout:          layout:TensorLayout -> ITensorFrontend<'T>
    /// returns a copy of the tensor
    abstract Copy:              ?order:TensorOrder -> ITensorFrontend<'T>
    /// Copies the specifed tensor into this tensor.
    abstract CopyFrom:          src:ITensorFrontend<'T> -> unit
    /// Transfers this tensor to the specifed device.
    abstract Transfer:          dev:ITensorDevice -> ITensorFrontend<'T>
    /// Transpose
    abstract T:                 ITensorFrontend<'T>

/// Tensor backend.
and ITensorBackend<'T> =
    abstract Item:              int64[] -> 'T with get, set

    abstract Copy:              trgt:ITensorFrontend<'T> * src:ITensorFrontend<'T> -> unit
    abstract Transfer:          trgt:ITensorFrontend<'T> * src:ITensorFrontend<'T> -> bool
    abstract Convert:           trgt:ITensorFrontend<'T> * src:ITensorFrontend<'T1> -> unit

    abstract FillConst:         value:'T * trgt:ITensorFrontend<'T> -> unit
    abstract FillIncrementing:  start:'T * incr:'T * trgt:ITensorFrontend<'T> -> unit
    
    abstract UnaryPlus:         trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract UnaryMinus:        trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Abs:               trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Sgn:               trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Log:               trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Log10:             trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Exp:               trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Sin:               trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Cos:               trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Tan:               trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Asin:              trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Acos:              trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Atan:              trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Sinh:              trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Cosh:              trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Tanh:              trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Sqrt:              trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Ceiling:           trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Floor:             trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Round:             trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract Truncate:          trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract IsFinite:          trgt:ITensorFrontend<bool> * src1:ITensorFrontend<'T> -> unit

    abstract Add:               trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract Subtract:          trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract Multiply:          trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract Divide:            trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract Modulo:            trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract Power:             trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract MaxElemwise:       trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract MinElemwise:       trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit

    abstract Equal:             trgt:ITensorFrontend<bool> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract NotEqual:          trgt:ITensorFrontend<bool> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract Less:              trgt:ITensorFrontend<bool> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract LessOrEqual:       trgt:ITensorFrontend<bool> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract Greater:           trgt:ITensorFrontend<bool> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract GreaterOrEqual:    trgt:ITensorFrontend<bool> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit

    abstract Negate:            trgt:ITensorFrontend<bool> * src1:ITensorFrontend<bool> -> unit
    abstract And:               trgt:ITensorFrontend<bool> * src1:ITensorFrontend<bool> * src2:ITensorFrontend<bool> -> unit
    abstract Or:                trgt:ITensorFrontend<bool> * src1:ITensorFrontend<bool> * src2:ITensorFrontend<bool> -> unit
    abstract Xor:               trgt:ITensorFrontend<bool> * src1:ITensorFrontend<bool> * src2:ITensorFrontend<bool> -> unit

    abstract IfThenElse:        trgt:ITensorFrontend<'T> * cond:ITensorFrontend<bool> * ifTrue:ITensorFrontend<'T> * ifFalse:ITensorFrontend<'T> -> unit  
    abstract Gather:            trgt:ITensorFrontend<'T> * srcIdxs:ITensorFrontend<int64> option list * src:ITensorFrontend<'T> -> unit
    abstract Scatter:           trgt:ITensorFrontend<'T> * trgtIdxs:ITensorFrontend<int64> option list * src:ITensorFrontend<'T> -> unit
    abstract MaskedGet:         trgt:ITensorFrontend<'T> * src:ITensorFrontend<'T> * masks:ITensorFrontend<bool> option [] -> unit
    abstract MaskedSet:         trgt:ITensorFrontend<'T> * masks:ITensorFrontend<bool> option [] * src:ITensorFrontend<'T> -> unit
    abstract TrueIndices:       trgt:ITensorFrontend<int64> * src1:ITensorFrontend<bool> -> unit

    abstract SumLastAxis:       trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract ProductLastAxis:   trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract MinLastAxis:       trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract MaxLastAxis:       trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract AllLastAxis:       trgt:ITensorFrontend<bool> * src1:ITensorFrontend<bool> -> unit
    abstract AnyLastAxis:       trgt:ITensorFrontend<bool> * src1:ITensorFrontend<bool> -> unit
    abstract CountTrueLastAxis: trgt:ITensorFrontend<int64> * src1:ITensorFrontend<bool> -> unit

    abstract ArgMinLastAxis:    trgt:ITensorFrontend<int64> * src1:ITensorFrontend<'T> -> unit
    abstract ArgMaxLastAxis:    trgt:ITensorFrontend<int64> * src1:ITensorFrontend<'T> -> unit
    abstract FindLastAxis:      value:'T * trgt:ITensorFrontend<int64> * src1:ITensorFrontend<'T> -> unit

    abstract VecVecDot:         trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract MatVecDot:         trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract MatMatDot:         trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit
    abstract BatchedMatMatDot:  trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> * src2:ITensorFrontend<'T> -> unit

    //abstract BatchedLU:             trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract BatchedSVD:            trgtS:ITensorFrontend<'T> * trgtUV:(ITensorFrontend<'T> * ITensorFrontend<'T>) option * src1:ITensorFrontend<'T> -> unit
    abstract BatchedInvert:         trgt:ITensorFrontend<'T> * src1:ITensorFrontend<'T> -> unit
    abstract SymmetricEigenDecomposition: part:MatrixPart * trgtEigVals:ITensorFrontend<'T> * trgtEigVecs:ITensorFrontend<'T> * 
                                          src:ITensorFrontend<'T> -> unit

/// Base tensor device.
[<AbstractClass>]
[<StructuredFormatDisplay("{Id}")>]
type BaseTensorDevice() =   
    abstract Id: string
    abstract Create: nElems:int64 -> ITensorStorage<'T>
    abstract Zeroed: bool

    interface ITensorDevice with
        member this.Id = this.Id
        member this.Create nElems = this.Create nElems
        member this.Zeroed = this.Zeroed

    interface IComparable<ITensorDevice> with
        member this.CompareTo other =
            compare (this :> ITensorDevice).Id other.Id
    interface IComparable with
        member this.CompareTo other =
            match other with
            | :? ITensorDevice as other -> 
                (this :> IComparable<ITensorDevice>).CompareTo other
            | _ -> failwithf "cannot compare to %A" (other.GetType())
    interface IEquatable<ITensorDevice> with
        member this.Equals other =
            (this :> ITensorDevice).Id = other.Id
    override this.Equals other =
        match other with
        | :? ITensorDevice as other ->
            (this :> IEquatable<ITensorDevice>).Equals other
        | _ -> false
    override this.GetHashCode () =
        hash (this :> ITensorDevice).Id
    override this.ToString () = this.Id
