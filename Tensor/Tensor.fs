namespace ArrayNDNS

open System.Collections
open System.Collections.Generic

open Basics
open System


/// singular matrix encountered
exception SingularMatrixError of string

/// operation requires tensors of same storage, but specified tensors had different storages
exception StorageMismatch of string

/// operation requires tensors of same shape, but specified tensor had different shapes
exception ShapeMismatch of string

/// a row-major layout was required for this operation, but the tensor has a different layout
exception RowMajorLayoutRequired of string

/// specified tensor index is out of range
exception PositionOutOfRange of string

/// sequence too short to fill tensor
exception SeqTooShort of string


/// memory ordering of tensor
type TensorOrder =
    /// row-major (C) order
    | RowMajor
    /// column-major (Fortran) order
    | ColumnMajor

/// Type-neutral interface to Tensor<'T> of any type
type ITensor =
    abstract Layout:            TensorLayout
    abstract Storage:           ITensorStorage
    abstract Shape:             int64 list
    abstract NDims:             int
    abstract NElems:            int64
    abstract CPPType:           string
    abstract Relayout:          TensorLayout -> ITensor
    abstract NewOfSameType:     TensorLayout -> ITensor
    abstract NewOfType:         TensorLayout -> System.Type -> ITensor
    abstract DataType:          System.Type
    abstract Copy:              ?order:TensorOrder -> ITensor
    abstract CopyTo:            ITensor -> unit
    abstract GetSlice:          [<System.ParamArray>] args: obj [] -> ITensor
    abstract SetSlice:          [<System.ParamArray>] args: obj [] -> unit
    abstract Item:              [<System.ParamArray>] allArgs: obj [] -> ITensor with get
    abstract Item:              obj -> ITensor with set
    abstract Item:              obj * obj -> ITensor with set
    abstract Item:              obj * obj * obj -> ITensor with set
    abstract Item:              obj * obj * obj * obj -> ITensor with set
    abstract Item:              obj * obj * obj * obj * obj -> ITensor with set
    abstract Item:              obj * obj * obj * obj * obj * obj -> ITensor with set
    abstract Item:              obj * obj * obj * obj * obj * obj * obj -> ITensor with set

and ITensorStorage =
    abstract Id:            string

and ITensorStorage<'T> =
    inherit ITensorStorage
    abstract Backend:       TensorLayout -> ITensorBackend<'T>
    abstract Factory:       ITensorStorageFactory
    //abstract Create:        nElems:int64 -> ITensorStorage<'T>

and ITensorBackend<'T> =
    abstract Item:          int64[] -> 'T with get, set
    abstract FillConst:     value:'T * trgt:Tensor<'T> -> unit
    abstract Copy:          trgt:Tensor<'T> * src:Tensor<'T> -> unit
    abstract Convert:       trgt:Tensor<'T> * src:Tensor<'T1> -> unit
    abstract Fill:          fn:(unit -> 'T) * trgt:Tensor<'T> * useThreads:bool -> unit
    abstract FillIndexed:   fn:(int64[] -> 'T) * trgt:Tensor<'T> * useThreads:bool -> unit
    abstract Map:           fn:('T1 -> 'T) * trgt:Tensor<'T> * src:Tensor<'T1> *
                            useThreads:bool -> unit
    abstract MapIndexed:    fn:(int64[] -> 'T1 -> 'T) * trgt:Tensor<'T> * src:Tensor<'T1> *
                            useThreads:bool -> unit
    abstract Map2:          fn:('T1 -> 'T2 -> 'T) * 
                            trgt:Tensor<'T> * src1:Tensor<'T1> * src2:Tensor<'T2> * 
                            useThreads:bool -> unit
    abstract MapIndexed2:   fn:(int64[] -> 'T1 -> 'T2 -> 'T) *
                            trgt:Tensor<'T> * src1:Tensor<'T1> * src2:Tensor<'T2> *
                            useThreads:bool -> unit

    abstract UnaryPlus:     trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract UnaryMinus:    trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Abs:           trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Sgn:           trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Log:           trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Log10:         trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Exp:           trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Sin:           trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Cos:           trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Tan:           trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Asin:          trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Acos:          trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Atan:          trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Sinh:          trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Cosh:          trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Tanh:          trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Sqrt:          trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Ceiling:       trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Floor:         trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Round:         trgt:Tensor<'T> * src1:Tensor<'T> -> unit
    abstract Truncate:      trgt:Tensor<'T> * src1:Tensor<'T> -> unit

    abstract Add:           trgt:Tensor<'T> * src1:Tensor<'T> * src2:Tensor<'T> -> unit
    abstract Subtract:      trgt:Tensor<'T> * src1:Tensor<'T> * src2:Tensor<'T> -> unit
    abstract Multiply:      trgt:Tensor<'T> * src1:Tensor<'T> * src2:Tensor<'T> -> unit
    abstract Divide:        trgt:Tensor<'T> * src1:Tensor<'T> * src2:Tensor<'T> -> unit
    abstract Modulo:        trgt:Tensor<'T> * src1:Tensor<'T> * src2:Tensor<'T> -> unit
    abstract Power:         trgt:Tensor<'T> * src1:Tensor<'T> * src2:Tensor<'T> -> unit


and ITensorStorageFactory =
    abstract Create:        nElems:int64 -> ITensorStorage<'T>
    abstract Zeroed:        bool


/// An N-dimensional array with elements of type 'T.
and [<StructuredFormatDisplay("{Pretty}")>] Tensor<'T> 
        (layout: TensorLayout, storage: ITensorStorage<'T>) =

    do TensorLayout.check layout
    let backend = storage.Backend layout

    /// value zero of type 'T
    static member Zero = conv<'T> 0

    /// value one of type 'T
    static member One = conv<'T> 1

    /// layout of this tensor (shape, offset and strides)
    member val Layout = layout

    /// layout
    static member inline layout (a: #ITensor) = a.Layout

    /// storage of this tensor
    member val Storage = storage

    /// backend
    member internal this.Backend = backend

    /// shape
    member inline this.Shape = this.Layout.Shape

    /// shape in elements
    static member inline shape (a: #ITensor) = a.Shape

    /// number of dimensions
    member inline this.NDims = this.Layout.NDims

    /// number of dimensions
    static member inline nDims (a: #ITensor) = a.NDims

    /// number of elements
    member inline this.NElems = this.Layout.NElems

    /// number of elements 
    static member inline nElems (a: #ITensor) = a.NElems

    /// type of data stored in this tensor
    member inline this.DataType = typeof<'T>

    /// type of data stored in the specified tensor
    static member inline dataType (a: #ITensor) = a.DataType

    /// a tensor with the same storage but new layout
    member internal this.Relayout (newLayout: TensorLayout) =
        Tensor<'T> (newLayout, storage)

    /// a tensor with the same storage but new layout
    static member relayout newLayout (a: 'A when 'A :> ITensor) : 'A =
        a.Relayout newLayout :?> 'A

    /// a view of this tensor over the given range 
    member internal this.Range (rng: TensorRng list) =
        this.Relayout (this.Layout |> TensorLayout.view rng)

    /// a view of the specified tensor over the given range 
    static member range (rng: TensorRng list) (a: 'A when 'A :> ITensor) : 'A =
        a |> Tensor<_>.relayout (a |> Tensor<_>.layout |> TensorLayout.view rng)
   
    /// checks that the given axis is valid
    member inline this.CheckAxis ax = this.Layout |> TensorLayout.checkAxis ax

    /// sequence of all indices 
    static member allIdx (a: #ITensor) = a.Layout |> TensorLayout.allIdx

    /// all indices of the given dimension
    static member allIdxOfDim dim (a: #ITensor) = a.Layout |> TensorLayout.allIdxOfDim dim 
            
    /// sequence of all elements stored in the tensor
    static member allElems (a: Tensor<'T>) = a |> Tensor<_>.allIdx |> Seq.map (fun idx -> a.[idx])

    ///// true if the ArrayND is contiguous
    //let inline isC a = layout a |> TensorLayout.isC

    ///// true if the ArrayND is in Fortran order
    //let inline isF a = layout a |> TensorLayout.isF

    ///// true if the memory of the ArrayND is a contiguous block
    //let inline hasContiguousMemory a = layout a |> TensorLayout.hasContiguousMemory

    /// checks that two ArrayNDs have the same shape
    static member inline internal CheckSameShape (a: #ITensor) (b: #ITensor) =
        if a.Shape <> b.Shape then
            raise (ShapeMismatch (sprintf "Tensors of shapes %A and %A were expected 
                                           to have same shape" a.Shape b.Shape))
      
    /// inserts a broadcastable dimension of size one as first dimension
    static member padLeft a =
        a |> Tensor<_>.relayout (a.Layout |> TensorLayout.padLeft)

    /// appends a broadcastable dimension of size one as last dimension
    static member padRight a =
        a |> Tensor<_>.relayout (a.Layout |> TensorLayout.padRight)

    /// Inserts an axis of size 1 before the specified position.
    static member insertAxis ax a =
        a |> Tensor<_>.relayout (a.Layout |> TensorLayout.insertAxis ax)

    /// removes the first dimension from the tensor
    static member cutLeft a =
        a |> Tensor<_>.relayout (a.Layout |> TensorLayout.cutLeft)
      
    /// removes the last dimension from the tensor
    static member cutRight a =
        a |> Tensor<_>.relayout (a.Layout |> TensorLayout.cutRight)

    /// broadcast the given dimension to the given size
    static member broadcastDim dim size a =
        a |> Tensor<_>.relayout (a.Layout |> TensorLayout.broadcastDim dim size)       

    /// Creates a new tensor of specifed shape with newly allocated storage using 
    /// the specified storage device.
    new (shape: int64 list, dev: ITensorStorageFactory, ?order: TensorOrder) =
        let order = defaultArg order RowMajor
        let layout = 
            match order with
            | RowMajor -> TensorLayout.newC shape
            | ColumnMajor -> TensorLayout.newF shape
        let storage = dev.Create layout.NElems
        Tensor<'T> (layout, storage)

    /// Applies the given function to the tensors' layouts.
    static member inline internal ApplyLayoutFn (fn, a, b) =
        let layouts = [Tensor<_>.layout a; Tensor<_>.layout b]
        let newLayouts = fn layouts
        match newLayouts with
        | [al; bl] -> 
            Tensor<_>.relayout al a, Tensor<_>.relayout bl b
        | _ -> failwith "unexpected layout function result"

    /// Applies the given function to the tensors' layouts.
    static member inline internal ApplyLayoutFn (fn, a, b, c) =
        let layouts = [Tensor<_>.layout a; Tensor<_>.layout b; Tensor<_>.layout c]
        let newLayouts = fn layouts
        match newLayouts with
        | [al; bl; cl] -> 
            Tensor<_>.relayout al a, Tensor<_>.relayout bl b, Tensor<_>.relayout cl c
        | _ -> failwith "unexpected layout function result"

    /// Applies the given function to the tensors' layouts.
    static member inline internal ApplyLayoutFn (fn, xs) =
        let layouts = fn (xs |> List.map Tensor<_>.layout)
        (layouts, xs) ||> List.map2 Tensor<_>.relayout

    /// pads the shapes of all tensors from the left until they have same rank
    static member padToSame (a, b) = 
        Tensor<_>.ApplyLayoutFn (TensorLayout.padToSameMany, a, b)

    /// pads the shapes of all tensors from the left until they have same rank
    static member padToSame (a, b, c) = 
        Tensor<_>.ApplyLayoutFn (TensorLayout.padToSameMany, a, b, c)

    /// pads the shapes of all tensors from the left until they have same rank
    static member padToSame (xs) = 
        Tensor<_>.ApplyLayoutFn (TensorLayout.padToSameMany, xs)

    /// broadcasts all tensors to the same shape 
    static member broadcastToSame (a, b) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameMany, a, b)

    /// broadcasts all tensors to the same shape if possible
    static member broadcastToSame (a, b, c) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameMany, a, b, c)

    /// broadcasts all tensors to the same shape if possible
    static member broadcastToSame (xs) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameMany, xs)

    /// broadcasts all tensors to the same sizes in the given dimensions
    static member broadcastToSameInDims (dims, a, b) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameInDimsMany dims, a, b)

    /// broadcasts all tensors to the same sizes in the given dimensions
    static member broadcastToSameInDims (dims, a, b, c) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameInDimsMany dims, a, b, c)

    /// broadcasts all tensors to the same sizes in the given dimensions
    static member broadcastToSameInDims (dims, xs) =
        Tensor<_>.ApplyLayoutFn (TensorLayout.broadcastToSameInDimsMany dims, xs)

    /// broadcasts the tensor to the given shape
    static member broadcastTo shp a =
        a |> Tensor<_>.relayout (a |> Tensor<_>.layout |> TensorLayout.broadcastToShape shp)

    /// returns true if at least one dimension is broadcasted
    static member isBroadcasted a =
        a |> Tensor<_>.layout |> TensorLayout.isBroadcasted 

    /// Tries to reshape the tensor without copying.
    /// For this to succeed, the tensor must have row-major layout.
    /// If this a reshape without copying is impossible, None is returned.
    static member tryReshapedView shp a =
        match a |> Tensor<_>.layout |> TensorLayout.tryReshape shp with
        | Some newLayout -> a |> Tensor<_>.relayout newLayout |> Some
        | None -> None

    /// Tries to reshape the tensor without copying.
    /// For this to succeed, the tensor must have row-major layout.
    /// If this a reshape without copying is impossible, an error is raised.
    static member reshapedView shp a =
        match Tensor<_>.tryReshapedView shp a with
        | Some res -> res
        | None -> 
            let msg =
                sprintf "cannot reshape tensor of shape %A and strides %A without copying"
                    (Tensor<_>.layout a).Shape (Tensor<_>.layout a).Stride
            raise (RowMajorLayoutRequired msg)

    /// Returns true if the tensor can be reshaped without copying.
    static member canReshapeWithoutCopy shp a =
        match Tensor<_>.tryReshapedView shp a with
        | Some _ -> true
        | None -> false

    /// Reshape array assuming a row-major order.
    /// If the array is currently not in row-major order, a reshaped copy is returned.
    /// Otherwise, a reshaped view of the same tensor is returned.
    /// The number of elements must not change.
    /// One element can be -1, in which case the size of that element is
    /// inferred automatically.
    static member reshape shp a =
        match a |> Tensor<_>.tryReshapedView shp with
        | Some res -> res
        | None ->
            a |> Tensor<_>.copy |> Tensor<_>.reshapedView shp

    /// Flattens the tensor into a vector assuming a row-major order.
    static member flatten a =
        Tensor<_>.reshape [Remainder] a

    /// swaps the given dimensions
    static member swapDim ax1 ax2 a =
        a |> Tensor<_>.relayout (a |> Tensor<_>.layout |> TensorLayout.swapDim ax1 ax2)

    /// Transposes the given matrix.
    /// If the tensor has more then two dimensions, the last two axes are swapped.
    static member transpose a =
        a |> Tensor<_>.relayout (a |> Tensor<_>.layout |> TensorLayout.transpose)

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the new position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    static member permuteAxes (permut: int list) a =
        a |> Tensor<_>.relayout (a |> Tensor<_>.layout |> TensorLayout.permuteAxes permut)

    /// Reverses the elements in the specified dimension.
    static member reverseAxis ax a =
        a |> Tensor<_>.relayout (a |> Tensor<_>.layout |> TensorLayout.reverseAxis ax)        

    /// Ensures that the tensor has at least minDims dimensions.
    /// If not, it is padded with size one dimensions from the left.
    static member atLeastND minDims a =
        let nd = Tensor<_>.nDims a
        if nd >= minDims then a
        else
            let newShp = List.init (minDims - nd) (fun _ -> 1L)
            a |> Tensor<_>.reshape newShp

    /// Ensures that the tensor has at least one dimension.
    static member atLeast1D a = a |> Tensor<_>.atLeastND 1

    /// Ensures that the tensor has at least two dimensions.
    /// If not, it is padded with size one dimensions from the left.
    static member atLeast2D a = a |> Tensor<_>.atLeastND 2

    /// Ensures that the tensor has at least three dimensions.
    /// If not, it is padded with size one dimensions from the left.
    static member atLeast3D a = a |> Tensor<_>.atLeastND 3


    /// checks that all tensors have the same storage
    static member internal CheckSameStorage (xs: ITensor list) =
        match xs with
        | x::rs when rs |> List.exists (fun r -> x.Storage.Id <> r.Storage.Id) ->
            let storages = xs |> List.map (fun x -> x.Storage.Id)
            raise (StorageMismatch (sprintf "Storages must be equal for this operation, 
                                             but they are %A." storages))
        | _ -> ()            

    /// prepares an elementwise operation by allocating a target of same size and storage
    static member internal PrepareElemwise (a: Tensor<'TA>, ?order) =
        let trgt = Tensor<_> (a.Shape, a.Storage.Factory, ?order=order)
        trgt, a

    /// prepares an elementwise operation by broadcasting both tensors to the same size
    /// and allocating a target of same size and storage
    static member internal PrepareElemwise (a: Tensor<'TA>, b: Tensor<'TB>, ?order) =
        Tensor<_>.CheckSameStorage [a; b]
        let a, b = Tensor<_>.broadcastToSame (a, b)
        let trgt = Tensor<_> (a.Shape, a.Storage.Factory, ?order=order)
        trgt, a, b

    /// prepares an elementwise operation by broadcasting all three tensors to the same size
    /// and allocating a target of same size and storage
    static member internal PrepareElemwise (a: Tensor<'TA>, b: Tensor<'TB>, c: Tensor<'TC>, ?order) =
        Tensor<_>.CheckSameStorage [a; b; c]
        let a, b, c = Tensor<_>.broadcastToSame (a, b, c)
        let trgt = Tensor<_> (a.Shape, a.Storage.Factory, ?order=order)
        trgt, a, b, c



    interface ITensor with
        member this.Layout = this.Layout
        member this.DataType = this.DataType
        member this.Shape = this.Shape
        member this.NDims = this.NDims
        member this.NElems = this.NElems
        member this.Storage = this.Storage :> ITensorStorage
        member this.CPPType = raise (System.NotImplementedException())
        member this.Copy (?order) = this.Copy (?order=order) :> ITensor
        member this.CopyTo(arg1) = raise (System.NotImplementedException())
        member this.GetSlice(args) = raise (System.NotImplementedException())
        member this.Item
            with get (allArgs: obj []): ITensor = 
                raise (System.NotImplementedException())
            and set (arg1: obj) (v: ITensor): unit = 
                raise (System.NotImplementedException())
        member this.Item
            with set (arg1: obj, arg2: obj) (v: ITensor): unit = 
                raise (System.NotImplementedException())
        member this.Item
            with set (arg1: obj, arg2: obj, arg3: obj) (v: ITensor): unit = 
                raise (System.NotImplementedException())
        member this.Item
            with set (arg1: obj, arg2: obj, arg3: obj, arg4: obj) (v: ITensor): unit = 
                raise (System.NotImplementedException())
        member this.Item
            with set (arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj) (v: ITensor): unit = 
                raise (System.NotImplementedException())
        member this.Item
            with set (arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj, arg6: obj) (v: ITensor): unit = 
                raise (System.NotImplementedException())
        member this.Item
            with set (arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj, arg6: obj, arg7: obj) (v: ITensor): unit = 
                raise (System.NotImplementedException())
        member this.NewOfSameType(arg1) = raise (System.NotImplementedException())
        member this.NewOfType arg1 arg2 = raise (System.NotImplementedException())
        member this.Relayout layout = this.Relayout layout :> ITensor
        member this.SetSlice(args) = raise (System.NotImplementedException())


    static member internal ApplyElemwise (fn, a: Tensor<'TA>, ?order) : Tensor<'R> =
        let trgt, a = Tensor<_>.PrepareElemwise (a, ?order=order)
        fn trgt a
        trgt       

    static member internal ApplyElemwise (fn, a: Tensor<'TA>, b: Tensor<'TB>, ?order) : Tensor<'R> =
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b, ?order=order)
        fn trgt a b
        trgt
       
    /// element-wise unary (prefix) plus
    static member (~+) (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.UnaryPlus (trgt=trgt, src1=a)
        trgt

    /// element-wise unary (prefix) minus
    static member (~-) (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.UnaryMinus (trgt=trgt, src1=a)
        trgt

    /// element-wise absolute value
    static member Abs (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Abs (trgt=trgt, src1=a)
        trgt

    /// element-wise sign (keeping type)
    static member Sgn (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Sgn (trgt=trgt, src1=a)
        trgt

    /// element-wise logarithm to base e
    static member Log (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Log (trgt=trgt, src1=a)
        trgt

    /// element-wise logarithm to base 10
    static member Log10 (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Log10 (trgt=trgt, src1=a)
        trgt

    /// element-wise exponential function
    static member Exp (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Exp (trgt=trgt, src1=a)
        trgt

    /// element-wise sinus function
    static member Sin (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Sin (trgt=trgt, src1=a)
        trgt

    /// element-wise cosinus function
    static member Cos (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Cos (trgt=trgt, src1=a)
        trgt

    /// element-wise tangens function
    static member Tan (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Tan (trgt=trgt, src1=a)
        trgt

    /// element-wise arcus sinus function
    static member Asin (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Asin (trgt=trgt, src1=a)
        trgt

    /// element-wise arcus cosinus function
    static member Acos (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Acos (trgt=trgt, src1=a)
        trgt

    /// element-wise arcus tangens function
    static member Atan (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Atan (trgt=trgt, src1=a)
        trgt

    /// element-wise sinus hyperbolicus function
    static member Sinh (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Sinh (trgt=trgt, src1=a)
        trgt

    /// element-wise cosinus hyperbolicus function
    static member Cosh (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Cosh (trgt=trgt, src1=a)
        trgt

    /// element-wise tangens hyperbolicus function
    static member Tanh (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Tanh (trgt=trgt, src1=a)
        trgt

    /// element-wise square root 
    static member Sqrt (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Sqrt (trgt=trgt, src1=a)
        trgt

    /// element-wise ceiling
    static member Ceiling (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Ceiling(trgt=trgt, src1=a)
        trgt

    /// element-wise ceiling
    static member Floor (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Floor(trgt=trgt, src1=a)
        trgt

    /// element-wise rounding
    static member Round (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Round(trgt=trgt, src1=a)
        trgt

    /// element-wise truncate
    static member Truncate (a: Tensor<'T>) = 
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Truncate(trgt=trgt, src1=a)
        trgt

    // element-wise unary logic
    //static member (~~~~)    (a: #Tensor<bool>) = map not a

    /// element-wise addition of two tensors
    static member (+) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
        trgt.Backend.Add (trgt=trgt, src1=a, src2=b)
        trgt

    /// element-wise subtraction of two tensors
    static member (-) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
        trgt.Backend.Subtract (trgt=trgt, src1=a, src2=b)
        trgt

    /// element-wise multiplication of two tensor
    static member (*) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
        trgt.Backend.Multiply (trgt=trgt, src1=a, src2=b)
        trgt

    /// element-wise division of two tensors
    static member (/) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
        trgt.Backend.Divide (trgt=trgt, src1=a, src2=b)
        trgt

    /// element-wise modulo of two tensors
    static member (%) (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
        trgt.Backend.Modulo (trgt=trgt, src1=a, src2=b)
        trgt

    /// element-wise power of two tensor
    static member Pow (a: Tensor<'T>, b: Tensor<'T>) = 
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
        trgt.Backend.Power (trgt=trgt, src1=a, src2=b)
        trgt


    //// element-wise binary logic
    //static member (&&&&) (a: #Tensor<bool>, b: #Tensor<bool>) = map2 (&&) a b
    //static member (||||) (a: #Tensor<bool>, b: #Tensor<bool>) = map2 (||) a b

    //// element-wise binary comparison
    //static member (====) (a: #Tensor<'T>, b: #Tensor<'T>) = typedMap2TypeChange (=) (=) (=) (=) (=) (=) a b
    //static member (<<<<) (a: #Tensor<'T>, b: #Tensor<'T>) = typedMap2TypeChange (<) (<) (<) (<) (<) (<) a b
    //static member (<<==) (a: #Tensor<'T>, b: #Tensor<'T>) = typedMap2TypeChange (<=) (<=) (<=) (<=) (<=) (<=) a b
    //static member (>>>>) (a: #Tensor<'T>, b: #Tensor<'T>) = typedMap2TypeChange (>) (>) (>) (>) (>) (>) a b
    //static member (>>==) (a: #Tensor<'T>, b: #Tensor<'T>) = typedMap2TypeChange (>=) (>=) (>=) (>=) (>=) (>=) a b
    //static member (<<>>) (a: #Tensor<'T>, b: #Tensor<'T>) = typedMap2TypeChange (<>) (<>) (<>) (<>) (<>) (<>) a b

    //// element-wise binary with scalars
    //static member inline (+) (a: #Tensor<'T>, b: 'T) = a + (scalarOfSameType a b)
    //static member inline (-) (a: #Tensor<'T>, b: 'T) = a - (scalarOfSameType a b)
    //static member inline (*) (a: #Tensor<'T>, b: 'T) = a * (scalarOfSameType a b)
    //static member inline (/) (a: #Tensor<'T>, b: 'T) = a / (scalarOfSameType a b)
    //static member inline (%) (a: #Tensor<'T>, b: 'T) = a % (scalarOfSameType a b)
    //static member inline Pow (a: #Tensor<'T>, b: 'T) = a ** (scalarOfSameType a b)        
    //static member inline (&&&&) (a: #Tensor<bool>, b: bool) = a &&&& (scalarOfSameType a b)
    //static member inline (||||) (a: #Tensor<bool>, b: bool) = a |||| (scalarOfSameType a b)
    //static member (====) (a: #Tensor<'T>, b: 'T) = typedMap2TypeChange (=) (=) (=) (=) (=) (=) a (scalarOfSameType a b)   
    //static member (<<<<) (a: #Tensor<'T>, b: 'T) = typedMap2TypeChange (<) (<) (<) (<) (<) (<) a (scalarOfSameType a b)   
    //static member (<<==) (a: #Tensor<'T>, b: 'T) = typedMap2TypeChange (<=) (<=) (<=) (<=) (<=) (<=)  a (scalarOfSameType a b)    
    //static member (>>>>) (a: #Tensor<'T>, b: 'T) = typedMap2TypeChange (>) (>) (>) (>) (>) (>) a (scalarOfSameType a b)   
    //static member (>>==) (a: #Tensor<'T>, b: 'T) = typedMap2TypeChange (>=) (>=) (>=) (>=) (>=) (>=) a (scalarOfSameType a b)   
    //static member (<<>>) (a: #Tensor<'T>, b: 'T) = typedMap2TypeChange (<>) (<>) (<>) (<>) (<>) (<>) a (scalarOfSameType a b)   

    //static member inline (+) (a: 'T, b: #Tensor<'T>) = (scalarOfSameType b a) + b
    //static member inline (-) (a: 'T, b: #Tensor<'T>) = (scalarOfSameType b a) - b
    //static member inline (*) (a: 'T, b: #Tensor<'T>) = (scalarOfSameType b a) * b
    //static member inline (/) (a: 'T, b: #Tensor<'T>) = (scalarOfSameType b a) / b
    //static member inline (%) (a: 'T, b: #Tensor<'T>) = (scalarOfSameType b a) % b
    //static member inline Pow (a: 'T, b: #Tensor<'T>) = (scalarOfSameType b a) ** b
    //static member inline (&&&&) (a: bool, b: #Tensor<bool>) = (scalarOfSameType b a) &&&& b
    //static member inline (||||) (a: bool, b: #Tensor<bool>) = (scalarOfSameType b a) |||| b
    //static member (====) (a: 'T, b: #Tensor<'T>) = typedMap2TypeChange (=) (=) (=) (=) (=) (=) (scalarOfSameType b a) b
    //static member (<<<<) (a: 'T, b: #Tensor<'T>) = typedMap2TypeChange (<) (<) (<) (<) (<) (<) (scalarOfSameType b a) b
    //static member (<<==) (a: 'T, b: #Tensor<'T>) = typedMap2TypeChange (<=) (<=) (<=) (<=) (<=) (<=) (scalarOfSameType b a) b
    //static member (>>>>) (a: 'T, b: #Tensor<'T>) = typedMap2TypeChange (>) (>) (>) (>) (>) (>) (scalarOfSameType b a) b
    //static member (>>==) (a: 'T, b: #Tensor<'T>) = typedMap2TypeChange (>=) (>=) (>=) (>=) (>=) (>=) (scalarOfSameType b a) b
    //static member (<<>>) (a: 'T, b: #Tensor<'T>) = typedMap2TypeChange (<>) (<>) (<>) (<>) (<>) (<>) (scalarOfSameType b a) b

    ///// dot product
    //static member (.*) (a: #Tensor<'T>, b: #Tensor<'T>) = typedApply2 (unsp) dotImpl dotImpl dotImpl dotImpl dotImpl a b

    ///// tensor product
    //static member (%*) (a: #Tensor<'T>, b: #Tensor<'T>) = typedApply2 (unsp) tensorProductImpl tensorProductImpl tensorProductImpl tensorProductImpl tensorProductImpl a b
        

    //// transposition
    //member this.T = transpose this


    /// returns a copy of the tensor
    member this.Copy (?order) =
        let trgt, src = Tensor<_>.PrepareElemwise (this)
        trgt.Backend.Copy (trgt=trgt, src=src)
        trgt      
        
    /// returns a copy of the tensor
    static member copy (a: 'A when 'A :> ITensor, ?order) =
        a.Copy (?order=order) :?> 'A

    /// Copies the specifed tensor into this tensor.
    /// Both tensors must have same shape and storage.
    member internal this.CopyFrom (src: Tensor<'T>) =
        Tensor<_>.CheckSameShape this src
        Tensor<_>.CheckSameStorage [this; src]
        this.Backend.Copy (trgt=this, src=src)

    /// maps all elements using the specified function into a new tensor
    static member map (fn: 'T -> 'R) (a: Tensor<'T>) =
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Map (fn=fn, trgt=trgt, src=a, useThreads=false)
        trgt       

    /// maps all elements using the specified indexed function into a new tensor
    static member mapi (fn: int64[] -> 'T -> 'R) (a: Tensor<'T>) =
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.MapIndexed (fn=fn, trgt=trgt, src=a, useThreads=false)
        trgt     

    /// maps all elements using the specified function into a new tensor
    static member map2 (fn: 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
        trgt.Backend.Map2 (fn=fn, trgt=trgt, src1=a, src2=b, useThreads=false)
        trgt       

    /// maps all elements using the specified indexed function into a new tensor
    static member mapi2 (fn: int64[] -> 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
        let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
        trgt.Backend.MapIndexed2 (fn=fn, trgt=trgt, src1=a, src2=b, useThreads=false)
        trgt       

    /// converts all elements to the specified type
    static member convert<'C> (a: Tensor<'T>) : Tensor<'C> =
        let trgt, a = Tensor<_>.PrepareElemwise (a)
        trgt.Backend.Convert (trgt=trgt, src=a)
        trgt    

    /// a view of this tensor with the given .NET range
    member inline internal this.GetRng (rngArgs: obj[]) =
        this.Range (TensorRng.ofItemOrSliceArgs rngArgs) 
    member inline internal this.GetRngWithRest (rngArgs: obj[]) (restArgs: obj[]) =
        Array.concat [rngArgs; restArgs] |> this.GetRng

    /// write into the view of this tensor with the given .NET range
    member inline internal this.SetRng (rngArgs: obj[]) (value: Tensor<'T>) =
        Tensor<_>.CheckSameStorage [this; value]
        let trgt = this.Range (TensorRng.ofItemOrSliceArgs rngArgs) 
        value |> Tensor<_>.broadcastTo trgt.Shape |> trgt.CopyFrom
    member inline internal this.SetRngWithRest (rngArgs: obj[]) (restArgs: obj[]) =
        let allArgs = Array.concat [rngArgs; restArgs]
        let value = Array.last allArgs :?> Tensor<'T>
        let args = allArgs.[0 .. allArgs.Length-2]
        this.SetRng args value

    /// access to a single item using an array of indices
    member this.Item
        with get (idx: int64[]) : 'T = backend.[idx]
        and set (idx: int64[]) (value: 'T) = backend.[idx] <- value

    /// access to a single item using a list of indices
    member this.Item
        with get (idx: int64 list) : 'T = backend.[Array.ofList idx]
        and set (idx: int64 list) (value: 'T) = backend.[Array.ofList idx] <- value

    /// n-dimensional slicing using a list of TensorRngs
    member this.Item
        with get (rng: TensorRng list) = this.GetRng [|rng|]
        and set (rng: TensorRng list) (value: Tensor<'T>) = this.SetRng [|rng|] value

    /// one-dimensional slicing using indices and special axes
    member this.Item
        with get (i0: int64) = this.GetRng [|i0|]
        and set (i0: int64) (value: Tensor<'T>) = this.SetRng [|i0|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option) = this.GetRng [|i0s; i0f|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, value: Tensor<'T>) = this.SetRng [|i0s; i0f|] value

    /// two-dimensional slicing using indices and special axes
    member this.Item
        with get (i0: int64, i1: int64) = this.GetRng [|i0; i1|]
        and set (i0: int64, i1: int64) (value: Tensor<'T>) = this.SetRng [|i0; i1|] value
    member this.GetSlice (i0: int64, i1s: int64 option, i1f: int64 option) = this.GetRng [|i0; i1s; i1f|]
    member this.SetSlice (i0: int64, i1s: int64 option, i1f: int64 option, value: Tensor<'T>) = this.SetRng [|i0; i1s; i1f|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1: int64) = this.GetRng [|i0s; i0f; i1|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1: int64, value: Tensor<'T>) = this.SetRng [|i0s; i0f; i1|] value
    member this.GetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option) = this.GetRng [|i0s; i0f; i1s; i1f|]
    member this.SetSlice (i0s: int64 option, i0f: int64 option, i1s: int64 option, i1f: int64 option, value: Tensor<'T>) = this.SetRng [|i0s; i0f; i1s; i1f|] value

    /// three-dimensional slicing using indices and special axes
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

    /// four- and more-dimensional slicing using indices and special axes
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

    /// get element value
    static member inline get (a: Tensor<_>) (pos: int64 list) = 
        a.[pos]
    
    /// set element value
    static member inline set (a: Tensor<_>) (pos: int64 list) value = 
        a.[pos] <- value

    /// value of scalar array
    member this.Value 
        with inline get () = this.[[||]]
        and inline set value = this.[[||]] <- value

    /// value of scalar array
    static member inline value (a: Tensor<_>) =
        a.Value

    /// Pretty string containing maxElems elements per dimension.
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
                let v = a.Value
                if   typeof<'T>.Equals(typeof<single>) then sprintf "%9.4f" (v |> box :?> single)
                elif typeof<'T>.Equals(typeof<double>) then sprintf "%9.4f" (v |> box :?> double)
                elif typeof<'T>.Equals(typeof<int>)    then sprintf "%4d"  (v |> box :?> int)
                elif typeof<'T>.Equals(typeof<int64>)  then sprintf "%4d"  (v |> box :?> int64)
                elif typeof<'T>.Equals(typeof<byte>)   then sprintf "%3d"  (v |> box :?> byte)
                elif typeof<'T>.Equals(typeof<bool>)   then if (v |> box :?> bool) then "true " else "false"
                else sprintf "%A;" v
            | 1 -> "[" + (String.concat " " (subStrs ())) + "]"
            | _ -> "[" + (String.concat ("\n" + lineSpace) (subStrs ())) + "]"
        prettyDim " " this                       

    /// pretty contents string
    member this.Pretty = this.ToString (maxElems=10L)
    override this.ToString() = this.Pretty

    /// full contents string
    member this.Full = this.ToString (maxElems=Int64.MaxValue)
   
    /// Fills the tensor with the specified constant.
    static member fillConst (value: 'T) (trgt: Tensor<'T>) =
        trgt.Backend.FillConst (value=value, trgt=trgt)

    /// Fills the tensor with the values returned by the function.
    static member fill (fn: unit -> 'T) (trgt: Tensor<'T>) =
        trgt.Backend.Fill (fn=fn, trgt=trgt, useThreads=false)

    /// Fills the tensor with the values returned by the function.
    static member fillIndexed (fn: int64[] -> 'T) (trgt: Tensor<'T>) =
        trgt.Backend.FillIndexed (fn=fn, trgt=trgt, useThreads=false)

    /// Fills the vector with linearly spaced values from start to (including) stop.
    static member inline fillLinSpaced (start: 'V) (stop: 'V) (a: Tensor<'V>) =
        if a.NDims <> 1 then raise (ShapeMismatch "tensor must be one dimensional")
        if a.NElems < 2L then raise (ShapeMismatch "tensor must have at least two elements")
        let step = (stop - start) / conv<'V> (a.NElems - 1L)
        a |> Tensor<_>.fillIndexed (fun idx -> start + conv<'V> idx.[0] * step)       

    /// Fills the tensor with the values returned by the given sequence.
    static member fillWithSeq (data: 'T seq) (trgt: Tensor<'T>) =
        use enumerator = data.GetEnumerator()
        trgt |> Tensor<_>.fill (fun () -> 
            if enumerator.MoveNext() then enumerator.Current
            else 
                let msg = 
                    sprintf "sequence ended before tensor of shape %A was filled"
                            trgt.Shape
                raise (SeqTooShort msg))

    /// Creates a new tensor of given shape filled with zeros.
    static member zeros<'T> (shape: int64 list, dev: ITensorStorageFactory) =
        let x = Tensor<'T> (shape, dev)
        if not dev.Zeroed then 
            x |> Tensor<_>.fillConst Tensor<'T>.Zero
        x
   
    /// Tensor of same shape as specifed tensor and filled with zeros.
    static member zerosLike (tmpl: Tensor<'T>) =
        Tensor<_>.zeros<'T> (tmpl.Shape, tmpl.Storage.Factory)

    /// Creates a new tensor of given shape filled with ones.
    static member ones<'T> (shape: int64 list, dev: ITensorStorageFactory) =
        let x = Tensor<'T> (shape, dev)
        x |> Tensor<_>.fillConst Tensor<'T>.One
        x
        
    /// Tensor of same shape as specifed tensor and filled with ones.
    static member onesLike (tmpl: Tensor<'T>) =
        Tensor<_>.ones<'T> (tmpl.Shape, tmpl.Storage.Factory)


           


#if false

    /// C++ type name
    member this.CPPType = 
        let dims = TensorLayout.nDims layout
        let shp = TensorLayout.shape layout
        let str = TensorLayout.stride layout
        let ofst = TensorLayout.offset layout
        let cppDataType = Util.cppType this.DataType
        let shapeStr = 
            if dims = 0 then "" 
            else "<" + (shp |> Seq.map (sprintf "%dLL") |> String.concat ",") + ">"
        let strideStr = 
            "<" + ((ofst :: str) |> Seq.map (sprintf "%dLL") |> String.concat ",") + ">"
        sprintf "ArrayND%dD<%s, ShapeStatic%dD%s, StrideStatic%dD%s>" 
            dims cppDataType dims shapeStr dims strideStr            




    abstract IfThenElseImpl: Tensor<bool> -> Tensor<'T> -> Tensor<'T> -> unit
    default this.IfThenElseImpl cond elseVal result =
        for idx in TensorLayout.allIdx this.Layout do
            result.[idx] <- if cond.[idx] then this.[idx] else elseVal.[idx]

    /// elementwise uses elements from this if cond is true, 
    /// otherwise elements from elseVal
    member this.IfThenElse (cond: #Tensor<bool>) (elseVal: #Tensor<'T>) =
        if elseVal.GetType() <> this.GetType() then
            failwithf "cannot use IfThenElse on ArrayNDTs of different types: %A and %A"
                (this.GetType()) (elseVal.GetType())
        if cond.GetType().GetGenericTypeDefinition() <> this.GetType().GetGenericTypeDefinition() then
            failwithf "cannot use IfThenElse on ArrayNDTs of different types: %A and %A"
                (this.GetType()) (cond.GetType())
        let ifVal, elseVal, cond = this.BroadcastToSame3 elseVal cond
        let res = this.NewOfSameType (TensorLayout.newC ifVal.Shape)
        ifVal.IfThenElseImpl cond elseVal res
        res

    abstract GatherImpl: #Tensor<int64> option list -> Tensor<'T> -> unit
    default trgt.GatherImpl indices src =
        for trgtIdx in TensorLayout.allIdx trgt.Layout do
            let srcIdx = 
                indices 
                |> List.mapi (fun dim idx ->
                    match idx with
                    | Some di -> di.[trgtIdx]
                    | None -> trgtIdx.[dim])
            trgt.[trgtIdx] <- src.[srcIdx]    
                       
    /// Sets the values of this array by selecting from the sources array according to the specified
    /// indices. If an index array is set to None then the target index is used as the source index.
    member trgt.Gather (indices: #Tensor<int64> option list) (src: #Tensor<'T>) =
        if src.GetType() <> trgt.GetType() then
            failwithf "cannot use IndexedSet on ArrayNDTs of different types: %A and %A"
                (trgt.GetType()) (src.GetType())
        match indices |> List.tryPick id with
        | Some ih ->
            if ih.GetType().GetGenericTypeDefinition() <> trgt.GetType().GetGenericTypeDefinition() then
                failwithf "cannot use IndexedSet on ArrayNDTs of different types: %A and %A"
                    (trgt.GetType()) (indices.GetType())
        | None -> ()
        if src.NDims <> indices.Length then
            failwithf "must specify an index array for each dimension of src"
        if indices |> List.skip trgt.NDims |> List.exists Option.isNone then
            failwithf "index dimensions beyond the number of target dimensions must not be None"
        let indices = indices |> List.map (Option.map (fun idx -> idx.BroadcastToShape trgt.Shape))
        trgt.GatherImpl indices src

    abstract ScatterImpl: #Tensor<int64> option list -> Tensor<'T> -> unit
    default trgt.ScatterImpl indices src = 
        let addInt a b = (a |> box |> unbox<int>) + (b |> box |> unbox<int>) |> box |> unbox<'T>
        let addInt64 a b = (a |> box |> unbox<int64>) + (b |> box |> unbox<int64>) |> box |> unbox<'T>
        let addSingle a b = (a |> box |> unbox<single>) + (b |> box |> unbox<single>) |> box |> unbox<'T>
        let addDouble a b = (a |> box |> unbox<double>) + (b |> box |> unbox<double>) |> box |> unbox<'T>
        let addBool a b = ((a |> box |> unbox<bool>) || (b |> box |> unbox<bool>)) |> box |> unbox<'T>
        let add =
            match typeof<'T> with
            | t when t=typeof<int> -> addInt
            | t when t=typeof<int64> -> addInt64
            | t when t=typeof<single> -> addSingle
            | t when t=typeof<double> -> addDouble
            | t when t=typeof<bool> -> addBool
            | t -> failwithf "unsupported type: %A" t
        for srcIdx in TensorLayout.allIdx src.Layout do
            let trgtIdx =
                indices
                |> List.mapi (fun dim idx ->
                    match idx with
                    | Some di -> di.[srcIdx]
                    | None -> srcIdx.[dim])
            trgt.[trgtIdx] <- add trgt.[trgtIdx] src.[srcIdx]

    /// Sets the values of this array by summing elements from the sources array into the elements
    /// of this array specified by the indices.
    /// If an index array is set to None then the target index is used as the source index.
    member trgt.Scatter (indices: #Tensor<int64> option list) (src: #Tensor<'T>) =
        if src.GetType() <> trgt.GetType() then
            failwithf "cannot use IndexedSum on ArrayNDTs of different types: %A and %A"
                (trgt.GetType()) (src.GetType())
        match indices |> List.tryPick id with
        | Some ih ->
            if ih.GetType().GetGenericTypeDefinition() <> trgt.GetType().GetGenericTypeDefinition() then
                failwithf "cannot use IndexedSum on ArrayNDTs of different types: %A and %A"
                    (trgt.GetType()) (indices.GetType())
            if ih.Shape <> src.Shape then
                failwithf "index arrays have shapes %A that do not match source shape %A"
                    (indices |> List.map (Option.map (fun a -> a.Shape))) src.Shape
        | None -> ()
        if trgt.NDims <> indices.Length then
            failwithf "must specify an index array for each dimension of the target"
        if indices |> List.skip src.NDims |> List.exists Option.isNone then
            failwithf "index dimensions beyond the number of source dimensions must not be None"
        let indices = indices |> List.map (Option.map (fun idx -> idx.BroadcastToShape src.Shape))
        trgt.ScatterImpl indices src

    /// invert the matrix
    abstract Invert : unit -> Tensor<'T>

    /// Computes the (real) eigenvalues and eigenvectors of the symmetric matrix.
    /// Returns (vals, vecs) where each column of 'vecs' is the eigenvector for the
    /// corresponding eigenvalue in 'vals'.
    abstract SymmetricEigenDecomposition: unit -> Tensor<'T> * Tensor<'T>

    // enumerator interfaces
    interface IEnumerable<'T> with
        member this.GetEnumerator() =
            TensorLayout.allIdx this.Layout
            |> Seq.map (fun idx -> this.[idx])
            |> fun s -> s.GetEnumerator()
        member this.GetEnumerator() =
            (this :> IEnumerable<'T>).GetEnumerator() :> IEnumerator

#endif

#if false
    interface ITensor with
        member this.Layout = this.Layout
        member this.CPPType = this.CPPType   
        member this.Shape = this.Shape
        member this.NDims = this.NDims
        member this.NElems = this.NElems      
        member this.NewView layout = this.NewView layout :> ITensor    
        member this.NewOfSameType layout = this.NewOfSameType layout :> ITensor
        member this.NewOfType layout typ = 
            let gm = this.GetType().GetMethod("NewOfType")
            let m = gm.MakeGenericMethod [|typ|]
            m.Invoke(this, [|box layout|]) :?> ITensor
        member this.DataType = this.DataType
        member this.Location = this.Location
        member this.Copy () = 
            let shp = TensorLayout.shape this.Layout
            let trgt = this.NewOfSameType (TensorLayout.newC shp)
            this.CopyTo trgt
            trgt :> ITensor
        member this.CopyTo dest = 
            match dest with
            | :? Tensor<'T> as dest -> this.CopyTo dest
            | _ -> failwith "destination must be of same type as source"
        member this.GetSlice ([<System.ParamArray>] allArgs: obj []) =
            this.GetSlice (allArgs) :> ITensor
        member this.SetSlice ([<System.ParamArray>] allArgs: obj []) =
            this.SetSlice (allArgs)
        member this.Item
            with get ([<System.ParamArray>] allArgs: obj []) = this.GetSlice (allArgs) :> ITensor
            and set (arg0: obj) (value: ITensor) = 
                this.SetSlice ([|arg0; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj) (value: ITensor) = 
                this.SetSlice ([|arg0; arg1; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj) (value: ITensor) = 
                this.SetSlice ([|arg0; arg1; arg2; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj) (value: ITensor) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj) (value: ITensor) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj) (value: ITensor) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; value :> obj|])
        member this.Item
            with set (arg0: obj, arg1: obj, arg2: obj, arg3: obj, arg4: obj, arg5: obj, arg6: obj) (value: ITensor) = 
                this.SetSlice ([|arg0; arg1; arg2; arg3; arg4; arg5; arg6; value :> obj|])
#endif            

#if false

module Tensor = 

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // array creation functions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// creates a scalar ArrayND of given value and type
    let scalarOfSameType (a: 'B when 'B :> Tensor<'T>) (value: 'T) : 'B =
        let ary = newCOfSameType [] a
        set [] value ary
        ary

    /// creates a scalar ArrayND of given value 
    let scalarOfType a (value: 'T) =
        let ary = newCOfType [] a
        set [] value ary
        ary


    /// fills the diagonal of a quadratic matrix with ones
    let inline fillDiagonalWithOnes (a: #Tensor<'T>) =
        match shape a with
        | [n; m] when n = m ->
            for i in 0L .. n-1L do
                set [i; i] Tensor<'T>.One a
        | _ -> invalidArg "a" "need a quadratic matrix"

    /// Creates a new ArrayNDT by selecting elements from `src` according to the specified `indices`.
    /// `indices` must be a list of ArrayNDTs, one per dimension of `src`. 
    /// If None is specified instead of an array in an dimension, the source index will match the 
    /// target index in that dimension.
    /// The result will have the shape of the (broadcasted) index arrays.
    let gather indices (src: #Tensor<'T>) =
        let someIndices = indices |> List.choose id
        if List.isEmpty someIndices then
            failwith "need to specify at least one index array"
        let bcSomeIndices = broadcastToSameMany someIndices
        let rec rebuild idxs repIdxs =
            match idxs, repIdxs with
            | Some idx :: rIdxs, repIdx :: rRepIdxs ->
                Some repIdx :: rebuild rIdxs rRepIdxs
            | None :: rIdxs, _ -> None :: rebuild rIdxs repIdxs
            | [], [] -> []
            | _ -> failwith "unbalanced idxs"
        let bcIndices = rebuild indices bcSomeIndices
        let trgtShp = bcSomeIndices.Head.Shape
        let trgt = newCOfSameType trgtShp src
        trgt.Gather bcIndices src
        trgt

    /// Creates a new ArrayNDT of shape `trgtShp` by dispersing elements from `src` according to 
    /// the specified target `indices`. If an index occurs multiple times the corresponding values are summed.
    /// Target elements that do not occur, are set to zero.
    /// `indices` must be a list of ArrayNDTs, one per dimension of `trgt` and of the same shape
    /// (or broadcastable to) as `src`.
    /// If None is specified instead of an array in an dimension, the source index will match the 
    /// target index in that dimension.
    let scatter indices trgtShp (src: #Tensor<'T>) =
        let bcIndices = indices |> List.map (Option.map (broadcastToShape src.Shape))
        let trgt = newCOfSameType trgtShp src
        trgt.Scatter bcIndices src
        trgt


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // element-wise operations
    ////////////////////////////////////////////////////////////////////////////////////////////////   
   
    let inline signImpl (x: 'T) =
        conv<'T> (sign x)

    /// sign keeping type
    let inline signt (a: #Tensor<'T>) =
        Tensor<'T>.SignT a 

    /// Elementwise check if two arrays have same (within machine precision) values.
    /// Check for exact equality when type is int or bool.
    let inline isCloseWithTol (aTol: 'T) (rTol: 'T) (a: Tensor<'T>) (b: Tensor<'T>) =
        match typeof<'T> with
        | t when t = typeof<bool> -> (box a :?> Tensor<bool>) ==== (box b :?> Tensor<bool>) 
        | t when t = typeof<int>  -> (box a :?> Tensor<int>)  ==== (box b :?> Tensor<int>) 
        | _ ->  abs (a - b) <<== aTol + rTol * abs b

    /// Elementwise check if two arrays have same (within machine precision) values.
    let inline isClose (a: Tensor<'T>) (b: Tensor<'T>) =
        isCloseWithTol (conv<'T> 1e-8) (conv<'T> 1e-5) a b

    /// Elementwise check if a value is finite (not NaN and not infinite).
    let inline isFinite (a: Tensor<'T>) =
        let isFiniteSingle v = not (System.Single.IsInfinity v || System.Single.IsNaN v)
        let isFiniteDouble v = not (System.Double.IsInfinity v || System.Double.IsNaN v)
        typedMapTypeChange (unsp) isFiniteDouble isFiniteSingle (unsp) (unsp) (unsp) a

    /// Elementwise picks the maximum of a or b.
    let inline maxElemwise (a: #Tensor<'T>) (b: #Tensor<'T>) =
        typedMap2 (max) (max) (max) (max) (max) (max) a b

    /// Elementwise picks the minimum of a or b.
    let inline minElemwise (a: #Tensor<'T>) (b: #Tensor<'T>) =
        typedMap2 (min) (min) (min) (min) (min) (min) a b

    /// Elementwise uses elements from ifTrue if cond is true, 
    /// otherwise elements from ifFalse.
    let inline ifThenElse (cond: #Tensor<bool>) (ifTrue: 'B when 'B :> Tensor<'T>) (ifFalse: 'B) : 'B =
        ifTrue.IfThenElse cond ifFalse :?> 'B

    /// converts the Array from one data type to another
    let convert (a: #Tensor<'T>) : Tensor<'C> =
        a |> mapTC (fun v -> conv<'C> v)

    /// converts to int
    let int (a: #Tensor<'T>) : Tensor<int> =
        convert a

    /// converts to float
    let float (a: #Tensor<'T>) : Tensor<float> =
        convert a

    /// converts to single
    let single (a: #Tensor<'T>) : Tensor<single> =
        convert a

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // reduction operations
    ////////////////////////////////////////////////////////////////////////////////////////////////         
     
    /// applies the given reduction function over the given dimension
    let inline axisReduceTypeChange (f: Tensor<'T> -> Tensor<'R>) dim (a: Tensor<'T>) : Tensor<'R> =
        checkAxis dim a
        let c = newCOfType (shape a |> List.without dim) a
        for srcRng, dstIdx in TensorLayout.allSrcRngsAndTrgtIdxsForAxisReduce dim (layout a) do
            set dstIdx (f (view srcRng a) |> get []) c
        c

    /// applies the given reduction function over the given dimension
    let inline axisReduce (f: Tensor<'T> -> Tensor<'T>) dim (a: 'A when 'A :> Tensor<'T>) : 'A =
        axisReduceTypeChange f dim a :?> 'A

    let inline private sumImpl (a: Tensor<'T>) =
        allElems a 
        |> Seq.fold (+) Tensor<'T>.Zero         
        |> scalarOfSameType a 

    /// element-wise sum
    let sum (a: #Tensor<'T>) =
        typedApply (unsp) sumImpl sumImpl sumImpl sumImpl sumImpl a 

    /// element-wise sum over given axis
    let sumAxis dim a = 
        axisReduce sum dim a

    /// mean 
    let mean (a: 'A when 'A :> Tensor<'T>) : 'A =
        let a = a :> Tensor<'T>
        sum a / scalarOfSameType a (conv<'T> (nElems a)) :?> 'A

    /// mean over given axis
    let meanAxis dim a = 
        axisReduce mean dim a

    /// standard deviation (maximum likelihood estimate for normally distributed variables)
    let std (a: 'A when 'A :> Tensor<'T>) : 'A =
        let a = a :> Tensor<'T>
        let v = a - mean a
        v * v |> mean |> sqrt :?> 'A

    /// standard deviation (maximum likelihood estimate for normally distributed variables) over given axis
    let stdAxis dim (a: 'A when 'A :> Tensor<'T>) : 'A =
        let a = a :> Tensor<'T>
        let means = a |> meanAxis dim |> insertAxis dim
        let v = a - means 
        v * v |> meanAxis dim |> sqrt :?> 'A
    
    /// tensor, matrix or vector norm of given order
    let ordNorm (ord: 'T) (a: 'A when 'A :> Tensor<'T>) : 'A =
        let ord = scalarOfType a ord
        let a = a :> Tensor<'T>
        let s = a ** ord |> sum
        s ** (onesLike ord / ord) :?> 'A

    /// tensor, matrix or vector norm of given order over given axis
    let ordNormAxis dim (ord: 'T) (a: 'A when 'A :> Tensor<'T>) : 'A =
        let ord = scalarOfType a ord
        let a = a :> Tensor<'T>
        let s = a ** ord |> sumAxis dim
        s ** (onesLike ord / ord) :?> 'A

    /// L2-norm of tensor, matrix or vector
    let norm (a: 'A when 'A :> Tensor<'T>) : 'A =
        ordNorm (conv<'T> 2) a

    /// L2-norm of tensor, matrix or vector over given axis
    let normAxis dim (a: 'A when 'A :> Tensor<'T>) : 'A =
        ordNormAxis dim (conv<'T> 2) a

    let inline private productImpl (a: Tensor<'T>) =
        allElems a 
        |> Seq.fold (*) Tensor<'T>.One
        |> scalarOfSameType a 

    /// element-wise product
    let product (a: #Tensor<'T>) =
        typedApply (unsp) productImpl productImpl productImpl productImpl productImpl a 

    /// element-wise product over given axis
    let productAxis dim a = 
        axisReduce product dim a

    let inline private maxImpl a =
        allElems a 
        |> Seq.reduce max
        |> scalarOfSameType a 

    /// maximum value
    let max a =
        if nElems a = 0L then invalidArg "a" "cannot compute max of empty ArrayNDT"
        typedApply (unsp) maxImpl maxImpl maxImpl maxImpl maxImpl a
    
    /// position of maximum value
    let argMax a =
        allIdx a
        |> Seq.maxBy (fun idx -> a |> get idx)

    /// maximum value over given axis
    let maxAxis dim a = 
        axisReduce max dim a

    let inline private argMaxAxisReduc (a: Tensor<'T>) =
        allIdx a
        |> Seq.maxBy (fun idx -> a |> get idx)
        |> fun idx -> idx.[0]
        |> scalarOfType a

    /// positions of maximum values along given axis
    let argMaxAxis dim (a: Tensor<'T>) : Tensor<int64> =
        let f a = axisReduceTypeChange argMaxAxisReduc dim a
        typedApplyTypeChange f f f f f f a

    let inline private minImpl a =
        allElems a 
        |> Seq.reduce min
        |> scalarOfSameType a 

    /// minimum value
    let min a =
        if nElems a = 0L then invalidArg "a" "cannot compute min of empty ArrayNDT"
        typedApply (unsp) minImpl minImpl minImpl minImpl minImpl a

    /// position of maximum value
    let argMin a =
        allIdx a
        |> Seq.minBy (fun idx -> a |> get idx)

    /// minimum value over given axis
    let minAxis dim a = 
        axisReduce min dim a

    let inline private argMinAxisReduc (a: Tensor<'T>) =
        allIdx a
        |> Seq.minBy (fun idx -> a |> get idx)
        |> fun idx -> idx.[0]
        |> scalarOfType a

    /// positions of maximum values along given axis
    let argMinAxis dim (a: Tensor<'T>) : Tensor<int64> =
        let f a = axisReduceTypeChange argMinAxisReduc dim a
        typedApplyTypeChange f f f f f f a

    /// true if all elements of the array are true
    let all a =
        let value = allElems a |> Seq.fold (&&) true
        scalarOfSameType a value

    /// true if all elements over given axis are true
    let allAxis dim a =
        axisReduce all dim a

    /// true if any element of the array is true
    let any a =
        let value = allElems a |> Seq.fold (||) false
        scalarOfSameType a value

    /// true if any element over given axis are true
    let anyAxis dim a =
        axisReduce any dim a
     
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // tensor operations
    ////////////////////////////////////////////////////////////////////////////////////////////////         

    /// Returns true if two arrays have same (within specified precision) values in all elements.
    /// If arrays have different shape, then false is returned.
    let inline almostEqualWithTol (aTol: 'T) (rTol: 'T) (a: Tensor<'T>) (b: Tensor<'T>) =
        if a.Shape = b.Shape then
            isCloseWithTol aTol rTol a b |> all
        else 
            let res = newCOfType [] a
            set [] false res
            res

    /// Returns true if two arrays have same (within machine precision) values in all elements.
    /// If arrays have different shape, then false is returned.
    let inline almostEqual (a: Tensor<'T>) (b: Tensor<'T>) =
        almostEqualWithTol (conv<'T> 1e-8) (conv<'T> 1e-5) a b

    /// Returns true if all values in the tensor are finite (not NaN and not infinite).
    let inline allFinite (a: Tensor<'T>) =
        a |> isFinite |> all

    /// dot product implementation between vec*vec, mat*vec, mat*mat, batched mat*vec, batched mat*mat
    let inline dotImpl (a: Tensor<'T>) (b: Tensor<'T>) =
        let inline matrixDot a b =
            let nI = (shape a).[0]
            let nJ = (shape a).[1]
            let nK = (shape b).[1]
            let c = newCOfSameType [nI; nK] a
            for k in 0L .. nK-1L do
                for i in 0L .. nI-1L do
                    let v = 
                        {0L .. nJ-1L}
                        |> Seq.map (fun j -> (get [i; j] a) * (get [j; k] b))
                        |> Seq.sum
                    set [i; k] v c
            c

        let inline batchedMatrixDot (a: Tensor<'T>) (b: Tensor<'T>) =
            let a, b = broadcastToSameInDims [0..nDims a - 3] a b
            let aRows, aCols = (shape a).[nDims a - 2], (shape a).[nDims a - 1]
            let bRows, bCols = (shape b).[nDims b - 2], (shape b).[nDims b - 1]
            if aCols <> bRows then
                failwithf "cannot compute batched dot product between arrays of shapes %A and %A" 
                    (shape a) (shape b)                
            let smplShape = (shape a).[0 .. nDims a - 3]
            let nSmpls = List.fold (*) 1L smplShape
            let a = reshape [nSmpls; aRows; aCols] a
            let b = reshape [nSmpls; bRows; bCols] b
            let c = newCOfSameType [nSmpls; aRows; bCols] a
            for smpl in 0L .. nSmpls-1L do
                c.[smpl, *, *] <- matrixDot a.[smpl, *, *] b.[smpl, *, *]
            c |> reshape (smplShape @ [aRows; bCols])         

        match nDims a, nDims b with
            | 1, 1 when shape a = shape b -> 
                map2 (*) a b |> sum
            | 2, 1 when (shape a).[1] = (shape b).[0] -> 
                matrixDot a (padRight b) |> view [RngAll; RngElem 0L] 
            | 2, 2 when (shape a).[1] = (shape b).[0] ->
                matrixDot a b
            | na, nb when na > 2 && na = nb+1 && (shape a).[na-1] = (shape b).[nb-1] ->
                // batched mat*vec
                (batchedMatrixDot a (padRight b)).[Fill, 0L]
            | na, nb when na > 2 && na = nb && (shape a).[na-1] = (shape b).[nb-2] ->
                // batched mat*mat
                batchedMatrixDot a b
            | _ -> 
                failwithf "cannot compute dot product between arrays of shapes %A and %A" 
                    (shape a) (shape b)

    /// dot product between vec*vec, mat*vec, mat*mat, batched mat*vec, batched mat*mat
    let inline dot a b =
        a .* b

    /// block array specification
    type BlockSpec<'T> =
        | Blocks of BlockSpec<'T> list
        | Array of Tensor<'T>

    /// array constructed of other arrays
    let inline blockArray bs =

        let rec commonShape joinDim shps =               
            match shps with
            | [shp] ->
                List.set joinDim -1L shp
            | shp::rShps ->
                let commonShp = commonShape joinDim [shp]
                if commonShp <> commonShape joinDim rShps then
                    failwithf "block array blocks must have same rank and be identical in all but the join dimension"
                commonShp
            | [] -> []

        let joinSize joinDim (shps: int64 list list) =
            shps |> List.map (fun shp -> shp.[joinDim]) |> List.sum

        let joinShape joinDim shps =
            commonShape joinDim shps 
                |> List.set joinDim (joinSize joinDim shps)

        let rec joinedBlocksShape joinDim bs =
            match bs with
            | Blocks blcks ->
                blcks |> List.map (joinedBlocksShape (joinDim + 1)) |> joinShape joinDim
            | Array ary ->
                ary |> shape

        let rec blockPosAndContents (joinDim: int) startPos bs = seq {
            match bs with
            | Blocks blcks ->
                let mutable pos = startPos
                for blck in blcks do
                    yield! blockPosAndContents (joinDim + 1) pos blck 
                    let blckShape = joinedBlocksShape joinDim blck
                    pos <- List.set joinDim (pos.[joinDim] + blckShape.[joinDim]) pos
            | Array ary ->
                yield startPos, ary
        }

        let rec anyArray bs =
            match bs with
            | Blocks b -> List.tryPick anyArray b
            | Array a -> Some a
                  
        let tmplArray = Option.get (anyArray bs)
        let joinedShape = joinedBlocksShape 0 bs
        let joined = newCOfSameType joinedShape tmplArray
        let startPos = List.replicate (List.length joinedShape) 0L

        for pos, ary in blockPosAndContents 0 startPos bs do
            let slice = List.map2 (fun p s -> Rng(Some p, Some (p + s))) pos (shape ary)
            let joinedSlice = joined |> view slice 
            copyTo ary joinedSlice

        joined
    
    /// tensor product
    let inline tensorProductImpl (a: Tensor<'T>) (b: Tensor<'T>) : Tensor<'T> =
        let a, b = padToSame a b
        let aShp = shape a

        let rec generate pos = 
            match List.length pos with
            | dim when dim = nDims a ->
                let aElem = get pos a
                Array (aElem * b)
            | dim ->
                seq {for p in 0L .. aShp.[dim] - 1L -> generate (pos @ [p])}
                    |> Seq.toList |> Blocks

        generate [] |> blockArray

    /// tensor product
    let inline tensorProduct (a: Tensor<'T>) (b: Tensor<'T>) : Tensor<'T> = a %* b

    /// Returns a view of the diagonal along the given axes.
    /// The diagonal replaces the first axis and the second axis is removed.
    let diagAxis ax1 ax2 (a: #Tensor<'T>) =
        relayout (TensorLayout.diagAxis ax1 ax2 a.Layout) a

    /// Returns a view of the diagonal of a matrix as a vector.
    /// If the specified tensor has more than two dimensions, the diagonals
    /// along the last two dimensions are returned.
    let diag (a: #Tensor<'T>) =
        if a.NDims < 2 then
            failwithf "need at least a two dimensional array for diagonal but got shape %A" a.Shape
        diagAxis (a.NDims-2) (a.NDims-1) a

    /// Creates a new array of same shape but with ax2 inserted.
    /// The diagonal over ax1 and ax2 is filled with the elements of the original ax1.
    /// The other elements are set to zero.
    let diagMatAxis ax1 ax2 (a: #Tensor<'T>) =
        if ax1 = ax2 then failwithf "axes to use for diagonal must be different"
        let ax1, ax2 = if ax1 < ax2 then ax1, ax2 else ax2, ax1
        checkAxis ax1 a
        if not (0 <= ax2 && ax2 <= a.NDims) then
            failwithf "cannot insert axis at position %d into array of shape %A" ax2 a.Shape
        let dShp = a.Shape |> List.insert ax2 a.Shape.[ax1]
        let d = newCOfSameType dShp a
        let dDiag = diagAxis ax1 ax2 d
        dDiag.[Fill] <- a
        d

    /// Creates a new matrix that has the specified diagonal.
    /// All other elements are zero.
    /// If the specified array has more than one dimension, the operation is
    /// performed batch-wise on the last dimension.
    let diagMat (a: #Tensor<'T>) =
        if a.NDims < 1 then
            failwithf "need at leat a one-dimensional array to create a diagonal matrix"
        diagMatAxis (a.NDims-1) a.NDims a

    /// Computes the traces along the given axes.
    let traceAxis ax1 ax2 (a: #Tensor<'T>) =
        let tax = if ax1 < ax2 then ax1 else ax1 - 1
        a |> diagAxis ax1 ax2 |> sumAxis tax

    /// Computes the trace of a matrix.
    /// If the specified tensor has more than two dimensions, the traces
    /// along the last two dimensions are returned.
    let trace (a: #Tensor<'T>) =
        if a.NDims < 2 then
            failwithf "need at least a two dimensional array for trace but got shape %A" a.Shape
        traceAxis (a.NDims-2) (a.NDims-1) a

    /// Returns the inverse of the given matrix.
    /// If the specified tensor has more than two dimensions, the matrices
    /// consisting of the last two dimensions are inverted.
    let invert (a: 'A when 'A :> Tensor<_>) : 'A  =
        a.Invert () :?> 'A

    /// Computes the (real) eigenvalues and eigenvectors of the symmetric matrix.
    /// Returns (vals, vecs) where each column of 'vecs' is the eigenvector for the
    /// corresponding eigenvalue in 'vals'.
    let symmetricEigenDecomposition (a: 'A when 'A :> Tensor<_>) : 'A * 'A =
        let eigVals, eigVecs = a.SymmetricEigenDecomposition () 
        eigVals :?> 'A, eigVecs :?> 'A

    /// calculates the pairwise differences along the given axis
    let diffAxis ax (a: #Tensor<'T>) =
        checkAxis ax a
        let shftRng = 
            [for d=0 to a.NDims-1 do
                if d = ax then yield Rng (Some 1L, None)
                else yield RngAll]
        let cutRng = 
            [for d=0 to a.NDims-1 do
                if d = ax then yield Rng (None, Some (a.Shape.[d] - 2L))
                else yield RngAll]
        a.[shftRng] - a.[cutRng]

    /// calculates the pairwise differences along the last axis
    let diff (a: #Tensor<'T>) =
        diffAxis (a.NDims-1) a

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // concatenation and replication
    ////////////////////////////////////////////////////////////////////////////////////////////////         

    /// Concatenates the list of tensors in the given axis.
    let concat dim (arys: #Tensor<'T> seq) =
        let arys = List.ofSeq arys
        if List.isEmpty arys then
            invalidArg "arys" "cannot concatenate empty list of tensors"

        // check for compatibility
        let shp = List.head arys |> shape
        if not (0 <= dim && dim < shp.Length) then
            failwithf "concatenation axis %d is out of range for shape %A" dim shp
        for aryIdx, ary in List.indexed arys do
            if List.without dim ary.Shape <> List.without dim shp then
                failwithf "concatentation element with index %d with shape %A must \
                    be equal to shape %A of the first element, except in the concatenation axis %d" 
                    aryIdx ary.Shape shp dim

        // calculate shape of concatenated tensors
        let totalSize = arys |> List.sumBy (fun ary -> ary.Shape.[dim])
        let concatShape = shp |> List.set dim totalSize

        // copy tensors into concatenated tensor
        let cc = List.head arys |> newCOfSameType concatShape
        let mutable pos = 0L
        for ary in arys do
            let aryLen = ary.Shape.[dim]
            if aryLen > 0L then
                let ccRng = 
                    List.init shp.Length (fun idx ->
                        if idx = dim then Rng (Some pos, Some (pos + aryLen - 1L))
                        else RngAll)
                cc.[ccRng] <- ary
                pos <- pos + aryLen
        cc

    /// Replicates the tensor the given number of repetitions along the given axis.
    let replicate dim reps (ary: #Tensor<'T>) =
        ary |> checkAxis dim
        if reps < 0L then
            invalidArg "reps" "number of repetitions cannot be negative"

        // 1. insert axis of size one left to repetition axis
        // 2. broadcast along the new axis to number of repetitions
        // 3. reshape to result shape
        ary 
        |> reshape (ary.Shape |> List.insert dim 1L)
        |> broadcastDim dim reps
        |> reshape (ary.Shape |> List.set dim (reps * ary.Shape.[dim]))



#endif


module Tensor =
    module Parallel = 

        /// Fills the tensor with the values returned by the function.
        let fill (fn: unit -> 'T) (trgt: Tensor<'T>) =
            trgt.Backend.Fill (fn=fn, trgt=trgt, useThreads=true)

        /// Fills the array with the values returned by the function.
        let fillIndexed (fn: int64[] -> 'T) (trgt: Tensor<'T>) =
            trgt.Backend.FillIndexed (fn=fn, trgt=trgt, useThreads=true)

        /// maps all elements using the specified function into a new tensor
        let map (fn: 'T -> 'R) (a: Tensor<'T>) =
            let trgt, a = Tensor<_>.PrepareElemwise (a)
            trgt.Backend.Map (fn=fn, trgt=trgt, src=a, useThreads=true)
            trgt       

        /// maps all elements using the specified indexed function into a new tensor
        let mapi (fn: int64[] -> 'T -> 'R) (a: Tensor<'T>) =
            let trgt, a = Tensor<_>.PrepareElemwise (a)
            trgt.Backend.MapIndexed (fn=fn, trgt=trgt, src=a, useThreads=true)
            trgt     

        /// maps all elements using the specified function into a new tensor
        let map2 (fn: 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
            let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
            trgt.Backend.Map2 (fn=fn, trgt=trgt, src1=a, src2=b, useThreads=true)
            trgt       

        /// maps all elements using the specified indexed function into a new tensor
        let mapi2 (fn: int64[] -> 'TA -> 'TB -> 'R) (a: Tensor<'TA>) (b: Tensor<'TB>) =
            let trgt, a, b = Tensor<_>.PrepareElemwise (a, b)
            trgt.Backend.MapIndexed2 (fn=fn, trgt=trgt, src1=a, src2=b, useThreads=true)
            trgt       
