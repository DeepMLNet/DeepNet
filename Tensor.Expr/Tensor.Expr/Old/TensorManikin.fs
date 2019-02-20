namespace Tensor.Expr.Compiler

open System
open System.Runtime.InteropServices
open ManagedCuda

open Tensor
open Tensor.Utils
open Tensor.Backend
open Tensor.Expr
open UExprTypes


/// C++ data type helpers
module internal Cpp =
    /// C++ data type for given type instance
    let cppTypeInst (typ: System.Type) = 
        match typ with
        | _ when typ = typeof<single>    -> "float"
        | _ when typ = typeof<double>    -> "double"
        | _ when typ = typeof<sbyte>     -> "int8_t"
        | _ when typ = typeof<byte>      -> "uint8_t"
        | _ when typ = typeof<int32>     -> "int32_t"
        | _ when typ = typeof<uint32>    -> "uint32_t"
        | _ when typ = typeof<int64>     -> "int64_t"
        | _ when typ = typeof<uint64>    -> "uint64_t"
        | _ when typ = typeof<bool>      -> "bool"
        | _ when typ = typeof<nativeint> -> "ptr_t"
        | _ -> failwithf "no C++ datatype for %A" typ

    /// C++ data type for given type 
    let cppType<'T> = cppTypeInst typeof<'T>


[<AutoOpen>]
module TensorManikinTypes = 

    // TODO: remove
    /// Kind of memory allocation.
    type MemAllocKind =
        /// device memory allocation
        | MemAllocDev
        /// registered host memory allocation
        | MemAllocRegHost

    /// Represents a memory allocation.
    type StorageAlloc = {
        /// allocation id
        Id:             int
        /// data type
        TypeName:       TypeName
        /// number of elements
        Elements:       int64
        // TODO: remove
        /// kind of allocation
        Kind:           MemAllocKind
        // TODO: add
        /// storage device
        //Device:         ITensorDevice
    } with
        /// size of allocation in bytes
        member this.ByteSize = this.Elements * TypeName.size64 this.TypeName

    /// Represents constant memory?
    type StorageConst = {
        /// allocation id
        Id:             int
        /// data type
        TypeName:       TypeName
    }

    /// Represents memory in which a tensor may be stored.
    /// Memory can either be internal to an expression workspace or external, e.g.
    /// used by a variable passed in variable at runtime.
    /// Memory can either be on the host or the accelerator.
    [<RequireQualifiedAccess; StructuredFormatDisplay("{Pretty}")>]
    type StorageManikin =
        /// no memory (represents a null pointer)
        | Zero of TypeName
        /// a memory allocation internal to the workspace
        | Alloc of StorageAlloc
        /// an external variable
        | External of Var
        /// a constant array
        | Const of StorageConst
        with 
            member this.Pretty = 
                match this with
                | StorageManikin.Zero t -> sprintf "StorageManikin.Zero %A" t.Type
                | StorageManikin.Alloc a -> sprintf "StorageManikin.Alloc %d (%d KB)" a.Id (a.ByteSize / 1024L)
                | StorageManikin.External vs -> sprintf "StorageManikin.External %A" vs
                | StorageManikin.Const c -> sprintf "StorageManikin.Const %d" c.Id

    /// represents an n-dimensional array that will be allocated or accessed during execution 
    [<StructuredFormatDisplay("{Pretty}")>]
    type TensorManikin (layout:           TensorLayout, 
                        storage:          StorageManikin) = 

        /// storage manikin
        member this.Storage = storage

        /// layout
        member this.Layout = layout

        /// shape
        member this.Shape = this.Layout.Shape

        /// number of dimensions
        member this.NDims = this.Layout.NDims

        // TODO: remove
        /// C++ type name
        member this.CPPType = 
            let dims = TensorLayout.nDims layout
            let shp = TensorLayout.shape layout
            let str = TensorLayout.stride layout
            let ofst = TensorLayout.offset layout
            let cppDataType = Cpp.cppTypeInst this.DataType
            let shapeStr = 
                if dims = 0 then "" 
                else "<" + (shp |> Seq.map (sprintf "%dLL") |> String.concat ",") + ">"
            let strideStr = 
                "<" + ((ofst :: str) |> Seq.map (sprintf "%dLL") |> String.concat ",") + ">"
            sprintf "ArrayND%dD<%s, ShapeStatic%dD%s, StrideStatic%dD%s>" 
                dims cppDataType dims shapeStr dims strideStr     

        // TODO: remove
        /// C++ type name for ArrayND with static shape and dynamic offset/strides
        member this.DynamicCPPType =
            let dims = TensorLayout.nDims layout
            let shp = TensorLayout.shape layout
            let cppDataType = Cpp.cppTypeInst this.DataType
            let shapeStr = 
                if dims = 0 then "" 
                else "<" + (shp |> Seq.map (sprintf "%dLL") |> String.concat ",") + ">"
            sprintf "ArrayND%dD<%s, ShapeStatic%dD%s, StrideDynamic%dD>" 
                dims cppDataType dims shapeStr dims   

        /// typename of the data stored in this tensor
        member this.TypeName = 
            match storage with
            | StorageManikin.Alloc {TypeName=tn} -> tn
            | StorageManikin.External vs -> vs.TypeName
            | StorageManikin.Const mc -> mc.TypeName
            | StorageManikin.Zero t -> t

        /// Creates a view of underlying data with a new layout.
        member this.NewView (layout: TensorLayout) = 
            TensorManikin(layout, storage) 

        /// type of stored data
        member this.DataType =
            TypeName.getType this.TypeName    

        /// transposed tensor
        member this.T = 
            TensorManikin (TensorLayout.transpose this.Layout, this.Storage)

        /// pretty string
        member this.Pretty = 
            sprintf "TensorManikin (Storage=%A; Shape=%A; Strides=%A)" 
                storage layout.Shape layout.Stride

        override this.Equals other =
            match other with
            | :? TensorManikin as other -> 
                this.Layout = other.Layout && this.Storage = other.Storage
            | _ -> false

        override this.GetHashCode () =
            hash (this.Layout, this.Storage)


module TensorManikin =

    /// creates a new TensorManikin using no storage
    let newZero typ shape =
        let layout = TensorLayout.newRowMajor shape
        TensorManikin (layout, StorageManikin.Zero typ)

    /// creates a new StorageManikin and a new TensorManikin with C-order
    let newRowMajor memAllocator typ shape = 
        let layout = TensorLayout.newRowMajor shape
        TensorManikin (layout, memAllocator typ (TensorLayout.nElems layout) MemAllocDev)

    /// creates a new StorageManikin and a new TensorManikin with Fortran-order
    let newColumnMajor memAllocator typ shape = 
        let layout = TensorLayout.newColumnMajor shape
        TensorManikin (layout, memAllocator typ (TensorLayout.nElems layout) MemAllocDev)

    /// creates a new StorageManikin and a new TensorManikin with specifed stride order
    let newOrdered memAllocator typ shape strideOrder =
        let layout = TensorLayout.newOrdered shape strideOrder
        TensorManikin (layout, memAllocator typ (TensorLayout.nElems layout) MemAllocDev)

    /// create a new StorageManikin and a new TensorManikin with layout suitable for being a BLAS target
    let newBlasTarget memAllocator typ shape = 
        let nd = List.length shape
        let smplShp = shape.[0..nd-3]
        let matRows, matCols = shape.[nd-2], shape.[nd-1]
        let matElems = matRows * matCols
        let rec smplStride (shp: int64 list) =
            match shp with
            | [] -> []
            | [l] -> [matElems]
            | l::(lp::lrest) ->
                match smplStride (lp::lrest) with 
                | sp::srest -> (lp*sp)::sp::srest
                | [] -> failwith "unexpected"           
        let stride = smplStride smplShp @ [1L; matRows]
        
        let layout = {Shape=shape; Stride=stride; Offset=0L}
        TensorManikin (layout, memAllocator typ (TensorLayout.nElems layout) MemAllocDev)

    /// creates a new TensorManikin with contiguous layout using the specified storage
    let externalRowMajor storage shape =
        let layout = TensorLayout.newRowMajor shape
        TensorManikin (layout, storage) 

    /// creates a new TensorManikin with specified strides and using the specified storage
    let external storage shape stride =
        let layout = {Shape=shape; Stride=stride; Offset=0L}
        TensorManikin (layout, storage)

    let layout (ary: TensorManikin) =
        ary.Layout

    let shape (ary: TensorManikin) =
        ary.Layout.Shape

    let nDims (ary: TensorManikin) =
        ary.Layout.NDims

    let nElems (ary: TensorManikin) =
        ary.Layout.NElems

    let stride (ary: TensorManikin) =
        ary.Layout.Stride

    let offset (ary: TensorManikin) =
        ary.Layout.Offset

    let relayout newLayout (ary: TensorManikin) =
        TensorManikin (newLayout, ary.Storage)

    let isRowMajor (ary: TensorManikin) =
        ary |> layout |> TensorLayout.isRowMajor

    let isColumnMajor (ary: TensorManikin) =
        ary |> layout |> TensorLayout.isColumnMajor
        
    /// a view of the specified tensor over the given range 
    let range (rng: Rng list) a =
        a |> relayout (a |> layout |> TensorLayout.view rng)

    /// Tries to reshape the tensor without copying.
    /// For this to succeed, the tensor must have row-major layout.
    /// If this a reshape without copying is impossible, None is returned.
    let tryReshapeView shp a =
        match a |> layout |> TensorLayout.tryReshape shp with
        | Some newLayout -> a |> relayout newLayout |> Some
        | None -> None

    /// Tries to reshape the tensor without copying.
    /// For this to succeed, the tensor must have row-major layout.
    /// If this a reshape without copying is impossible, an error is raised.
    let reshapeView shp a =
        match tryReshapeView shp a with
        | Some res -> res
        | None -> 
            let msg =
                sprintf "cannot reshape tensor of shape %A and strides %A without copying"
                    (layout a).Shape (layout a).Stride
            raise (InvalidOperationException msg)

    /// Returns true if the tensor can be reshaped without copying.
    let canReshapeView shp a =
        match tryReshapeView shp a with
        | Some _ -> true
        | None -> false

    /// Permutes the axes as specified.
    /// Each entry in the specified permutation specifies the new position of 
    /// the corresponding axis, i.e. to which position the axis should move.
    let permuteAxes (permut: int list) a =
        a |> relayout (a |> layout |> TensorLayout.permuteAxes permut)

    /// inserts a broadcastable dimension of size one as first dimension
    let padLeft a =
        a |> relayout (a.Layout |> TensorLayout.padLeft)

    /// appends a broadcastable dimension of size one as last dimension
    let padRight a =
        a |> relayout (a.Layout |> TensorLayout.padRight)

    /// Inserts an axis of size 1 before the specified position.
    let insertAxis ax a =
        a |> relayout (a.Layout |> TensorLayout.insertAxis ax)

    /// removes the first dimension from the tensor
    let cutLeft a =
        a |> relayout (a.Layout |> TensorLayout.cutLeft)
      
    /// removes the last dimension from the tensor
    let cutRight a =
        a |> relayout (a.Layout |> TensorLayout.cutRight)

    /// transpose
    let transpose (a: TensorManikin) =
        a.T

    /// C++ type string
    let cppType (a: TensorManikin) = 
        a.CPPType

    /// Reverses the elements in the specified dimension.
    let reverseAxis ax a =
        a |> relayout (a |> layout |> TensorLayout.reverseAxis ax)      

    /// Returns a view of the diagonal along the given axes.
    /// The diagonal replaces the first axis and the second axis is removed.
    let diagAxis ax1 ax2 a =
        a |> relayout (a |> layout |> TensorLayout.diagAxis ax1 ax2)

    /// broadcasts the tensor to the given shape
    let broadcastTo shp a =
        a |> relayout (a |> layout |> TensorLayout.broadcastToShape shp)

    /// returns true if at least one dimension is broadcasted
    let isBroadcasted a =
        a |> layout |> TensorLayout.isBroadcasted 

    /// storage
    let storage (ary: TensorManikin) =
        ary.Storage

    /// used data type name
    let typeName (ary: TensorManikin) =
        ary.TypeName

    /// size of the used data type 
    let typeSize ary =
        ary |> typeName |> TypeName.size

    /// size of the used data type as int64
    let typeSize64 ary =
        ary |> typeName |> TypeName.size64

    /// offset in bytes
    let offsetInBytes (ary: TensorManikin) =
        typeSize64 ary * ary.Layout.Offset

    /// address of given element in bytes (relative to start of array)
    let addrInBytes idx (ary: TensorManikin) =
        typeSize64 ary * (ary.Layout |> TensorLayout.addr idx)

    /// size in bytes 
    let sizeInBytes (ary: TensorManikin) =
        typeSize64 ary * TensorLayout.nElems ary.Layout

    /// True if array can be target of BLAS operation.
    let canBeBlasTarget (ary: TensorManikin) =
        let nd = ary.NDims
        if nd >= 2 then
            let st = ary.Layout.Stride
            let shp = ary.Shape
            match st.[nd-2 ..] with
            | [1L; ld] when ld >= 1L && ld >= shp.[nd-2] -> true
            | _ -> false
        else false

    /// true if a and b may overlap
    let maybeOverlapping a b =    
        storage a = storage b
        
