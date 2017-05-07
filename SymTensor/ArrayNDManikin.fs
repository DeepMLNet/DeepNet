namespace SymTensor.Compiler

open System.Runtime.InteropServices
open ManagedCuda
open Basics
open Tensor
open SymTensor
open UExprTypes


[<AutoOpen>]
module ArrayNDManikinTypes = 

    /// memory allocation type
    type MemAllocKindT =
        /// device memory allocation
        | MemAllocDev
        /// registered host memory allocation
        | MemAllocRegHost

    /// represents a memory allocation exclusively for this expression (used for temporary results)
    type MemAllocManikinT = {
        Id:             int
        TypeName:       TypeNameT
        Elements:       int64
        Kind:           MemAllocKindT
    } with
        member this.ByteSize = this.Elements * TypeName.size64 this.TypeName

    type MemConstManikinT = {
        Id:             int
        TypeName:       TypeNameT
    }

    /// Represents memory. 
    /// Memory can either be internal to this expression or external (passed in variable at runtime).
    /// Memory can either be on the host or the accelerator.
    [<StructuredFormatDisplay("{Pretty}")>]
    type MemManikinT =
        /// no memory (represents a null pointer)
        | MemZero of TypeNameT
        /// a memory allocation internal to the workspace
        | MemAlloc of MemAllocManikinT
        /// an external variable
        | MemExternal of VarSpecT
        /// a constant array
        | MemConst of MemConstManikinT
        with 
            member this.Pretty = 
                match this with
                | MemZero t -> sprintf "MemZero %A" t.Type
                | MemAlloc a -> sprintf "MemAlloc %d (%d KB)" a.Id (a.ByteSize / 1024L)
                | MemExternal vs -> sprintf "MemExternal %A" vs
                | MemConst c -> sprintf "MemConst %d" c.Id

    /// represents an n-dimensional array that will be allocated or accessed during execution 
    [<StructuredFormatDisplay("{Pretty}")>]
    type ArrayNDManikinT (layout:           TensorLayout, 
                          storage:          MemManikinT) = 

        /// storage manikin
        member this.Storage = storage

        member this.Layout = layout

        member this.Shape = this.Layout.Shape

        member this.NDims = this.Layout.NDims


        /// typename of the data stored in this array
        member this.TypeName = 
            match storage with
            | MemAlloc {TypeName=tn} -> tn
            | MemExternal vs -> vs.TypeName
            | MemConst mc -> mc.TypeName
            | MemZero t -> t

        member this.NewView (layout: TensorLayout) = 
            ArrayNDManikinT(layout, storage) 

        member this.DataType =
            TypeName.getType this.TypeName

        /// C++ type name for ArrayND with static shape and dynamic offset/strides
        member this.DynamicCPPType =
            let dims = TensorLayout.nDims layout
            let shp = TensorLayout.shape layout
            let cppDataType = Util.cppType this.DataType
            let shapeStr = 
                if dims = 0 then "" 
                else "<" + (shp |> Seq.map (sprintf "%dLL") |> String.concat ",") + ">"
            sprintf "ArrayND%dD<%s, ShapeStatic%dD%s, StrideDynamic%dD>" 
                dims cppDataType dims shapeStr dims        

        member this.Pretty = 
            sprintf "ArrayNDManikinT (Storage=%A; Shape=%A; Strides=%A)" 
                storage layout.Shape layout.Stride

        override this.Equals other =
            match other with
            | :? ArrayNDManikinT as other -> 
                this.Layout = other.Layout && this.Storage = other.Storage
            | _ -> false

        override this.GetHashCode () =
            hash (this.Layout, this.Storage)


module ArrayNDManikin =

    /// creates a new ArrayNDManikinT using no storage
    let newZero typ shape =
        let layout = TensorLayout.newC shape
        ArrayNDManikinT (layout, MemZero typ)

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with C-order
    let newC memAllocator typ shape = 
        let layout = TensorLayout.newC shape
        ArrayNDManikinT (layout, memAllocator typ (TensorLayout.nElems layout) MemAllocDev)

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with Fortran-order
    let newF memAllocator typ shape = 
        let layout = TensorLayout.newF shape
        ArrayNDManikinT (layout, memAllocator typ (TensorLayout.nElems layout) MemAllocDev)

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with specifed stride order
    let newOrdered memAllocator typ shape strideOrder =
        let layout = TensorLayout.newOrdered shape strideOrder
        ArrayNDManikinT (layout, memAllocator typ (TensorLayout.nElems layout) MemAllocDev)

    /// create a new MemoryManikinT and a new ArrayNDManikinT with layout suitable for being a BLAS target
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
        ArrayNDManikinT (layout, memAllocator typ (TensorLayout.nElems layout) MemAllocDev)

    /// creates a new ArrayNDManikinT with contiguous layout using the specified storage
    let externalC storage shape =
        let layout = TensorLayout.newC shape
        ArrayNDManikinT (layout, storage) 

    /// creates a new ArrayNDManikinT with specified strides and using the specified storage
    let external storage shape stride =
        let layout = {Shape=shape; Stride=stride; Offset=0L}
        ArrayNDManikinT (layout, storage)

    /// storage
    let storage (ary: ArrayNDManikinT) =
        ary.Storage

    /// used data type name
    let typeName (ary: ArrayNDManikinT) =
        ary.TypeName

    /// size of the used data type 
    let typeSize ary =
        ary |> typeName |> TypeName.size

    /// size of the used data type as int64
    let typeSize64 ary =
        ary |> typeName |> TypeName.size64

    /// offset in bytes
    let offsetInBytes (ary: ArrayNDManikinT) =
        typeSize64 ary * ary.Layout.Offset

    /// address of given element in bytes (relative to start of array)
    let addrInBytes idx (ary: ArrayNDManikinT) =
        typeSize64 ary * (ary.Layout |> TensorLayout.addr idx)

    /// size in bytes 
    let sizeInBytes (ary: ArrayNDManikinT) =
        typeSize64 ary * TensorLayout.nElems ary.Layout

    /// True if array can be target of BLAS operation.
    let canBeBlasTarget (ary: ArrayNDManikinT) =
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


        
