namespace SymTensor.Compiler

open System.Runtime.InteropServices
open ManagedCuda
open Basics
open ArrayNDNS
open SymTensor


[<AutoOpen>]
module ArrayNDManikinTypes = 
    open ArrayND

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
        Elements:       int
        Kind:           MemAllocKindT
    } with
        member this.ByteSize =
            this.Elements * TypeName.size this.TypeName

    /// Represents memory. 
    /// Memory can either be internal to this expression or external (passed in variable at runtime).
    /// Memory can either be on the host or the accelerator.
    type MemManikinT =
        | MemAlloc of MemAllocManikinT
        | MemExternal of UVarSpecT

    /// represents an n-dimensional array that will be allocated or accessed during execution 
    type ArrayNDManikinT (layout:           ArrayNDLayoutT, 
                          storage:          MemManikinT) = 
        inherit ArrayNDT<int> (layout)  // generic type does not matter since we do not store data

        /// storage manikin
        member this.Storage = storage

        /// typename of the data stored in this array
        member this.TypeName = 
            match storage with
            | MemAlloc {TypeName=tn} -> tn
            | MemExternal vs -> vs.TypeName

        override this.Item
            with get pos = failwith "ArrayNDManikin does not store data"
            and set pos value = failwith "ArrayNDManikin does not store data"

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            failwith "ArrayNDManikin cannot allocate memory on its own"

        override this.NewOfType<'N> (layout: ArrayNDLayoutT) : ArrayNDT<'N> = 
            failwith "ArrayNDManikin cannot allocate memory on its own"

        override this.NewView (layout: ArrayNDLayoutT) = 
            ArrayNDManikinT(layout, storage) :> ArrayNDT<int>

        override this.DataType =
            TypeName.getType this.TypeName

        override this.Location = ArrayLoc "Manikin"

        /// C++ type name for ArrayND with static shape and dynamic offset/strides
        member this.DynamicCPPType =
            let dims = ArrayNDLayout.nDims layout
            let shp = ArrayNDLayout.shape layout
            let cppDataType = Util.cppType this.DataType
            let shapeStr = 
                if dims = 0 then "" 
                else "<" + (shp |> Util.intToStrSeq |> String.concat ",") + ">"
            sprintf "ArrayND%dD<%s, ShapeStatic%dD%s, StrideDynamic%dD>" 
                dims cppDataType dims shapeStr dims        

        override this.Invert () = 
            failwith "ArrayNDManikin does not store data"


module ArrayNDManikin =
    open ArrayND

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with contiguous layout
    let newC memAllocator typ shape = 
        let layout = ArrayNDLayout.newC shape
        ArrayNDManikinT (layout, memAllocator typ (ArrayNDLayout.nElems layout) MemAllocDev)

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with Fortran layout
    let newF memAllocator typ shape = 
        let layout = ArrayNDLayout.newF shape
        ArrayNDManikinT (layout, memAllocator typ (ArrayNDLayout.nElems layout) MemAllocDev)

    /// create a new MemoryManikinT and a new ArrayNDManikinT with layout suitable for being a BLAS target
    let newBlasTarget memAllocator typ shape = 
        let nd = List.length shape
        let smplShp = shape.[0..nd-3]
        let matRows, matCols = shape.[nd-2], shape.[nd-1]
        let matElems = matRows * matCols
        let rec smplStride (shp: int list) =
            match shp with
            | [] -> []
            | [l] -> [matElems]
            | l::(lp::lrest) ->
                match smplStride (lp::lrest) with 
                | sp::srest -> (lp*sp)::sp::srest
                | [] -> failwith "unexpected"           
        let stride = smplStride smplShp @ [1; matRows]
        
        let layout = {Shape=shape; Stride=stride; Offset=0}
        ArrayNDManikinT (layout, memAllocator typ (ArrayNDLayout.nElems layout) MemAllocDev)

    /// creates a new ArrayNDManikinT with contiguous layout using the specified storage
    let externalC storage shape =
        let layout = ArrayNDLayout.newC shape
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

    /// offset in bytes
    let offsetInBytes ary =
        (typeSize ary) * (ArrayND.offset ary)

    /// address of given element in bytes (relative to start of array)
    let addrInBytes idx ary =
        (typeSize ary) * (ary |> ArrayND.layout |> ArrayNDLayout.addr idx)

    /// size in bytes 
    let sizeInBytes ary =
        (typeSize ary) * (ArrayND.nElems ary)

    /// True if array can be target of BLAS operation.
    let canBeBlasTarget ary =
        let nd = ArrayND.nDims ary
        if nd >= 2 then
            let st = ArrayND.stride ary
            let shp = ArrayND.shape ary
            match st.[nd-2 ..] with
            | [1; ld] when ld >= 1 && ld >= shp.[nd-2] -> true
            | _ -> false
        else false

            

        
