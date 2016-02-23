namespace SymTensor.Compiler

open System.Runtime.InteropServices
open ManagedCuda
open Util
open ArrayNDNS
open SymTensor


[<AutoOpen>]
module ArrayNDManikinTypes = 
    open ArrayND

    /// represents a memory allocation execlusively for this expression (used for temporary results)
    type MemAllocManikinT = {
        Id:             int; 
        TypeName:       TypeNameT; 
        Elements:       int;
    }

    /// Represents memory. 
    /// Memory can either be internal to this expression or external (passed in variable at runtime).
    /// Memory can either be on the host or the accelerator.
    type MemManikinT =
        | MemAlloc of MemAllocManikinT
        | MemExternal of IVarSpec

    type IArrayNDManikin =
        abstract member Storage : MemManikinT
        abstract member TypeName : TypeNameT

    /// represents an n-dimensional array that will be allocated or accessed during execution 
    type ArrayNDManikinT (layout: ArrayNDLayoutT, storage: MemManikinT) = 
        inherit ArrayNDT<int> (layout)  // generic type does not matter since we do not store data

        /// storage manikin
        member this.Storage = storage

        /// typename of the data stored in this array
        member this.TypeName = 
            match storage with
            | MemAlloc {TypeName=tn} -> tn
            | MemExternal vs -> vs.TypeName

        interface IHasLayout with
            member this.Layout = this.Layout
        interface IArrayNDManikin with
            member this.Storage = this.Storage
            member this.TypeName = this.TypeName

        override this.Item
            with get pos = failwith "ArrayNDManikin does not store data"
            and set pos value = failwith "ArrayNDManikin does not store data"

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            failwith "ArrayNDManikin cannot allocate memory on its own"

        override this.NewView (layout: ArrayNDLayoutT) = 
            ArrayNDManikinT(layout, storage) :> ArrayNDT<int>




module ArrayNDManikin =
    open ArrayND

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with contiguous layout
    let inline newContiguous memAllocator typ shape = 
        let layout = ArrayNDLayout.newContiguous shape
        ArrayNDManikinT (layout, 
                         (memAllocator typ (ArrayNDLayout.nElems layout)))
//            :> ArrayNDT<int>

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with Fortran layout
    let inline newColumnMajor memAllocator typ shape = 
        let layout = ArrayNDLayout.newColumnMajor shape
        ArrayNDManikinT (layout, 
                         (memAllocator typ (ArrayNDLayout.nElems layout)))
            //:> ArrayNDT<int>

    /// creates a new ArrayNDManikinT with contiguous layout using the specified storage
    let inline externalContiguous storage shape =
        let layout = ArrayNDLayout.newContiguous shape
        ArrayNDManikinT (layout, storage) 

    /// used data type name
    let inline typeName (ary: ArrayNDManikinT) =
        ary.TypeName

    /// size of the used data type 
    let inline typeSize ary =
        ary |> typeName |> TypeName.getType |> Marshal.SizeOf

    /// offset in bytes
    let inline offsetInBytes ary =
        (typeSize ary) * (ArrayND.offset ary)

    /// size in bytes for contiguous arrays
    let inline sizeInBytes ary =
        if ArrayND.isContiguous ary then
            (typeSize ary) * (ArrayND.nElems ary)
        else failwith "size calculation currently only possible for continguous arrays"

        
