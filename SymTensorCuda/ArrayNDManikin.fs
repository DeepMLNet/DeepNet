namespace SymTensor.Compiler

open ArrayNDNS
open SymTensor


[<AutoOpen>]
module ArrayNDManikinTypes = 
    open ArrayND

    /// represents a memory allocation execlusively for this expression (used for temporary results)
    type MemAllocManikinT = {Id: int; Size: int; Type: System.Type}

    /// memory can either be internal to this expression or external (passed in variable at runtime)
    type MemManikinT =
        | MemAlloc of MemAllocManikinT
        | ExternalMem of IVarSpec

    type IArrayNDManikin =
        abstract member Storage : MemManikinT

    /// represents an n-dimensional array 
    type ArrayNDManikinT<'T> (layout: ArrayNDLayoutT, storage: MemManikinT) = 
        inherit ArrayNDT<'T> (layout)

        /// storage
        member this.Storage = storage

        interface IHasLayout with
            member this.Layout = this.Layout
        interface IArrayNDManikin with
            member this.Storage = storage

        override this.Item
            with get pos = failwith "ArrayNDManikin does not store data"
            and set pos value = failwith "ArrayNDManikin does not store data"

        override this.NewOfSameType (layout: ArrayNDLayoutT) = 
            failwith "ArrayNDManikin cannot allocate memory by its own"

        override this.NewView (layout: ArrayNDLayoutT) = 
            ArrayNDManikinT<'T>(layout, storage) :> ArrayNDT<'T>




module ArrayNDManikin =
    open ArrayND

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with continguous layout
    let inline newContinguous<'T> memAllocator shape = 
        let layout = ArrayNDLayout.newContiguous shape
        ArrayNDManikinT<'T> (layout, 
                             memAllocator (typeof<'T>, ArrayNDLayout.nElems layout)) :> ArrayNDT<'T>

    /// creates a new MemoryManikinT and a new ArrayNDManikinT with Fortran layout
    let inline newColumnMajor<'T> memAllocator shape = 
        let layout = ArrayNDLayout.newColumnMajor shape
        ArrayNDManikinT<'T> (layout, 
                             memAllocator (typeof<'T>, ArrayNDLayout.nElems layout)) :> ArrayNDT<'T>

