namespace ArrayNDNS

open Basics


/// Functions for storing ArrayNDTs in HDF5 files.
module ArrayNDHDF =

    /// Writes the given ArrayNDT into the HDF5 file under the given name.
    let write (hdf5: HDF5) name (ary: ArrayNDT<'T>) =
        let ary = ArrayND.makeContiguous ary
        
        match ary with
        | :? ArrayNDHostT<'T> as hostAry ->
            match hostAry.Storage with
            | :? ManagedArrayStorageT<'T> as storage ->
                hdf5.Write (name, storage.Data, ArrayND.shape ary)
            | _ -> failwith "currently only ManagedArrayStorage is supported"
        | _ -> failwith "can only write ArrayNDHostT to HDF5"

    /// Reads the ArrayNDT with the given name from an HDF5 file.
    let read<'T> (hdf5: HDF5) name =
        let (data: 'T array), shape = hdf5.Read (name)       
        ArrayNDHostT (ArrayNDLayout.newContiguous shape, ManagedArrayStorageT (data)) :> ArrayNDT<'T>

        
