namespace ArrayNDNS

open System.Reflection
open Basics


[<AutoOpen>]
module ArrayNDHDFTypes =

    /// Functions for storing ArrayNDTs in HDF5 files.
    type ArrayNDHDF =

        /// Writes the given ArrayNDHostT into the HDF5 file under the given name.
        static member write<'T> (hdf5: HDF5) name (hostAry: ArrayNDHostT<'T>) =
            let hostAry = Tensor.ensureCAndOffsetFree hostAry
            hdf5.Write (name, hostAry.Data, Tensor.shape hostAry)

        /// Reads the ArrayNDHostT with the given name from an HDF5 file.
        static member read<'T> (hdf5: HDF5) name =
            let (data: 'T array), shape = hdf5.Read (name)       
            ArrayNDHostT (TensorLayout.newC shape, data) 

        /// Writes the given IArrayNDHostT into the HDF5 file under the given name.
        static member writeUntyped (hdf5: HDF5) (name: string) (hostAry: IArrayNDHostT) =
            let gm = typeof<ArrayNDHDF>.GetMethod ("write",  BindingFlags.Public ||| BindingFlags.Static)
            let m = gm.MakeGenericMethod ([| hostAry.DataType |])
            m.Invoke(null, [| box hdf5; box name; box hostAry|]) |> ignore
        
        /// Reads the IArrayNDHostT with the given name and type from an HDF5 file.
        static member readUntyped (hdf5: HDF5) (name: string) (dataType: System.Type) =
            let gm = typeof<ArrayNDHDF>.GetMethod ("read",  BindingFlags.Public ||| BindingFlags.Static)
            let m = gm.MakeGenericMethod ([| dataType |])
            m.Invoke(null, [| box hdf5; box name |]) :?> IArrayNDHostT

        