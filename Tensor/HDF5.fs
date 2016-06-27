namespace Basics

open System
open System.IO
open System.Runtime.InteropServices
open HDF.PInvoke

 
module HDF5Support =

    let inline check retVal =
        if retVal < 0 then failwithf "HDF5 function failed" 
        else retVal

    do
        H5.``open`` () |> check |> ignore

    let hdfType<'T> =        
        if   typeof<'T> = typeof<bool> then H5T.NATIVE_UINT8
        elif typeof<'T> = typeof<int8> then H5T.NATIVE_INT8
        elif typeof<'T> = typeof<int16> then H5T.NATIVE_INT16
        elif typeof<'T> = typeof<int32> then H5T.NATIVE_INT32
        elif typeof<'T> = typeof<int64> then H5T.NATIVE_INT64
        elif typeof<'T> = typeof<uint8> then H5T.NATIVE_UINT8
        elif typeof<'T> = typeof<uint16> then H5T.NATIVE_UINT16
        elif typeof<'T> = typeof<uint32> then H5T.NATIVE_UINT32
        elif typeof<'T> = typeof<uint64> then H5T.NATIVE_UINT64
        elif typeof<'T> = typeof<single> then H5T.NATIVE_FLOAT
        elif typeof<'T> = typeof<double> then H5T.NATIVE_DOUBLE
        else failwithf "unknown type for HDF5: %A" typeof<'T>

    let hdfShape shape =
        shape |> List.map uint64 |> List.toArray

    let intShape (shape: uint64 array) =
        shape |> Array.toList |> List.map int


[<AutoOpen>]
module HDF5Types =

    open HDF5Support

    type Mode =
        /// read HDF5 file
        | HDF5Read
        /// (over-)write HDF5 file
        | HDF5Overwrite


    /// A HDF5 file.
    type HDF5 (path: string, mode: Mode) = 
        
        let mutable disposed = false

        let fileHnd =
            match mode with
            | HDF5Read -> 
                if not (File.Exists path) then 
                    raise (FileNotFoundException (sprintf "HDF5 file not found: %s" path, path))
                H5F.``open`` (path, H5F.ACC_RDONLY)
            | HDF5Overwrite -> H5F.create (path, H5F.ACC_TRUNC)
            |> check

        let checkShape data shape =
            let nElems = List.fold (*) 1 shape
            if Array.length data <> nElems then
                failwithf "shape %A does not match number of elements in data array" shape
            if List.exists ((>) 0) shape then
                failwithf "shape %A has negative elements" shape

        /// opens a HDF5 file for reading
        new (path: string) = new HDF5 (path, HDF5Read)

        /// closes the HDF5 file
        member this.Dispose () = 
            if not disposed then             
                if fileHnd >= 0 then
                    H5F.close fileHnd |> check |> ignore
                disposed <- true

        interface IDisposable with
            member this.Dispose () = this.Dispose ()

        override this.Finalize () =
            this.Dispose ()
                    
        /// opens the specified HDF5 file for reading
        static member OpenRead  path = new HDF5 (path, HDF5Read)

        /// Opens the specified HDF5 file for writing.
        /// If the file already exists it will be overwritten.
        static member OpenWrite path = new HDF5 (path, HDF5Overwrite)

        /// Splits a HDF5 path string into a list.
        static member private SplitPath (path: string) =
            path.Split('/') |> List.ofArray

        /// Combines a list of groups into a HDF5 path string.
        static member private CombinePath (dirs: string list) =
            String.concat "/" dirs
            
        /// Checks whether an object (array or group) with the given name exists.
        member this.Exists (name: string) =
            if disposed then raise (ObjectDisposedException("HDF5", "HDF5 file was previously disposed"))
            let rec exists prefix dirs =
                match dirs with
                | [] -> true
                | dir::dirs ->
                    let nextPrefix = prefix @ [dir]
                    if H5L.exists (fileHnd, HDF5.CombinePath nextPrefix) |> check <= 0 then false
                    else
                        exists nextPrefix dirs
            exists [] (HDF5.SplitPath name) 

        /// Creates the given group path. All necessary parent groups are created automatically.
        /// If the group with the given path already exists, nothing happens.
        member this.CreateGroups (path: string) =
            let rec create prefix dirs =
                match dirs with
                | [] -> ()
                | dir::dirs ->
                    let nextPrefix = prefix @ [dir]
                    let nextPrefixPath = HDF5.CombinePath nextPrefix
                    if not (this.Exists nextPrefixPath) then
                        let groupHnd = H5G.create(fileHnd, nextPrefixPath) |> check 
                        H5G.close groupHnd |> check |> ignore
                    create nextPrefix dirs
            create [] (HDF5.SplitPath path)                

        /// Create all necessary parent groups for the given path.
        member private this.CreateParentGroups (path: string) =
            match HDF5.SplitPath path with
            | [] -> ()
            | [_] -> ()
            | pl ->
                pl.[0 .. pl.Length-2]
                |> HDF5.CombinePath
                |> this.CreateGroups

        /// Write data array using specified name and shape.
        member this.Write (name: string, data: 'T array, shape: int list) =
            if disposed then raise (ObjectDisposedException("HDF5", "HDF5 file was previously disposed"))

            if mode <> HDF5Overwrite then
                failwithf "HDF5 file %s is opened for reading" path
            checkShape data shape
            this.CreateParentGroups name
            
            let typeHnd = H5T.copy hdfType<'T> |> check
            let shapeHnd = H5S.create_simple (List.length shape, hdfShape shape, hdfShape shape) |> check
            let dataHnd = H5D.create (fileHnd, name, typeHnd, shapeHnd) |> check

            let gcHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
            H5D.write (dataHnd, typeHnd, H5S.ALL, H5S.ALL, H5P.DEFAULT, gcHnd.AddrOfPinnedObject()) |> check |> ignore
            gcHnd.Free ()

            H5D.close dataHnd |> check |> ignore
            H5S.close shapeHnd |> check |> ignore
            H5T.close typeHnd |> check |> ignore

        /// Read data array using specified name. Returns tuple of data and shape.
        member this.Read<'T> (name: string) =            
            if disposed then raise (ObjectDisposedException("HDF5", "HDF5 file was previously disposed"))

            if not (this.Exists name) then
                failwithf "HDF5 dataset %s does not exist in file %s" name path
            let dataHnd = H5D.``open`` (fileHnd, name) |> check
            let typeHnd = H5D.get_type dataHnd |> check
            let shapeHnd = H5D.get_space (dataHnd) |> check

            if H5T.equal (hdfType<'T>, typeHnd) = 0 then
                failwithf "HDF5 dataset %s has other type than %A" name typeof<'T>

            if H5S.is_simple (shapeHnd) = 0 then
                failwithf "HDF5 dataset %s is not simple" name
            let nDims = H5S.get_simple_extent_ndims (shapeHnd) |> check
            let shape : uint64 array = Array.zeroCreate nDims
            let maxShape : uint64 array = Array.zeroCreate nDims
            H5S.get_simple_extent_dims(shapeHnd, shape, maxShape) |> check |> ignore
            let nElems = Array.fold (*) (uint64 1) shape |> int

            let data : 'T array = Array.zeroCreate nElems
            let gcHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
            H5D.read (dataHnd, typeHnd, H5S.ALL, H5S.ALL, H5P.DEFAULT, gcHnd.AddrOfPinnedObject()) |> check |> ignore
            gcHnd.Free ()

            H5D.close dataHnd |> check |> ignore
            H5S.close shapeHnd |> check |> ignore
            H5T.close typeHnd |> check |> ignore

            data, shape |> intShape



