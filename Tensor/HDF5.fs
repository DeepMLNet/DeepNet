namespace Basics

open System
open System.IO
open System.Runtime.InteropServices
open FSharp.Reflection
open HDF.PInvoke

 
module HDF5Support =

    let inline check retVal =
        if retVal < 0 then failwithf "HDF5 function failed" 
        else retVal

    do
        H5.``open`` () |> check |> ignore

    let hdfTypeInst t =        
        if   t = typeof<bool> then H5T.NATIVE_UINT8
        elif t = typeof<int8> then H5T.NATIVE_INT8
        elif t = typeof<int16> then H5T.NATIVE_INT16
        elif t = typeof<int32> then H5T.NATIVE_INT32
        elif t = typeof<int64> then H5T.NATIVE_INT64
        elif t = typeof<uint8> then H5T.NATIVE_UINT8
        elif t = typeof<uint16> then H5T.NATIVE_UINT16
        elif t = typeof<uint32> then H5T.NATIVE_UINT32
        elif t = typeof<uint64> then H5T.NATIVE_UINT64
        elif t = typeof<single> then H5T.NATIVE_FLOAT
        elif t = typeof<double> then H5T.NATIVE_DOUBLE
        elif t = typeof<string> then H5T.C_S1
        else failwithf "unknown type for HDF5: %A" t

    let hdfType<'T> =  
        hdfTypeInst typeof<'T>

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
            if Array.length data < nElems then
                failwithf "shape %A does not match number of elements in data array" shape
            if List.exists ((>) 0) shape then
                failwithf "shape %A has negative elements" shape

        let checkDisposed () =
            if disposed then raise (ObjectDisposedException("HDF5", "HDF5 file was previously disposed"))

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
            path.Split('/') 
            |> List.ofArray
            |> List.filter (fun d -> String.length d > 0)

        /// Combines a list of groups into a HDF5 path string.
        static member private CombinePath (dirs: string list) =
            dirs
            |> List.filter (fun d -> String.length d > 0)
            |> String.concat "/" 
            
        /// Checks whether an object (array or group) with the given name exists.
        member this.Exists (name: string) =
            checkDisposed ()
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
            checkDisposed ()
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
            if path.Length > 0 then
                create [] (HDF5.SplitPath path)                

        /// Create all necessary parent groups for the given path.
        member private this.CreateParentGroups (path: string) =
            checkDisposed ()
            match HDF5.SplitPath path with
            | [] -> ()
            | [_] -> ()
            | pl ->
                pl.[0 .. pl.Length-2]
                |> HDF5.CombinePath
                |> this.CreateGroups

        /// Write data array using specified name and shape.
        member this.Write (name: string, data: 'T array, shape: int list) =
            checkDisposed ()
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
            checkDisposed ()
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

        /// Sets the HDF5 attribute with the specified `atrName` on object specified by `name`.
        member this.SetAttribute (name: string, atrName: string, value: 'T) =
            checkDisposed ()
            if not (this.Exists name) then
                failwithf "HDF5 object %s does not exist in file %s" name path

            let elementType, data =
                if typeof<'T>.IsArray then typeof<'T>.GetElementType(), box value :?> Array
                else 
                    let ary = Array.CreateInstance(typeof<'T>, 1)
                    ary.SetValue (value, 0)
                    typeof<'T>, ary

            let typeHnd = H5T.copy (hdfTypeInst elementType) |> check
            let shapeHnd = H5S.create_simple (1, [|uint64 data.Length|], [|uint64 data.Length|]) |> check

            if H5A.exists_by_name (fileHnd, name, atrName) > 0 then
                H5A.delete_by_name (fileHnd, name, atrName) |> check |> ignore
            let atrHnd = H5A.create_by_name (fileHnd, name, atrName, typeHnd, shapeHnd) |> check

            let gcHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
            H5A.write (atrHnd, typeHnd, gcHnd.AddrOfPinnedObject()) |> check |> ignore
            gcHnd.Free ()

            H5A.close atrHnd |> check |> ignore
            H5S.close shapeHnd |> check |> ignore
            H5T.close typeHnd |> check |> ignore

        /// Gets the HDF5 attribute with the specified `name` on object specified by `path`.
        member this.GetAttribute (name: string, atrName: string) : 'T =
            checkDisposed ()
            if not (this.Exists name) then
                failwithf "HDF5 object %s does not exist in file %s" name path
            if not (H5A.exists_by_name (fileHnd, name, atrName) > 0) then
                failwithf "HDF5 attribute %s does not exist on object %s in file %s" atrName name path

            let elementType =
                if typeof<'T>.IsArray then typeof<'T>.GetElementType()
                else typeof<'T>

            let atrHnd = H5A.open_by_name (fileHnd, name, atrName) |> check
            let typeHnd = H5A.get_type atrHnd |> check
            let shapeHnd = H5A.get_space (atrHnd) |> check

            if H5T.equal (hdfTypeInst elementType, typeHnd) = 0 then
                failwithf "HDF5 attribute %s on object %s has other type than %A" atrName name elementType

            if H5S.is_simple (shapeHnd) = 0 then
                failwithf "HDF5 attribute %s on object %s is not simple" atrName name
            let nDims = H5S.get_simple_extent_ndims (shapeHnd) |> check
            if nDims <> 1 then
                failwithf "HDF5 attribute %s on object %s is not of rank 1" atrName name
            let shape : uint64 array = Array.zeroCreate nDims
            let maxShape : uint64 array = Array.zeroCreate nDims
            H5S.get_simple_extent_dims(shapeHnd, shape, maxShape) |> check |> ignore
            let nElems = shape.[0] |> int
            if nElems <> 1 && not typeof<'T>.IsArray then
                failwithf "HDF5 attribute %s on object %s has %d elements, but a scalar is expected"
                    atrName name nElems

            let data = Array.CreateInstance(elementType, nElems)
            let gcHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
            H5A.read (atrHnd, typeHnd, gcHnd.AddrOfPinnedObject()) |> check |> ignore
            gcHnd.Free()

            H5A.close atrHnd |> check |> ignore
            H5S.close shapeHnd |> check |> ignore
            H5T.close typeHnd |> check |> ignore

            if typeof<'T>.IsArray then
                box data :?> 'T
            else
                data.GetValue(0) :?> 'T

        /// Attaches the specified record as attributes to the object with `name`.
        member this.SetRecord (name: string, record: 'R) =
            if not (FSharpType.IsRecord typeof<'R>) then
                failwith "must specify a value of record type"
            for fi, value in Array.zip (FSharpType.GetRecordFields typeof<'R>) 
                                       (FSharpValue.GetRecordFields record) do
                callGenericInst<HDF5, unit> this "SetAttribute" [fi.PropertyType]
                    (name, fi.Name, value)

        /// Reads the record attached as attributes to the object with `name`.
        member this.GetRecord (name: string) : 'R =
            if not (FSharpType.IsRecord typeof<'R>) then
                failwith "must specify a value of record type"
            let values =
                FSharpType.GetRecordFields typeof<'R>
                |> Array.map (fun fi ->
                    callGenericInst<HDF5, obj> this "GetAttribute" [fi.PropertyType]
                        (name, fi.Name)
                )
            FSharpValue.MakeRecord (typeof<'R>, values) :?> 'R


            

            


