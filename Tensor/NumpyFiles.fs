namespace Tensor

open System
open System.IO
open System.IO.Compression
open System.Text.RegularExpressions

open Tensor.Utils


/// methods for accessing Numpy .npy data files.
module NPYFile =

    type internal EndianT =
        | LittleEndian
        | BigEndian

    type internal NumpyTypeT =
        | Bool
        | SignedInt
        | UnsignedInt
        | Float

    /// loads a .npy file from the specified stream
    let loadFromStream (stream: Stream) name : Tensor<'T> =
        // read and check prelude
        let inline checkByte req value =
            if byte req <> byte value then failwithf "not a valid npy file header in %s" name
        stream.ReadByte () |> checkByte 0x93
        stream.ReadByte () |> checkByte 'N'
        stream.ReadByte () |> checkByte 'U'
        stream.ReadByte () |> checkByte 'M'
        stream.ReadByte () |> checkByte 'P'
        stream.ReadByte () |> checkByte 'Y'
        let verMajor = stream.ReadByte ()
        let verMinor = stream.ReadByte ()
        if verMajor <> 1 && verMinor <> 0 then
            failwithf "npy file format version %d.%d is not supported in %s" verMajor verMinor name
        let lenLo, lenHi = stream.ReadByte (), stream.ReadByte ()
        let headerLen = (lenHi <<< 8) ||| lenLo

        // read header
        let headerBytes : byte[] = Array.zeroCreate headerLen
        stream.Read(headerBytes, 0, headerLen) |> ignore
        let header = System.Text.Encoding.ASCII.GetString headerBytes
        
        // parse header
        // example header: {'descr': '<f8', 'fortran_order': True, 'shape': (23L, 2201L), }
        let headerRegex = 
            @"^{'descr': '([0-9a-z<>\|]+)', 'fortran_order': (False|True), 'shape': \(([0-9L, ]*)\), }"
        let m = Regex.Match (header, headerRegex)
        if not m.Success then failwithf "cannot parse npy header in %s: %s" name header
        let descrStr = m.Groups.[1].Value
        let fortranOrder = (m.Groups.[2].Value = "True")
        let shapeStr = m.Groups.[3].Value

        // parse data description
        let endian =
            match descrStr.[0] with
            | v when v = '<' || v = '|' -> LittleEndian
            | v when v = '>' -> BigEndian
            | _ -> failwithf "unsupported endianness \"%s\" in %s" descrStr name
        let numpyType =
            match descrStr.[1] with
            | v when v = 'b' -> Bool
            | v when v = 'i' -> SignedInt
            | v when v = 'u' -> UnsignedInt
            | v when v = 'f' -> Float
            | _ -> failwithf "Unsupported numpy data type \"%s\" in %s" descrStr name
        let typeBits = (int (descrStr.[2..])) * 8

        // parse shape
        // example shape: 23L, 2201L
        let shapeRegex = @"([\d]+)"
        let m = Regex.Match(shapeStr, shapeRegex)
        let rec extractShape (m: Match) =
            if m.Success then
                (int64 m.Value) :: extractShape (m.NextMatch())
            else []
        let shp = extractShape m
        
        // check that data format matches
        match endian with
        | LittleEndian -> ()
        | BigEndian -> failwithf "numpy big endian format is unsupported in %s" name
        match numpyType with
        | Bool when typeBits = 8 && typeof<'T> = typeof<bool> -> ()
        | SignedInt when typeBits = 8  && typeof<'T> = typeof<int8>  -> ()
        | SignedInt when typeBits = 16 && typeof<'T> = typeof<int16> -> ()
        | SignedInt when typeBits = 32 && typeof<'T> = typeof<int32> -> ()
        | SignedInt when typeBits = 64 && typeof<'T> = typeof<int64> -> ()
        | UnsignedInt when typeBits = 8  && typeof<'T> = typeof<uint8>  -> ()
        | UnsignedInt when typeBits = 16 && typeof<'T> = typeof<uint16> -> ()
        | UnsignedInt when typeBits = 32 && typeof<'T> = typeof<uint32> -> ()
        | UnsignedInt when typeBits = 64 && typeof<'T> = typeof<uint64> -> ()
        | Float when typeBits = 32 && typeof<'T> = typeof<single> -> ()
        | Float when typeBits = 64 && typeof<'T> = typeof<double> -> ()
        | t -> 
            failwithf "numpy data type \"%s\" does not match type %A or is unsupported in %s" 
                descrStr typeof<'T> name

        // create layout
        let layout =
            if fortranOrder then TensorLayout.newF shp
            else TensorLayout.newC shp
        let nElems = TensorLayout.nElems layout

        // read data
        let sizeInBytes = nElems * sizeof64<'T>
        let dataBytes : byte[] = Array.zeroCreate (int32 sizeInBytes)
        let nRead = stream.Read (dataBytes, 0, int32 sizeInBytes)
        if nRead <> int32 sizeInBytes then
            failwithf "premature end of .npy file after reading %d bytes but expecting %d bytes in %s"
                nRead sizeInBytes name

        // convert to correct data type
        let data : 'T[] = Array.zeroCreate (int32 nElems)
        Buffer.BlockCopy (dataBytes, 0, data, 0, int32 sizeInBytes)

        // create HostTensor
        Tensor<'T> (layout, TensorHostStorage<'T> data)


    /// loads a .npy file from the specified path 
    let load path = 
        use fs = File.OpenRead path
        loadFromStream fs path



/// A Numpy .npz data file.
type NPZFile (path: string) =
        
    let zipFile = ZipFile.OpenRead path

    /// path to this .npz file
    member this.Path = path

    /// returns all variable names in the .npz file
    member this.Names = [
        for entry in zipFile.Entries do
            let filename = entry.Name
            if not (filename.EndsWith ".npy") then
                failwithf "invalid zip entry %s in npz file %s" filename path
            let name = filename.[0 .. filename.Length-5]
            yield name
    ]

    /// gets the variable with the specified name from the .npz file
    member this.Get name =
        let filename = name + ".npy"
        match zipFile.GetEntry filename with
        | null -> failwithf "variable %s does not exist in %s" name path
        | entry ->
            use stream = entry.Open()
            NPYFile.loadFromStream stream (path + ":" + name)

    /// opens the specified .npz file
    static member Open path = new NPZFile (path)

    interface IDisposable with
        member this.Dispose () = zipFile.Dispose()



        

    
    

