namespace DeepNet.Utils

open System
open System.IO
open System.Linq
open MBrace.FsPickler
open System.Threading
open System.Security.Cryptography



/// a filesystem backed map for arbitrary keys and values
type DiskMap<'TKey, 'TValue> (baseDir: string, keyFilename: string, valueFilename: string) =
    
    static let binarySerializer = FsPickler.CreateBinarySerializer()
    static let sha1 = new SHA1CryptoServiceProvider()

    let binaryMap = DiskBinaryMap(baseDir, keyFilename, valueFilename)

    let toBinary data =
        match box data with
        | :? (byte[]) as binData -> binData
        | _ -> binarySerializer.Pickle data

    let toKey binData : 'TKey =
        if typeof<'TKey>.Equals(typeof<byte[]>) then box binData :?> 'TKey
        else binarySerializer.UnPickle<'TKey> binData

    let toValue binData : 'TValue =
        if typeof<'TValue>.Equals(typeof<byte[]>) then box binData :?> 'TValue
        else binarySerializer.UnPickle<'TValue> binData

    member this.TryGet (key: 'TKey) =
        match binaryMap.TryGet (toBinary key) with
        | Some binValue -> Some (toValue binValue)
        | None -> None

    member this.Get (key: 'TKey) =
        toValue (binaryMap.Get (toBinary key))

    member this.Remove (key: 'TKey) =
        binaryMap.Remove (toBinary key)

    member this.Set (key: 'TKey) (value: 'TValue) =
        binaryMap.Set (toBinary key) (toBinary value)
    
    member this.Clear () =
        binaryMap.Clear ()

       


