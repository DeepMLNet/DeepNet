module DiskMap

open System
open System.IO
open System.Linq
open Nessos.FsPickler


let binarySerializer = FsPickler.CreateBinarySerializer()


/// a filesystem backed map for binary keys and values
type DiskBinaryMap (baseDir: string, keyFilename: string, valueFilename: string) =
    
    do
        if baseDir.Length = 0 then 
            invalidArg "baseDir" "baseDir cannot be empty"

    let stableHash data =
        // TODO: is this stable?
        hash data

    let hashDirForKey key =
        Path.Combine(baseDir, sprintf "%016x" (stableHash key))

    let newGuidStr () =
        Guid.NewGuid().ToString()

    let tryGetDirAndValueForKey (key: byte []) =
        if Directory.Exists (hashDirForKey key) then
            Directory.EnumerateDirectories (hashDirForKey key)
            |> Seq.tryPick (fun dir ->
                let keyPath = Path.Combine (dir, keyFilename)
                let valuePath = Path.Combine (dir, valueFilename)
                if File.Exists keyPath && File.Exists valuePath then
                    if key.SequenceEqual (File.ReadAllBytes valuePath) then
                        Some (dir, File.ReadAllBytes valuePath)
                    else None
                else None)
        else None

    new (baseDir: string) = DiskBinaryMap(baseDir, "key.dat", "value.dat")
        
    member this.TryGet key =
        match tryGetDirAndValueForKey key with
        | Some (_, value) -> Some value
        | None -> None
        
    member this.Get key =
        match this.TryGet key with
        | Some value -> value
        | None -> raise (System.Collections.Generic.KeyNotFoundException())

    member this.Remove key =
        match tryGetDirAndValueForKey key with
        | Some (dir, _) -> Directory.Delete (dir, true)
        | None -> raise (System.Collections.Generic.KeyNotFoundException())

    member this.Set key value =
        let dir =
            match tryGetDirAndValueForKey key with
            | Some (dir, _) -> dir
            | None -> Path.Combine (hashDirForKey key, newGuidStr ())
        Directory.CreateDirectory dir |> ignore

        let keyPath = Path.Combine (dir, keyFilename)
        let valuePath = Path.Combine (dir, valueFilename)
        File.WriteAllBytes (keyPath, key)
        File.WriteAllBytes (valuePath, value)

    member this.Clear () =
        Directory.Delete (baseDir, true)


/// a filesystem backed map for arbitrary keys and values
type DiskMap<'TKey, 'TValue> (baseDir: string, keyFilename: string, valueFilename: string) =
    
    let binaryMap = DiskBinaryMap(baseDir, keyFilename, valueFilename)

    let toBinary data =
        binarySerializer.Pickle data

    let toKey binData =
        binarySerializer.UnPickle<'TKey> binData

    let toValue binData =
        binarySerializer.UnPickle<'TValue> binData

    member this.TryGet key =
        match binaryMap.TryGet (toBinary key) with
        | Some binValue -> Some (toValue binValue)
        | None -> None

    member this.Get key =
        toValue (binaryMap.Get (toBinary key))

    member this.Remove key =
        binaryMap.Remove (toBinary key)

    member this.Set key value =
        binaryMap.Set (toBinary key) (toBinary value)
    
    member this.Clear () =
        binaryMap.Clear ()

       


