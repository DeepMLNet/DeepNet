namespace DeepNet.Utils

open System
open System.IO
open System.Linq
open System.Threading
open System.Security.Cryptography


/// a filesystem backed map for binary keys and values
type internal DiskBinaryMap (baseDir: string, keyFilename: string, valueFilename: string) =

    let sha1 = new SHA1CryptoServiceProvider()
    
    do
        if baseDir.Length = 0 then 
            invalidArg "baseDir" "baseDir cannot be empty"

    let hashDirForKey (key: byte[]) =
        let keyHash = 
            sha1.ComputeHash key 
            |> Convert.ToBase64String
            |> fun s -> [yield 'K'
                         for c in s.ToLower() do 
                             if 'a' <= c && c <= 'z' then yield c]
            |> List.toArray 
            |> String
        Path.Combine(baseDir, keyHash)
       
    let tryGetDirAndValueForKey (key: byte []) =
        try
            if Directory.Exists (hashDirForKey key) then
                Directory.EnumerateDirectories (hashDirForKey key)
                |> Seq.tryPick (fun dir ->
                    let keyPath = Path.Combine (dir, keyFilename)
                    let valuePath = Path.Combine (dir, valueFilename)
                    if File.Exists keyPath && File.Exists valuePath then
                        if key.SequenceEqual (File.ReadAllBytes keyPath) then
                            Some (dir, File.ReadAllBytes valuePath)
                        else None
                    else None)
            else None
        with :? IOException as excp ->
            printfn "DiskMap: IOException while reading: %s" excp.Message
            None

    let tryIOWrite ioFunc = 
        let rng = System.Random ()
        let rec performTry nRetries =
            try ioFunc ()
            with :? IOException as excp when nRetries < 10 ->
                printfn "DiskMap: retrying due to IOException while writing: %s" excp.Message
                Thread.Sleep 100 
                Thread.Sleep (rng.Next(100))
                performTry (nRetries + 1)
        performTry 0

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
        | Some (dir, _) -> tryIOWrite (fun () -> Directory.Delete (dir, true))
        | None -> raise (System.Collections.Generic.KeyNotFoundException())

    member this.Set key value =
        let dir =
            match tryGetDirAndValueForKey key with
            | Some (dir, _) -> dir
            | None -> Path.Combine (hashDirForKey key, Guid.NewGuid().ToString())
        let keyPath = Path.Combine (dir, keyFilename)
        let valuePath = Path.Combine (dir, valueFilename)

        tryIOWrite (fun () ->
            Directory.CreateDirectory dir |> ignore
            File.WriteAllBytes (keyPath, key)
            File.WriteAllBytes (valuePath, value))

    member this.Clear () =
        tryIOWrite (fun () -> Directory.Delete (baseDir, true))


       


