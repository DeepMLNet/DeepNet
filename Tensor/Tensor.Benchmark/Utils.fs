namespace Tensor.Benchmark

open System
open System.Reflection
open System.IO
open System.Collections.Generic

open Tensor


type Cache (name: string) =
    let cacheDir = Path.Combine(Path.GetTempPath(), "TensorBenchmarkCache")
    do if not (Directory.Exists cacheDir) then Directory.CreateDirectory cacheDir |> ignore
    let path = (Path.Combine (cacheDir, name) + ".h5") |> Path.GetFullPath
    //do printfn "Cache %s is at %s" name path
    let onDisk = if File.Exists path then Some (HDF5.OpenRead path) else None
    let contents = Dictionary<string, ITensor> ()
    let mutable changed = false

    member __.Get name createFn =
        match onDisk with
        | _ when contents.ContainsKey name -> 
            //printfn "Cache hit: %s" name
            contents.[name] :?> Tensor<_>
        | Some d when d.Exists name -> 
            //printfn "Cache hit: %s" name
            let v = HostTensor.read d name
            contents.[name] <- v
            v
        | _ -> 
            //printfn "Cache miss: %s" name
            changed <- true
            let v = createFn ()
            contents.[name] <- v 
            v

    member __.Dispose () =
        match onDisk with
        | Some d -> d.Dispose ()
        | _ -> ()
        if changed then
            use toDisk = HDF5.OpenWrite path
            for KeyValue (name, data) in contents do
                HostTensor.write toDisk name data

    interface IDisposable with
        member this.Dispose () = this.Dispose ()
