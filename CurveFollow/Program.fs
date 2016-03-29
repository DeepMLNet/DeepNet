open Microsoft.VisualStudio.Profiler
open System
open System.IO
open Argu
open FSharp.Charting
open System.Diagnostics

open ArrayNDNS
open Datasets
open Data


[<EntryPoint>]
let main argv = 
    DataCollection.StopProfile (ProfileLevel.Global, DataCollection.CurrentId) |> ignore

    let srcDir = __SOURCE_DIRECTORY__ + "/../Data/DeepBraille/curv2"

    let sw = Stopwatch.StartNew()
    let allCurves = loadCurves srcDir
    printfn "Curve loading took %A" sw.Elapsed

    let sw = Stopwatch.StartNew()
    let allPoints = loadPoints srcDir |> Seq.toList
    printfn "Point loading took %A" sw.Elapsed
    printfn "number of points: %d" (List.length allPoints)

//    DataCollection.StartProfile (ProfileLevel.Global, DataCollection.CurrentId) |> ignore
//    let sw = Stopwatch.StartNew()
//    let dataset = allPoints |> Dataset.FromSamples
//    printfn "Dataset building took %A" sw.Elapsed
//    dataset.Save (srcDir + "/dataset.h5")
//    //DataCollection.StopProfile (ProfileLevel.Global, DataCollection.CurrentId) |> ignore

    let sw = Stopwatch.StartNew()
    let datasetLoad : Dataset<TactilePoint> = Dataset.Load (srcDir + "/dataset.h5")
    printfn "Dataset loading from HDF5 took %A" sw.Elapsed

    0 
