module DatasetTests

open Xunit
open FsUnit.Xunit
open System.IO

open Basics
open ArrayNDNS
open Datasets


let dataDir = Util.assemblyDirectory + "/../../TestData/curve"

type Arrayf = ArrayNDHostT<float>
type CurveSample = {Time: Arrayf; Pos: Arrayf; Vels: Arrayf; Biotac: Arrayf}

let dataSamples = 
    seq {
        for filename in Directory.EnumerateFiles(dataDir, "*.npz") do
            use tactile = NPZFile.Open filename
            yield  {Time=tactile.Get "time"
                    Pos=tactile.Get "pos"
                    Vels=tactile.Get "vels"
                    Biotac=tactile.Get "biotac"}
    } |> Seq.cache

[<Fact>]
let ``Loading curve dataset`` () =
    let dataset = Dataset.FromSamples dataSamples
    printfn "Number of samples: %d" dataset.NSamples

type CurveDataset () =
    let dataset = Dataset.FromSamples dataSamples

    [<Fact>]
    member this.``Accessing elements`` () =
        printfn "\naccessing elements:"
        for idx, (smpl, orig) in Seq.indexed (Seq.zip dataset dataSamples) |> Seq.take 3 do
            printfn "idx %d has sample biotac %A pos %A" 
                idx (smpl.Biotac |> ArrayND.shape) (smpl.Pos |> ArrayND.shape)
            smpl.Biotac ==== orig.Biotac |> ArrayND.all |> ArrayND.value |> should equal true
            smpl.Pos ==== orig.Pos |> ArrayND.all |> ArrayND.value |> should equal true
            smpl.Vels ==== orig.Vels |> ArrayND.all |> ArrayND.value |> should equal true
            smpl.Biotac ==== orig.Biotac |> ArrayND.all |> ArrayND.value |> should equal true

    [<Fact>]
    member this.``Partitioning`` () =
        let parts = TrnValTst.Of dataset
        printfn "Training   set size: %d" parts.Trn.NSamples
        printfn "Validation set size: %d" parts.Val.NSamples
        printfn "Test       set size: %d" parts.Tst.NSamples
        
        parts.Trn.NSamples + parts.Val.NSamples + parts.Tst.NSamples 
        |> should equal dataset.NSamples
    
    [<Fact>]
    member this.``Batching`` () =
        let batchSize = 10
        let batchGen = dataset.Batches batchSize
        printfn "\nbatching:"
        for idx, batch in Seq.indexed (batchGen()) do
            printfn "batch: %d has biotac: %A;  pos: %A" 
                idx (batch.Biotac |> ArrayND.shape) (batch.Pos |> ArrayND.shape)
            batch.Biotac |> ArrayND.shape |> List.last |> should equal batchSize

    [<Fact>]
    member this.``To CUDA GPU`` () =
        let dsCuda = dataset.ToCuda()
        printfn "copied to CUDA: %A" dsCuda

    [<Fact>]
    member this.``Saving and loading`` () =
        dataset.Save "DatasetTests.h5"
        printfn "Saved"
        let dataset2 : Dataset<CurveSample> = Dataset.Load "DatasetTests.h5"
        printfn "Loaded."
