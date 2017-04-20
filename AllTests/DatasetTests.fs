module DatasetTests

open Xunit
open FsUnit.Xunit
open System.IO

open Basics
open ArrayNDNS
open Datasets


let dataDir = Util.assemblyDirectory + "/../../TestData/curve"

type Arrayf = ArrayNDT<float>
type CurveSample = {Time: Arrayf; Pos: Arrayf; Vels: Arrayf; Biotac: Arrayf}

let dataSamples = 
    seq {
        for filename in Directory.EnumerateFiles(dataDir, "*.npz") |> Seq.sort do
            use tactile = NPZFile.Open filename
            yield  {Time=tactile.Get "time"
                    Pos=tactile.Get "pos"
                    Vels=tactile.Get "vels"
                    Biotac=tactile.Get "biotac"}
    } |> Seq.cache

[<Fact>]
let ``Loading curve dataset`` () =
    let dataset = Dataset.ofSamples dataSamples
    printfn "Number of samples: %d" dataset.NSamples

type CurveDataset () =
    let dataset = Dataset.ofSamples dataSamples

    [<Fact>]
    member this.``Accessing elements`` () =
        printfn "\naccessing elements:"
        for idx, (smpl, orig) in Seq.indexed (Seq.zip dataset dataSamples) |> Seq.take 3 do
            printfn "idx %d has sample biotac %A pos %A" 
                idx (smpl.Biotac |> Tensor.shape) (smpl.Pos |> Tensor.shape)
            smpl.Biotac ==== orig.Biotac |> Tensor.all |> Tensor.value |> should equal true
            smpl.Pos ==== orig.Pos |> Tensor.all |> Tensor.value |> should equal true
            smpl.Vels ==== orig.Vels |> Tensor.all |> Tensor.value |> should equal true
            smpl.Biotac ==== orig.Biotac |> Tensor.all |> Tensor.value |> should equal true

    [<Fact>]
    member this.``Partitioning`` () =
        let parts = TrnValTst.ofDataset dataset
        printfn "Training   set size: %d" parts.Trn.NSamples
        printfn "Validation set size: %d" parts.Val.NSamples
        printfn "Test       set size: %d" parts.Tst.NSamples
        
        parts.Trn.NSamples + parts.Val.NSamples + parts.Tst.NSamples 
        |> should equal dataset.NSamples
    
    [<Fact>]
    member this.``Batching`` () =
        let batchSize = 11L
        let batchGen = dataset.PaddedBatches batchSize
        printfn "\nbatching:"
        for idx, batch in Seq.indexed (batchGen()) do
            printfn "batch: %d has biotac: %A;  pos: %A" 
                idx (batch.Biotac |> Tensor.shape) (batch.Pos |> Tensor.shape)
            batch.Biotac |> Tensor.shape |> List.head |> should equal batchSize

    [<Fact>]
    [<Trait("Category", "Skip_CI")>]
    member this.``To CUDA GPU`` () =
        let dsCuda = dataset |> Dataset.toCuda
        printfn "copied to CUDA: %A" dsCuda

    [<Fact>]
    member this.``Saving and loading`` () =
        dataset.Save "DatasetTests.h5"
        printfn "Saved"
        let dataset2 : Dataset<CurveSample> = Dataset.Load "DatasetTests.h5"
        printfn "Loaded."




[<Fact>]
let ``Loading CSV datasets`` () =
    
    let paths = ["abalone.txt",       CsvLoader.DefaultParameters 
                 //"arrhythmia.txt.gz", {CsvLoader.DefaultParameters 
                 //                      with CsvLoader.IntTreatment=CsvLoader.IntAsNumerical}
                 "imports-85.data",   {CsvLoader.DefaultParameters 
                                       with CsvLoader.IntTreatment=CsvLoader.IntAsNumerical}
                 "SPECT.txt",         CsvLoader.DefaultParameters]
    for path, pars in paths do
        printfn "Loading %s" path
        let data = CsvLoader.loadFile pars path |> Seq.cache
        let ds = Dataset.ofSamples data
        printfn "%A" ds
        for smpl in data |> Seq.take 10 do
            printfn "Input: %s\nTarget: %s" smpl.Input.Full smpl.Target.Full
        printfn ""


type SeqSample = {SeqData: ArrayNDT<int64>}

type SequenceDataset () = 
    let smpl1 = {SeqData = ArrayNDHost.arange 98L}
    let smpl2 = {SeqData = 100L + ArrayNDHost.arange 98L}
    let smpl3 = {SeqData = 200L + ArrayNDHost.arange 98L}
    let dataset = Dataset.ofSeqSamples [smpl1; smpl2; smpl3]

    [<Fact>]
    member this.``Slot batches`` () =
        printfn "Dataset: %A" dataset
        for idx, sb in Seq.indexed (dataset.SlotBatches 2L 4L) do
            printfn "Slotbatch %d:" idx
            printfn "%A" sb.SeqData
        printfn ""

    [<Fact>]
    member this.``Cut sequence`` () =
        let minSmpls = 15L
        printfn "Original: %A" dataset
        printfn "cutting to minimum of %d samples" minSmpls
        let ds = dataset |> Dataset.cutToMinSamples minSmpls
        printfn "Cut: %A" ds
        for idx, sb in Seq.indexed (ds.SlotBatches 2L 4L) do
            printfn "Slotbatch %d:" idx
            printfn "%A" sb.SeqData
        printfn ""

