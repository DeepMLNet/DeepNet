module DatasetTests

open Xunit
open FsUnit.Xunit
open Xunit.Abstractions
open System.IO

open DeepNet.Utils
open Tensor
open Tensor.Algorithm
open Tensor.Expr.ML


let dataDir = Util.assemblyDir + "/TestData/"

type Tensorf = Tensor<float>
type CurveSample = {Time: Tensorf; Pos: Tensorf; Vels: Tensorf; Biotac: Tensorf}

let curveSamples = 
    seq {
        for filename in Directory.EnumerateFiles(dataDir + "curve/", "*.npz") |> Seq.sort do
            use tactile = NPZFile.Open filename
            yield  {Time=tactile.Get "time"
                    Pos=tactile.Get "pos"
                    Vels=tactile.Get "vels"
                    Biotac=tactile.Get "biotac"}
    } |> Seq.cache


type CurveDatasetLoading (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    [<Fact>]
    let ``Loading curve dataset`` () =
        let dataset = Dataset.ofSamples curveSamples
        printfn "Number of samples: %d" dataset.NSamples


type CurveDataset (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let dataset = Dataset.ofSamples curveSamples

    [<Fact>]
    member this.``Accessing elements`` () =
        printfn "\naccessing elements:"
        for idx, (smpl, orig) in Seq.indexed (Seq.zip dataset curveSamples) |> Seq.take 3 do
            printfn "idx %d has sample biotac %A pos %A" 
                idx (smpl.Biotac |> Tensor.shape) (smpl.Pos |> Tensor.shape)
            smpl.Biotac ==== orig.Biotac |> Tensor.all |> should equal true
            smpl.Pos ==== orig.Pos |> Tensor.all |> should equal true
            smpl.Vels ==== orig.Vels |> Tensor.all |> should equal true
            smpl.Biotac ==== orig.Biotac |> Tensor.all |> should equal true

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
        let dsCuda = dataset |> Dataset.transfer CudaTensor.Dev
        printfn "copied to CUDA: %A" dsCuda

    [<Fact>]
    member this.``Saving and loading`` () =
        dataset.Save "DatasetTests.h5"
        printfn "Saved"
        let dataset2 : Dataset<CurveSample> = Dataset.load "DatasetTests.h5"
        printfn "Loaded."



type CsvDataset (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let paths = [dataDir + "abalone.csv",      Loader.Csv.DefaultParameters 
                 dataDir + "imports-85.csv",  {Loader.Csv.DefaultParameters with Loader.Csv.IntTreatment=Loader.Csv.IntAsNumerical}
                 dataDir + "SPECT.csv",        Loader.Csv.DefaultParameters]

    [<Fact>]
    let ``Loading CSV datasets`` () =
        for path, pars in paths do
            printfn "Loading %s" path
            let data = Loader.Csv.loadFile pars path |> Seq.cache
            let ds = Dataset.ofSamples data
            printfn "%A" ds
            for smpl in data |> Seq.take 10 do
                printfn "Input: %s\nTarget: %s" smpl.Input.Full smpl.Target.Full
            printfn ""



type SeqSample = {SeqData: Tensor<int64>}

type SequenceDataset (output: ITestOutputHelper) =
    let printfn format = Printf.kprintf (fun msg -> output.WriteLine(msg)) format 

    let smpl1 = {SeqData = HostTensor.counting 98L}
    let smpl2 = {SeqData = 100L + HostTensor.counting 98L}
    let smpl3 = {SeqData = 200L + HostTensor.counting 98L}
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

