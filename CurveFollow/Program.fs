open Microsoft.VisualStudio.Profiler
open System
open System.IO
open System.Diagnostics
open Argu
open FSharp.Charting

open Basics
open ArrayNDNS
open Datasets
open SymTensor
open SymTensor.Compiler.Cuda
open Optimizers
open Models
open Data


/// command line arguments
type CLIArgs =
    | [<Mandatory>] Mode of string
    | Cfg of string
    | Dir of string
    | NoCache 
with interface IArgParserTemplate with 
        member x.Usage = 
            match x with
            | Mode _ -> "mode: train, follow"
            | Cfg _ -> "configuration file"         
            | Dir _ -> "data (movments) directory"
            | NoCache -> "disables loading a Dataset.h5 cache file"
let parser = ArgumentParser.Create<CLIArgs>("Curve following")
let args = parser.Parse(errorHandler=ProcessExiter())

// configuration
type CurveFollowCfg = {
    DatasetDir:         string
    MLPControllerCfg:   Controller.MLPControllerCfg    
}


/// loads the dataset specified in the configuration
let loadPointDataset datasetDir =
    let noCache = args.Contains <@ NoCache @>
    let sw = Stopwatch.StartNew()
    let cache = Path.Combine (datasetDir, "PointDataset.h5")
    let dataset : Dataset<TactilePoint> = 
        if File.Exists cache && not noCache then
            Dataset.Load cache
        else
            let dataset = loadPoints datasetDir |> Dataset.FromSamples 
            dataset.Save cache
            dataset
        |> Dataset.ToCuda
    printfn "Point %A loaded in %A" dataset sw.Elapsed   
    dataset

/// loads the dataset specified in the configuration
let loadCurveDataset datasetDir =
    let noCache = args.Contains <@ NoCache @>
    let sw = Stopwatch.StartNew()
    let cache = Path.Combine (datasetDir, "CurveDataset.h5")
    let dataset : Dataset<TactileCurve> = 
        if File.Exists cache && not noCache then
            Dataset.Load cache
        else
            let dataset = loadCurves datasetDir |> Dataset.FromSamples 
            dataset.Save cache
            dataset
        |> Dataset.ToCuda
    printfn "Curve %A loaded in %A" dataset sw.Elapsed   
    dataset


let doTrain () =  
    let cfg : CurveFollowCfg = Config.loadAndChdir (args.GetResult <@ Cfg @>)

    let mlpController = Controller.MLPController cfg.MLPControllerCfg
    let dataset = loadPointDataset cfg.DatasetDir
    mlpController.Train dataset
    mlpController.Save "model.h5"


let saveChart (path: string) (chart:FSharp.Charting.ChartTypes.GenericChart) =
    use control = new FSharp.Charting.ChartTypes.ChartControl(chart)
    control.Size <- Drawing.Size(1280, 720)
    chart.CopyAsBitmap().Save(path, Drawing.Imaging.ImageFormat.Png)


let doPlot () =
    let cfg : CurveFollowCfg = Config.loadAndChdir (args.GetResult <@ Cfg @>)   

    let mlpController = Controller.MLPController cfg.MLPControllerCfg
    mlpController.Load "model.h5"

    let dataset = loadCurveDataset cfg.DatasetDir   
    for idx, smpl in Seq.indexed (dataset |> Seq.take 10) do
        let predVel = mlpController.Predict smpl.Biotac

        let posChart = 
            Chart.Line (Seq.zip smpl.Time smpl.DrivenPos.[*, 1])
            |> Chart.WithXAxis (Title="Time")
            |> Chart.WithYAxis (Title="Position", Min=0.0, Max=12.0)
        
        let controlCharts =
            Chart.Combine(
                [Chart.Line (Seq.zip smpl.Time predVel.[*, 1], Name="pred")
                 Chart.Line (Seq.zip smpl.Time smpl.OptimalVel.[*, 1], Name="optimal")
                 Chart.Line (Seq.zip smpl.Time smpl.DrivenVel.[*, 1], Name="driven") ])
            |> Chart.WithXAxis (Title="Time")
            |> Chart.WithYAxis (Title="Velocity", Min=(-2.0), Max=2.0)
            |> Chart.WithLegend ()

        let chart = Chart.Rows [posChart; controlCharts]
        chart |> saveChart (sprintf "curve%03d.png" idx)   
 
let doFollow () =
    ()



let doMovement () =
    let cfg : Movement.GenCfg = Config.loadAndChdir (args.GetResult <@ Cfg @>)  
    Movement.generateMovementUsingCfg cfg

let doRecord () =
    let dir = args.GetResult <@ Dir @>
    Movement.recordMovements dir


[<EntryPoint>]
let main argv = 
    DataCollection.StopProfile (ProfileLevel.Global, DataCollection.CurrentId) |> ignore

    let mode = args.GetResult <@ Mode @>
    match mode with
    | _ when mode = "train" -> doTrain () ; doPlot ()
    | _ when mode = "plot" -> doPlot ()
    | _ when mode = "follow" -> doFollow ()
    | _ when mode = "movement" -> doMovement ()
    | _ when mode = "record" -> doRecord ()
    | _ -> parser.Usage ("unknown mode") |> printfn "%s"

    // shutdown
    Basics.Cuda.CudaSup.shutdown ()    
    Async.Sleep 1000 |> Async.RunSynchronously
    0 
