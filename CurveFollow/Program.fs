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
    | CfgDir of string
    | Curves of string
    | NoCache 
with interface IArgParserTemplate with 
        member x.Usage = 
            match x with
            | Mode _ -> "mode: train, follow"
            | CfgDir _ -> "configuration directory"         
            | NoCache -> "disables loading a Dataset.h5 cache file"
            | Curves _ -> "directory where curve files (*.cur.npz) are stored"
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
    // configuration
    let cfgDir = args.GetResult <@ CfgDir @>
    printfn "Using configuration direction %s" (Path.GetFullPath cfgDir)
    let cfg : CurveFollowCfg = Config.load cfgDir   

    // model
    let mlpController = Controller.MLPController cfg.MLPControllerCfg

    // train
    let dataset = loadPointDataset cfg.DatasetDir
    mlpController.Train dataset

    // save
    let modelFile = Path.Combine (cfgDir, "model.h5")
    mlpController.Save modelFile
    printfn "Saved model to %s" (Path.GetFullPath modelFile)


let saveChart (path: string) (chart:FSharp.Charting.ChartTypes.GenericChart) =
    use control = new FSharp.Charting.ChartTypes.ChartControl(chart)
    control.Size <- Drawing.Size(1280, 720)
    chart.CopyAsBitmap().Save(path, Drawing.Imaging.ImageFormat.Png)


let doPlot () =
    let cfgDir = args.GetResult <@ CfgDir @>
    printfn "Using configuration direction %s" (Path.GetFullPath cfgDir)
    let cfg : CurveFollowCfg = Config.load cfgDir   

    // model
    let mlpController = Controller.MLPController cfg.MLPControllerCfg
    mlpController.Load (Path.Combine (cfgDir, "model.h5"))

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

        let filename = sprintf "curve%03d.png" idx
        chart |> saveChart (Path.Combine (cfgDir, filename))


    // need to make         
    //let posChart = Chart.Line (Seq.zip allData.[3].Time.Data allData.[3].Pos.Data) |> Chart.WithTitle "pos" 
    //let velChart = Chart.Line allData.[3].Vels.Data |> Chart.WithTitle "vels"
    //Chart.Rows [posChart; velChart]
    //|> Chart.Save (__SOURCE_DIRECTORY__ + "/chart.pdf")

    ()
    
 
let doFollow () =
    ()

let doMovement () =
    let curveDir = args.GetResult <@ Curves @> 
    let cfg = {
        Movement.Dt             = 0.01
        Movement.Accel          = 10.0 
        Movement.VelX           = 1.0 
        Movement.MaxVel         = 10.0
        Movement.MaxControlVel  = 5.0
        //Movement.Mode           = Movement.FixedOffset 0.2
        Movement.Mode = 
            Movement.Distortions {
                Movement.DistortionsPerSec = 1.0
                Movement.MaxOffset         = 0.3
                Movement.MinHold           = 0.1
                Movement.MaxHold           = 0.3            
                Movement.NotAgainFor       = 0.5
            }         
        Movement.IndentorPos    = -10.        
    }
    Movement.generateMovementForDir [cfg] curveDir


[<EntryPoint>]
let main argv = 
    DataCollection.StopProfile (ProfileLevel.Global, DataCollection.CurrentId) |> ignore

    let mode = args.GetResult <@ Mode @>
    match mode with
    | _ when mode = "train" -> doTrain () ; doPlot ()
    | _ when mode = "plot" -> doPlot ()
    | _ when mode = "follow" -> doFollow ()
    | _ when mode = "movement" -> doMovement ()
    | _ -> parser.Usage ("unknown mode") |> printfn "%s"

    // shutdown
    Basics.Cuda.CudaSup.shutdown ()    
    Async.Sleep 1000 |> Async.RunSynchronously
    0 
