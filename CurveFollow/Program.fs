//open Microsoft.VisualStudio.Profiler
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
    let cfg : Controller.Cfg = Config.load (args.GetResult <@ Cfg @>)
    Controller.train cfg

let doPlotPredictions () =
    let cfg : Controller.Cfg = Config.load (args.GetResult <@ Cfg @>)
    let dir = args.GetResult <@ Dir @>
    Controller.plotCurvePredictions cfg dir
 
let doEvalController () =
    BRML.Drivers.Devices.init ()
    let cfg : Controller.Cfg = Config.load (args.GetResult <@ Cfg @>)
    let dir = args.GetResult <@ Dir @>
    ControllerEval.evalController cfg dir

let doMovement () =
    let cfg : Movement.GenCfg = Config.load (args.GetResult <@ Cfg @>)  
    Movement.generateMovementUsingCfg cfg

let doDistortions () =
    let cfg : ControllerEval.GenCfg = Config.load (args.GetResult <@ Cfg @>)  
    ControllerEval.generateDistortionsUsingCfg cfg

let doRecord () =
    BRML.Drivers.Devices.init ()
    let dir = args.GetResult <@ Dir @>
    Movement.recordMovements dir

let doPlotRecorded () =
    let dir = args.GetResult <@ Dir @>
    Movement.plotRecordedMovements dir


[<EntryPoint>]
let main argv = 
    //DataCollection.StopProfile (ProfileLevel.Global, DataCollection.CurrentId) |> ignore

    let mode = args.GetResult <@ Mode @>
    match mode with
    | _ when mode = "train" -> doTrain () 
    | _ when mode = "plotPredictions" -> doPlotPredictions ()
    | _ when mode = "evalController" -> doEvalController ()
    | _ when mode = "movement" -> doMovement ()
    | _ when mode = "distortions" -> doDistortions ()
    | _ when mode = "record" -> doRecord ()
    | _ when mode = "plotRecorded" -> doPlotRecorded ()
    | _ -> parser.Usage ("unknown mode") |> printfn "%s"

    // shutdown
    Basics.Cuda.CudaSup.shutdown ()    
    Async.Sleep 1000 |> Async.RunSynchronously
    0 
