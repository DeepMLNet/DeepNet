open Microsoft.VisualStudio.Profiler
open System
open System.IO
open System.Diagnostics
open Argu
open FSharp.Charting

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
    | [<Mandatory>] CfgDir of string
    | NoCache 
with interface IArgParserTemplate with 
        member x.Usage = 
            match x with
            | Mode _ -> "mode: train, follow"
            | CfgDir _ -> "configuration directory"         
            | NoCache -> "disables loading a Dataset.h5 cache file"


/// configuration
type CurveFollowCfg = {
    DatasetDir:         string
    MLPControllerCfg:   Controller.MLPControllerCfg    
}


let doTrain (args: ParseResults<CLIArgs>) =
    let cfgDir = args.GetResult <@ CfgDir @>
    let noCache = args.Contains <@ NoCache @>

    // load configuration
    printfn "Doing training using configuration direction %s" (Path.GetFullPath cfgDir)
    let cfg : CurveFollowCfg = Config.load cfgDir   

    // load data set
    let sw = Stopwatch.StartNew()
    let cache = Path.Combine (cfg.DatasetDir, "Dataset.h5")
    let dataset : Dataset<TactilePoint> = 
        if File.Exists cache && not noCache then
            Dataset.Load cache
        else
            let dataset = loadPoints cfg.DatasetDir |> Dataset.FromSamples 
            dataset.Save cache
            dataset
        |> Dataset.ToCuda
    printfn "Dataset %A loaded in %A" dataset sw.Elapsed   
    
    // train
    let mlpController = Controller.MLPController cfg.MLPControllerCfg
    mlpController.Train dataset

    // save
    let modelFile = Path.Combine (cfgDir, "model.h5")
    mlpController.Save modelFile
    printfn "Saved model to %s" (Path.GetFullPath modelFile)

 
let doFollow (args: ParseResults<CLIArgs>) =
    ()



[<EntryPoint>]
let main argv = 
    DataCollection.StopProfile (ProfileLevel.Global, DataCollection.CurrentId) |> ignore

    // parse command line arguments
    let parser = ArgumentParser.Create<CLIArgs>("Curve following")
    let args = parser.Parse(errorHandler=ProcessExiter())
    let mode = args.GetResult <@ Mode @>

    match mode with
    | _ when mode = "train" -> doTrain args
    | _ when mode = "follow" -> doFollow args
    | _ -> parser.Usage ("unknown mode") |> printfn "%s"

    // shutdown
    Basics.Cuda.CudaSup.shutdown ()    
    Async.Sleep 1000 |> Async.RunSynchronously
    0 
