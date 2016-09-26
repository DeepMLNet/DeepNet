// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
namespace GPTransfer
open FSharp.Configuration
open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor

module ConfigTraining = 
    type TrainConfig = YamlConfig<"Config.yaml">

    let dataFromConfig (config:TrainConfig)=
        let pars = {CsvLoader.DefaultParameters with CsvLoader.TargetCols = config.Data.TargetCols |> List.map (fun x -> int x)}
        let fullData = CsvLoader.loadFile pars config.Data.Path
        let fullDataset = Dataset.FromSamples fullData
        let splitDataHost = TrnValTst.Of(fullDataset)
        match config.Training.Device with
        | DevCuda -> splitDataHost.ToCuda()
        | _ -> splitDataHost
    [<EntryPoint>]
    let main argv = 
        let config = TrainConfig()
        let netName = config.Network.Name
        printfn "NetworkName = %s" netName
        0 // return an integer exit code
