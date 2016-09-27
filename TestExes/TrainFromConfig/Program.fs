// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
namespace TrainFromConfig
open FSharp.Configuration
open Datasets
open Models
open Optimizers
open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open System.Text.RegularExpressions
open GPTransfer

module ConfigTraining = 
    
    [<EntryPoint>]
    let main argv = 
        let config = TrainConfig()
        config.Load(__SOURCE_DIRECTORY__ +  @"\RuntimeConfig.yaml");
        let data = dataFromConfig config
        let pred,loss,mi,input,target = modelFromConfig config data

        let opt, optCfg = optFromConfig config loss mi.ParameterVector

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg

        let trainCfg = trainConfigFromConfig config

        let result = Train.train trainable data trainCfg
        result.Save "result.json"
        0 // return an integer exit code
