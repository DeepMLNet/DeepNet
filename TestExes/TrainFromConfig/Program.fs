// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
namespace TrainFromConfig

open FSharp.Configuration
open Datasets
open Models
open Optimizers
open SymTensor
open SymTensor.Compiler.Cuda
open Argu
open Basics

type CLIArguments = 
    | Config_Path of path:string
    with
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Config_Path _ -> "specify a configuration file."


module ConfigTraining = 
    
    [<EntryPoint>]
    let main argv = 
        
        ///parsing command line input
        let parser = ArgumentParser.Create<CLIArguments>(programName = "TrainFromConfig.exe", errorHandler = ProcessExiter())
        let usage = parser.PrintUsage()
        let results = parser.ParseCommandLine argv

        let mutable configPath = __SOURCE_DIRECTORY__ +  @"\RuntimeConfig.yaml"
        let newPath = results.TryGetResult <@Config_Path@>
        match newPath with
        | Some p -> configPath <- p
        | None -> ()

        ///loading config file
        let config = TrainConfig()
        config.Load(__SOURCE_DIRECTORY__ +  @"\RuntimeConfig.yaml");
        
        ///loading dataset
        let data = dataFromConfig config

        ///creating model instance
        let pred,loss,mi,input,target = modelFromConfig config data configPath

        ///creating optimizer
        let opt, optCfg = optFromConfig config loss mi.ParameterVector

        let smplVarEnv (smpl: CsvLoader.CsvSample) =
            VarEnv.empty
            |> VarEnv.add input smpl.Input
            |> VarEnv.add target smpl.Target

        ///initialize training instance
        let trainable =
            Train.trainableFromLossExpr mi loss smplVarEnv opt optCfg
        
        /// generate taining configuration
        let trainCfg = trainConfigFromConfig config

        //start training
        let result = Train.train trainable data trainCfg
        ///save train results
        result.Save "result.json"
        Cuda.CudaSup.shutdown ()

        0 // return an integer exit code
