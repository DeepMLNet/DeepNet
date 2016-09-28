namespace TrainFromConfig


open Datasets
open Models
open Optimizers
open SymTensor
open SymTensor.Compiler.Cuda
open Argu
open Basics


type CLIArguments = 
    | [<MainCommand; ExactlyOnce; Last>] Config_Path of path:string
    with
    interface IArgParserTemplate with
        member s.Usage =
            match s with
            |  Config_Path _ -> "configuration file"



module Main = 
    
    [<EntryPoint>]
    let main argv = 
        
        // parsing command line input
        let parser = ArgumentParser.Create<CLIArguments> (helpTextMessage="Trains a model using a config file.",
                                                          errorHandler = ProcessExiter())
        let results = parser.ParseCommandLine argv

        // build model
        let cfgPath = results.GetResult <@Config_Path@>
        let mi, prenFn, trainFn = ConfigLoader.buildModel cfgPath
   
        // start training
        let result = trainFn ()

        // save train results
        result.Save "result.json"
        

        // shutdown
        Cuda.CudaSup.shutdown ()
        0 
