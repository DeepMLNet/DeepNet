namespace GPAct

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
        
        // Debug options:
        //SymTensor.Debug.Timing <- true
        //SymTensor.Debug.TraceCompile <- true
        SymTensor.Debug.EnableCheckFinite <- false
        //SymTensor.Debug.PrintOptimizerStatistics <- true
        //SymTensor.Compiler.Cuda.Debug.Timing <- true
        //SymTensor.Compiler.Cuda.Debug.TraceCalls <- true
        //SymTensor.Compiler.Cuda.Debug.TraceCompile <- true
        //SymTensor.Compiler.Cuda.Debug.DebugCompile <- true
        //SymTensor.Compiler.Cuda.Debug.GenerateLineInfo <- true
        //SymTensor.Compiler.Cuda.Debug.KeepCompileDir <- true
        //SymTensor.Compiler.Cuda.Debug.DisableKernelCache <- true
        //SymTensor.Compiler.Cuda.Debug.ResourceUsage <- true
        SymTensor.Compiler.Cuda.Debug.DisableEvents <- true
        SymTensor.Compiler.Cuda.Debug.DisableStreams <- true
        //SymTensor.Compiler.Cuda.Debug.TerminateWhenNonFinite <- false
        //SymTensor.Compiler.Cuda.Debug.DumpCode <- true
        //SymTensor.Compiler.Cuda.Debug.TerminateAfterRecipeGeneration <- true
//        SymTensor.Compiler.Cuda.Debug.FastKernelMath <- true


        // parsing command line input
        let parser = ArgumentParser.Create<CLIArguments> (helpTextMessage="Trains a model using a config file.",
                                                          errorHandler = ProcessExiter())
        let results = parser.ParseCommandLine argv

        // build model
        let cfgPath = results.GetResult <@Config_Path@>
        let mi, predFn, trainFn = ConfigLoader.buildModel cfgPath
   
        // start training
        let result = trainFn ()

        // save train results
        result.Save "result.json"
        

        // shutdown
        Cuda.CudaSup.shutdown ()
        0 
