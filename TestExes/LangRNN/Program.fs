namespace LangRNN

open System.IO
open Argu

open Basics
open ArrayNDNS
open Models


module Program =

    type CLIArgs = 
        | Generate of int 
        | Train
        | Slack of string
        | MaxSamples of int
        | MaxIters of int
        with
        interface IArgParserTemplate with
            member s.Usage =
                match s with
                | Generate _ -> "generates samples from trained model using the specified seed"
                | Train -> "train model"
                | Slack _ -> "connect as a slack bot using the specified key"
                | MaxSamples _ -> "limits the number of training samples"
                | MaxIters _ -> "limits the number of training epochs"

    [<EntryPoint>]
    let main argv = 
        // debug
        Util.disableCrashDialog ()
        //SymTensor.Compiler.Cuda.Debug.ResourceUsage <- true
        //SymTensor.Compiler.Cuda.Debug.SyncAfterEachCudaCall <- true
        SymTensor.Compiler.Cuda.Debug.FastKernelMath <- true
        //SymTensor.Debug.VisualizeUExpr <- true
        //SymTensor.Debug.TraceCompile <- true
        //SymTensor.Debug.Timing <- true
        //SymTensor.Compiler.Cuda.Debug.Timing <- true
        //SymTensor.Compiler.Cuda.Debug.TraceCompile <- true

        // required for SlackBot
        Cuda.CudaSup.setContext ()

        // tests
        //verifyRNNGradientOneHot DevCuda
        //verifyRNNGradientIndexed DevCuda
        //TestUtils.compareTraces verifyRNNGradientIndexed false |> ignore
        //exit 0

        let parser = ArgumentParser.Create<CLIArgs> (helpTextMessage="Language learning RNN",
                                                     errorHandler = ProcessExiter())
        let args = parser.ParseCommandLine argv

        // load data
//        let data = WordData (dataPath      = "../../Data/Songs.txt",
//                             vocSizeLimit  = None,
//                             stepsPerSmpl  = 25,
//                             maxSamples    = args.TryGetResult <@ MaxSamples @>,
//                             useChars      = false
//                             )
        let data = WordData (dataPath      = "../../Data/Gutenberg10000000.txt",
                             vocSizeLimit  = None,
                             //stepsPerSmpl  = 25,
                             stepsPerSmpl  = 50,
                             maxSamples    = args.TryGetResult <@ MaxSamples @>,
                             useChars      = true
                             )



        let model = GRUTrain (VocSize      = data.VocSize,
                              EmbeddingDim = 181)
                              //EmbeddingDim = 128)

        // train model or load checkpoint
        let trainCfg = {
            Train.defaultCfg with
                MinIters           = args.TryGetResult <@ MaxIters @>
                LearningRates      = [1e-3; 1e-4; 1e-5; 1e-6]
                //BatchSize          = 150
                BatchSize          = 200
                BestOn             = Training
                CheckpointDir      = Some "."
                CheckpointInterval = Some 1
                //CheckpointInterval = Some 10
                PerformTraining    = args.Contains <@ Train @>
        }
        model.Train data.Dataset 0.02 trainCfg |> ignore
        //model.Train data.Dataset 0.1 trainCfg |> ignore

        // generate some word sequences
        match args.TryGetResult <@ Generate @> with
        | Some seed ->
            printfn "Generating..."
            let NStart  = 30
            let NPred   = 20

            let rng = System.Random seed
            let allWords = data.Words |> Array.ofList
            let startIdxs = rng.Seq (0, allWords.Length-100) |> Seq.take NPred
        
            let startWords = 
                startIdxs
                |> Seq.map (fun startIdx ->
                    let mutable pos = startIdx
                    if not data.UseChars then
                        while pos+2*NStart >= allWords.Length || 
                              allWords.[pos+NStart-1] <> ">" ||
                              (allWords.[pos .. pos+NStart-1] |> Array.contains "===") do
                            pos <- pos + 1
                            if pos >= allWords.Length then pos <- 0
                    allWords.[pos .. pos+2*NStart-1] |> List.ofArray
                    )
                |> Seq.map data.Tokenize
                |> List.ofSeq
                |> ArrayNDHost.ofList2D

            let genWords = model.Generate 1001 {Words=startWords |> ArrayNDCuda.toDev}
            let genWords = genWords.Words |> ArrayNDHost.fetch
            for s=0 to NPred-1 do
                printfn "======================= Sample %d ====================================" s
                printfn "====> prime:      \n%s" (data.ToStr startWords.[s, 0..NStart-1])
                printfn "\n====> generated:\n> %s" (data.ToStr genWords.[s, *])
                printfn "\n====> original: \n> %s" (data.ToStr startWords.[s, NStart..])
                printfn ""
        | None -> ()

        // slack bot
        match args.TryGetResult <@ Slack @> with
        | Some slackKey -> 
            let bot = SlackBot (data, model, slackKey)
            printfn "\nSlackBot is connected. Press Ctrl+C to quit."
            while true do
               Async.Sleep 10000 |> Async.RunSynchronously
        | None -> ()

        // shutdown
        Cuda.CudaSup.shutdown ()
        0 





