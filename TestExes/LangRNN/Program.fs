namespace LangRNN

open System.IO
open Argu

open Basics
open Tensor
open Models


module Program =

    type CLIArgs = 
        | Generate of int 
        | Train
        | Slack of string
        | TokenLimit of int
        | MaxIters of int
        | BatchSize of int64
        | [<Mandatory>] Data of string
        | Steps of int64
        | Hiddens of int64
        | CheckpointInterval of int
        | DropState of float
        | PrintSamples
        | MultiStepLoss
        | UseChars
        with
        interface IArgParserTemplate with
            member s.Usage =
                match s with
                | Generate _ -> "generates samples from trained model using the specified seed"
                | Train -> "train model"
                | Slack _ -> "connect as a slack bot using the specified key"
                | TokenLimit _ -> "limits the number of training tokens"
                | MaxIters _ -> "limits the number of training epochs"
                | BatchSize _ -> "training batch size"
                | Data _ -> "path to data file"
                | Steps _ -> "number of steps to back-propagate gradient for"
                | Hiddens _ -> "number of hidden units"
                | CheckpointInterval _ -> "number of epochs between writing checkpoint"
                | DropState _ -> "probability of setting latent state to zero at the start of a mini-batch"
                | PrintSamples -> "prints some samples from the training set"
                | MultiStepLoss -> "use multi-step loss"
                | UseChars -> "uses chars as tokens (instead of words)"

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
        CudaSup.setContext ()

        // tests
        //verifyRNNGradientOneHot DevCuda
        //verifyRNNGradientIndexed DevCuda
        //TestUtils.compareTraces verifyRNNGradientIndexed false |> ignore
        //exit 0

        let parser = ArgumentParser.Create<CLIArgs> (helpTextMessage="Language learning RNN",
                                                     errorHandler = ProcessExiter())
        let args = parser.ParseCommandLine argv
        let batchSize = args.GetResult (<@BatchSize@>, 250L)
        let stepsPerSmpl = args.GetResult (<@Steps@>, 25L)
        let embeddingDim = args.GetResult (<@Hiddens@>, 128L)
        let checkpointInterval = args.GetResult (<@CheckpointInterval@>, 10)
        let dropState = args.GetResult (<@DropState@>, 0.0)
        let multiStepLoss = args.Contains <@MultiStepLoss@>

        // load data
        let data = WordData (dataPath      = args.GetResult <@Data@>,
                             vocSizeLimit  = None,
                             stepsPerSmpl  = stepsPerSmpl,
                             minSamples    = int64 (float batchSize / 0.90),
                             tokenLimit    = args.TryGetResult <@TokenLimit@>,
                             useChars      = args.Contains <@UseChars@>)

        // instantiate model
        let model = GRUInst (VocSize       = int64 data.VocSize,
                             EmbeddingDim  = embeddingDim,
                             MultiStepLoss = multiStepLoss)

        // output some training samples
        if args.Contains <@PrintSamples@> then
            for smpl in 0L .. 3L do
                for i, s in Seq.indexed (data.Dataset.Trn.SlotBatches batchSize stepsPerSmpl) do
                    let words = s.Words.[smpl, *] |> data.ToStr
                    printfn "Batch %d, sample %d:\n%s\n" i smpl words

        // train model or load checkpoint
        printfn "Training with %d steps per slot" stepsPerSmpl
        let trainCfg = {
            Train.defaultCfg with
                MinIters           = Some 150
                MaxIters           = args.TryGetResult <@ MaxIters @>
                LearningRates      = [1e-2; 1e-3; 1e-4; 1e-5; 1e-6]
                //LearningRates      = [1e-3; 1e-4; 1e-5; 1e-6]
                //LearningRates      = [1e-4; 1e-5; 1e-6]
                BatchSize          = System.Int64.MaxValue
                SlotSize           = Some stepsPerSmpl
                BestOn             = Training
                CheckpointFile     = Some "LangRNN-%ITER%.h5"
                CheckpointInterval = Some checkpointInterval
                PerformTraining    = args.Contains <@Train@>
        }
        model.Train data.Dataset dropState trainCfg |> ignore

        // generate some word sequences
        match args.TryGetResult <@Generate@> with
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
                |> HostTensor.ofList2D

            let genWords = model.Generate 1001 {Words=startWords |> CudaTensor.transfer}
            let genWords = genWords.Words |> HostTensor.transfer
            for s in 0 .. NPred-1 do
                printfn "======================= Sample %d ====================================" s
                printfn "====> prime:      \n%s" (data.ToStr startWords.[int64 s, 0L .. int64 NStart-1L])
                printfn "\n====> generated:\n> %s" (data.ToStr genWords.[int64 s, *])
                printfn "\n====> original: \n> %s" (data.ToStr startWords.[int64 s, int64 NStart ..])
                printfn ""
        | None -> ()

        // slack bot
        match args.TryGetResult <@Slack@> with
        | Some slackKey -> 
            let bot = SlackBot (data, model, slackKey)
            printfn "\nSlackBot is connected. Press Ctrl+C to quit."
            while true do
               Async.Sleep 10000 |> Async.RunSynchronously
        | None -> ()

        // shutdown
        CudaSup.shutdown ()
        0 


