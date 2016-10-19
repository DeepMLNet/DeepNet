namespace LangRNN

open Basics
open System.IO

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Optimizers
open Datasets


module Program =



    [<EntryPoint>]
    let main argv = 
        Util.disableCrashDialog ()
        //SymTensor.Compiler.Cuda.Debug.ResourceUsage <- true
        //SymTensor.Compiler.Cuda.Debug.SyncAfterEachCudaCall <- true
        SymTensor.Compiler.Cuda.Debug.FastKernelMath <- true
        //SymTensor.Debug.VisualizeUExpr <- true
        //SymTensor.Debug.TraceCompile <- true
        //SymTensor.Debug.Timing <- true
        //SymTensor.Compiler.Cuda.Debug.Timing <- true
        //SymTensor.Compiler.Cuda.Debug.TraceCompile <- true

        // tests
        //verifyRNNGradientOneHot DevCuda
        //verifyRNNGradientIndexed DevCuda
        //TestUtils.compareTraces verifyRNNGradientIndexed false |> ignore

//        let data = WordData (dataPath      = "../../Data/reddit-comments-2015-08-tokenized.txt",
//                             vocSizeLimit  = Some 8000,
//                             stepsPerSmpl  = 20,
//                             //maxSamples    = Some 1000
//                             maxSamples    = None
//                             )

        let data = WordData (dataPath      = "../../Data/Songs/Songs",
                             vocSizeLimit  = None,
                             stepsPerSmpl  = 20,
                             maxSamples    = Some 1000
                             //maxSamples    = None
                             )

        let model = GRUTrain (VocSize      = data.VocSize,
                              EmbeddingDim = 128)

        // train model
        let trainCfg = {
            Train.defaultCfg with
                //MinIters  = Some 1000
                //MaxIters  = Some 10
                //MaxIters  = Some 1000
                LearningRates      = [1e-3; 1e-4; 1e-5]
                BatchSize          = 150
                BestOn             = Training
                CheckpointDir      = Some "."
                CheckpointInterval = Some 10
                PerformTraining    = false
        }
        model.Train data.Dataset trainCfg |> ignore

        // generate some word sequences
        printfn "Generating..."
        let NPred   = 10
        let NStart  = 30

        let rng = System.Random 123
        let allWords = data.Words |> Array.ofList
        let startIdxs = rng.Seq (0, allWords.Length-100) |> Seq.take NPred
        
        let startWords = 
            startIdxs
            |> Seq.map (fun startIdx ->
                let mutable pos = startIdx
                while allWords.[pos] <> "---" ||
                        (allWords.[pos .. pos+NStart-1] |> Array.contains "===") do
                    pos <- pos + 1
                allWords.[pos .. pos+2*NStart-1] |> List.ofArray
                )
            |> Seq.map data.Tokenize
            |> List.ofSeq
            |> ArrayNDHost.ofList2D

        let genWords = model.Generate 1001 {Words=startWords |> ArrayNDCuda.toDev}
        let genWords = genWords.Words |> ArrayNDHost.fetch
        for s=0 to NPred-1 do
            printfn "%3d: prime:     %s" s (data.ToStr startWords.[s, 0..NStart-1])
            printfn "%3d: generated: %s" s (data.ToStr genWords.[s, *])
            printfn "%3d: original:  %s" s (data.ToStr startWords.[s, NStart-1..])
            printfn ""

        // shutdown
        Cuda.CudaSup.shutdown ()
        0 





