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

    let dataPath = "../../Data/reddit-comments-2015-08-tokenized.txt"
//    let NWords      = 8000
//    let NBatch      = 20
//    let NSteps      = 35
//    let NRecurrent  = 650
    
//    let NWords      = 7
//    let NWords      = 100
//    let NBatch      = 1000
//    let NSteps      = 20
//    let NRecurrent  = 100
//    let NMaxSamples = 10000

//    let NWords      = 8000
//    let NBatch      = 1000
//    let NSteps      = 35
//    let NRecurrent  = 100
//    let NMaxSamples = 10000


    let NWords      = 5
    let NSteps      = 7
    let NBatch      = 10
    let NMaxSamples = 10

    let NRecurrent  = 8


    type WordSeq = {
        Words:  ArrayNDT<int>
    }

    type WordSeqOneHot = {
        Words:  ArrayNDT<single>
    }
    
    let readData path = 
        seq {
            for line in File.ReadLines path do
                let words = line.Split ([|' '|]) |> List.ofArray
                yield words
        }

    let wordFreqs words =
        let freqs = Dictionary<string, int> ()
        for word in words do
            freqs.[word] <- (freqs.GetOrDefault word 0) + 1
        Map.ofDictionary freqs

    let tokenize idForWord sentences =
        let nWords = idForWord |> Map.toSeq |> Seq.length        
        sentences |> Seq.map (List.map (fun word ->
            match idForWord |> Map.tryFind word with
            | Some id -> id
            | None -> nWords
        ))

    let detokenize (wordForId: Map<int, string>) tokenizedSentences =
        tokenizedSentences |> Seq.map (List.map (fun id -> wordForId.[id]))


    let verifyRNNGradientOneHot () =
        printfn "Verifying RNN gradient with one-hot class encoding..."
        let device = DevHost

        let mb = ModelBuilder<single> ("Lang")
        let nBatch     = mb.Size "nBatch"
        let nSteps     = mb.Size "nSteps"
        let nWords     = mb.Size "nWords"
        let nRecurrent = mb.Size "nRecurrent"

        let input   = mb.Var<single>  "Input"   [nBatch; nSteps; nWords]
        let initial = mb.Var<single>  "Initial" [nBatch; nRecurrent]
        let target  = mb.Var<single>  "Target"  [nBatch; nSteps; nWords]
        
        let rnn = RecurrentLayer.pars (mb.Module "RNN") {
            RecurrentLayer.defaultHyperPars with
                NInput                  = nWords
                NRecurrent              = nRecurrent
                NOutput                 = nWords
                RecurrentActivationFunc = Tanh
                OutputActivationFunc    = SoftMax
                OneHotIndexInput        = false 
        }
        let _, pred = (initial, input) ||> RecurrentLayer.pred rnn
        let loss = -target * log pred |> Expr.mean

        let NBatch, NSteps, NWords, NRecurrent = 2, 2, 2, 2
        let mi = mb.Instantiate (device, Map [nWords, NWords; nRecurrent, NRecurrent])
        mi.InitPars 100

        let rng = System.Random 123
        let vInput = rng.UniformArrayND (-1.0f, 1.0f) [NBatch; NSteps; NWords] |> device.ToDev
        let vInitial = ArrayNDHost.zeros [NBatch; NRecurrent] |> device.ToDev
        let vTarget = rng.UniformArrayND (-1.0f, 1.0f) [NBatch; NSteps; NWords] |> device.ToDev
        let varEnv = VarEnv.ofSeq [input, vInput; initial, vInitial; target, vTarget] 

        DerivCheck.checkExprTree device 1e-4f 1e-1f (varEnv |> mi.Use) (loss |> mi.Use)
        printfn "Done."

    
    //let trainModel (dataset: TrnValTst<WordSeq>) =
    let trainModel (dataset: TrnValTst<WordSeqOneHot>) =
        let mb = ModelBuilder<single> ("Lang")

        let nBatch     = mb.Size "nBatch"
        let nSteps     = mb.Size "nSteps"
        let nWords     = mb.Size "nWords"
        let nRecurrent = mb.Size "nRecurrent"

        //let input   = mb.Var<int>     "Input"   [nBatch; nSteps]
        let input   = mb.Var<single>  "Input"   [nBatch; nSteps; nWords]
        let initial = mb.Var<single>  "Initial" [nBatch; nRecurrent]
        let target  = mb.Var<single>  "Target"  [nBatch; nSteps; nWords]
        //let target  = mb.Var<int>     "Target"  [nBatch; nSteps]
        
        let rnn = RecurrentLayer.pars (mb.Module "RNN") {
            RecurrentLayer.defaultHyperPars with
                NInput                  = nWords
                NRecurrent              = nRecurrent
                NOutput                 = nWords
                RecurrentActivationFunc = Tanh
                OutputActivationFunc    = SoftMax
                OneHotIndexInput        = false // true
        }

        // final [smpl, recUnit]
        // pred  [smpl, step, word] - probability of word
        let final, pred = (initial, input) ||> RecurrentLayer.pred rnn

        // [smpl, step]
        //let targetProb = pred |> Expr.gather [None; None; Some target]            
        //let stepLoss = -log targetProb 
        let stepLoss = -target * log pred

        let loss = Expr.mean stepLoss

        let dLoss = Deriv.compute loss
        let dLossDInitial = dLoss |> Deriv.ofVar rnn.InputWeights |> Expr.sum
        //printfn "loss:\n%A" loss

        let mi = mb.Instantiate (DevCuda, Map [nWords,     NWords
                                               nRecurrent, NRecurrent])


        let lossFn = mi.Func (loss) |> arg3 initial input target

        let dLossDInitialFn = mi.Func (loss, dLossDInitial) |> arg3 initial input target

        //let smplVarEnv stateOpt (smpl: WordSeq) =
        let smplVarEnv stateOpt (smpl: WordSeqOneHot) =
            let zeroInitial = ArrayNDCuda.zeros<single> [smpl.Words.Shape.[0]; NRecurrent]
            let state =
                match stateOpt with
                | Some state -> state :> IArrayNDT
                | None -> zeroInitial :> IArrayNDT
            //let n = smpl.Words.Shape.[1]
            let n = smpl.Words.Shape.[1]
            //printfn "smpl.Words: %A" smpl.Words.Shape
            VarEnv.ofSeq [input,   smpl.Words.[*, 0 .. n-2, *] :> IArrayNDT
                          target,  smpl.Words.[*, 1 .. n-1, *] :> IArrayNDT
                          initial, state]
//
//            VarEnv.ofSeq [input,   smpl.Words.[*, 0 .. n-2] :> IArrayNDT
//                          target,  smpl.Words.[*, 1 .. n-1] :> IArrayNDT
//                          initial, state]
                          
        let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv GradientDescent.New GradientDescent.DefaultCfg
        //let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv Adam.New Adam.DefaultCfg

        let trainCfg = {
            Train.defaultCfg with
                MinIters  = Some 1000
                MaxIters  = Some 1000
                BatchSize = NBatch
                LearningRates = [1e-4; 1e-5]
                //CheckpointDir = Some "."
                BestOn    = Training
        }
        Train.train trainable dataset trainCfg |> ignore

//        for i=1 to 1 do
//            printfn "Calculating loss:"
//            let lossVal = lossFn zeroInitial dataset.Trn.[0 .. NBatch-1].Words dataset.Trn.[0 .. NBatch-1].Words
//            printfn "loss=%f" (lossVal |> ArrayND.value)

        //let tr = Trace.startSessionWithRng "trc" (Some 900, None) 

//        for smpl in dataset.Trn.Batches NBatch do
//            let zeroInitial = ArrayNDCuda.zeros<single> [smpl.Words.Shape.[0]; NRecurrent]
//            printfn "Calculating and dloss:"
//            //let lossVal, dLossVal = dLossDInitialFn zeroInitial dataset.Trn.[0 .. NBatch-1].Words dataset.Trn.[0 .. NBatch-1].Words
//            let lossVal, dLossVal = dLossDInitialFn zeroInitial smpl.Words smpl.Words
//            printfn "loss=%f   dloss/dInitial=%f" (lossVal |> ArrayND.value) (dLossVal |> ArrayND.value)

        //let ts = tr.End ()
        //ts |> Trace.dumpToFile "trc.txt"



    [<EntryPoint>]
    let main argv = 

        Util.disableCrashDialog ()

        //SymTensor.Compiler.Cuda.Debug.ResourceUsage <- true
        //SymTensor.Compiler.Cuda.Debug.SyncAfterEachCudaCall <- true
        //SymTensor.Debug.VisualizeUExpr <- true

//        let sentences = readData dataPath
//
//        let freqs = sentences |> Seq.concat |> wordFreqs
//        printfn "Found %d unique words." (Map.toSeq freqs |> Seq.length)
//        
//        let freqsSorted = freqs |> Map.toList |> List.sortByDescending snd 
//                          |> List.take (NWords-1)
//        
//        let idForWord = freqsSorted |> Seq.mapi (fun i (word, _) -> word, i) |> Map.ofSeq
//        let wordForId = freqsSorted |> Seq.mapi (fun i (word, _) -> i, word) |> Map.ofSeq
//                        |> Map.add (NWords-1) "UNKNOWN_TOKEN" 
//
//        let tokenizedSentences = sentences |> tokenize idForWord
//        let detokenizedSentences = tokenizedSentences |> detokenize wordForId
//
//        printfn "Using vocabulary of size %d with least common word %A."
//                NWords (List.last freqsSorted)       
        //printfn "%A" (sentences |> Seq.take 10 |> Seq.toList)
        //printfn "%A" (tokenizedSentences |> Seq.take 10 |> Seq.toList)
        //printfn "%A" (detokenizedSentences |> Seq.take 10 |> Seq.toList)

//        // create dataset
//        let dataset = 
//            tokenizedSentences 
//            //|> Seq.take NMaxSamples
//            |> List.concat
//            |> List.chunkBySize NSteps
//            |> List.take NMaxSamples
//            |> List.map (fun smplWords -> {Words = smplWords |> ArrayNDHost.ofList})
//            |> List.filter (fun {Words=words} -> words.Shape = [NSteps])
//            |> Dataset.FromSamples
//            |> TrnValTst.Of
//            |> TrnValTst.ToCuda

        let rng = System.Random 123

        // generate random data
        let dataset : TrnValTst<WordSeqOneHot> =
            Seq.init NMaxSamples (fun _ -> 
                {Words =
                    rng.Seq (0, NWords-1)
                    |> Seq.take NSteps
                    |> Seq.map (fun w -> 
                        ArrayNDHost.initIndexed [1; NWords] (fun p -> if p.[1] = w then 1.0f else 0.0f))
                    |> ArrayND.concat 0})
            |> Dataset.FromSamples
            |> TrnValTst.Of
            |> TrnValTst.ToCuda


//            {Words = ArrayNDHost.arange 5 |> ArrayND.convert}
//            |> Seq.replicate 10
//            |> Dataset.FromSamples
//            |> TrnValTst.Of
//            |> TrnValTst.ToCuda


        verifyRNNGradientOneHot ()

        // train model
        //let res = trainModel dataset


        // shutdown
        Cuda.CudaSup.shutdown ()
        0 





