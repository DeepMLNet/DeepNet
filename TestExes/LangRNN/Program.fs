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
    
    let NWords      = 200
    let NBatch      = 100
    let NSteps      = 35
    let NRecurrent  = 100
    let NMaxBatches = 1000



    type WordSeq = {
        Words:  ArrayNDT<int>
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

    
    let trainModel (dataset: TrnValTst<WordSeq>) =
        let mb = ModelBuilder<single> ("Lang")

        let nBatch     = mb.Size "nBatch"
        let nSteps     = mb.Size "nSteps"
        let nWords     = mb.Size "nWords"
        let nRecurrent = mb.Size "nRecurrent"

        let input   = mb.Var<int>     "Input"   [nBatch; nSteps]
        let initial = mb.Var<single>  "Initial" [nBatch; nRecurrent]
        let target  = mb.Var<int>     "Target"  [nBatch; nSteps]
        
        let rnn = RecurrentLayer.pars (mb.Module "RNN") {
            RecurrentLayer.defaultHyperPars with
                NInput                  = nWords
                NRecurrent              = nRecurrent
                NOutput                 = nWords
                RecurrentActivationFunc = Tanh
                OutputActivationFunc    = SoftMax
                OneHotIndexInput        = true
        }

        // final [smpl, recUnit]
        // pred  [smpl, step, word] - probability of word
        let final, pred = (initial, input) ||> RecurrentLayer.pred rnn

        // [smpl, step]
        let targetProb = pred |> Expr.gather [None; None; Some target]            
        let stepLoss = -log targetProb 
        let loss = Expr.mean stepLoss

        //printfn "loss:\n%A" loss

        let mi = mb.Instantiate (DevCuda, Map [nWords,     NWords
                                               nRecurrent, NRecurrent])

        let lossFn = mi.Func (loss) |> arg3 initial input target

        let zeroInitial = ArrayNDCuda.zeros<single> [NBatch; NRecurrent]
        let smplVarEnv stateOpt (smpl: WordSeq) =
            let state =
                match stateOpt with
                | Some state -> state :> IArrayNDT
                | None -> zeroInitial :> IArrayNDT
            let n = smpl.Words.Shape.[1]
            //printfn "smpl.Words: %A" smpl.Words.Shape
            VarEnv.ofSeq [input,   smpl.Words.[*, 0 .. n-2] :> IArrayNDT
                          target,  smpl.Words.[*, 1 .. n-1] :> IArrayNDT
                          initial, state]
                          
        let trainable = Train.newStatefulTrainable mi [loss] final smplVarEnv Adam.New Adam.DefaultCfg

        let trainCfg = {
            Train.defaultCfg with
                MaxIters  = Some 10
                BatchSize = NBatch
        }
        //Train.train trainable dataset trainCfg

        printfn "Calculating loss:"
        let lossVal = lossFn zeroInitial dataset.Trn.[0 .. NBatch-1].Words dataset.Trn.[0 .. NBatch-1].Words
        printfn "loss=%f" (lossVal |> ArrayND.value)


    [<EntryPoint>]
    let main argv = 

        let sentences = readData dataPath

        let freqs = sentences |> Seq.concat |> wordFreqs
        printfn "Found %d unique words." (Map.toSeq freqs |> Seq.length)
        
        let freqsSorted = freqs |> Map.toList |> List.sortByDescending snd 
                          |> List.take (NWords-1)
        
        let idForWord = freqsSorted |> Seq.mapi (fun i (word, _) -> word, i) |> Map.ofSeq
        let wordForId = freqsSorted |> Seq.mapi (fun i (word, _) -> i, word) |> Map.ofSeq
                        |> Map.add (NWords-1) "UNKNOWN_TOKEN"

        let tokenizedSentences = sentences |> tokenize idForWord
        let detokenizedSentences = tokenizedSentences |> detokenize wordForId

        printfn "Using vocabulary of size %d with least common word %A."
                NWords (List.last freqsSorted)       
        //printfn "%A" (sentences |> Seq.take 10 |> Seq.toList)
        //printfn "%A" (tokenizedSentences |> Seq.take 10 |> Seq.toList)
        //printfn "%A" (detokenizedSentences |> Seq.take 10 |> Seq.toList)

        // create dataset
        let dataset = 
            tokenizedSentences 
            |> List.concat
            |> List.chunkBySize NSteps
            |> List.take NMaxBatches
            |> List.map (fun smplWords -> {Words = smplWords |> ArrayNDHost.ofList})
            |> List.filter (fun {Words=words} -> words.Shape = [NSteps])
            |> Dataset.FromSamples
            |> TrnValTst.Of
            |> TrnValTst.ToCuda

        // train model
        let res = trainModel dataset


        // shutdown
        Cuda.CudaSup.shutdown ()
        0 





