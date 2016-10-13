namespace LangRNN

open Basics
open System.IO

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Optimizers


module Program =

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

    
    let buildModel () =
        let mb = ModelBuilder ("Lang")

        let nBatch     = mb.Size "nBatch"
        let nSteps     = mb.Size "nSteps"
        let nWords     = mb.Size "nWords"
        let nRecurrent = mb.Size "nRecurrent"

        let input  = mb.Var<int> "Input"  [nBatch; nSteps]
        let target = mb.Var<int> "Target" [nBatch; nSteps]
        
        let rnn = RecurrentLayer.pars (mb.Module "RNN") {
            RecurrentLayer.defaultHyperPars with
                NInput                  = nWords
                NRecurrent              = nRecurrent
                NOutput                 = nWords
                RecurrentActivationFunc = Tanh
                OutputActivationFunc    = SoftMax
                OneHotIndexInput        = true
        }

        // pred [smpl, step, word] - probability of word
        let pred = input |> RecurrentLayer.pred rnn

        // [smpl, step]
        let targetProb = pred |> Expr.gather [None; None; Some target]            
        let stepLoss = -log targetProb 
        let loss = Expr.mean stepLoss

        pred, loss


    [<EntryPoint>]
    let main argv = 
        let dataPath = "../../Data/reddit-comments-2015-08-tokenized.txt"
        let vocabularySize = 8000

        let sentences = readData dataPath

        let freqs = sentences |> Seq.concat |> wordFreqs
        printfn "Found %d unique words." (Map.toSeq freqs |> Seq.length)
        
        let freqsSorted = freqs |> Map.toList |> List.sortByDescending snd 
                          |> List.take (vocabularySize-1)
        
        let idForWord = freqsSorted |> Seq.mapi (fun i (word, _) -> word, i) |> Map.ofSeq
        let wordForId = freqsSorted |> Seq.mapi (fun i (word, _) -> i, word) |> Map.ofSeq
                        |> Map.add (vocabularySize-1) "UNKNOWN_TOKEN"

        let tokenizedSentences = sentences |> tokenize idForWord
        let detokenizedSentences = tokenizedSentences |> detokenize wordForId

        printfn "Using vocabulary of size %d with least common word %A."
            vocabularySize (List.last freqsSorted)       
        //printfn "%A" (sentences |> Seq.take 10 |> Seq.toList)
        //printfn "%A" (tokenizedSentences |> Seq.take 10 |> Seq.toList)
        //printfn "%A" (detokenizedSentences |> Seq.take 10 |> Seq.toList)


        // build model


        0





