namespace LangRNN


open Basics
open System.IO

open ArrayNDNS
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Optimizers
open Datasets



type WordSeq = {
    Words:  ArrayNDT<int>
}

//type WordSeqOneHot = {
//    Words:  ArrayNDT<single>
//}


module Dataset = 
    let dataPath     = "../../Data/reddit-comments-2015-08-tokenized.txt"
    let StepsPerSmpl = 20
    let VocSize      = 2000
    //let NMaxSamples  = 40000
    //let NMaxSamples  = Some 10000
    let NMaxSamples  = None

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

    let load () =
        let sentences = readData dataPath
        let freqs = sentences |> Seq.concat |> wordFreqs
        printfn "Found %d unique words." (Map.toSeq freqs |> Seq.length)
        
        let freqsSorted = freqs |> Map.toList |> List.sortByDescending snd 
                          |> List.take (VocSize-1)       
        let idForWord = freqsSorted |> Seq.mapi (fun i (word, _) -> word, i) |> Map.ofSeq
        let wordForId = freqsSorted |> Seq.mapi (fun i (word, _) -> i, word) |> Map.ofSeq
                        |> Map.add (VocSize-1) "UNKNOWN_TOKEN" 

        let tokenizedSentences = sentences |> tokenize idForWord |> List.ofSeq
        printfn "Having %d sentences with vocabulary of size %d with least common word %A."
                (tokenizedSentences.Length) VocSize (List.last freqsSorted)       

        tokenizedSentences 
        |> List.concat
        |> List.chunkBySize StepsPerSmpl
        |> List.filter (fun chunk -> chunk.Length = StepsPerSmpl)
        |> fun b -> printfn "Would have %d samples in total." b.Length; b
        |> fun b -> match NMaxSamples with | Some n -> b |> List.take n | None -> b
        |> fun b -> printfn "Using %d samples with %d steps per sample." b.Length b.Head.Length; b
        |> List.map (fun smplWords -> {Words = smplWords |> ArrayNDHost.ofList})
        |> Dataset.FromSamples
        |> TrnValTst.Of
        |> TrnValTst.ToCuda

    let random () =
        let rng = System.Random 123
        Seq.init NMaxSamples.Value (fun _ -> 
            {WordSeq.Words = rng.Seq (0, VocSize-1) |> ArrayNDHost.ofSeqWithShape [StepsPerSmpl]})
        |> Dataset.FromSamples
        |> TrnValTst.Of
        |> TrnValTst.ToCuda

//        let dataset : TrnValTst<WordSeqOneHot> =
//            Seq.init NMaxSamples (fun _ -> 
//                {Words =
//                    rng.Seq (0, NWords-1)
//                    |> Seq.take StepsPerSmpl
//                    |> Seq.map (fun w -> 
//                        ArrayNDHost.initIndexed [1; NWords] (fun p -> if p.[1] = w then 1.0f else 0.0f))
//                    |> ArrayND.concat 0})
//            |> Dataset.FromSamples
//            |> TrnValTst.Of
//            |> TrnValTst.ToCuda


