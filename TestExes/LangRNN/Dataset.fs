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


type WordData (dataPath:      string,
               vocSize:       int,
               stepsPerSmpl:  int,
               maxSamples:    int option) =

    do printfn "Reading text %s" dataPath
    let sentences = 
        seq {
            for line in File.ReadLines dataPath do
                let words = line.Split ([|' '|]) |> List.ofArray
                yield words
        } |> Seq.cache
    let words = List.concat sentences
    let wordFreqs =
        let freqs = Dictionary<string, int> ()
        for word in words do
            freqs.[word] <- (freqs.GetOrDefault word 0) + 1
        Map.ofDictionary freqs
    do printfn "Found %d unique words." (Map.toSeq wordFreqs |> Seq.length)
        
    let freqsSorted = wordFreqs |> Map.toList |> List.sortByDescending snd 
                      |> List.take (vocSize-1)       
    let idForWord = freqsSorted |> Seq.mapi (fun i (word, _) -> word, i) |> Map.ofSeq
    let wordForId = freqsSorted |> Seq.mapi (fun i (word, _) -> i, word) |> Map.ofSeq
                    |> Map.add (vocSize-1) "###" 
    let tokenize words =
        let nWords = idForWord |> Map.toSeq |> Seq.length        
        words |> List.map (fun word ->
            match idForWord |> Map.tryFind word with
            | Some id -> id
            | None -> nWords)
    let detokenize tokens =
        tokens |> List.map (fun id -> wordForId.[id])
    do printfn "Using vocabulary of size %d with least common word %A." 
               vocSize (List.last freqsSorted)    

    let dataset = 
        words 
        |> tokenize
        |> List.chunkBySize stepsPerSmpl
        |> List.filter (fun chunk -> chunk.Length = stepsPerSmpl)
        |> fun b -> printfn "Would have %d samples in total." b.Length; b
        |> fun b -> match maxSamples with | Some n -> b |> List.take n | None -> b
        |> fun b -> printfn "Using %d samples with %d steps per sample." b.Length b.Head.Length; b
        |> List.map (fun smplWords -> {Words = smplWords |> ArrayNDHost.ofList})
        |> Dataset.FromSamples
        |> TrnValTst.Of
        |> TrnValTst.ToCuda

    do printfn "%A" dataset

    member this.Dataset = dataset
    member this.VocSize = vocSize

    member this.Random =
        let rng = System.Random 123
        Seq.init maxSamples.Value (fun _ -> 
            {WordSeq.Words = rng.Seq (0, vocSize-1) |> ArrayNDHost.ofSeqWithShape [stepsPerSmpl]})
        |> Dataset.FromSamples
        |> TrnValTst.Of
        |> TrnValTst.ToCuda

    member this.Tokenize words = tokenize words
    member this.Detokenize tokens = detokenize tokens
    member this.ToStr tokens = 
        tokens 
        |> List.ofSeq 
        |> this.Detokenize 
        |> List.map (fun s -> s.Replace("SENTENCE_START", ">>").Replace("SENTENCE_END", "<<"))
        |> String.concat " "
