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

exception UnknownWord of string

type WordData (dataPath:      string,
               vocSizeLimit:  int option,
               stepsPerSmpl:  int,
               maxSamples:    int option) =

    do printfn "Reading text %s" dataPath
    let sentences = 
        seq {
            for line in File.ReadLines dataPath do
                let words = line.Split (' ') |> List.ofArray
                let words = words |> List.filter (fun w -> w.Trim().Length > 0)
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
    let freqsSorted = match vocSizeLimit with Some vs -> freqsSorted |> List.take (vs-1) | None -> freqsSorted
    do printfn "Most common words:\n%A" (freqsSorted |> List.take 10)
    let idForWord = freqsSorted |> Seq.mapi (fun i (word, _) -> word, i) |> Map.ofSeq
    let wordForId = freqsSorted |> Seq.mapi (fun i (word, _) -> i, word) |> Map.ofSeq
    let wordForId = match vocSizeLimit with Some vs -> wordForId |> Map.add (vs-1) "###"  | None -> wordForId
    let tokenize words =
        let nWords = idForWord |> Map.toSeq |> Seq.length        
        words |> List.map (fun word ->
            match idForWord |> Map.tryFind word with
            | Some id -> id
            | None when vocSizeLimit.IsSome -> nWords
            | None -> raise (UnknownWord word))
    let detokenize tokens =
        tokens |> List.map (fun id -> wordForId.[id])
    let vocSize = 
        match vocSizeLimit with
        | Some vs -> vs
        | None -> wordForId.Count
    do printfn "Using vocabulary of size %d (limit: %A) with least common word %A." 
               vocSize vocSizeLimit (List.last freqsSorted)    

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
        |> fun ds -> TrnValTst.Of (ds, trnRatio=0.95, valRatio=0.05, tstRatio=0.0)
        |> TrnValTst.ToCuda

    do printfn "%A" dataset

    member this.Dataset = dataset
    member this.VocSize = vocSize

    member this.Random =
        let rng = System.Random 123
        Seq.init maxSamples.Value (fun _ -> 
            {WordSeq.Words = rng.Seq (0, vocSize-1) |> ArrayNDHost.ofSeqWithShape [stepsPerSmpl]})
        |> Dataset.FromSamples
        |> fun ds -> TrnValTst.Of (ds, trnRatio=0.95, valRatio=0.05, tstRatio=0.0)
        |> TrnValTst.ToCuda

    member this.Tokenize words = tokenize words
    member this.Detokenize tokens = detokenize tokens
    member this.ToStr tokens = 
        tokens 
        |> List.ofSeq 
        |> this.Detokenize 
        |> List.map (function
                     | ">" -> "\n>"
                     | "===" -> "\n==="
                     | "---" -> "\n---"
                     | w -> w)
        |> String.concat " "

    member this.Words = words
    member this.Lines = 
        let mutable rWords = words
        while rWords.Head <> ">" do
            rWords <- rWords.Tail

        seq {
            let mutable line = []       
            for word in rWords do
                if word = ">" then
                    if line.Length > 0 then yield line
                    line <- []
                if word <> "===" && word <> "---" then
                    line <- line @ [word]            
        }
