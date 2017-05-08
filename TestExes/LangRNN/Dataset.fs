namespace LangRNN


open Basics
open System.IO
open System.Text

open Tensor
open SymTensor
open SymTensor.Compiler.Cuda
open Models
open Optimizers
open Datasets



type WordSeq = {
    Words:  Tensor<int>
}

exception UnknownWord of string

type WordData (dataPath:      string,
               vocSizeLimit:  int option,
               stepsPerSmpl:  int64,
               minSamples:    int64,
               tokenLimit:    int option,
               useChars:      bool) =

    do printfn "Reading text %s" dataPath
    let sentences = 
        seq {
            for line in File.ReadLines (dataPath, Encoding.GetEncoding 65001) do
                if useChars then
                    for ch in line do yield [string ch]
                    yield ["\n"]
                else
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
    do printfn "Found %d unique tokens." (Map.toSeq wordFreqs |> Seq.length)
        
    let freqsSorted = wordFreqs |> Map.toList |> List.sortByDescending snd 
    let freqsSorted = match vocSizeLimit with Some vs -> freqsSorted |> List.take (vs-1) | None -> freqsSorted
    do printfn "Most common tokens:\n%A" (freqsSorted |> List.truncate 60)
    let idForWord = freqsSorted |> Seq.mapi (fun i (word, _) -> word, i+1) |> Map.ofSeq |> Map.add "%" 0
    let wordForId = freqsSorted |> Seq.mapi (fun i (word, _) -> i+1, word) |> Map.ofSeq |> Map.add 0 "%"
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
    do printfn "Using vocab-/charabulary of size %d (limit: %A) with least common token %A." 
               vocSize vocSizeLimit (List.last freqsSorted)    

    let dataset = 
        words 
        |> tokenize
        |> fun t -> printfn "Having %d tokens in total." t.Length; t
        |> fun t -> match tokenLimit with | Some n -> t |> List.truncate n | None -> t
        |> fun t -> printfn "Using %d tokens." t.Length; t
        |> fun t -> Seq.singleton {Words = HostTensor.ofList t}
        |> Dataset.ofSeqSamples
        |> Dataset.cutToMinSamples minSamples
        |> TrnValTst.ofDatasetWithRatios (0.95, 0.04, 0.01)
        |> TrnValTst.toCuda

    do printfn "%A" dataset

    member this.Dataset = dataset
    member this.VocSize = vocSize

    member this.Random =
        let rng = System.Random 123
        Seq.init tokenLimit.Value (fun _ -> 
            {WordSeq.Words = rng.Seq (0, vocSize-1) |> HostTensor.ofSeqWithShape [stepsPerSmpl]})
        |> Dataset.ofSeqSamples
        |> TrnValTst.ofDatasetWithRatios (0.90, 0.05, 0.05)
        |> TrnValTst.toCuda

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
        |> String.concat (if useChars then "" else " ")

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

    member this.UseChars = useChars

