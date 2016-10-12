namespace LangRNN

open Basics
open System.IO


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




        0





