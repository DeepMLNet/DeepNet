open System
open System.IO
open System.Text.RegularExpressions

let outfile = File.CreateText "Data/Songs.txt"

let mutable prevEmpty = false

let words = [
    for filename in Directory.GetFiles("Data/Songs", "*.txt") do
        for line in File.ReadLines filename do
            let line = line.ToLower()
            let line = line.Replace("\"", "").Replace("-", "").Replace("!", ".").Replace(",", "").Replace(".", "")
            let line = line.Replace("„", "").Replace("“", "").Replace("…", "").Replace("–", "").Replace("=", "")
            let line = line.Replace(":", "").Replace("refrain", "").Replace("—", "")
            let line = line.Replace("ref","").Replace("courus","").Replace("corus", "").Replace("chorus", "")
            let line = line.Replace(">", "").Replace("<", "").Replace(" x ", "").Replace("&", " ").Replace("#", "")
            let line = Regex.Replace(line, @"(\w)'(\w)", "$1 '$2")
            let line = Regex.Replace(line, @"(\w)'", "$1")
            let line = Regex.Replace(line, @"\[.*\]", "")
            let line = Regex.Replace(line, @"\(.*\)", "")
            let line = Regex.Replace(line, @"\d", "")
            let line = Regex.Replace(line, @" [^ ia] ", " ")
            let line = Regex.Replace(line, @"(\w)\?", "$1 ?")
            let line = Regex.Replace(line, @"^ ", "")
            let line = Regex.Replace(line, @"  ", " ")
            let line = line.Trim()
            if line.Length > 0 then
                yield ">"
                prevEmpty <- false
                for word in line.Split(' ') do
                    yield word
            elif not prevEmpty then
                yield "---"
                prevEmpty <- true
        yield "==="
]

let vocSize = 10000

// count words
let wordCount = words |> List.countBy id 
let vocabulary = wordCount |> List.sortByDescending snd |> List.take (vocSize-1)
let leastWord, leastCount = List.last vocabulary
printfn "Vocabulary is of size %d with least common word %s occuring %d times."
        vocSize leastWord leastCount
printfn "There are %d words not represented in the vocabulary." (wordCount.Length - vocSize)

let usableWords = vocabulary |> List.map fst |> Set.ofList

let skipUnknownLine = true

let mutable lastLine = ""
let mutable line = ""
let mutable hasUnknown = false
for word in words do   
    let word = 
        if usableWords.Contains word then word 
        else hasUnknown <- true; "###"
    if word = ">" || word = "---" || word = "===" then 
        if not (skipUnknownLine && hasUnknown) then
            if lastLine <> line then
                outfile.WriteLine line
            lastLine <- line
        line <- ""
        hasUnknown <- false
    if word = "---" || word = "===" then 
        outfile.WriteLine word    
    else line <- line + word + " "

outfile.Dispose ()
