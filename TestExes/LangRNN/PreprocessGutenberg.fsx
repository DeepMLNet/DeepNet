open System
open System.IO
open System.Text
open System.Text.RegularExpressions

let knownChars = System.Collections.Generic.Dictionary<char, bool> ()
let isChar c =
    if not (knownChars.ContainsKey c) then
        let isc = string c = Encoding.ASCII.GetString(Encoding.ASCII.GetBytes(string c))
        let isc = isc || c = 'ß' || c = 'ä' || c='ö' || c='ü'
        knownChars.[c] <- isc
    knownChars.[c]
        
let chars = Seq.cache (seq {
    for filename in Directory.GetFiles("Data/Gutenberg", "*.txt") do // |> Seq.take 10 do
        let encoding = 
            if filename.EndsWith("-8.txt") && 
                not (File.Exists (filename.Replace("-8", "-0"))) then Some 28591
            elif filename.EndsWith("-0.txt") then Some 65001
            else None
        printfn "%s %A" filename encoding

        let mutable inText = false
        let mutable lastChar = char 0
        let mutable charReps = 0
        match encoding with
        | Some enc ->   
            for line in File.ReadLines (filename, Encoding.GetEncoding enc) do
                let line = line.ToLower().Trim()
                let line = line.Replace("\009", "").Replace("\022", "").Replace("\029", "").Replace("\\", "/")
                let line = line.Replace("@", "/").Replace("`", "'").Replace("{", "(").Replace("}", ")")
                let line = line.Replace("[", "(").Replace("]", ")").Replace("|", "/").Replace("~", "-")
                let line = line.Replace("_", "").Replace("&", "").Replace("<", "\"").Replace(">", "\"")
                let line = line.Replace("^", "**").Replace("%", "/")
                //printfn "%s" line
                if line.StartsWith "*** start of this project gutenberg ebook" then inText <- true
                if line.StartsWith "*** end of this project gutenberg ebook" then inText <- false
                
                if inText  then
                    let line = line.Replace("„", "\"").Replace("“", "\"").Replace("…", "...").Replace("–", "-")
                    let line = line.Trim()
                    for c in line do
                        if lastChar = c then charReps <- charReps + 1
                        else charReps <- 0
                        lastChar <- c
                        if charReps <= 4 then
                            if isChar c then yield c
                            //else yield '#'
                    yield '\n'
            for char in "\n\nnext file\n\n" do yield char
        | None -> ()
})

// write output
let limit = Some 1000000
let fn, writeChars = 
    match limit with
    Some l -> sprintf "Data/Gutenberg%d.txt" l, chars |> Seq.truncate l
    | None -> sprintf "Data/Gutenberg.txt", chars

// count chars
let charCount = writeChars |> Seq.countBy id |> Seq.toList
let charcabulary = charCount |> List.sortByDescending snd 
printfn "Charcabulary with frequencies:\n%A" charcabulary 

let outfile = File.CreateText fn
for c in writeChars do
    outfile.Write c
outfile.Dispose ()
printfn "Written to %s" fn


