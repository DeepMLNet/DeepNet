open System
open System.IO
open System.Text
open System.Text.RegularExpressions
open FSharp.Data
open Newtonsoft.Json
open Tensor.Benchmark
open Argu

[<Measure>] type ms

type Benchmark = {
    Lib: string
    Method: string
    Type: string
    Shape: string
    Dev: string    

    Mean: float<ms>
    StdDev: float<ms>
}

type TensorCsv = CsvProvider<"TensorCsv.csv", ";">

let parseTensorCsv (lib: string) (filename: string) =
    let csv = TensorCsv.Load filename
    csv.Rows |> Seq.map (fun row ->
        {
            Lib = lib
            Method = row.Method.[0..0].ToLower() + row.Method.[1..]
            Type = row.Type
            Shape = row.Shape
            Dev = row.Dev
            Mean = row.``Mean [ms]`` * 1.0<ms>
            StdDev = row.``StdDev [ms]`` * 1.0<ms>
        }
    )

type NumpyJson = JsonProvider<"NumPy.json">

let numpyNames = 
    ["Sign", "sgn"; "Arcsin", "asin"; "Arccos", "acos"; "Arctan", "atan"; "CountTrue", "countTrueAxis";]
    |> Map.ofList

let parseNumpy (lib: string) (filename: string) =
    let mapName (name: string) =
        let name = name.Split ('_') |> Seq.map (fun n -> n.[0..0].ToUpper() + n.[1..]) |> String.concat ""
        match numpyNames |> Map.tryFind name with
        | Some n -> n
        | _ -> name
    let json = NumpyJson.Load filename
    json.Benchmarks |> Seq.map (fun row ->
        {
            Lib = lib
            Method = (Regex.Match (row.Name, @"test_(.+)\[")).Groups.[1].Value |> mapName
            Type = row.Params.Typ |> Option.defaultValue "bool"
            Shape = row.Params.Shape
            Dev = "Host"
            Mean = row.Stats.Mean * 1000.0<ms>
            StdDev = row.Stats.Stddev * 1000.0<ms>
        }
    )
    
let merge (bm1: seq<Benchmark>) (bm2: seq<Benchmark>) =
    let bm2 =
        bm2 |> Seq.map (fun b2 -> 
            let method = 
                bm1 |> Seq.tryPick (fun b1 -> 
                    if b1.Method.ToLowerInvariant() = b2.Method.ToLowerInvariant() then Some b1.Method 
                    else None)
            match method with
            | Some m -> {b2 with Method=m}
            | None -> b2)
    Seq.concat [bm1; bm2]


let typOrder = ["bool"; "int32"; "int64"; "single"; "double"]
let devOrder = ["Host"; "Cuda"]

let generateHtml (columns: (string * Info) list) (bms: seq<Benchmark>) =
    let sb = StringBuilder()
    let out fmt = Printf.kprintf (fun s -> sb.Append s |> ignore) fmt

    out "<!DOCTYPE html><html><head>"
    out "<title>Tensor benchmarks</title>\n" 
    out "<style>\n"
    out "body {margin: auto auto auto auto; font-family: 'Segoe UI', Tahoma, Helvetica, sans-serif; text-align: center;}\n"
    out "table {margin: 0px auto;}\n"
    out "table, th, td {border-width: 1px; border-color: black; border-collapse: collapse;}\n"
    out "th, td {padding-left: 0.5em; padding-right: 0.5em;}\n"
    out ".lb {border-left-style: solid;}\n"
    out ".rb {border-right-style: solid;}\n"
    out ".tb {border-top-style: solid;}\n"
    out ".txt {text-align: left;}\n"
    out ".num {text-align: right;}\n"
    out ".lib {border-left-style: solid; text-align: center; vertical-align: top; font-size: 108%%;}\n"
    out ".info {text-align: left; border: none; font-weight: normal; vertical-align: top; font-size: 80%%}\n"
    out "th.info {font-weight: bold;}\n"
    out "table.info {width: 230px}\n"
    out ".min {width: 1px; white-space: nowrap; padding-left: 0px;}\n"
    out ".npr {padding-right: 0px;}\n"
    out "</style>\n"
    out "</head><body>\n"

    out "<table>\n"
    out "<tr><th colspan=4></th>\n"
    for id, info in columns do 
        out "<th colspan=2 class=lib><!-- %s -->%s<br/><br/>" id info.Library
        out "<table class=info>\n"
        let writeInfo title value = out "<tr><th class=info>%s:</th><td class=info>%s</td></tr>\n" title value
        writeInfo "Hostname" info.Name
        writeInfo "CPU" info.CPU
        writeInfo "GPU" info.GPU
        writeInfo "OS" info.OS
        writeInfo "Runtime" info.Runtime
        writeInfo "Time" (info.Date.ToString("u"))
        out "</table><br/></th>\n"
    out "</tr>\n"
    out "<tr><th class='txt'>Function</th><th class=txt>Shape</th><th class=txt>Type</th><th class=txt>Device</th>"
    for _ in columns do 
        out "<th class='num lb npr'>Mean</th><th class='num min'>StdDev</th>"
    out "</tr>\n"

    let funcs = bms |> Seq.map (fun bm -> bm.Method) |> Seq.distinct
    for func in funcs do
        out "\n<tr><td colspan=4 class='rb tb'></td>"
        for _ in columns do
            out "<td colspan=2 class='lb tb'></td>"
        out "</tr>\n"
        let idxOf entry lst = lst |> List.tryFindIndex ((=) entry) |> Option.defaultValue 100
        let funcBms = 
            bms 
            |> Seq.filter (fun bm -> bm.Method = func)
            |> Seq.groupBy (fun bm -> bm.Shape, idxOf bm.Type typOrder, idxOf bm.Dev devOrder)
            |> Seq.sortBy fst
        let lastFunc, lastShape, lastTyp  = ref "", ref "", ref ""
        let lbl last current = if !last = current then "" else last := current; current
        for _, libBms in funcBms do
            let bm = Seq.head libBms                    
            out "<tr><th class='txt'>%s</th><td class=txt>%s</td><td class=txt>%s</td><td class=txt>%s</td>" 
                (lbl lastFunc bm.Method) (lbl lastShape bm.Shape) (lbl lastTyp bm.Type) bm.Dev
            for id, _ in columns do
                let mean, std = 
                    match libBms |> Seq.tryFind (fun bm -> bm.Lib = id) with
                    | Some bm -> sprintf "%.02f ms" bm.Mean, sprintf "&plusmn;%.02f ms" bm.StdDev
                    | None -> "&mdash;", "&mdash;"                   
                out "<td class='num lb npr'>%s</td>" mean
                out "<td class='num min'>%s</td>" std
            out "</tr>\n"                            

    out "</table>\n"    
    out "<br/>\n"
    out "Report generated on %s." (DateTime.Now.ToUniversalTime().ToString("R"))
    out "</body></html>\n"
    sb.ToString()


type CLIArguments =
    | Output of output:string
    | [<MainCommand; ExactlyOnce; Last>] Srcs of dirs:string list
with 
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Output _ -> "Output path for generated HTML report."
            | Srcs _ -> "Benchmark directories to include. Supports wildcards."

[<EntryPoint>]
let main argv =
    let parser = ArgumentParser.Create<CLIArguments>()
    let results = 
        try parser.ParseCommandLine()
        with e -> printfn "%s" e.Message ; exit 1

    let outputFile = results.GetResult (Output, defaultValue="report.html") |> Path.GetFullPath
    let srcs = results.GetResult Srcs
    let srcsExpanded = [
        for src in srcs do
            if src.Contains '?' || src.Contains '*' then
                let p = src.LastIndexOfAny([|Path.DirectorySeparatorChar; Path.AltDirectorySeparatorChar|])
                let dir, pattern =
                    if p <> -1 then src.[..p], src.[p+1..]
                    else "", src
                yield! Directory.GetDirectories(dir, pattern)
            else
                yield src
    ]

    let infos, benchmarks =
        srcsExpanded
        |> List.map Path.GetFullPath
        |> List.choose (fun dir ->
            let infoFile = Path.Combine (dir, "Info.json")
            if File.Exists infoFile then
                let info = infoFile |> File.ReadAllText |> JsonConvert.DeserializeObject<Info>                
                let tensorBenchmarksFile = Path.Combine (dir, "TensorBenchmark-report.csv") 
                let numpyBenchmarksFile = Path.Combine (dir, "NumPy.json") 
                if File.Exists tensorBenchmarksFile then
                    let tensorBenchmarks = parseTensorCsv dir tensorBenchmarksFile 
                    printfn "Loaded Tensor benchmark from %s" dir
                    Some ((dir, info), tensorBenchmarks)
                elif File.Exists numpyBenchmarksFile then
                    let numpyBenchmarks = parseNumpy dir numpyBenchmarksFile 
                    printfn "Loaded NumPy benchmark from %s" dir
                    Some ((dir, info), numpyBenchmarks)
                else 
                    printfn "Skipping %s because it contains no benchmark" dir
                    None
            else 
                printfn "Skipping %s because %s does not exist" dir infoFile
                None)
        |> List.unzip

    if List.isEmpty benchmarks then
        printfn "No benchmarks found."
        exit 2

    let benchmarks = 
        benchmarks 
        |> List.reduce merge
        |> Seq.filter (fun bm -> Double.IsFinite (bm.Mean / 1.0<ms>))
        |> Seq.cache

    File.WriteAllText (outputFile, generateHtml infos benchmarks)
    printfn "Wrote report to %s" outputFile
    0 
