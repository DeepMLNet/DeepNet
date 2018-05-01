open System
open System.IO
open System.Text.RegularExpressions
open FSharp.Data
open System.Text

[<Measure>] type ms

type MachineInfo = JsonProvider<"MachineInfo.json">

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

let generate (bms: seq<Benchmark>) =
    let sb = StringBuilder()
    let out fmt = Printf.kprintf (fun s -> sb.Append s |> ignore) fmt
    let libs = bms |> Seq.map (fun bm -> bm.Lib) |> Seq.distinct
    out "<table style=\"broder: none; border-collapse: collapse;\">\n"
    out "<tr><th style=\"text-align: left; border-left: solid; border-width: 1px;\" colspan=4></th>"
    for lib in libs do out "<th colspan=2 style=\"border-left: solid; border-right: solid; border-width: 1px; text-align: center; padding-left: 1em; padding-right: 1em\">%s</th>" lib
    out "</tr>\n"
    out "<tr><th style=\"text-align: left; border-left: solid; border-width: 1px; padding-left: 1em\">Method</th><th style=\"text-align: left; padding-left: 1em\">Shape</th><th>Type</th><th style=\"text-align: left; padding-left: 1em; padding-right:1em;\">Device</th>"
    for _ in libs do out "<th style=\"text-align:right; border-left: solid; border-width: 1px;\">Mean</th><th style=\"text-align:right; border-right: solid; border-width: 1px; padding-right: 1em;\">StdDev</th>"
    out "</tr>\n"

    let funcs = bms |> Seq.map (fun bm -> bm.Method) |> Seq.distinct
    for func in funcs do
        out "\n<tr><td colspan=%d style=\"border-top: solid; border-width: 1px\"></td></tr>\n" (4 + 2 * Seq.length libs)
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
            out "<tr><th style=\"text-align: left; border-left: solid; border-width: 1px; padding-left: 1em\">%s</th><td style=\"text-align: left; padding-left: 1em;\">%s</td><td style=\"text-align: left; padding-left: 1em\">%s</td><td style=\"text-align: left; padding-left: 1em\">%s</td>" 
                (lbl lastFunc bm.Method) (lbl lastShape bm.Shape) (lbl lastTyp bm.Type) bm.Dev
            for lib in libs do
                let mean, std = 
                    match libBms |> Seq.tryFind (fun bm -> bm.Lib = lib) with
                    | Some bm -> sprintf "%.02f ms" bm.Mean, sprintf "%.02f ms" bm.StdDev
                    | None -> "&mdash;", "&mdash;"                   
                out "<td style=\"text-align: right; border-left: solid; border-width: 1px;\">%s</td>" mean
                out "<td style=\"text-align: right; border-right: solid; border-width: 1px; padding-right: 1em\">%s</td>" std
            out "</tr>\n"                            

    out "</table>\n"
    sb.ToString()


let generateHeader (mi: MachineInfo.Root) =
    let sb = StringBuilder()
    let out fmt = Printf.kprintf (fun s -> sb.Append s |> ignore) fmt
    out "<!DOCTYPE html><html><head>"
    out "<title>Tensor benchmarks on %s</title>\n" mi.Name
    //out "<style>table, th, td { border: 1px solid black; border-collapse: collapse; };</style>\n"
    //out "<style>td.num {text-align: right;}</style>\n"
    out "</head><body>\n"
    out "<h1>Tensor benchmarks on %s</h1>\n" mi.Name
    out "<table style=\"text-align:left\">\n"
    out "<tr><th>Hostname</th><td>%s</td></tr>\n" mi.Name
    out "<tr><th>OS</th><td>%s</td></tr>\n" mi.Os
    out "<tr><th>CPU</th><td>%s</td></tr>\n" mi.Cpu
    out "<tr><th>GPU</th><td>%s</td></tr>\n" mi.Gpu
    out "</table>\n"
    out "<br/>\n"
    sb.ToString()

let generateFooter () =
    "</body></html>\n"

[<EntryPoint>]
let main [|dir|] =
    let machineInfoFile = Path.Combine (dir, "MachineInfo.json") |> Path.GetFullPath
    let machineInfo = MachineInfo.Load machineInfoFile

    let tensorBenchmarksFile = Path.Combine (dir, "TensorBenchmark-report.csv") |> Path.GetFullPath
    let tensorBenchmarks = 
        if File.Exists tensorBenchmarksFile then Some (parseTensorCsv machineInfo.Tensor tensorBenchmarksFile) else None

    let numpyBenchmarksFile = Path.Combine (dir, "NumPy.json") |> Path.GetFullPath
    let numpyBenchmarks = 
        if File.Exists numpyBenchmarksFile then Some (parseNumpy machineInfo.NumPy numpyBenchmarksFile) else None

    let allBenchmarks =
        match tensorBenchmarks, numpyBenchmarks with
        | Some tb, Some nb -> merge tb nb
        | Some tb, None -> tb
        | None, Some nb -> nb
        | None, None -> failwith "no benchmarks"
        |> Seq.filter (fun bm -> Double.IsFinite (bm.Mean / 1.0<ms>))
        |> Seq.cache

    let htmlFile = Path.Combine (dir, "report.html") |> Path.GetFullPath
    let header, footer = generateHeader machineInfo, generateFooter ()
    let benchHtml = generate allBenchmarks
    File.WriteAllText (htmlFile, header + benchHtml + footer)

    0 
