namespace Datasets

open System.IO
open System.IO.Compression
open System.Net
open FSharp.Data
open FSharp.Data.CsvExtensions
open ArrayNDNS


module private TryParser =
    // convenient, functional TryParse wrappers returning option<'a>
    let tryParseWith tryParseFunc = tryParseFunc >> function
        | true, v    -> Some v
        | false, _   -> None

    let parseDate   = tryParseWith System.DateTime.TryParse
    let parseInt    = tryParseWith System.Int32.TryParse
    let parseSingle = tryParseWith System.Single.TryParse
    let parseDouble = tryParseWith System.Double.TryParse
    // etc.

    // active patterns for try-parsing strings
    let (|Date|_|)   = parseDate
    let (|Int|_|)    = parseInt
    let (|Single|_|) = parseSingle
    let (|Double|_|) = parseDouble


module CsvLoader =
    type private Category = string

    [<StructuredFormatDisplay("{Pretty}")>]
    type private ColumnType =
        | Categorical of Set<Category>
        | Numerical
        | Unknown

        member this.Pretty = 
            match this with
            | Categorical cs -> sprintf "{%s}" (cs |> Set.toList |> String.concat ", ")
            | Numerical -> "Numerical"
            | Unknown -> "Unknown"

    type private RowTypes = ColumnType list

    type IntTreatment =
        | IntAsNumerical
        | IntAsCategorical

    [<StructuredFormatDisplay("{Pretty}")>]
    type private ColumnData =
        | Category of Category
        | Number of float
        | Missing

        member this.Pretty =
            match this with
            | Category c -> c
            | Number n -> sprintf "%g" n
            | Missing -> "Missing"

    type private RowData = ColumnData list

    type CategoryEnconding =
        | OneHot
        | OrderedInt

    type MissingHandling =
        | UseZero
        | SkipRow

    type Parameters = {
        TargetCols:         int list
        IntTreatment:       IntTreatment
        CategoryEncoding:   CategoryEnconding
        Missing:            MissingHandling
        Separators:         string
    }

    let DefaultParameters = {
        TargetCols          = [0]
        IntTreatment        = IntAsCategorical
        CategoryEncoding    = OneHot
        Missing             = SkipRow
        Separators          = " \t,"    
    }

//    type CsvSample = {
//        Input:              ArrayNDT<single>
//        Target:             ArrayNDT<single>
//    }


    let private loadRowTypes (intTreatment: IntTreatment) (csv: CsvFile) : RowTypes =
        csv.Rows
        |> Seq.map (fun row ->
            row.Columns
            |> Array.map (fun col ->
                let col = col.Trim()
                match col, intTreatment with
                | TryParser.Int _, IntAsNumerical -> Numerical
                | TryParser.Int n, IntAsCategorical -> Categorical (Set [n.ToString()])
                | TryParser.Double d, _ -> Numerical
                | _ when col = "?" -> Unknown
                | _ -> Categorical (Set [col]))
            |> Array.toList)
        |> Seq.indexed
        |> Seq.reduce (fun (rowIdx1, r1) (rowIdx2, r2) ->
            let r = 
                List.zip r1 r2
                |> List.indexed
                |> List.map (fun (colIdx, (c1, c2)) ->
                    match c1, c2 with
                    | Categorical c1s, Categorical c2s -> Categorical (Set.union c1s c2s)
                    | Numerical, Numerical -> Numerical
                    | Categorical cs, Numerical | Numerical, Categorical cs when
                        intTreatment = IntAsCategorical && 
                        cs |> Set.forall (function TryParser.Int _ -> true | _ -> false) -> Numerical                        
                    | Unknown, other | other, Unknown -> other
                    | _ -> failwithf "inconsistent column type in row %d, column %d (zero-based): %A and %A" 
                                     rowIdx2 colIdx c1 c2)
            rowIdx2, r)
        |> snd
            


    let private loadData (rowTypes: RowTypes) (csv: CsvFile) : seq<RowData> =
        csv.Rows
        |> Seq.map (fun row ->            
            Seq.zip row.Columns rowTypes
            |> Seq.map (fun (col, typ) ->
                let col = col.Trim()
                match typ with
                | _ when col = "?" -> Missing
                | Categorical _ -> Category col
                | Numerical -> Number (TryParser.parseDouble col).Value
                | Unknown -> failwith "a column had only missing values")
            |> Seq.toList
        )

    let private buildCategoryTables (rowTypes: RowTypes) =
        rowTypes
        |> List.map (fun ct ->
            match ct with
            | Categorical cs ->
                Set.toList cs
                |> List.sort
                |> List.indexed
                |> List.map (fun (idx, cat) -> cat, idx)
                |> Map.ofList
            | _ -> Map.empty)

    let private categoryArrayND (categoryEncoding: CategoryEnconding) categorySet isTarget idx =
        match categoryEncoding with
        | OrderedInt -> idx |> single |> ArrayNDHost.scalar
        | OneHot ->
            match Set.count categorySet with
            | 1 -> ArrayNDHost.zeros<single> [0]
            | 2 when idx=0 && not isTarget -> ArrayNDHost.zeros<single> [1]
            | 2 when idx=1 && not isTarget -> ArrayNDHost.ones<single> [1]
            | _ ->
                let v = ArrayNDHost.zeros<single> [Set.count categorySet]
                v.[[idx]] <- 1.0f
                v        

    let private fieldsToArrayNDs (missing: MissingHandling) (categoryEncoding: CategoryEnconding) 
                                 (rowTypes: RowTypes) (categoryTables: Map<Category, int> list) 
                                 (targetCols: int list) (data: seq<RowData>) =                
        data 
        |> Seq.filter (fun smpl ->
            match missing with
            | UseZero -> true
            | SkipRow ->
                smpl |> List.exists (function Missing -> true | _ -> false) |> not)
        |> Seq.map (fun smpl ->
            List.zip3 smpl rowTypes categoryTables
            |> List.indexed
            |> List.map (fun (colIdx, (col, colType, categoryTable)) ->
                let isTarget = targetCols |> List.contains colIdx
                match col, colType with
                | Category c, Categorical cs -> categoryArrayND categoryEncoding cs isTarget categoryTable.[c]
                | Number n, Numerical -> n |> single |> ArrayNDHost.scalar |> ArrayND.reshape [1]
                | Missing, Categorical cs -> categoryArrayND categoryEncoding cs isTarget 0
                | Missing, Numerical -> ArrayNDHost.zeros<single> [1]
                | _ -> failwithf "data inconsistent with column type: %A for %A" col colType))

    let private toCsvSamples (targetCols: int list) (data: seq<ArrayNDHostT<single> list>) =
        data 
        |> Seq.map (fun smpl ->
            let targets, inputs =
                List.indexed smpl
                |> List.partition (fun (idx, _) -> targetCols |> List.contains idx)
            let targets = targets |> List.map snd |> List.map ArrayND.atLeast1D
            let inputs = inputs |> List.map snd |> List.map ArrayND.atLeast1D
            {Input=ArrayND.concat 0 inputs; Target=ArrayND.concat 0  targets}
        )
    

    let printInfo (pars: Parameters) (rowTypes: RowTypes) (data: RowData seq) =
        let printField isHead s = 
            let fieldLen = 17
            let pad = if isHead then "-" else " " 
            let ps =
                if String.length s > fieldLen then 
                    s.[0 .. fieldLen-1]
                else
                    let padLeft = (fieldLen - String.length s) / 2
                    let padRight = fieldLen - padLeft - String.length s
                    String.replicate padLeft pad + s + String.replicate padRight pad
            printf "%s|" ps
        let fieldPrintf format = Printf.kprintf (printField false) format
        let headPrintf format = Printf.kprintf (printField true) format

        printfn "CSV dataset information:"
        for i, rt in Seq.indexed rowTypes do
            if pars.TargetCols |> List.contains i then
                printfn "Target %3d: %A" i rt
            else
                printfn "Column %3d: %A" i rt
        for i, _ in List.indexed rowTypes do
            if pars.TargetCols |> List.contains i then
                headPrintf "*%d*" i
            else
                headPrintf "%d" i
        printfn ""
        for row in Seq.take 5 data do
            for d in row do
                fieldPrintf "%A" d
            printfn ""
        printfn ""

    let loadFromReader (pars: Parameters) (reader: unit -> TextReader) = 
        let csv () = CsvFile.Load(reader(), separators=pars.Separators, quote='"', hasHeaders=false)
        let rowTypes = loadRowTypes pars.IntTreatment (csv ())
        let categoryTables = buildCategoryTables rowTypes
        let data = loadData rowTypes (csv ()) 
        let dataArrays = fieldsToArrayNDs pars.Missing pars.CategoryEncoding rowTypes categoryTables pars.TargetCols data 
        let samples = toCsvSamples pars.TargetCols dataArrays 
        printInfo pars rowTypes data
        samples

    let loadTextFile (pars: Parameters) path =
        loadFromReader pars (fun () -> File.OpenText path :> TextReader)

    let loadGzipFile (pars: Parameters) path =
        loadFromReader pars (fun () ->
            let gzStream = File.OpenRead path
            let stream = new GZipStream (gzStream, CompressionMode.Decompress)
            new StreamReader (stream) :> TextReader)

    let loadFile (pars: Parameters) path =
        if Path.GetExtension(path).EndsWith(".gz") then
            loadGzipFile pars path
        else
            loadTextFile pars path

    let loadURI (pars: Parameters) (path: string) =
        if path.StartsWith("http://") || path.StartsWith("https://") || path.StartsWith("ftp://") then
            use wc = new WebClient()
            let tmpPath = Path.GetTempFileName()
            wc.DownloadFile (path, tmpPath)
            if path.EndsWith(".gz") then loadGzipFile pars tmpPath
            else loadTextFile pars tmpPath
        else
            loadFile pars path



