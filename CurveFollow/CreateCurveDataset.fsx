#if IDE
fsi.ShowDeclarationValues <- false
fsi.ShowProperties <- false
let cmdLine = [|"--srcdir"; __SOURCE_DIRECTORY__ + "/../Data/DeepBraille/curv2"|]
#else
let cmdLine = fsi.CommandLineArgs.[1..]
#endif

#load "../DeepNet.fsx"

open System
open System.IO
open Argu
open FSharp.Charting

open ArrayNDNS
open Datasets


type Arrayf = ArrayNDHostT<float>


type CLIArgs =
    | [<Mandatory>] SrcDir of string
with interface IArgParserTemplate with 
        member x.Usage = 
            match x with
            | SrcDir _ -> "source directory"
          
let parser = ArgumentParser.Create<CLIArgs>("Creates a curve dataset.")
let args = parser.Parse(cmdLine, errorHandler=ProcessExiter())

let srcDir = args.GetResult <@ SrcDir @>

type CurveSample = {Time: Arrayf; Pos: Arrayf; Vels: Arrayf; Biotac: Arrayf}

let loadData srcDir =
    let files = seq {
        for pageDir in Directory.EnumerateDirectories srcDir do
            for curveDir in Directory.EnumerateDirectories pageDir do
                yield curveDir + "/tactile.npz"   
    }
    let loadFile filename = async {
        use tactile = NPZFile.Open filename
        return {Time=tactile.Get "time"
                Pos=tactile.Get "pos"
                Vels=tactile.Get "vels"
                Biotac=tactile.Get "biotac"}
    }
    files
    |> Seq.map loadFile
    |> Async.Parallel
    |> Async.RunSynchronously

let allData = loadData srcDir 

let dataset = Dataset.FromSamples allData



//for smpl in allData do
//    printfn "Time: %A" (ArrayND.shape smpl.Time)
//    printfn "Pos:  %A" (ArrayND.shape smpl.Pos)
//    printfn "Vels: %A" (ArrayND.shape smpl.Vels)
//    printfn "Biotac: %A" (ArrayND.shape smpl.Biotac)
//    printfn ""

// need to make         
let posChart = Chart.Line (Seq.zip allData.[3].Time.Data allData.[3].Pos.Data) |> Chart.WithTitle "pos" 
let velChart = Chart.Line allData.[3].Vels.Data |> Chart.WithTitle "vels"
Chart.Rows [posChart; velChart]
//|> Chart.Save (__SOURCE_DIRECTORY__ + "/chart.pdf")

