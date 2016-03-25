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


type Arrayf = ArrayNDT<float>

type CurveSample = {Time: Arrayf; Pos: Arrayf; Vels: Arrayf; Biotac: Arrayf}

let loadData srcDir =
    seq {
        for pageDir in Directory.EnumerateDirectories srcDir do
            for curveDir in Directory.EnumerateDirectories pageDir do
                let filename = curveDir + "/tactile.npz"   
                yield async {
                    use tactile = NPZFile.Open filename
                    return {Time=tactile.Get "time"
                            Pos=tactile.Get "pos"
                            Vels=tactile.Get "vels"
                            Biotac=tactile.Get "biotac"}
                }
    }
    |> Async.Parallel |> Async.RunSynchronously



type CLIArgs =
    | [<Mandatory>] SrcDir of string
with interface IArgParserTemplate with 
        member x.Usage = 
            match x with
            | SrcDir _ -> "source directory"
          
let parser = ArgumentParser.Create<CLIArgs>("Creates a curve dataset.")
let args = parser.Parse(cmdLine, errorHandler=ProcessExiter())

let srcDir = args.GetResult <@ SrcDir @>

let allData = loadData srcDir |> Seq.toList

let dataset = allData |> Dataset.FromSamples |> Dataset.ToCuda


// next step?
// define models

// input data for neural network is computed in python code
// do the same here during dataset loading

dataset.[0..2].Biotac |> ArrayND.shape
// how is the target velocity calculated? where is it stored?

//for smpl in allData do
//    printfn "Time: %A" (ArrayND.shape smpl.Time)
//    printfn "Pos:  %A" (ArrayND.shape smpl.Pos)
//    printfn "Vels: %A" (ArrayND.shape smpl.Vels)
//    printfn "Biotac: %A" (ArrayND.shape smpl.Biotac)
//    printfn ""

// need to make         
//let posChart = Chart.Line (Seq.zip allData.[3].Time.Data allData.[3].Pos.Data) |> Chart.WithTitle "pos" 
//let velChart = Chart.Line allData.[3].Vels.Data |> Chart.WithTitle "vels"
//Chart.Rows [posChart; velChart]
//|> Chart.Save (__SOURCE_DIRECTORY__ + "/chart.pdf")

