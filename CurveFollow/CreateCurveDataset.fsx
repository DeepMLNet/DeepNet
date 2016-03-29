#if IDE
fsi.ShowDeclarationValues <- false
fsi.ShowProperties <- false
let cmdLine = [|"--srcdir"; __SOURCE_DIRECTORY__ + "/../Data/DeepBraille/curv2"|]
#else
let cmdLine = fsi.CommandLineArgs.[1..]
#endif

#load "../DeepNet.fsx"
#load "Data.fs"

open System
open System.IO
open Argu
open FSharp.Charting

open ArrayNDNS
open Datasets
open SymTensor
open SymTensor.Compiler.Cuda
open Optimizers
open Models
open Data


// argument parsing
type CLIArgs =
    | [<Mandatory>] SrcDir of string
with interface IArgParserTemplate with 
        member x.Usage = 
            match x with
            | SrcDir _ -> "source directory"         
let parser = ArgumentParser.Create<CLIArgs>("Creates a curve dataset.")
let args = parser.Parse(cmdLine, errorHandler=ProcessExiter())
let srcDir = args.GetResult <@ SrcDir @>

// load data set
let allData = loadPoints srcDir |> Seq.toList
let dataset = allData |> Dataset.FromSamples |> Dataset.ToCuda


dataset.[0].Biotac.Shape
dataset.[0].OptimalVel.Shape

// next step?
// define models

let batches = dataset.Batches 10
let tmpl = batches () |> Seq.head


// how is the target velocity calculated? where is it stored?

// define model
let mc = ModelBuilder<single> "CurveFollow"
    
// symbolic sizes
let batchSize   = mc.Size "BatchSize"
let nBiotac     = mc.Size "nBiotac"
let nOptimalVel = mc.Size "nOptimalVel"

// model parameters
let pars = NeuralLayer.pars (mc.Module "Layer1") nBiotac nOptimalVel
    
// input / output variables
let biotac     = mc.Var "Biotac"     [nBiotac;     batchSize]
let optimalVel = mc.Var "OptimalVel" [nOptimalVel; batchSize]
let md = mc.ParametersComplete ()

// expressions
let loss = NeuralLayer.loss pars biotac optimalVel |> md.Subst
let dLoss = md.WrtParameters loss

// infer sizes and variable locations from dataset
md.UseTmplVal biotac     tmpl.Biotac
md.UseTmplVal optimalVel tmpl.OptimalVel

printfn "inferred sizes: %A" md.SymSizeEnv
printfn "inferred locations: %A" md.VarLocs

// instantiate model
let mi = md.Instantiate DevCuda

// compile functions
let lossFun = mi.Func (loss) |> arg2 biotac optimalVel
let opt = GradientDescent.minimize {Step=1e-5f} loss md.ParameterSet.Flat   
let optFun = mi.Func opt |> arg2 biotac optimalVel


// calculate test loss on MNIST
for itr, batch in Seq.indexed (batches()) do
    let loss = lossFun batch.Biotac batch.OptimalVel
    printfn "Loss after %d iterations: %A" itr loss
    optFun batch.Biotac batch.OptimalVel |> ignore

    
printfn "Training complete."



// need to make         
//let posChart = Chart.Line (Seq.zip allData.[3].Time.Data allData.[3].Pos.Data) |> Chart.WithTitle "pos" 
//let velChart = Chart.Line allData.[3].Vels.Data |> Chart.WithTitle "vels"
//Chart.Rows [posChart; velChart]
//|> Chart.Save (__SOURCE_DIRECTORY__ + "/chart.pdf")

