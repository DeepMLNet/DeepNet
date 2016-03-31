open Microsoft.VisualStudio.Profiler
open System
open System.IO
open System.Diagnostics
open Argu
open FSharp.Charting

open ArrayNDNS
open Datasets
open SymTensor
open SymTensor.Compiler.Cuda
open Optimizers
open Models
open Data



/// command line arguments
type CLIArgs =
    | [<Mandatory>] SrcDir of string
    | NoCache 
with interface IArgParserTemplate with 
        member x.Usage = 
            match x with
            | SrcDir _ -> "source directory"         
            | NoCache -> "disables loading a Dataset.h5 cache file"





[<EntryPoint>]
let main argv = 
    DataCollection.StopProfile (ProfileLevel.Global, DataCollection.CurrentId) |> ignore

    let parser = ArgumentParser.Create<CLIArgs>("Curve following")
    let args = parser.Parse(errorHandler=ProcessExiter())
    let srcDir = args.GetResult <@ SrcDir @>
    let noCache = args.Contains <@ NoCache @>

    // load data set
    let sw = Stopwatch.StartNew()
    let cache = srcDir + "/Dataset.h5"
    let dataset : Dataset<TactilePoint> = 
        if File.Exists cache && not noCache then
            Dataset.Load cache
        else
            let dataset = loadPoints srcDir |> Dataset.FromSamples 
            dataset.Save cache
            dataset
        |> Dataset.ToCuda
    printfn "Dataset %A loaded in %A" dataset sw.Elapsed   
    let ds = TrnValTst.Of dataset

    // training minibatch generation
    let trnBatches = ds.Trn.Batches 10000
    let tmpl = trnBatches () |> Seq.head
    let tstBatch = trnBatches() |> Seq.head

    // define model
    let mc = ModelBuilder<single> "CurveFollow"
    
    // symbolic sizes
    let batchSize   = mc.Size "BatchSize"
    let nBiotac     = mc.Size "nBiotac"
    let nOptimalVel = mc.Size "nOptimalVel"
    let nHidden     = mc.Size "nHidden"

    // model parameters
    let pars = MLP.pars mc {
        MLP.Layers = 
            [ { NInput=nBiotac; NOutput=nHidden;     TransferFunc=NeuralLayer.Tanh }
              { NInput=nHidden; NOutput=nOptimalVel; TransferFunc=NeuralLayer.Identity } ]
        MLP.LossMeasure = LossLayer.MSE
    }
    
    // input / output variables
    let biotac     = mc.Var "Biotac"     [batchSize; nBiotac]
    let optimalVel = mc.Var "OptimalVel" [batchSize; nOptimalVel]

    // expressions
    let loss = MLP.loss pars biotac.T optimalVel.T

    // infer sizes and variable locations from dataset
    mc.UseTmplVal biotac     tmpl.Biotac
    mc.UseTmplVal optimalVel tmpl.OptimalVel
    mc.SetSize    nHidden    100
    //printfn "inferred sizes: %A" md.SymSizeEnv
    //printfn "inferred locations: %A" md.VarLocs

    // instantiate model
    let mi = mc.Instantiate DevCuda
    printfn "Number of parameters in model: %d" (ArrayND.nElems mi.ParameterValues)

    // initialize parameters
    let rng = Random (10)
    let initPars : ArrayNDHostT<single> = 
        ArrayNDHost.zeros mi.ParameterValues.Shape
        |> ArrayND.map (fun _ -> rng.NextDouble() - 0.5 |> single)    
    mi.ParameterValues.[Fill] <- initPars |> ArrayNDCuda.toDev   

    // compile functions
    let lossFun = mi.Func (loss) |> arg2 biotac optimalVel
    let opt = GradientDescent.minimize {Step=1e-5f} loss mi.ParameterVector
    let optFun = mi.Func opt |> arg2 biotac optimalVel

    // train
    let iters = 100
    printfn "Training for %d iterations..." iters
    for itr = 0 to iters do
        let loss = lossFun tstBatch.Biotac tstBatch.OptimalVel
        printfn "Loss after %d iterations: %A" itr loss

        for trnBatch in trnBatches() do
            optFun trnBatch.Biotac trnBatch.OptimalVel |> ignore  
    printfn "Training complete."


    // need to make         
    //let posChart = Chart.Line (Seq.zip allData.[3].Time.Data allData.[3].Pos.Data) |> Chart.WithTitle "pos" 
    //let velChart = Chart.Line allData.[3].Vels.Data |> Chart.WithTitle "vels"
    //Chart.Rows [posChart; velChart]
    //|> Chart.Save (__SOURCE_DIRECTORY__ + "/chart.pdf")


    Basics.Cuda.CudaSup.shutdown ()    
    Async.Sleep 1000 |> Async.RunSynchronously
    0 
